"""Build an EPUB 3 from transcript segments. No third-party deps -- an EPUB is a zip.

The interesting part is not the container format, it is deciding where the
chapters go. An audiobook narrator pauses noticeably longer between chapters
than between paragraphs, so the silences the transcript already carries are a
usable table of contents if you pick the threshold adaptively.
"""
from __future__ import annotations

import html
import re
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PARA_GAP_S = 1.6      # pause that reads as a paragraph break
MIN_CHAPTER_S = 900.0  # never cut two chapters closer than 15 min apart
MIN_GAP_S = 1.8        # a shorter pause is never a chapter break

# --------------------------------------------------------------------------
# Spoken heading detection.
#
# When the narrator actually announces "Глава первая", that is an exact chapter
# boundary -- far better than inferring one from a pause. Whisper writes ordinals
# as words about as often as digits, so both forms have to be matched.
# --------------------------------------------------------------------------
_ORDINAL_STEMS = (
    "перв|втор|трет|четв[её]рт|пят|шест|седьм|восьм|девят|десят|"
    "одиннадцат|двенадцат|тринадцат|четырнадцат|пятнадцат|шестнадцат|"
    "семнадцат|восемнадцат|девятнадцат|двадцат|тридцат|сороков"
)
_NUMBERED_KW = "глава|часть|ступень|раздел|книга|том|урок|лекция"
_STANDALONE_KW = (
    "введение|вступление|предисловие|пролог|эпилог|заключение|"
    "послесловие|приложение|оглавление"
)

# A heading only counts at the start of a sentence -- that single anchor is what
# separates a real heading from prose like "в этой главе мы поговорим...".
_SENTENCE_START = r"(?:^|(?<=[.!?…])\s+|(?<=\n)\s*)"
_HEADING_RE = re.compile(
    _SENTENCE_START
    + r"(?P<title>(?:"
    # Ordinals inflect: третья, четвёртой, двадцатыми. Match the stem, then let
    # any short Cyrillic ending follow -- an explicit letter class keeps missing
    # cases like the soft sign in "третья".
    + rf"(?:{_NUMBERED_KW})\s+(?:\d{{1,3}}|(?:{_ORDINAL_STEMS})[а-яё]{{0,4}})"
    # Standalone words are common nouns too ("заключение было очевидным"), so
    # they only count when they stand alone as a sentence.
    + rf"|(?:{_STANDALONE_KW})(?=\s*[.!?…]|\s*$)"
    + r"))\b",
    re.IGNORECASE | re.UNICODE,
)


@dataclass
class Chapter:
    title: str            # clean, e.g. "Глава вторая" -- no timestamp baked in
    start: float
    paragraphs: list[str]

    def display(self) -> str:
        """Title plus timestamp, for navigation entries."""
        return f"{self.title} · {_stamp(self.start)}"

    def words(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)


@dataclass
class Heading:
    index: int      # segment the heading was found in
    time: float     # interpolated timestamp of the heading itself
    title: str


def _stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def detect_headings(segments: list[dict], pattern: re.Pattern | None = None,
                    min_apart_s: float = 30.0) -> list[Heading]:
    """Find spoken chapter announcements, e.g. 'Глава 1' / 'Часть вторая'."""
    rx = pattern or _HEADING_RE
    found: list[Heading] = []
    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue
        for m in rx.finditer(text):
            # Interpolate where inside the segment the phrase falls, so the
            # anchor lands on the heading rather than on the segment start.
            frac = m.start() / max(len(text), 1)
            t = seg["start"] + frac * (seg["end"] - seg["start"])
            title = " ".join(m.group("title").split()).strip(" .,:;—-")
            title = title[:1].upper() + title[1:]
            # Only suppress a genuine repeat. Two *different* headings close
            # together are legitimate -- "Ступень первая. Глава первая."
            if found and t - found[-1].time < min_apart_s \
                    and title.lower() == found[-1].title.lower():
                continue
            found.append(Heading(i, t, title))
    return found


def _paragraphs(block: list[dict]) -> list[str]:
    paras, cur, prev_end = [], [], block[0]["start"]
    for seg in block:
        if cur and seg["start"] - prev_end > PARA_GAP_S:
            paras.append(" ".join(cur).strip())
            cur = []
        cur.append(seg["text"].strip())
        prev_end = seg["end"]
    if cur:
        paras.append(" ".join(cur).strip())
    return [p for p in paras if p]


def chapters_from_headings(segments: list[dict], headings: list[Heading],
                           intro_title: str = "Начало") -> list[Chapter]:
    """Cut at announced headings, titled with what the narrator actually said."""
    bounds = [h.index for h in headings]
    titles = [h.title for h in headings]

    # Anything spoken before the first heading is front matter, not chapter one.
    if not bounds or bounds[0] > 0:
        bounds = [0, *bounds]
        titles = [intro_title, *titles]

    bounds.append(len(segments))
    chapters: list[Chapter] = []
    for title, lo, hi in zip(titles, bounds, bounds[1:]):
        block = segments[lo:hi]
        if not block:
            continue
        chapters.append(Chapter(title, block[0]["start"], _paragraphs(block)))
    return chapters


def split_chapters(segments: list[dict], min_chapter_s: float = MIN_CHAPTER_S,
                   label: str = "Глава", mode: str = "auto",
                   pattern: re.Pattern | None = None,
                   on_report=lambda msg: None) -> list[Chapter]:
    """Split into chapters. mode: auto | keywords | pauses.

    'auto' prefers spoken headings and falls back to pauses when the narrator
    does not announce chapters -- which is common in non-fiction.
    """
    if not segments:
        return []

    if mode in ("auto", "keywords"):
        headings = detect_headings(segments, pattern)
        if len(headings) >= 2:
            on_report(f"headings: {len(headings)} spoken chapter markers found")
            return chapters_from_headings(segments, headings)
        if mode == "keywords":
            on_report("headings: none found — writing a single chapter")
            return chapters_from_headings(segments, headings)
        on_report(f"headings: only {len(headings)} found — falling back to pauses")

    return _split_by_pauses(segments, min_chapter_s, label)


def _split_by_pauses(segments: list[dict], min_chapter_s: float,
                     label: str) -> list[Chapter]:
    """Cut at the longest pauses, keeping chapters at least min_chapter_s apart."""

    gaps = [(segments[i + 1]["start"] - segments[i]["end"], i + 1)
            for i in range(len(segments) - 1)
            if segments[i + 1]["start"] - segments[i]["end"] >= MIN_GAP_S]
    gaps.sort(reverse=True)  # longest silence first -- the strongest candidates

    total = segments[-1]["end"] - segments[0]["start"]
    accepted: list[int] = []
    for _gap, idx in gaps:
        if len(accepted) >= max(1, int(total // min_chapter_s)):
            break
        t = segments[idx]["start"]
        if all(abs(t - segments[a]["start"]) >= min_chapter_s for a in accepted) \
                and t - segments[0]["start"] >= min_chapter_s \
                and segments[-1]["end"] - t >= min_chapter_s:
            accepted.append(idx)

    bounds = [0, *sorted(accepted), len(segments)]
    chapters: list[Chapter] = []
    for n, (lo, hi) in enumerate(zip(bounds, bounds[1:]), 1):
        block = segments[lo:hi]
        if not block:
            continue
        chapters.append(Chapter(f"{label} {n}", block[0]["start"],
                                _paragraphs(block)))
    return chapters


_CSS = """\
body { font-family: Georgia, 'Times New Roman', serif; line-height: 1.6;
       margin: 0 6%; text-align: justify; hyphens: auto; }
h1 { font-size: 1.3em; font-weight: normal; text-align: left;
     margin: 2.2em 0 1.2em; padding-bottom: .35em;
     border-bottom: 1px solid #bbb; page-break-before: always; }
p { text-indent: 1.4em; margin: 0 0 .35em; }
p.first { text-indent: 0; }
p.ts { text-indent: 0; font-size: .8em; color: #888; margin: -.8em 0 1.6em; }

body.tocpage { text-align: left; }
h1.toch1 { page-break-before: avoid; border: 0; margin-bottom: .2em; }
p.tocsub { text-indent: 0; font-size: .9em; color: #777; font-style: italic;
           margin: 0 0 2em; }
ol.toclist { list-style: none; padding: 0; margin: 0; }
ol.toclist li { display: flex; align-items: baseline; margin: 0 0 1em;
                text-indent: 0; }
ol.toclist a { text-decoration: none; color: inherit; }
span.tocdot { flex: 1; border-bottom: 1px dotted #bbb; margin: 0 .5em;
              min-width: 1.5em; }
span.toctime { font-size: .82em; color: #888; white-space: nowrap;
               font-variant-numeric: tabular-nums; }
"""


def _chapter_xhtml(ch: Chapter) -> str:
    # Built without an f-string expression: escapes inside those are 3.12+ only.
    body = "\n".join(
        '  <p class="first">' + html.escape(p) + "</p>" if i == 0
        else "  <p>" + html.escape(p) + "</p>"
        for i, p in enumerate(ch.paragraphs))
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ru" lang="ru">\n'
        f'<head><meta charset="utf-8"/><title>{html.escape(ch.title)}</title>\n'
        '<link rel="stylesheet" type="text/css" href="style.css"/></head>\n'
        f'<body>\n  <h1>{html.escape(ch.title)}</h1>\n'
        f'  <p class="ts">{_stamp(ch.start)}</p>\n{body}\n</body>\n</html>\n'
    )


def _toc_xhtml(chapters: list[Chapter], names: list[str], title: str) -> str:
    """A *readable* contents page.

    nav.xhtml drives the reader's own menu but is normally excluded from the
    reading order, so without this the book has no visible contents at all.
    """
    rows = "\n".join(
        f'    <li><a href="{n}">{html.escape(c.title)}</a>'
        f'<span class="tocdot"></span>'
        f'<span class="toctime">{_stamp(c.start)}</span></li>'
        for n, c in zip(names, chapters))
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ru" lang="ru">\n'
        f'<head><meta charset="utf-8"/><title>Оглавление</title>\n'
        '<link rel="stylesheet" type="text/css" href="style.css"/></head>\n'
        '<body class="tocpage">\n'
        f'  <h1 class="toch1">Оглавление</h1>\n'
        f'  <p class="tocsub">{html.escape(title)}</p>\n'
        f'  <ol class="toclist">\n{rows}\n  </ol>\n'
        '</body>\n</html>\n'
    )


def build_epub(segments: list[dict], out_path: Path, title: str,
               author: str = "Расшифровка аудиокниги",
               language: str = "ru", min_chapter_s: float = MIN_CHAPTER_S,
               mode: str = "auto", pattern: re.Pattern | None = None,
               on_report=lambda msg: None) -> Path:
    chapters = split_chapters(segments, min_chapter_s, mode=mode,
                              pattern=pattern, on_report=on_report)
    if not chapters:
        raise ValueError("no segments to write")

    book_id = f"urn:uuid:{uuid.uuid4()}"
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    names = [f"ch{i:03d}.xhtml" for i in range(1, len(chapters) + 1)]

    manifest = "\n".join(
        f'    <item id="ch{i:03d}" href="{n}" media-type="application/xhtml+xml"/>'
        for i, n in enumerate(names, 1))
    # The contents page leads the reading order, so opening the book lands on it.
    spine = "\n".join(['    <itemref idref="toc"/>']
                      + [f'    <itemref idref="ch{i:03d}"/>'
                         for i in range(1, len(chapters) + 1)])

    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">{book_id}</dc:identifier>
    <dc:title>{html.escape(title)}</dc:title>
    <dc:creator>{html.escape(author)}</dc:creator>
    <dc:language>{language}</dc:language>
    <meta property="dcterms:modified">{modified}</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="toc" href="toc.xhtml" media-type="application/xhtml+xml"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="css" href="style.css" media-type="text/css"/>
{manifest}
  </manifest>
  <spine toc="ncx">
{spine}
  </spine>
</package>
"""

    nav_items = "\n".join(
        f'      <li><a href="{n}">{html.escape(c.display())}</a></li>'
        for n, c in zip(names, chapters))
    nav = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"
      xml:lang="{language}" lang="{language}">
<head><meta charset="utf-8"/><title>Оглавление</title></head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Оглавление</h1>
    <ol>
{nav_items}
    </ol>
  </nav>
  <nav epub:type="landmarks" hidden="hidden">
    <ol>
      <li><a epub:type="toc" href="toc.xhtml">Оглавление</a></li>
      <li><a epub:type="bodymatter" href="{names[0]}">Начало книги</a></li>
    </ol>
  </nav>
</body>
</html>
"""

    # Older readers (and many Android apps) still navigate by NCX, not nav.xhtml.
    ncx_points = "\n".join(
        ['    <navPoint id="np0" playOrder="1">\n'
         '      <navLabel><text>Оглавление</text></navLabel>\n'
         '      <content src="toc.xhtml"/>\n    </navPoint>']
        + [f'    <navPoint id="np{i}" playOrder="{i + 1}">\n'
           f'      <navLabel><text>{html.escape(c.display())}</text></navLabel>\n'
           f'      <content src="{n}"/>\n    </navPoint>'
           for i, (n, c) in enumerate(zip(names, chapters), 1)])
    ncx = f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="{book_id}"/></head>
  <docTitle><text>{html.escape(title)}</text></docTitle>
  <navMap>
{ncx_points}
  </navMap>
</ncx>
"""

    container = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        # The spec requires mimetype to be the first entry AND stored uncompressed.
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container)
        z.writestr("OEBPS/content.opf", opf)
        z.writestr("OEBPS/nav.xhtml", nav)
        z.writestr("OEBPS/toc.xhtml", _toc_xhtml(chapters, names, title))
        z.writestr("OEBPS/toc.ncx", ncx)
        z.writestr("OEBPS/style.css", _CSS)
        for n, c in zip(names, chapters):
            z.writestr(f"OEBPS/{n}", _chapter_xhtml(c))
    return out_path


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Build an .epub from a transcript .json produced by engine.py")
    ap.add_argument("json_file", type=Path)
    ap.add_argument("--title", default=None)
    ap.add_argument("--author", default="Расшифровка аудиокниги")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--chapter-minutes", type=float, default=15.0,
                    help="minimum chapter length for pause mode")
    ap.add_argument("--mode", choices=["auto", "keywords", "pauses"], default="auto",
                    help="auto: spoken headings, falling back to pauses")
    ap.add_argument("--chapter-regex", default=None,
                    help="custom heading pattern; must contain a 'title' group")
    ap.add_argument("--list-headings", action="store_true",
                    help="only print detected headings, write nothing")
    args = ap.parse_args()

    segments = json.loads(args.json_file.read_text(encoding="utf-8"))
    pattern = (re.compile(args.chapter_regex, re.IGNORECASE | re.UNICODE)
               if args.chapter_regex else None)

    if args.list_headings:
        found = detect_headings(segments, pattern)
        print(f"{len(found)} headings detected:")
        for h in found:
            print(f"  {_stamp(h.time):>8}  {h.title}")
        return 0

    title = args.title or args.json_file.stem
    out = args.json_file.with_suffix(".epub")
    chapters = split_chapters(segments, args.chapter_minutes * 60,
                              mode=args.mode, pattern=pattern,
                              on_report=lambda m: print(m))
    build_epub(segments, out, title, args.author, args.language,
               args.chapter_minutes * 60, args.mode, pattern)
    words = sum(c.words() for c in chapters)
    print(f"{out.name}  —  {len(chapters)} chapters, {words:,} words")
    for c in chapters[:12]:
        print(f"  {c.display():<34} {c.words():>7,} words")
    if len(chapters) > 12:
        print(f"  … +{len(chapters) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
