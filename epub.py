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
@dataclass(frozen=True)
class LangSpec:
    """Vocabulary needed to recognise a spoken chapter heading in one language."""
    numbered: str      # words that take a number: chapter, part, book...
    standalone: str    # words that are a heading by themselves: prologue...
    counters: str      # ordinal/cardinal stems: first, one, перв, erste...
    suffix: str        # inflection allowed after a counter stem
    toc_title: str
    intro_title: str
    chapter_label: str


LANGUAGES: dict[str, LangSpec] = {
    "ru": LangSpec(
        numbered="глава|часть|ступень|раздел|книга|том|урок|лекция",
        standalone=("введение|вступление|предисловие|пролог|эпилог|"
                    "заключение|послесловие|приложение|оглавление"),
        counters=("перв|втор|трет|четв[её]рт|пят|шест|седьм|восьм|девят|десят|"
                  "одиннадцат|двенадцат|тринадцат|четырнадцат|пятнадцат|"
                  "шестнадцат|семнадцат|восемнадцат|девятнадцат|двадцат|"
                  "тридцат|сороков"),
        # Russian ordinals inflect heavily (третья, четвёртой, двадцатыми), so
        # the stem is matched and a short ending is allowed to follow.
        suffix=r"[а-яё]{0,4}",
        toc_title="Оглавление", intro_title="Начало", chapter_label="Глава"),

    "uk": LangSpec(
        numbered="глава|розділ|частина|книга|том|урок|лекція",
        standalone=("вступ|пролог|епілог|висновок|висновки|передмова|"
                    "післямова|додаток|зміст"),
        counters=("перш|друг|трет|четверт|п'ят|шост|сьом|восьм|дев'ят|десят|"
                  "одинадцят|дванадцят|тринадцят|чотирнадцят|п'ятнадцят|"
                  "двадцят|тридцят"),
        suffix=r"[а-яіїєґ']{0,4}",
        toc_title="Зміст", intro_title="Початок", chapter_label="Розділ"),

    "en": LangSpec(
        numbered="chapter|part|book|volume|section|lesson|act|episode",
        standalone=("introduction|prologue|epilogue|conclusion|preface|"
                    "foreword|afterword|appendix|contents"),
        # English puts cardinals after the noun ("Chapter One") as often as
        # ordinals ("the First Chapter"), so both are listed.
        counters=("one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                  "twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
                  "eighteen|nineteen|twenty|thirty|forty|fifty|"
                  "first|second|third|fourth|fifth|sixth|seventh|eighth|"
                  "ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
                  "fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|"
                  "twentieth|thirtieth"),
        suffix=r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?",
        toc_title="Contents", intro_title="Beginning", chapter_label="Chapter"),

    "es": LangSpec(
        numbered="cap[íi]tulo|parte|libro|volumen|secci[óo]n|lecci[óo]n",
        standalone=("introducci[óo]n|pr[óo]logo|ep[íi]logo|conclusi[óo]n|"
                    "prefacio|ap[ée]ndice|[íi]ndice"),
        counters=("uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|"
                  "doce|trece|catorce|quince|dieciséis|dieciseis|veinte|"
                  "primer|segund|tercer|cuart|quint|sext|s[ée]ptim|octav|"
                  "noven|d[ée]cim"),
        suffix=r"[oa]?s?",
        toc_title="Índice", intro_title="Comienzo", chapter_label="Capítulo"),

    "fr": LangSpec(
        numbered="chapitre|partie|livre|volume|section|le[çc]on",
        standalone=("introduction|prologue|[ée]pilogue|conclusion|pr[ée]face|"
                    "avant-propos|annexe|sommaire"),
        counters=("un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|"
                  "douze|treize|quatorze|quinze|seize|vingt|"
                  "premi[èe]r|deuxi[èe]m|second|troisi[èe]m|quatri[èe]m|"
                  "cinqui[èe]m|sixi[èe]m|septi[èe]m|huiti[èe]m|neuvi[èe]m|"
                  "dixi[èe]m|onzi[èe]m|douzi[èe]m|vingti[èe]m"),
        suffix=r"[e]?s?",
        toc_title="Sommaire", intro_title="Début", chapter_label="Chapitre"),

    "de": LangSpec(
        numbered="kapitel|teil|buch|band|abschnitt|lektion",
        standalone=("einleitung|prolog|epilog|schluss|vorwort|nachwort|"
                    "anhang|inhalt|inhaltsverzeichnis"),
        counters=("eins|zwei|drei|vier|f[üu]nf|sechs|sieben|acht|neun|zehn|"
                  "elf|zw[öo]lf|dreizehn|vierzehn|f[üu]nfzehn|zwanzig|"
                  "erste|zweite|dritte|vierte|f[üu]nfte|sechste|siebte|"
                  "achte|neunte|zehnte"),
        suffix=r"[nsrm]{0,2}",
        toc_title="Inhalt", intro_title="Anfang", chapter_label="Kapitel"),

    "it": LangSpec(
        numbered="capitolo|parte|libro|volume|sezione|lezione",
        standalone=("introduzione|prologo|epilogo|conclusione|prefazione|"
                    "appendice|indice"),
        counters=("uno|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|"
                  "undici|dodici|tredici|quattordici|quindici|venti|"
                  "prim|second|terz|quart|quint|sest|settim|ottav|non|decim"),
        suffix=r"[oaie]?",
        toc_title="Indice", intro_title="Inizio", chapter_label="Capitolo"),

    "pt": LangSpec(
        numbered="cap[íi]tulo|parte|livro|volume|se[çc][ãa]o|li[çc][ãa]o",
        standalone=("introdu[çc][ãa]o|pr[óo]logo|ep[íi]logo|conclus[ãa]o|"
                    "pref[áa]cio|ap[êe]ndice|sum[áa]rio"),
        counters=("um|dois|tr[êe]s|quatro|cinco|seis|sete|oito|nove|dez|onze|"
                  "doze|treze|catorze|quinze|vinte|"
                  "primeir|segund|terceir|quart|quint|sext|s[ée]tim|oitav|"
                  "non|d[ée]cim"),
        suffix=r"[oa]?s?",
        toc_title="Sumário", intro_title="Início", chapter_label="Capítulo"),

    "pl": LangSpec(
        numbered="rozdzia[łl]|cz[ęe][śs][ćc]|ksi[ęe]ga|tom|lekcja",
        standalone=("wst[ęe]p|prolog|epilog|zako[ńn]czenie|przedmowa|"
                    "pos[łl]owie|dodatek|spis"),
        counters=("pierwsz|drug|trzec|czwart|pi[ąa]t|sz[óo]st|si[óo]dm|[óo]sm|"
                  "dziewi[ąa]t|dziesi[ąa]t|jedena[śs]t|dwuna[śs]t|dwudziest"),
        suffix=r"[a-ząćęłńóśźż]{0,3}",
        toc_title="Spis treści", intro_title="Początek", chapter_label="Rozdział"),
}

DEFAULT_LANG = "ru"

# A heading only counts at the start of a sentence -- that single anchor is what
# separates a real heading from prose like "в этой главе мы поговорим...".
_SENTENCE_START = r"(?:^|(?<=[.!?…])\s+|(?<=\n)\s*)"


def build_heading_re(lang: str = DEFAULT_LANG) -> re.Pattern:
    """Compile the heading pattern for one language."""
    spec = LANGUAGES.get(lang) or LANGUAGES[DEFAULT_LANG]
    return re.compile(
        _SENTENCE_START
        + r"(?P<title>(?:"
        # A counter may be digits, Roman numerals, or a spelled-out word.
        + rf"(?:{spec.numbered})\s+"
        + rf"(?:\d{{1,3}}|[IVXLCDM]{{1,7}}|(?:{spec.counters}){spec.suffix})"
        # Standalone words are ordinary nouns too ("the conclusion was clear"),
        # so they only count when they stand alone as a sentence.
        + rf"|(?:{spec.standalone})(?=\s*[.!?…]|\s*$)"
        + r"))\b",
        re.IGNORECASE | re.UNICODE,
    )


_HEADING_RE = build_heading_re(DEFAULT_LANG)


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
                    min_apart_s: float = 30.0,
                    lang: str = DEFAULT_LANG) -> list[Heading]:
    """Find spoken chapter announcements, e.g. 'Глава 1' / 'Chapter Two'."""
    rx = pattern or build_heading_re(lang)
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
                           intro_title: str | None = None,
                           lang: str = DEFAULT_LANG) -> list[Chapter]:
    """Cut at announced headings, titled with what the narrator actually said."""
    spec = LANGUAGES.get(lang) or LANGUAGES[DEFAULT_LANG]
    intro_title = intro_title or spec.intro_title
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
                   label: str | None = None, mode: str = "auto",
                   pattern: re.Pattern | None = None,
                   lang: str = DEFAULT_LANG,
                   on_report=lambda msg: None) -> list[Chapter]:
    """Split into chapters. mode: auto | keywords | pauses.

    'auto' prefers spoken headings and falls back to pauses when the narrator
    does not announce chapters -- which is common in non-fiction.
    """
    if not segments:
        return []

    spec = LANGUAGES.get(lang) or LANGUAGES[DEFAULT_LANG]
    label = label or spec.chapter_label

    if mode in ("auto", "keywords"):
        headings = detect_headings(segments, pattern, lang=lang)

        # A mislabelled language would silently produce zero headings, so when
        # the declared one finds nothing, try the others before giving up.
        if len(headings) < 2 and pattern is None:
            for alt in LANGUAGES:
                if alt == lang:
                    continue
                other = detect_headings(segments, lang=alt)
                if len(other) > len(headings):
                    headings, lang, spec = other, alt, LANGUAGES[alt]
            if len(headings) >= 2:
                on_report(f"headings: matched '{lang}' patterns, not the "
                          "declared language")

        if len(headings) >= 2:
            on_report(f"headings: {len(headings)} spoken chapter markers found")
            return chapters_from_headings(segments, headings, lang=lang)
        if mode == "keywords":
            on_report("headings: none found — writing a single chapter")
            return chapters_from_headings(segments, headings, lang=lang)
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


def _chapter_xhtml(ch: Chapter, lang: str = DEFAULT_LANG) -> str:
    # Built without an f-string expression: escapes inside those are 3.12+ only.
    body = "\n".join(
        '  <p class="first">' + html.escape(p) + "</p>" if i == 0
        else "  <p>" + html.escape(p) + "</p>"
        for i, p in enumerate(ch.paragraphs))
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{lang}" lang="{lang}">\n'
        f'<head><meta charset="utf-8"/><title>{html.escape(ch.title)}</title>\n'
        '<link rel="stylesheet" type="text/css" href="style.css"/></head>\n'
        f'<body>\n  <h1>{html.escape(ch.title)}</h1>\n'
        f'  <p class="ts">{_stamp(ch.start)}</p>\n{body}\n</body>\n</html>\n'
    )


def _toc_xhtml(chapters: list[Chapter], names: list[str], title: str,
               lang: str = DEFAULT_LANG) -> str:
    """A *readable* contents page.

    nav.xhtml drives the reader's own menu but is normally excluded from the
    reading order, so without this the book has no visible contents at all.
    """
    spec = LANGUAGES.get(lang) or LANGUAGES[DEFAULT_LANG]
    rows = "\n".join(
        f'    <li><a href="{n}">{html.escape(c.title)}</a>'
        f'<span class="tocdot"></span>'
        f'<span class="toctime">{_stamp(c.start)}</span></li>'
        for n, c in zip(names, chapters))
    heading = html.escape(spec.toc_title)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{lang}" lang="{lang}">\n'
        f'<head><meta charset="utf-8"/><title>{heading}</title>\n'
        '<link rel="stylesheet" type="text/css" href="style.css"/></head>\n'
        '<body class="tocpage">\n'
        f'  <h1 class="toch1">{heading}</h1>\n'
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
                              pattern=pattern, lang=language, on_report=on_report)
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
        z.writestr("OEBPS/toc.xhtml", _toc_xhtml(chapters, names, title, language))
        z.writestr("OEBPS/toc.ncx", ncx)
        z.writestr("OEBPS/style.css", _CSS)
        for n, c in zip(names, chapters):
            z.writestr(f"OEBPS/{n}", _chapter_xhtml(c, language))
    return out_path


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Build an .epub from a transcript .json produced by engine.py")
    ap.add_argument("json_file", type=Path)
    ap.add_argument("--title", default=None)
    ap.add_argument("--author", default="Расшифровка аудиокниги")
    ap.add_argument("--language", default=DEFAULT_LANG,
                    choices=sorted(LANGUAGES) + ["other"],
                    help="drives both EPUB metadata and heading detection")
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
        if pattern is not None:
            langs = [(args.language, detect_headings(segments, pattern))]
        else:
            langs = [(code, detect_headings(segments, lang=code))
                     for code in LANGUAGES]
            langs.sort(key=lambda p: len(p[1]), reverse=True)
        best = langs[0]
        print(f"{len(best[1])} headings detected using '{best[0]}' patterns:")
        for h in best[1]:
            print(f"  {_stamp(h.time):>8}  {h.title}")
        others = [f"{c}:{len(v)}" for c, v in langs[1:] if v]
        if others:
            print("  other languages: " + ", ".join(others))
        return 0

    title = args.title or args.json_file.stem
    out = args.json_file.with_suffix(".epub")
    chapters = split_chapters(segments, args.chapter_minutes * 60,
                              mode=args.mode, pattern=pattern,
                              lang=args.language, on_report=lambda m: print(m))
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
