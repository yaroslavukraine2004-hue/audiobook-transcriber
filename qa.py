"""Sanity-check a finished transcript: encoding, coverage, hallucination loops."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main(stem_path: Path) -> int:
    segs = json.loads(stem_path.with_suffix(".json").read_text(encoding="utf-8"))
    txt = stem_path.with_suffix(".txt").read_text(encoding="utf-8")

    words = txt.split()
    cyr = sum(1 for ch in txt if "Ѐ" <= ch <= "ӿ")
    total_alpha = sum(1 for ch in txt if ch.isalpha())

    print(f"segments        : {len(segs)}")
    print(f"words           : {len(words):,}")
    print(f"characters      : {len(txt):,}")
    print(f"paragraphs      : {txt.count(chr(10) + chr(10)) + 1}")
    print(f"cyrillic share  : {cyr / max(total_alpha, 1) * 100:.1f}%  (should be ~99%)")

    covered = sum(s["end"] - s["start"] for s in segs)
    span = segs[-1]["end"] - segs[0]["start"] if segs else 0
    print(f"speech covered  : {covered / 60:.0f} min over a {span / 3600:.2f} h span")

    # Hallucination loop check: whisper's classic long-file failure is emitting
    # the same line over and over. Consecutive repeats are the signal.
    texts = [s["text"].strip() for s in segs]
    runs, best, cur = [], 1, 1
    for a, b in zip(texts, texts[1:]):
        cur = cur + 1 if a == b and a else 1
        best = max(best, cur)
        if cur > 2:
            runs.append(a)
    print(f"longest repeat  : {best} consecutive identical segments"
          f"  {'<- OK' if best <= 2 else '<- INSPECT'}")

    dupes = [t for t, n in Counter(texts).most_common(5) if n > 3 and t]
    print(f"repeated lines  : {len(dupes)} distinct lines appearing >3x")
    for d in dupes[:3]:
        print(f"                  {texts.count(d):3d}x  {d[:60]!r}")

    # Timing gaps larger than a minute can mean a chunk silently produced nothing.
    gaps = [(segs[i]["end"], segs[i + 1]["start"] - segs[i]["end"])
            for i in range(len(segs) - 1)
            if segs[i + 1]["start"] - segs[i]["end"] > 60]
    print(f"gaps > 60s      : {len(gaps)}")
    for at, g in gaps[:5]:
        print(f"                  {g:5.0f}s of silence at {at / 3600:.2f} h")

    print(f"\nfirst line  : {texts[0][:70]}")
    print(f"last line   : {texts[-1][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
