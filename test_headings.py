"""Check heading detection per language, including what must NOT match."""
from __future__ import annotations

import epub

# (language, [phrases that ARE headings], [prose that must NOT match])
CASES = [
    ("ru",
     ["Глава первая.", "Глава 7.", "Часть третья.", "Раздел IV.", "Пролог.",
      "Ступень вторая.", "Заключение."],
     ["В этой главе мы поговорим о другом.", "Заключение было очевидным.",
      "Он прочитал книгу за вечер."]),
    ("uk",
     ["Розділ перший.", "Частина друга.", "Глава 5.", "Вступ.", "Епілог."],
     ["У цьому розділі йдеться про інше.", "Висновок був простий."]),
    ("en",
     ["Chapter One.", "Chapter 12.", "Part Three.", "Book II.", "Prologue.",
      "Chapter Twenty-One.", "Introduction."],
     ["In this chapter we will see.", "The conclusion was obvious.",
      "She wrote a book about it."]),
    ("es",
     ["Capítulo primero.", "Capítulo 9.", "Parte segunda.", "Prólogo.",
      "Introducción."],
     ["En este capítulo veremos.", "La conclusión fue clara."]),
    ("fr",
     ["Chapitre premier.", "Chapitre 4.", "Partie deuxième.", "Prologue.",
      "Introduction."],
     ["Dans ce chapitre nous verrons.", "La conclusion était claire."]),
    ("de",
     ["Kapitel eins.", "Kapitel 3.", "Teil zweite.", "Prolog.", "Einleitung."],
     ["In diesem Kapitel sehen wir.", "Der Schluss war klar."]),
    ("it",
     ["Capitolo primo.", "Capitolo 8.", "Parte seconda.", "Prologo.",
      "Introduzione."],
     ["In questo capitolo vedremo.", "La conclusione era ovvia."]),
    ("pt",
     ["Capítulo primeiro.", "Capítulo 6.", "Parte segunda.", "Prólogo.",
      "Introdução."],
     ["Neste capítulo veremos.", "A conclusão foi clara."]),
    ("pl",
     ["Rozdział pierwszy.", "Rozdział 2.", "Część druga.", "Prolog.", "Wstęp."],
     ["W tym rozdziale zobaczymy.", "Zakończenie było jasne."]),
]


def main() -> int:
    total_fail = 0
    for lang, positives, negatives in CASES:
        rx = epub.build_heading_re(lang)
        misses = [p for p in positives if not rx.search(p)]
        false_hits = [n for n in negatives if rx.search(n)]
        ok = not misses and not false_hits
        total_fail += len(misses) + len(false_hits)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {lang}  {len(positives) - len(misses)}/{len(positives)} "
              f"headings, {len(negatives) - len(false_hits)}/{len(negatives)} "
              f"prose correctly ignored")
        for m in misses:
            print(f"        MISSED      {m!r}")
        for f in false_hits:
            print(f"        FALSE HIT   {f!r}  -> {rx.search(f).group('title')!r}")

    print(f"\n{'ALL PASS' if not total_fail else str(total_fail) + ' FAILURES'}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
