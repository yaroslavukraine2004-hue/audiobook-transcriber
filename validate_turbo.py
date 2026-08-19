"""Does turbo's accuracy hold up across the book, or only on one easy slice?

A single sample proves nothing -- narration difficulty varies, and a distilled
model tends to fail on the hard parts specifically. This samples several
positions and reports the worst one, not the average.
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import time
from pathlib import Path

import engine

APP_DIR = Path(__file__).resolve().parent


def normalise(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def slice_audio(src: Path, start: float, secs: int):
    wav = APP_DIR / "output" / "_val_slice.wav"
    subprocess.run(
        [engine.ffmpeg_bin(), "-y", "-ss", str(start), "-t", str(secs),
         "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio = engine.read_wav_slice(wav, 0, secs)
    wav.unlink(missing_ok=True)
    return audio


def transcribe(model, audio, beam: int) -> tuple[str, float]:
    t0 = time.monotonic()
    segs, _ = model.transcribe(
        audio, language="ru", beam_size=beam, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False)
    text = " ".join(s.text.strip() for s in segs)
    return text, time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--secs", type=int, default=480)
    ap.add_argument("--points", type=int, default=4)
    args = ap.parse_args()

    duration = engine.probe_duration(args.src)
    starts = [duration * (i + 1) / (args.points + 1) for i in range(args.points)]
    print(f"{args.src.name}: {duration / 3600:.2f} h, sampling {args.points} "
          f"x {args.secs}s\n")

    clips = [(s, slice_audio(args.src, s, args.secs)) for s in starts]

    ref_model, _, _ = engine.load_model("large-v3", lambda p: None)
    refs = []
    for start, audio in clips:
        text, el = transcribe(ref_model, audio, 5)
        refs.append(text)
        print(f"  large-v3       @{start / 3600:5.2f}h  {el:5.0f}s  "
              f"{args.secs / el:5.2f}x  {len(normalise(text))} words")
    del ref_model

    turbo_model, _, _ = engine.load_model("large-v3-turbo", lambda p: None)
    print()
    scores = []
    for (start, audio), ref in zip(clips, refs):
        text, el = transcribe(turbo_model, audio, 5)
        ratio = difflib.SequenceMatcher(None, normalise(ref), normalise(text)).ratio()
        scores.append(ratio)
        print(f"  turbo          @{start / 3600:5.2f}h  {el:5.0f}s  "
              f"{args.secs / el:5.2f}x  {ratio * 100:6.2f}% agreement")

    print(f"\n{'=' * 52}")
    print(f"  mean agreement : {sum(scores) / len(scores) * 100:.2f}%")
    print(f"  worst slice    : {min(scores) * 100:.2f}%")
    print(f"{'=' * 52}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
