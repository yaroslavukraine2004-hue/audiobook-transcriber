"""Cut a slice from a file, transcribe it, report real throughput.

Run this once on a new machine to replace guesswork with a number.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import engine

APP_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--start", default="600", help="seek offset in seconds")
    ap.add_argument("--secs", type=int, default=600, help="slice length")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="ru")
    args = ap.parse_args()

    slice_wav = APP_DIR / "output" / "_bench_slice.wav"
    slice_wav.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cutting {args.secs}s from {args.src.name} at +{args.start}s ...")
    subprocess.run(
        [engine.ffmpeg_bin(), "-y", "-ss", str(args.start), "-t", str(args.secs),
         "-i", str(args.src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(slice_wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    audio = engine.read_wav_slice(slice_wav, 0, args.secs)
    real_secs = len(audio) / engine.SAMPLE_RATE
    print(f"Slice is {real_secs:.0f}s of audio\n")

    t = time.monotonic()
    model, device, compute = engine.load_model(args.model, lambda p: None)
    print(f"\nLoaded {args.model} on {device} ({compute}) in {time.monotonic() - t:.0f}s")

    # Language probe: what is actually in this file?
    seg_iter, info = model.transcribe(audio[:engine.SAMPLE_RATE * 30], beam_size=5)
    list(seg_iter)
    print(f"Detected language: {info.language} (confidence {info.language_probability:.2f})\n")

    print("Transcribing slice ...")
    t0 = time.monotonic()
    seg_iter, _ = model.transcribe(
        audio, language=args.language or None, beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    segments = []
    for s in seg_iter:
        segments.append(s)
        done = s.end
        el = time.monotonic() - t0
        print(f"\r  {done:6.0f}s / {real_secs:.0f}s   {done / max(el, 1e-9):5.2f}x realtime",
              end="", flush=True)

    elapsed = time.monotonic() - t0
    speed = real_secs / elapsed
    print(f"\n\n{'=' * 58}")
    print(f"  {real_secs:.0f}s audio in {elapsed:.0f}s  ->  {speed:.2f}x realtime")
    print(f"  {len(segments)} segments")
    print(f"{'=' * 58}")
    for hours in (1, 5, 10, 30):
        print(f"  {hours:2d} h of audio  ->  ~{hours * 3600 / speed / 60:.0f} min")
    print(f"{'=' * 58}")

    sample = " ".join(s.text.strip() for s in segments[:4])[:300]
    print(f"\nFirst words (sanity check):\n  {sample}\n")

    slice_wav.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
