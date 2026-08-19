"""Compare decoding configurations on this GPU and report what actually helps.

Every configuration runs on the same audio slice. Order matters on a laptop:
the GPU heats up, so a config measured last is measured under a handicap. The
baseline is therefore repeated at the end to expose any thermal drift.
"""
from __future__ import annotations

import argparse
import gc
import subprocess
import time
from pathlib import Path

import engine

APP_DIR = Path(__file__).resolve().parent

COMMON = dict(
    language="ru",
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500},
)


def vram_mb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def run_sequential(model, audio, beam: int) -> tuple[float, int]:
    t0 = time.monotonic()
    segs, _ = model.transcribe(audio, beam_size=beam,
                               condition_on_previous_text=False, **COMMON)
    n = sum(1 for _ in segs)
    return time.monotonic() - t0, n


def run_batched(model, audio, beam: int, batch: int) -> tuple[float, int]:
    from faster_whisper import BatchedInferencePipeline

    pipe = BatchedInferencePipeline(model=model)
    t0 = time.monotonic()
    segs, _ = pipe.transcribe(audio, beam_size=beam, batch_size=batch, **COMMON)
    n = sum(1 for _ in segs)
    return time.monotonic() - t0, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--start", default="3600")
    ap.add_argument("--secs", type=int, default=300)
    ap.add_argument("--model", default="large-v3")
    args = ap.parse_args()

    wav = APP_DIR / "output" / "_speed_slice.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [engine.ffmpeg_bin(), "-y", "-ss", str(args.start), "-t", str(args.secs),
         "-i", str(args.src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio = engine.read_wav_slice(wav, 0, args.secs)
    real = len(audio) / engine.SAMPLE_RATE

    base_vram = vram_mb()
    model, device, compute = engine.load_model(args.model, lambda p: None)
    print(f"\n{args.model} on {device} ({compute}); slice = {real:.0f}s")
    print(f"VRAM before {base_vram:.0f} MB, after model load {vram_mb():.0f} MB\n")

    configs = [
        ("sequential  beam=5", lambda: run_sequential(model, audio, 5)),
        ("sequential  beam=1", lambda: run_sequential(model, audio, 1)),
        ("batched=4   beam=5", lambda: run_batched(model, audio, 5, 4)),
        ("batched=8   beam=5", lambda: run_batched(model, audio, 5, 8)),
        ("batched=8   beam=1", lambda: run_batched(model, audio, 1, 8)),
        ("batched=16  beam=5", lambda: run_batched(model, audio, 5, 16)),
        ("sequential  beam=5 (repeat)", lambda: run_sequential(model, audio, 5)),
    ]

    print(f"{'config':<28}{'time':>8}{'speed':>9}{'segs':>7}{'VRAM':>9}")
    print("-" * 61)
    results = []
    for name, fn in configs:
        gc.collect()
        try:
            elapsed, n = fn()
            speed = real / elapsed
            print(f"{name:<28}{elapsed:>7.0f}s{speed:>8.2f}x{n:>7}{vram_mb():>8.0f}M")
            results.append((name, speed))
        except Exception as exc:  # noqa: BLE001
            short = str(exc).split("\n")[0][:34]
            print(f"{name:<28}{'FAILED':>8}  {short}")

    if results:
        base = results[0][1]
        best = max(results, key=lambda r: r[1])
        print("-" * 61)
        print(f"baseline {base:.2f}x  ->  best '{best[0].strip()}' {best[1]:.2f}x "
              f"({best[1] / base:.2f}x faster)")

    wav.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
