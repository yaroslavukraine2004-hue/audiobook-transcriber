"""large-v3 vs large-v3-turbo: speed AND accuracy on the same Russian slice.

Speed alone would be a useless comparison -- turbo is only interesting if the
Russian text it produces is close enough to large-v3 to be worth the tradeoff.
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


def run(model_name: str, audio, real: float, beam: int):
    model, device, compute = engine.load_model(model_name, lambda p: None)
    t0 = time.monotonic()
    segs, _ = model.transcribe(
        audio, language="ru", beam_size=beam, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False)
    text = " ".join(s.text.strip() for s in segs)
    elapsed = time.monotonic() - t0
    del model
    return elapsed, real / elapsed, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--start", default="3600")
    ap.add_argument("--secs", type=int, default=300)
    args = ap.parse_args()

    wav = APP_DIR / "output" / "_turbo_slice.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [engine.ffmpeg_bin(), "-y", "-ss", str(args.start), "-t", str(args.secs),
         "-i", str(args.src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio = engine.read_wav_slice(wav, 0, args.secs)
    real = len(audio) / engine.SAMPLE_RATE

    results = {}
    for name, beam in [("large-v3", 5), ("large-v3", 1), ("large-v3-turbo", 5),
                       ("large-v3-turbo", 1)]:
        key = f"{name} beam={beam}"
        try:
            elapsed, speed, text = run(name, audio, real, beam)
            results[key] = (elapsed, speed, text)
            print(f"{key:<26}{elapsed:>6.0f}s{speed:>7.2f}x   {len(normalise(text)):>5} words")
        except Exception as exc:  # noqa: BLE001
            print(f"{key:<26}FAILED  {str(exc).splitlines()[0][:40]}")

    ref_key = "large-v3 beam=5"
    if ref_key in results:
        ref = normalise(results[ref_key][2])
        print(f"\nAgreement vs {ref_key} (reference = {len(ref)} words):")
        for key, (_e, _s, text) in results.items():
            if key == ref_key:
                continue
            hyp = normalise(text)
            ratio = difflib.SequenceMatcher(None, ref, hyp).ratio()
            print(f"  {key:<26}{ratio * 100:6.2f}% word agreement")

        base_speed = results[ref_key][1]
        print(f"\nSpeedups vs {ref_key} ({base_speed:.2f}x):")
        for key, (_e, speed, _t) in results.items():
            if key != ref_key:
                print(f"  {key:<26}{speed / base_speed:5.2f}x faster")

    wav.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
