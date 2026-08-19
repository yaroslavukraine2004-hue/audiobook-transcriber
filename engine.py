"""Transcription engine: ffmpeg normalise -> silence-aware chunking -> faster-whisper.

Chunking on silence gives three things a single long decode does not:
fine-grained progress, crash-resume, and a bound on how much work a thermal
stall can throw away.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHUNK_TARGET_S = 1200.0   # aim for 20-minute chunks
CHUNK_SEARCH_S = 120.0    # snap the boundary to any silence within +/- 2 min
SILENCE_DB = -30
SILENCE_MIN_S = 0.4

AUDIO_EXTS = {".mp3", ".m4a", ".m4b", ".wav", ".flac", ".ogg", ".opus",
              ".aac", ".wma", ".mp4", ".mkv", ".webm", ".mov"}


# --------------------------------------------------------------------------
# CUDA runtime DLLs ship as pip packages (nvidia-cublas-cu12 / nvidia-cudnn-cu12).
# CTranslate2 will not find them on Windows unless we register the dirs first.
# --------------------------------------------------------------------------
def add_cuda_dll_dirs() -> None:
    """Make the pip-installed cuBLAS/cuDNN DLLs findable.

    add_dll_directory alone is not enough: CTranslate2 delay-loads cublas64_12
    through the plain OS loader, which consults PATH and ignores the
    add_dll_directory list. So we do both.
    """
    if sys.platform != "win32":
        return
    import site

    roots: list[str] = list(sys.path)
    try:
        roots += list(site.getsitepackages())
        roots.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001 - getsitepackages is absent in some embeds
        pass

    found: list[str] = []
    seen: set[str] = set()
    for root in roots:
        nvidia = Path(root) / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            binv = pkg / "bin"
            key = str(binv).lower()
            if binv.is_dir() and key not in seen:
                seen.add(key)
                found.append(str(binv))
                try:
                    os.add_dll_directory(str(binv))
                except OSError:
                    pass

    if found:
        os.environ["PATH"] = os.pathsep.join(found) + os.pathsep + os.environ.get("PATH", "")


add_cuda_dll_dirs()


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(
        f"{name} not found on PATH. Install it (winget install Gyan.FFmpeg) "
        "and reopen the terminal."
    )


_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


@dataclass
class Progress:
    stage: str = "idle"        # convert | analyse | transcribe | write | done
    fraction: float = 0.0      # 0..1 within the current file
    message: str = ""
    audio_done_s: float = 0.0
    audio_total_s: float = 0.0
    speed: float = 0.0         # x realtime
    eta_s: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Job:
    src: Path
    outdir: Path
    workdir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.workdir = self.outdir / (self.src.stem + ".work")


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


_TIME_RE = re.compile(r"time=(\d+):(\d\d):(\d\d\.\d+)")
_SIL_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END_RE = re.compile(r"silence_end:\s*([\d.]+)")


def normalise_and_scan(src: Path, wav_out: Path, duration: float, report) -> list[float]:
    """One ffmpeg pass: 16 kHz mono WAV + a list of silence midpoints.

    silencedetect is an analysis filter, so it does not alter the samples --
    we get the conversion and the split-point scan for the price of one decode.
    """
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-y", "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}",
        "-c:a", "pcm_s16le", str(wav_out),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW,
    )

    silences: list[float] = []
    pending_start: float | None = None
    assert proc.stderr is not None
    for line in proc.stderr:
        m = _SIL_START_RE.search(line)
        if m:
            pending_start = float(m.group(1))
        m = _SIL_END_RE.search(line)
        if m and pending_start is not None:
            silences.append((pending_start + float(m.group(1))) / 2.0)
            pending_start = None
        m = _TIME_RE.search(line)
        if m and duration > 0:
            done = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            report(Progress("convert", min(done / duration, 1.0),
                            "Converting to 16 kHz mono + scanning for silence"))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src.name} (exit {proc.returncode})")
    return silences


def plan_chunks(duration: float, silences: list[float]) -> list[tuple[float, float]]:
    """Cut at the silence nearest each 20-minute mark; hard-cut if none is close."""
    if duration <= CHUNK_TARGET_S:
        return [(0.0, duration)]

    bounds = [0.0]
    while True:
        target = bounds[-1] + CHUNK_TARGET_S
        if target >= duration - CHUNK_SEARCH_S:
            break
        window = [s for s in silences
                  if abs(s - target) <= CHUNK_SEARCH_S and s > bounds[-1] + 60]
        bounds.append(min(window, key=lambda s: abs(s - target)) if window else target)
    bounds.append(duration)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def read_wav_slice(wav_path: Path, start_s: float, end_s: float) -> np.ndarray:
    """Pull one chunk as float32 mono without loading the whole file."""
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        wf.setpos(max(0, int(start_s * rate)))
        frames = wf.readframes(max(0, int((end_s - start_s) * rate)))
    pcm = np.frombuffer(frames, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0)


def free_vram_mb() -> float:
    """Free VRAM in MB, or nan if it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def load_model(model_name: str, report):
    from faster_whisper import WhisperModel

    # On Windows, oversubscribing VRAM does not raise -- WDDM silently pages to
    # system RAM over PCIe and throughput collapses. Warning beats a silent 3x
    # slowdown discovered hours later.
    need = 2600 if "turbo" not in model_name else 1500
    free = free_vram_mb()
    if free == free and free < need:  # not-nan check
        report(Progress("load", 0.0,
                        f"Only {free:.0f} MB VRAM free, {model_name} wants ~{need} MB "
                        "— close browsers or expect heavy slowdown"))

    # 4 GB of VRAM cannot hold large-v3 in fp16, so int8_float16 is the real
    # target; the rest of the chain is a graceful climb-down, not a preference.
    attempts = [
        ("cuda", "int8_float16"),
        ("cuda", "int8"),
        ("cpu", "int8"),
    ]
    last: Exception | None = None
    for device, compute in attempts:
        try:
            report(Progress("load", 0.0, f"Loading {model_name} on {device} ({compute})"))
            model = WhisperModel(model_name, device=device, compute_type=compute)
            return model, device, compute
        except Exception as exc:  # noqa: BLE001 - want the fallback chain
            last = exc
            report(Progress("load", 0.0, f"{device}/{compute} unavailable: {exc}"))
    raise RuntimeError(f"Could not load model {model_name}: {last}")


def fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(segments: list[Segment], outdir: Path, stem: str,
                  language: str | None = None) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    txt = outdir / f"{stem}.txt"
    srt = outdir / f"{stem}.srt"
    jsn = outdir / f"{stem}.json"

    with txt.open("w", encoding="utf-8") as fh:
        para: list[str] = []
        prev_end = 0.0
        for seg in segments:
            # A pause longer than a beat reads as a paragraph break in prose.
            if para and seg.start - prev_end > 1.6:
                fh.write(" ".join(para).strip() + "\n\n")
                para = []
            para.append(seg.text.strip())
            prev_end = seg.end
        if para:
            fh.write(" ".join(para).strip() + "\n")

    with srt.open("w", encoding="utf-8") as fh:
        for i, seg in enumerate(segments, 1):
            fh.write(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{seg.text.strip()}\n\n")

    payload = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    jsn.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    out = {"txt": txt, "srt": srt, "json": jsn}

    # An .epub is a nice-to-have, so a failure here must not lose the transcript
    # we just spent hours producing.
    try:
        import epub as epub_mod

        out["epub"] = epub_mod.build_epub(
            payload, outdir / f"{stem}.epub", stem,
            language=language or epub_mod.DEFAULT_LANG)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[warn] epub not written: {exc}")

    return out


def transcribe_file(
    src: Path,
    outdir: Path,
    model,
    language: str = "ru",
    report=lambda p: None,
    should_stop=lambda: False,
    keep_wav: bool = False,
    beam_size: int = 5,
) -> dict:
    """Transcribe one file. Safe to re-run: finished chunks are reused."""
    job = Job(src, outdir)
    job.workdir.mkdir(parents=True, exist_ok=True)
    wav = job.workdir / "audio16k.wav"
    plan_path = job.workdir / "plan.json"
    parts_dir = job.workdir / "parts"
    parts_dir.mkdir(exist_ok=True)

    duration = probe_duration(src)

    if plan_path.exists() and wav.exists():
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        chunks = [tuple(c) for c in saved["chunks"]]
        duration = saved.get("duration", duration)
        report(Progress("analyse", 1.0, f"Resuming — {len(chunks)} chunks already planned"))
    else:
        silences = normalise_and_scan(src, wav, duration, report)
        if duration <= 0:
            duration = probe_duration(wav)
        chunks = plan_chunks(duration, silences)
        plan_path.write_text(
            json.dumps({"duration": duration, "chunks": [list(c) for c in chunks]}),
            encoding="utf-8",
        )
        report(Progress("analyse", 1.0,
                        f"{len(chunks)} chunks from {len(silences)} silences"))

    all_segments: list[Segment] = []
    t0 = time.monotonic()
    audio_done = 0.0

    for idx, (start, end) in enumerate(chunks):
        if should_stop():
            report(Progress("stopped", audio_done / max(duration, 1e-9),
                            "Stopped — progress saved, re-run to resume"))
            return {"status": "stopped", "segments": len(all_segments)}

        part = parts_dir / f"{idx:04d}.json"
        if part.exists():
            saved = json.loads(part.read_text(encoding="utf-8"))
            all_segments.extend(Segment(**s) for s in saved)
            audio_done = end
            report(Progress("transcribe", audio_done / max(duration, 1e-9),
                            f"Chunk {idx + 1}/{len(chunks)} already done — skipping",
                            audio_done, duration))
            continue

        audio = read_wav_slice(wav, start, end)
        seg_iter, _info = model.transcribe(
            audio,
            language=language or None,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # Both of these guard the failure that ruins long runs: whisper
            # latching onto its own output and repeating a phrase for hours.
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
        )

        chunk_segments: list[Segment] = []
        for seg in seg_iter:
            if should_stop():
                break
            chunk_segments.append(Segment(start + seg.start, start + seg.end, seg.text))
            audio_done = start + seg.end
            elapsed = time.monotonic() - t0
            speed = (audio_done / elapsed) if elapsed > 1 else 0.0
            eta = ((duration - audio_done) / speed) if speed > 0.05 else None
            report(Progress("transcribe", min(audio_done / max(duration, 1e-9), 1.0),
                            f"Chunk {idx + 1}/{len(chunks)}",
                            audio_done, duration, speed, eta))

        if should_stop():
            report(Progress("stopped", audio_done / max(duration, 1e-9),
                            "Stopped — finished chunks saved, re-run to resume"))
            return {"status": "stopped", "segments": len(all_segments)}

        part.write_text(
            json.dumps([{"start": s.start, "end": s.end, "text": s.text}
                        for s in chunk_segments], ensure_ascii=False),
            encoding="utf-8",
        )
        all_segments.extend(chunk_segments)
        audio_done = end

    report(Progress("write", 1.0, "Writing .txt / .srt / .json", duration, duration))
    paths = write_outputs(all_segments, outdir, src.stem, language)

    if not keep_wav:
        wav.unlink(missing_ok=True)

    elapsed = time.monotonic() - t0
    return {
        "status": "done",
        "segments": len(all_segments),
        "duration": duration,
        "elapsed": elapsed,
        "speed": duration / elapsed if elapsed > 0 else 0.0,
        "outputs": {k: str(v) for k, v in paths.items()},
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Transcribe audio with faster-whisper")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--beam", type=int, default=5,
                    help="1 is ~1.2x faster with near-identical output")
    args = ap.parse_args()

    def report(p: Progress) -> None:
        bar = f"{p.fraction * 100:5.1f}%"
        extra = f" {p.speed:.1f}x" if p.speed else ""
        print(f"\r[{p.stage:10}] {bar}{extra}  {p.message[:60]:<60}", end="", flush=True)

    model, device, compute = load_model(args.model, report)
    print(f"\nModel {args.model} on {device} ({compute})\n")

    for f in args.files:
        print(f"\n=== {f.name} ===")
        res = transcribe_file(f, args.out, model, args.language, report,
                              beam_size=args.beam)
        print(f"\n{res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
