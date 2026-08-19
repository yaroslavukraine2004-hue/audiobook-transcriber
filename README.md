# Audiobook Transcriber

Turn long audiobooks into text, subtitles, and a proper EPUB — locally, with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Built for
multi-hour files on modest hardware, where the usual advice quietly falls apart.

Drop files on the window, press Start:

```
┌──────────────────────────────────────────────────────────────┐
│  Audiobook Transcriber          large-v3-turbo · cuda · int8 │
│  Model [large-v3-turbo ▾]  Language [ru ▾]  Beam [5 ▾]       │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │            Drop audio files here                         │ │
│ │      mp3 · m4a · m4b · wav · flac · opus · mp4           │ │
│ └──────────────────────────────────────────────────────────┘ │
│  File            Length   Status       Progress  Speed   ETA │
│  book-01.mp3      8:36:42 transcribe        41%  5.2x  1:12  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
└──────────────────────────────────────────────────────────────┘
```

## Why this exists

Running Whisper on a 30-minute podcast is easy. Running it on a nine-hour
audiobook surfaces problems that short-clip tutorials never mention:

- **It repeats itself forever.** Whisper's classic long-file failure is
  latching onto its own output during a quiet passage and emitting the same
  sentence for hours. Handled with VAD filtering and
  `condition_on_previous_text=False`.
- **A crash costs you everything.** Five hours in, a reboot should not mean
  starting over. Work is checkpointed per chunk and resumes automatically.
- **The output is an unreadable wall of text.** 47,000 words in one blob is
  not a book. This produces a real EPUB with chapters, a contents page, and
  timestamps back into the audio.
- **Published benchmarks lie to you.** See [Performance](#performance).

## Features

- **Drag-and-drop GUI** with per-file and overall progress, live speed and ETA
- **Resumable** — a crash, reboot, or Stop costs at most one chunk
- **Silence-aware chunking** — cuts at the nearest pause, never mid-word
- **EPUB with real chapters** — detected from spoken headings ("Глава третья"),
  falling back to pause-based splitting
- **Four output formats** — `.txt`, `.srt`, `.json`, `.epub`
- **Benchmark scripts** so you can measure your own hardware instead of
  trusting anyone's numbers

## Requirements

- Python 3.9+ (developed on 3.11)
- [ffmpeg](https://ffmpeg.org/) on `PATH` — `winget install Gyan.FFmpeg`,
  `brew install ffmpeg`, or `apt install ffmpeg`
- Optional: an NVIDIA GPU. CPU works but is roughly 10x slower.

## Install

```bash
git clone https://github.com/<you>/audiobook-transcriber
cd audiobook-transcriber

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

CPU-only? Drop the two `nvidia-*` lines from `requirements.txt`.

## Usage

**GUI** — `python app.py`, or double-click `run.bat` on Windows.

**CLI:**

```bash
python engine.py "book.mp3" --out output --model large-v3-turbo --language ru
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `large-v3-turbo` | any faster-whisper model name |
| `--language` | `ru` | ISO code, or omit to auto-detect |
| `--beam` | `5` | `1` is ~1.2x faster, near-identical output |
| `--out` | `output` | output directory |

**Rebuild an EPUB** without re-transcribing — the transcription is the
expensive part, so chapter tuning works off the saved `.json`:

```bash
python epub.py "output/book.json" --title "My Book" --chapter-minutes 10
python epub.py "output/book.json" --list-headings   # preview detection
```

## Output

| File | Contents |
|---|---|
| `<name>.txt` | Plain prose, paragraph breaks at long pauses |
| `<name>.srt` | Subtitles with timestamps |
| `<name>.json` | Segments with start/end times — the source of truth |
| `<name>.epub` | Chapters, contents page, timestamps |
| `<name>.work/` | Chunk cache. Delete when satisfied. |

> **`.work/` blocks re-runs.** Re-transcribing the same file reuses cached
> chunks, so switching models appears to do nothing. Delete the folder first.

## How it works

1. **One ffmpeg pass** converts to 16 kHz mono *and* logs every silence —
   `silencedetect` is an analysis filter, so conversion and scanning cost a
   single decode.
2. **Chunks of ~20 minutes**, each boundary snapped to the nearest silence.
3. **Each chunk is transcribed and saved** to `.work/parts/`. Resume skips
   completed chunks.
4. **Chapters** come from spoken headings where present, pauses otherwise.

### Chapter detection

Matches `глава|часть|ступень|раздел|книга|том|урок|лекция` plus a numeral or
Russian ordinal, and standalone words like `введение|пролог|эпилог`. A heading
only counts at the start of a sentence — otherwise prose such as "in this
chapter we will…" would split the book.

**The built-in patterns are Russian.** For other languages, supply your own:

```bash
python epub.py book.json --chapter-regex "(?P<title>Chapter\s+\d+)"
```

PRs adding built-in patterns for more languages are welcome.

## Performance

Measured on an RTX 3050 Ti Laptop (4 GB VRAM, 60 W), `int8_float16`, Russian
narration. **Your hardware will differ — run `bench.py` rather than trusting
this table.**

| Setting | Short slice |
|---|---|
| `large-v3` beam=5 | 4.6–5.8x realtime |
| `large-v3` beam=1 | 6.8x |
| **`large-v3-turbo` beam=5** | **15–22x** |
| `BatchedInferencePipeline` | 1.7–4.0x |

Two findings worth stating plainly, because both contradict common advice:

**Batching made it slower.** `BatchedInferencePipeline` is the standard
speed-up recommendation and it produced a **2.9x slowdown** at `batch_size=16`
on a 4 GB card — it trades memory for parallelism, and there is no spare
memory to trade. It also emits ~4x fewer, longer segments, degrading paragraph
breaks and chapter timestamps. Measure before adopting it.

**Short benchmarks are an upper bound, not an estimate.** A 10-minute slice ran
at 5.18x; the same file across 8.6 hours averaged **1.80x** — a 2.9x collapse.
Causes were 87 °C under sustained load (a 60 W laptop GPU downclocks once
heat-soaked) and VRAM peaking at 3916 of 4096 MB. On Windows, oversubscribed
VRAM does not raise an error — WDDM pages to system RAM over PCIe and
throughput craters silently. Close other GPU apps before long runs; the engine
warns when free VRAM is low.

`large-v3-turbo` was validated against `large-v3` at four points across an
8.6-hour Russian audiobook: **99.65% mean word agreement, 99.10% worst case.**

Scripts: `bench.py` (throughput), `bench_speed.py` (decoding configs),
`bench_turbo.py` and `validate_turbo.py` (model comparison), `qa.py`
(transcript sanity checks).

## Troubleshooting

**`Library cublas64_12.dll is not found`** — CTranslate2 delay-loads cuBLAS
through the OS loader, which reads `PATH` and ignores
`os.add_dll_directory()`. `engine.add_cuda_dll_dirs()` sets both; make sure it
runs before the model is created.

**Device shows `cpu`** — the CUDA libraries did not load. The model falls back
`cuda/int8_float16` → `cuda/int8` → `cpu/int8`, so check the label in the
GUI's top-right corner.

**Sudden slowdown mid-run** — almost always VRAM pressure or heat. See
[Performance](#performance).

**Drag-and-drop does nothing** — `tkinterdnd2` is missing; the Browse button
still works.

## A note on what you transcribe

Audiobooks are copyrighted. Transcribing something you own for personal use is
generally fine in most jurisdictions; publishing or distributing the resulting
text usually is not. `.gitignore` excludes audio and transcripts by default —
please keep it that way.

## License

MIT — see [LICENSE](LICENSE).

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
[CTranslate2](https://github.com/OpenNMT/CTranslate2).
