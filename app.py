"""Drag-and-drop transcription desk.

Drop audio files on the window, press Start, watch it work. The heavy lifting
lives in engine.py; this module is queue + progress + log.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, ttk

import engine

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:  # the Browse button still covers everything
    HAS_DND = False

APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "output"

BG = "#16181d"
PANEL = "#1e2128"
EDGE = "#2e323c"
FG = "#e6e8ec"
MUTED = "#8b919e"
ACCENT = "#5b9dff"
OK = "#4ec98a"
WARN = "#e5a53f"
ERR = "#e5605f"


@dataclass
class Item:
    path: Path
    duration: float = 0.0
    status: str = "queued"
    fraction: float = 0.0
    speed: float = 0.0
    eta: float | None = None
    note: str = ""
    iid: str = field(default="", init=False)


def human_time(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class App:
    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.items: list[Item] = []
        self.events: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker: threading.Thread | None = None
        self.model = None
        self.device_label = tk.StringVar(value="model not loaded yet")

        root.title("Audiobook Transcriber — faster-whisper")
        root.geometry("1020x720")
        root.minsize(880, 620)
        root.configure(bg=BG)

        self._build_style()
        self._build_ui()
        self.root.after(80, self._drain_events)

    # ---------------------------------------------------------------- style
    def _build_style(self) -> None:
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL,
                     bordercolor=EDGE, lightcolor=PANEL, darkcolor=PANEL)
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Muted.TLabel", background=BG, foreground=MUTED)
        st.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED)
        st.configure("Head.TLabel", background=BG, foreground=FG,
                     font=("Segoe UI Semibold", 15))
        st.configure("TButton", background=EDGE, foreground=FG, borderwidth=0,
                     focuscolor=EDGE, padding=(14, 7))
        st.map("TButton", background=[("active", "#3a3f4b"), ("disabled", "#23262d")],
               foreground=[("disabled", MUTED)])
        st.configure("Go.TButton", background=ACCENT, foreground="#0b1220")
        st.map("Go.TButton", background=[("active", "#7ab0ff"), ("disabled", "#23262d")])
        st.configure("TProgressbar", background=ACCENT, troughcolor=PANEL,
                     borderwidth=0, thickness=8)
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                     foreground=FG, borderwidth=0, rowheight=27)
        st.configure("Treeview.Heading", background=BG, foreground=MUTED,
                     borderwidth=0, font=("Segoe UI", 9))
        st.map("Treeview", background=[("selected", "#2c3a55")])
        st.configure("TCombobox", fieldbackground=PANEL, background=PANEL)

    # ------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        pad = {"padx": 16}

        head = ttk.Frame(self.root)
        head.pack(fill="x", pady=(14, 8), **pad)
        ttk.Label(head, text="Audiobook Transcriber", style="Head.TLabel").pack(side="left")
        ttk.Label(head, textvariable=self.device_label, style="Muted.TLabel").pack(side="right")

        # settings row
        row = ttk.Frame(self.root)
        row.pack(fill="x", pady=(0, 10), **pad)
        ttk.Label(row, text="Model", style="Muted.TLabel").pack(side="left")
        self.model_var = tk.StringVar(value="large-v3-turbo")
        ttk.Combobox(row, textvariable=self.model_var, width=16, state="readonly",
                     values=["large-v3-turbo", "large-v3", "large-v2",
                             "medium", "small"]
                     ).pack(side="left", padx=(6, 18))
        ttk.Label(row, text="Language", style="Muted.TLabel").pack(side="left")
        self.lang_var = tk.StringVar(value="ru")
        ttk.Combobox(row, textvariable=self.lang_var, width=8, state="readonly",
                     values=["ru", "en", "uk", "de", "fr", "auto"]
                     ).pack(side="left", padx=(6, 18))
        ttk.Label(row, text="Beam", style="Muted.TLabel").pack(side="left")
        self.beam_var = tk.StringVar(value="5")
        ttk.Combobox(row, textvariable=self.beam_var, width=4, state="readonly",
                     values=["5", "1"]).pack(side="left", padx=(6, 18))
        ttk.Button(row, text="Open output folder",
                   command=self.open_output).pack(side="right")

        # drop zone
        self.drop = tk.Frame(self.root, bg=PANEL, highlightthickness=2,
                             highlightbackground=EDGE, highlightcolor=EDGE)
        self.drop.pack(fill="x", ipady=18, **pad)
        msg = ("Drop audio files here" if HAS_DND
               else "Use Browse — drag-and-drop needs tkinterdnd2")
        tk.Label(self.drop, text=msg, bg=PANEL, fg=FG,
                 font=("Segoe UI", 12)).pack(pady=(4, 2))
        tk.Label(self.drop, text="mp3 · m4a · m4b · wav · flac · opus · mp4",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack()
        ttk.Button(self.drop, text="Browse…", command=self.browse).pack(pady=(8, 2))

        if HAS_DND:
            for w in (self.drop, *self.drop.winfo_children()):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self.on_drop)
                w.dnd_bind("<<DropEnter>>", lambda e: self._drop_glow(True))
                w.dnd_bind("<<DropLeave>>", lambda e: self._drop_glow(False))

        # queue table
        cols = ("file", "length", "status", "progress", "speed", "eta")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", height=8)
        for c, txt, w, anchor in (
            ("file", "File", 380, "w"), ("length", "Length", 90, "center"),
            ("status", "Status", 190, "w"), ("progress", "Progress", 90, "center"),
            ("speed", "Speed", 80, "center"), ("eta", "ETA", 90, "center"),
        ):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor=anchor,
                             stretch=(c in ("file", "status")))
        self.tree.pack(fill="both", expand=True, pady=(12, 8), **pad)
        self.tree.tag_configure("done", foreground=OK)
        self.tree.tag_configure("error", foreground=ERR)
        self.tree.tag_configure("active", foreground=ACCENT)
        self.tree.tag_configure("stopped", foreground=WARN)

        # progress block
        prog = ttk.Frame(self.root)
        prog.pack(fill="x", pady=(0, 6), **pad)
        self.now_label = ttk.Label(prog, text="Idle", style="Muted.TLabel")
        self.now_label.pack(anchor="w")
        self.file_bar = ttk.Progressbar(prog, maximum=1000)
        self.file_bar.pack(fill="x", pady=(4, 8))
        self.overall_label = ttk.Label(prog, text="Overall  0 / 0", style="Muted.TLabel")
        self.overall_label.pack(anchor="w")
        self.overall_bar = ttk.Progressbar(prog, maximum=1000)
        self.overall_bar.pack(fill="x", pady=(4, 0))

        # buttons
        btns = ttk.Frame(self.root)
        btns.pack(fill="x", pady=10, **pad)
        self.start_btn = ttk.Button(btns, text="Start", style="Go.TButton",
                                    command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        ttk.Button(btns, text="Remove selected", command=self.remove_selected).pack(side="left")
        ttk.Button(btns, text="Clear finished", command=self.clear_done).pack(side="left", padx=8)

        # log
        self.log = tk.Text(self.root, height=7, bg=PANEL, fg=MUTED, bd=0,
                           insertbackground=FG, font=("Consolas", 9), wrap="word")
        self.log.pack(fill="both", padx=16, pady=(0, 14))
        self.log.configure(state="disabled")
        self._log("Ready. Drop files above, pick a model, press Start.")
        self._log(f"Output goes to {OUT_DIR}")

    def _drop_glow(self, on: bool) -> None:
        self.drop.configure(highlightbackground=ACCENT if on else EDGE,
                            highlightcolor=ACCENT if on else EDGE)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- files
    def on_drop(self, event) -> None:
        self._drop_glow(False)
        self.add_paths(self.root.tk.splitlist(event.data))

    def browse(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose audio files",
            filetypes=[("Audio/Video", "*.mp3 *.m4a *.m4b *.wav *.flac *.ogg *.opus "
                                       "*.aac *.wma *.mp4 *.mkv *.webm *.mov"),
                       ("All files", "*.*")])
        self.add_paths(paths)

    def add_paths(self, paths) -> None:
        added = 0
        for raw in paths:
            p = Path(str(raw).strip("{}"))
            if p.is_dir():
                candidates = [c for c in sorted(p.rglob("*"))
                              if c.suffix.lower() in engine.AUDIO_EXTS]
            else:
                candidates = [p]
            for c in candidates:
                if c.suffix.lower() not in engine.AUDIO_EXTS:
                    self._log(f"Skipped (not audio): {c.name}")
                    continue
                if any(i.path == c for i in self.items):
                    continue
                item = Item(path=c)
                self.items.append(item)
                item.iid = self.tree.insert("", "end", values=(c.name, "…", "queued", "0%", "—", "—"))
                added += 1
                threading.Thread(target=self._probe, args=(item,), daemon=True).start()
        if added:
            self._log(f"Added {added} file(s)")
            self._refresh_overall()

    def _probe(self, item: Item) -> None:
        try:
            item.duration = engine.probe_duration(item.path)
        except Exception as exc:  # noqa: BLE001
            self.events.put(("log", f"Could not read {item.path.name}: {exc}"))
        self.events.put(("row", item))
        self.events.put(("overall", None))

    def remove_selected(self) -> None:
        if self.worker and self.worker.is_alive():
            self._log("Stop the run before changing the queue.")
            return
        for iid in self.tree.selection():
            self.items = [i for i in self.items if i.iid != iid]
            self.tree.delete(iid)
        self._refresh_overall()

    def clear_done(self) -> None:
        for item in [i for i in self.items if i.status == "done"]:
            self.tree.delete(item.iid)
            self.items.remove(item)
        self._refresh_overall()

    def open_output(self) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(OUT_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(OUT_DIR)])

    # ------------------------------------------------------------------ run
    def start(self) -> None:
        pending = [i for i in self.items if i.status in ("queued", "stopped", "error")]
        if not pending:
            self._log("Nothing to do — add files first.")
            return
        self.stop_flag.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.worker = threading.Thread(target=self._run, args=(pending,), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_flag.set()
        self.stop_btn.configure(state="disabled")
        self._log("Stopping after the current chunk — finished chunks are kept.")

    def _run(self, pending: list[Item]) -> None:
        try:
            if self.model is None:
                self.events.put(("log", f"Loading {self.model_var.get()} — "
                                        "first run downloads ~3 GB of weights."))
                model, device, compute = engine.load_model(
                    self.model_var.get(), lambda p: self.events.put(("stage", p)))
                self.model = model
                self.events.put(("device", f"{self.model_var.get()} · {device} · {compute}"))
                self.events.put(("log", f"Model ready on {device} ({compute})."))

            lang = self.lang_var.get()
            for item in pending:
                if self.stop_flag.is_set():
                    break
                item.status = "running"
                item.fraction = 0.0
                self.events.put(("row", item))
                self.events.put(("log", f"Starting {item.path.name}"))

                def report(p: engine.Progress, it: Item = item) -> None:
                    it.fraction = p.fraction
                    it.speed = p.speed
                    it.eta = p.eta_s
                    it.note = p.message
                    it.status = p.stage
                    self.events.put(("row", it))
                    self.events.put(("stage", p))

                try:
                    res = engine.transcribe_file(
                        item.path, OUT_DIR, self.model,
                        language=None if lang == "auto" else lang,
                        report=report, should_stop=self.stop_flag.is_set,
                        beam_size=int(self.beam_var.get()))
                except Exception as exc:  # noqa: BLE001 - surface, keep the queue alive
                    item.status = "error"
                    item.note = str(exc)
                    self.events.put(("row", item))
                    self.events.put(("log", f"FAILED {item.path.name}: {exc}"))
                    continue

                if res["status"] == "stopped":
                    item.status = "stopped"
                    self.events.put(("row", item))
                    self.events.put(("log", f"Stopped {item.path.name} — re-run to resume."))
                    break

                item.status = "done"
                item.fraction = 1.0
                item.eta = 0
                self.events.put(("row", item))
                self.events.put((
                    "log",
                    f"Done {item.path.name} — {human_time(res['duration'])} audio in "
                    f"{human_time(res['elapsed'])} ({res['speed']:.1f}x realtime), "
                    f"{res['segments']} segments"))
        finally:
            self.events.put(("finished", None))

    # --------------------------------------------------------------- events
    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "row":
                    self._update_row(payload)
                    self._refresh_overall()
                elif kind == "stage":
                    p: engine.Progress = payload
                    self.file_bar["value"] = p.fraction * 1000
                    speed = f" · {p.speed:.1f}x" if p.speed else ""
                    eta = f" · ETA {human_time(p.eta_s)}" if p.eta_s else ""
                    self.now_label.configure(
                        text=f"{p.stage}: {p.message}{speed}{eta}")
                elif kind == "log":
                    self._log(payload)
                elif kind == "device":
                    self.device_label.set(payload)
                elif kind == "overall":
                    self._refresh_overall()
                elif kind == "finished":
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.now_label.configure(text="Idle")
                    self.file_bar["value"] = 0
                    self._log("Run finished.")
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _update_row(self, item: Item) -> None:
        if not self.tree.exists(item.iid):
            return
        tag = {"done": "done", "error": "error", "stopped": "stopped"}.get(
            item.status, "active" if item.fraction else "")
        label = item.note if item.status in ("error",) else item.status
        self.tree.item(item.iid, tags=(tag,), values=(
            item.path.name,
            human_time(item.duration) if item.duration else "…",
            label[:60],
            f"{item.fraction * 100:.0f}%",
            f"{item.speed:.1f}x" if item.speed else "—",
            human_time(item.eta),
        ))

    def _refresh_overall(self) -> None:
        total = sum(i.duration for i in self.items) or 0.0
        done = sum(i.duration * (1.0 if i.status == "done" else i.fraction)
                   for i in self.items)
        frac = (done / total) if total else 0.0
        self.overall_bar["value"] = frac * 1000
        n_done = sum(1 for i in self.items if i.status == "done")
        self.overall_label.configure(
            text=f"Overall  {n_done} / {len(self.items)} files  ·  "
                 f"{human_time(done)} of {human_time(total)} audio  ·  {frac * 100:.0f}%")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
