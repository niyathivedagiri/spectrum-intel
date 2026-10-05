"""Local web dashboard for the real-time link (Python standard library only).

    python scripts/dashboard.py            # opens http://127.0.0.1:8765

The server listens on 127.0.0.1 only (your own computer). A background thread
sends one packet at a time at the chosen pace; the browser page polls the state
a few times per second and draws it on <canvas> elements (no external libraries,
works offline).

API (JSON):
  GET  /api/state                       live snapshot (see LinkEngine.state)
  POST /api/control {"action": ...}     start | pause | step | reset | save
                    {"action": "update", "config": {...}}   change settings
  POST /api/image   <raw image file>    use your own image (PNG/JPEG)
"""
from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from spectrum_intel import experiments as ex
from spectrum_intel import payload as pl
from spectrum_intel.engine import EngineConfig, LinkEngine

PAGE = Path(__file__).resolve().parent / "web" / "dashboard.html"
MAX_UPLOAD = 10 * 1024 * 1024


class DashboardApp:
    """Engine + lock + pacing thread. All engine access goes through the lock."""

    def __init__(self, config: EngineConfig | None = None, image: np.ndarray | None = None,
                 results_base: Path | None = None):
        self.engine = LinkEngine(config, image)
        self.lock = threading.Lock()
        self.running = False
        self.results_base = results_base
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start_thread(self):
        self._thread.start()

    def stop_thread(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self.lock:
                if self.running:
                    if self.engine.step() is None:
                        self.running = False
                period = 1.0 / self.engine.config.rate_pps
            time.sleep(max(0.005, period - (time.perf_counter() - t0)) if self.running else 0.05)

    # ------------------------------------------------------------------ actions
    def control(self, msg: dict) -> dict:
        action = msg.get("action")
        with self.lock:
            e = self.engine
            if action == "start":
                if e.done:
                    e.reset()
                self.running = True
            elif action == "pause":
                self.running = False
            elif action == "step":
                self.running = False
                e.step()
            elif action == "reset":
                self.running = False
                e.reset()
            elif action == "update":
                if e.update(**(msg.get("config") or {})):
                    self.running = False
            elif action == "save":
                return {"ok": True, "saved": str(self._save())}
            else:
                raise ValueError(f"unknown action {action!r}")
            return {"ok": True, "running": self.running}

    def set_image(self, raw: bytes):
        from PIL import Image
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        with self.lock:
            self.running = False
            self.engine.set_image(img)
            self.engine.update(source="image")
            self.engine.reset()

    def state(self) -> dict:
        with self.lock:
            s = self.engine.state()
            s["running"] = self.running
            return s

    def _save(self):
        """Write this run to a new results/experiments/dashboard/<time>-seed<N>/ folder (lock held)."""
        e = self.engine
        cfg = {k: v for k, v in e.state()["config"].items()}
        run = ex.new_run("dashboard", cfg | {"packets_sent": len(e.records), "n_packets": e.n_packets},
                         e.config.seed, base=self.results_base)
        ex.save_metrics(run, {"report": e.report(), "complete": e.done, "corrupted_kept": e.corrupted})
        hist = e.state(history=len(e.records) or 1).get("history", [])
        ex.save_csv(run, "packets.csv", hist)
        data, _ = e.received()
        if e.config.source == "image":
            from PIL import Image
            img, _ = pl.bytes_to_image(data, e.image.shape)
            Image.fromarray(img).save(run / "received.png")
            Image.fromarray(e.image).save(run / "original.png")
        else:
            (run / "received.txt").write_text(data.decode("utf-8", errors="replace"))
            (run / "original.txt").write_text(e.config.text)
        return run


def make_handler(app: DashboardApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):          # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(200, app.state())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                return self._json(413, {"error": "upload too large"})
            body = self.rfile.read(n)
            try:
                if self.path == "/api/control":
                    self._json(200, app.control(json.loads(body or b"{}")))
                elif self.path == "/api/image":
                    app.set_image(body)
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as err:            # bad input -> message in the page, server keeps running
                self._json(400, {"error": str(err)})

    return Handler


def make_server(app: DashboardApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(app))
