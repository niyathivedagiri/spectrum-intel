"""Live dashboard: watch a message travel over the simulated radio link, packet by packet.

    python scripts/dashboard.py                         # text, QPSK, opens your browser
    python scripts/dashboard.py --source image --snr 3  # the sample photo
    python scripts/dashboard.py --image my.jpg --size 64
    python scripts/dashboard.py --port 9000 --no-browser

Stop it with Ctrl+C in the terminal. Runs on your own computer only (127.0.0.1).
"Save run" in the page writes results/experiments/dashboard/<time>-seed<N>/.
"""
import argparse
import webbrowser

import _common  # noqa: F401

from spectrum_intel.dashboard import DashboardApp, make_server
from spectrum_intel.engine import EngineConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["text", "image"], default="text")
    ap.add_argument("--text", default=None)
    ap.add_argument("--image", default=None, help="your own image file (implies --source image)")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--mod", choices=["bpsk", "qpsk", "qam16"], default="qpsk")
    ap.add_argument("--snr", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rate", type=float, default=8.0, help="packets per second on screen")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    cfg = EngineConfig(source="image" if a.image else a.source, image_size=a.size, mod=a.mod,
                       snr_db=a.snr, seed=a.seed, rate_pps=a.rate, tolerant=bool(a.image) or a.source == "image")
    if a.text:
        cfg.text = a.text
    img = None
    if a.image:
        from PIL import Image
        import numpy as np
        img = np.asarray(Image.open(a.image).convert("RGB"), dtype=np.uint8)
    app = DashboardApp(cfg, img)
    app.start_thread()
    server = make_server(app, port=a.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Dashboard running at {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        app.stop_thread()
        server.server_close()


if __name__ == "__main__":
    main()
