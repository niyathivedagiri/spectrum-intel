import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from spectrum_intel import link, signals
from spectrum_intel import metrics as m
from spectrum_intel.dashboard import DashboardApp, make_server
from spectrum_intel.engine import EngineConfig, LinkEngine


def _same(engine, cfg):
    ref = link.send(engine.data, cfg.mod, cfg.snr_db, np.random.default_rng(cfg.seed),
                    keep_corrupted=cfg.tolerant, fill=engine.fill, cfo=cfg.cfo)
    return (engine.received()[0] == ref.received and engine.corrupted == ref.corrupted
            and [vars(a) for a in engine.records] == [vars(b) for b in ref.packets])


@pytest.mark.parametrize("cfg", [EngineConfig(snr_db=1.0, seed=3),
                                 EngineConfig(source="image", image_size=24, mod="qam16", snr_db=8.0,
                                              tolerant=True, seed=5, cfo=0.001)])
def test_engine_is_identical_to_link_send(cfg):
    assert _same(LinkEngine(cfg).run_to_end(), cfg)


def test_live_change_of_modulation_and_snr():
    e = LinkEngine(EngineConfig(snr_db=15.0))
    for _ in range(3):
        e.step()
    assert e.update(mod="bpsk", snr_db=12.0) is False            # live: no restart
    e.run_to_end()
    mods = [r.mod for r in e.records]
    assert mods[:3] == ["qpsk"] * 3 and set(mods[3:]) == {"bpsk"}
    assert e.snr_set[:3] == [15.0] * 3 and e.snr_set[3] == 12.0
    r = e.report()
    assert r["mod"] == "mixed" and 125e3 < r["phy_rate_bps"] < 250e3


def test_restart_fields_and_validation():
    e = LinkEngine(EngineConfig())
    e.step()
    assert e.update(text="new message") is True and len(e.records) == 0 and e.data == b"new message"
    with pytest.raises(ValueError):
        e.update(volume=3)
    with pytest.raises(ValueError):
        e.update(mod="qam64")
    e.update(snr_db=500)
    assert e.config.snr_db == 40                                    # clipped to a sane range


def test_state_is_strict_json_even_when_frames_are_lost():
    e = LinkEngine(EngineConfig(snr_db=-15.0, seed=2))
    json.dumps(e.state(), allow_nan=False)                         # before any packet
    e.run_to_end()
    s = e.state()
    json.dumps(s, allow_nan=False)
    assert s["done"] and s["report"]["packet_success_rate"] == 0.0
    assert all(seg["state"] == "lost" for seg in s["payload"]["segments"])


def test_text_payload_builds_up():
    e = LinkEngine(EngineConfig(snr_db=20.0))
    e.step()
    segs = e.payload_view()["segments"]
    assert segs[0]["state"] == "ok" and segs[1]["state"] == "pending" and set(segs[1]["text"]) == {"·"}
    e.run_to_end()
    assert "".join(s["text"] for s in e.payload_view()["segments"]) == e.config.text


def test_rrc_cache_returns_safe_copies():
    signals._TAP_CACHE.clear()
    a = signals.rrc_taps(0.35, 8, 8)
    a[:] = 0                                                       # caller damages its copy
    b = signals.rrc_taps(0.35, 8, 8)
    assert np.allclose(b, signals._rrc_taps_compute(0.35, 8, 8)) and np.any(b)


def test_link_report_mixed_modulation_weighting():
    e = LinkEngine(EngineConfig(snr_db=20.0, text="x" * 128))   # 2 packets
    e.update(mod="bpsk")
    e.step()
    e.update(mod="qam16")
    e.step()
    k = sum(r.n_symbols * {"bpsk": 1, "qam16": 4}[r.mod] for r in e.records) / sum(r.n_symbols for r in e.records)
    assert m.link_report(type("R", (), {"packets": e.records})())["phy_rate_bps"] == pytest.approx(125e3 * k)


@pytest.fixture
def server(tmp_path):
    app = DashboardApp(EngineConfig(snr_db=20.0, rate_pps=200), results_base=tmp_path)
    app.start_thread()
    srv = make_server(app, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app, tmp_path
    srv.shutdown()
    app.stop_thread()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post(url, body, raw=False):
    req = urllib.request.Request(url, data=body if raw else json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_server_page_state_and_controls(server):
    url, app, _ = server
    code, page = _get(url + "/")
    assert code == 200 and b"<canvas" in page
    assert b"<script src" not in page and b"https://" not in page          # self-contained, works offline
    s = json.loads(_get(url + "/api/state")[1])
    assert s["sent"] == 0 and not s["running"]
    assert _post(url + "/api/control", {"action": "step"})[0] == 200
    assert json.loads(_get(url + "/api/state")[1])["sent"] == 1
    assert _post(url + "/api/control", {"action": "update", "config": {"snr_db": 7}})[1]["ok"]
    assert app.engine.config.snr_db == 7
    assert _post(url + "/api/control", {"action": "fly"})[0] == 400
    assert _post(url + "/api/control", {"action": "update", "config": {"mod": "x"}})[0] == 400


def test_server_runs_to_completion_and_saves(server):
    url, app, tmp = server
    _post(url + "/api/control", {"action": "start"})
    for _ in range(200):
        if app.engine.done:
            break
        threading.Event().wait(0.05)
    assert app.engine.done and not app.running
    code, j = _post(url + "/api/control", {"action": "save"})
    assert code == 200
    files = {p.name for p in (tmp / "dashboard").glob("*/*")}
    assert {"config.json", "metrics.json", "packets.csv", "received.txt", "original.txt"} <= files


def test_server_accepts_an_uploaded_image(server):
    from PIL import Image
    url, app, _ = server
    buf = io.BytesIO()
    Image.fromarray(np.full((40, 60, 3), 200, np.uint8)).save(buf, format="PNG")
    assert _post(url + "/api/image", buf.getvalue(), raw=True)[0] == 200
    assert app.engine.config.source == "image" and app.engine.image.shape[:2] == (48, 48)
    assert _post(url + "/api/image", b"not an image", raw=True)[0] == 400


def test_text_view_follows_the_receiver_when_a_sequence_number_is_hit():
    # seed 1 at 1.5 dB: packet 1 arrives with one bit error in its sequence number (1 -> 9);
    # tolerant mode files it under slot 9, the clean packet 9 later replaces it, slot 1 stays empty
    e = LinkEngine(EngineConfig(snr_db=1.5, tolerant=True, seed=1)).run_to_end()
    assert e.outcomes[1] == "damaged" and 1 not in e._good and 1 not in e._rough
    segs = e.payload_view()["segments"]
    assert segs[1]["state"] == "lost" and set(segs[1]["text"]) == {"·"}
    assert all(sg["state"] == "ok" for i, sg in enumerate(segs) if i != 1)
