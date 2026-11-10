import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import base64
import numpy as np
import pytest
from PIL import Image

from slicer_mcp.bridge_client import BridgeClient, BridgeError, png_size
from slicer_mcp.imaging import changed, compare, ssim


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(b, format="PNG")
    return b.getvalue()


def test_compare_identical_and_changed():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64, 3))
    b = a.copy()
    b[10:30, 10:30] = 0
    assert compare(png(a), png(a))["identical"]
    c = compare(png(a), png(b))
    assert not c["identical"] and c["n_diff_pixels"] == 400 and c["ssim"] < 0.99
    assert changed(png(a), png(b)) and not changed(png(a), png(a))
    assert png_size(png(a)) == (64, 64)


def test_ssim_bounds():
    rng = np.random.default_rng(1)
    x = rng.random((50, 50)) * 255
    assert ssim(x, x) == pytest.approx(1.0)
    assert ssim(x, 255 - x) < 0.5


class _Fake(BaseHTTPRequestHandler):
    frame = png(np.zeros((8, 8, 3)))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = self.path.split("/")[-1]
        if name == "missing":
            self.send_response(404); self.end_headers(); return
        if name == "bad":
            out = {"ok": False, "error": {"type": "invalid_argument", "message": "nope"}}
        else:
            out = {"ok": True, "result": {"echo": body["args"]}, "state": {"x": 1}, "warnings": [],
                   "images": {v: base64.b64encode(self.frame).decode() for v in body["capture"]},
                   "latency_ms": {"total": 1.0}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def fake_url():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_client_roundtrip_and_errors(fake_url):
    c = BridgeClient(fake_url, obs_size=8, connect_retries=0)
    r = c.call("orbit", {"a": 1}, capture=("3d",))
    assert r.result == {"echo": {"a": 1}} and png_size(r.images["3d"]) == (8, 8)
    with pytest.raises(BridgeError) as e:
        c.call("bad")
    assert e.value.type == "invalid_argument"
    with pytest.raises(BridgeError) as e:
        c.call("missing")
    assert e.value.type == "not_found"
    with pytest.raises(BridgeError):
        BridgeClient(fake_url, obs_size=512, connect_retries=0).call("x", capture=("3d",))
