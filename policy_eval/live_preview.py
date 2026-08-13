"""Dependency-free browser preview for live LIBERO observations.

The simulator already renders RGB observations offscreen for policy inference. This
module publishes the latest two camera frames and evaluation status through a small
local HTTP server, so a run can be watched without enabling a desktop OpenGL viewer.
"""

from __future__ import annotations

import json
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def encode_rgb_png(image: Any) -> bytes:
    """Encode a contiguous HWC uint8 RGB array as PNG using only the stdlib."""

    shape = tuple(int(value) for value in getattr(image, "shape", ()))
    if len(shape) != 3 or shape[2] != 3 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"live-preview image must have shape [H, W, 3], got {shape}")

    dtype = getattr(image, "dtype", None)
    dtype_name = getattr(dtype, "name", str(dtype))
    if dtype_name != "uint8":
        raise ValueError(f"live-preview image must have dtype uint8, got {dtype_name!r}")

    try:
        raw = image.tobytes(order="C")
    except TypeError:
        raw = image.tobytes()
    height, width, _ = shape
    row_bytes = width * 3
    expected = height * row_bytes
    if len(raw) != expected:
        raise ValueError(f"live-preview image has {len(raw)} bytes; expected {expected}")

    # PNG filter type 0 keeps encoding inexpensive. Compression level 1 favors the
    # low latency needed for a live preview over the smallest possible response.
    scanlines = bytearray(height * (row_bytes + 1))
    for row in range(height):
        source_start = row * row_bytes
        target_start = row * (row_bytes + 1)
        scanlines[target_start] = 0
        scanlines[target_start + 1 : target_start + 1 + row_bytes] = raw[
            source_start : source_start + row_bytes
        ]

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=1)),
            _png_chunk(b"IEND", b""),
        )
    )


class _PreviewState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: Dict[str, Tuple[int, bytes]] = {}
        self._sequence = 0
        self._status: Dict[str, Any] = {
            "phase": "waiting_for_first_frame",
            "frame_sequence": 0,
        }

    def publish(
        self, *, frames: Mapping[str, bytes], status: Optional[Mapping[str, Any]] = None
    ) -> None:
        with self._lock:
            if frames:
                self._sequence += 1
                for name, payload in frames.items():
                    self._frames[name] = (self._sequence, payload)
            if status:
                self._status.update(status)
            self._status["frame_sequence"] = self._sequence
            self._status["updated_at_unix"] = time.time()

    def update_status(self, status: Mapping[str, Any]) -> None:
        self.publish(frames={}, status=status)

    def frame(self, name: str) -> Optional[Tuple[int, bytes]]:
        with self._lock:
            return self._frames.get(name)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)


def _preview_html(refresh_ms: int) -> bytes:
    page = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LIBERO live preview</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #0d1117; color: #e6edf3; }
    header { padding: 16px 20px; border-bottom: 1px solid #30363d; }
    h1 { margin: 0; font-size: 18px; }
    #summary { margin-top: 7px; color: #9da7b3; font-size: 13px; }
    main { padding: 18px; display: grid; gap: 18px; }
    .cameras { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    figure { margin: 0; padding: 10px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
    figcaption { margin: 0 0 8px; color: #9da7b3; font-size: 13px; }
    img { display: block; width: 100%; aspect-ratio: 1; object-fit: contain; background: #010409; image-rendering: auto; }
    pre { margin: 0; padding: 13px; overflow: auto; background: #161b22; border: 1px solid #30363d; border-radius: 8px; font-size: 12px; }
    .ok { color: #3fb950; } .pending { color: #d29922; } .error { color: #f85149; }
  </style>
</head>
<body>
  <header>
    <h1>LIBERO 实时场景</h1>
    <div id="summary" class="pending">等待首帧…</div>
  </header>
  <main>
    <section class="cameras">
      <figure><figcaption>Agent view</figcaption><img id="agentview" alt="Agent view"></figure>
      <figure><figcaption>Wrist / eye-in-hand</figcaption><img id="wrist" alt="Wrist view"></figure>
    </section>
    <pre id="status">{}</pre>
  </main>
  <script>
    const refreshMs = __REFRESH_MS__;
    async function refreshFrame(elementId, camera) {
      const response = await fetch(`/frame/${camera}.png?t=${Date.now()}`, {cache: "no-store"});
      if (response.status !== 200) return;
      const image = document.getElementById(elementId);
      const nextUrl = URL.createObjectURL(await response.blob());
      const previousUrl = image.dataset.objectUrl;
      image.onload = () => { if (previousUrl) URL.revokeObjectURL(previousUrl); };
      image.dataset.objectUrl = nextUrl;
      image.src = nextUrl;
    }
    async function refreshStatus() {
      const response = await fetch(`/api/status?t=${Date.now()}`, {cache: "no-store"});
      if (!response.ok) return;
      const value = await response.json();
      document.getElementById("status").textContent = JSON.stringify(value, null, 2);
      const summary = document.getElementById("summary");
      const parts = [value.phase || "unknown"];
      if (value.episode_index !== undefined) parts.push(`episode ${value.episode_index}`);
      if (value.step !== undefined && value.max_steps !== undefined) parts.push(`step ${value.step}/${value.max_steps}`);
      if (value.policy_queries !== undefined) parts.push(`queries ${value.policy_queries}`);
      if (value.success !== undefined) parts.push(`success ${value.success}`);
      summary.textContent = parts.join(" · ");
      summary.className = value.error ? "error" : (value.success ? "ok" : "pending");
    }
    async function tick() {
      await Promise.all([
        refreshFrame("agentview", "agentview"),
        refreshFrame("wrist", "wrist"),
        refreshStatus(),
      ]).catch(() => {});
    }
    tick();
    setInterval(tick, refreshMs);
  </script>
</body>
</html>
"""
    return page.replace("__REFRESH_MS__", str(refresh_ms)).encode("utf-8")


class _PreviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: Tuple[str, int],
        state: _PreviewState,
        refresh_ms: int,
    ) -> None:
        self.preview_state = state
        self.preview_page = _preview_html(refresh_ms)
        super().__init__(address, _PreviewHandler)


class _PreviewHandler(BaseHTTPRequestHandler):
    server: _PreviewHTTPServer

    def _send(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", self.server.preview_page)
            return
        if path == "/healthz":
            self._send(200, "application/json", b'{"status":"ok"}\n')
            return
        if path == "/api/status":
            payload = json.dumps(
                self.server.preview_state.status(), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if path in ("/frame/agentview.png", "/frame/wrist.png"):
            name = path.rsplit("/", 1)[-1].split(".", 1)[0]
            frame = self.server.preview_state.frame(name)
            if frame is None:
                self._send(204, "image/png", b"")
            else:
                sequence, payload = frame
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("ETag", f'"frame-{sequence}"')
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)
            return
        if path == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
            return
        self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def log_message(self, format: str, *args: Any) -> None:
        return


class LivePreviewServer:
    """Serve the most recent LIBERO camera observations to a browser."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        refresh_hz: float = 10.0,
    ) -> None:
        if not host:
            raise ValueError("live-preview host cannot be empty")
        if not 0 <= port <= 65535:
            raise ValueError("live-preview port must be in [0, 65535]")
        if not 0 < refresh_hz <= 60:
            raise ValueError("live-preview refresh rate must be in (0, 60]")
        self.host = host
        self.port = port
        self.refresh_hz = refresh_hz
        self._state = _PreviewState()
        self._server: Optional[_PreviewHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("live-preview server has not been started")
        bound_host, bound_port = self._server.server_address[:2]
        display_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
        if ":" in display_host and not display_host.startswith("["):
            display_host = f"[{display_host}]"
        return f"http://{display_host}:{bound_port}/"

    @property
    def bound_address(self) -> Tuple[str, int]:
        if self._server is None:
            raise RuntimeError("live-preview server has not been started")
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> str:
        if self._server is not None:
            raise RuntimeError("live-preview server is already running")
        refresh_ms = max(16, int(round(1000.0 / self.refresh_hz)))
        self._server = _PreviewHTTPServer(
            (self.host, self.port), self._state, refresh_ms=refresh_ms
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="libero-live-preview",
            daemon=True,
        )
        self._thread.start()
        return self.url

    def publish(
        self,
        *,
        agentview_rgb: Any,
        wrist_rgb: Any,
        status: Optional[Mapping[str, Any]] = None,
    ) -> None:
        frames = {
            "agentview": encode_rgb_png(agentview_rgb),
            "wrist": encode_rgb_png(wrist_rgb),
        }
        self._state.publish(frames=frames, status=status)

    def update_status(self, status: Mapping[str, Any]) -> None:
        self._state.update_status(status)

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)

    def __enter__(self) -> "LivePreviewServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
