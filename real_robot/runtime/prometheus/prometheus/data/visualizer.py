from __future__ import annotations

import binascii
import json
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from prometheus.data.raw_episode import image_to_array


@dataclass
class DataVisualizer:
    data: Any
    host: str = "127.0.0.1"
    port: int = 7860
    refresh_ms: int = 200

    def __post_init__(self) -> None:
        self.port = int(self.port)
        self.refresh_ms = int(self.refresh_ms)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, *, block: bool = False) -> None:
        if self._server is not None:
            return
        visualizer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                visualizer._handle(self)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        if block:
            self._server.serve_forever()
            return
        self._thread = threading.Thread(target=self._server.serve_forever, name="data_visualizer", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None

    def _handle(self, request: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(request.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            self._send(request, "text/html; charset=utf-8", self._html().encode())
            return
        if parsed.path == "/api/status":
            self._send_json(request, self._status_payload())
            return
        if parsed.path == "/api/timeline":
            self._send_json(request, self._timeline_payload())
            return
        if parsed.path == "/api/latest":
            self._send_json(request, self._latest_payload(t_ms=_optional_int(query, "t_ms")))
            return
        if parsed.path == "/api/image":
            stream = query.get("stream", [""])[0]
            self._send_image(request, stream, t_ms=_optional_int(query, "t_ms"), index=_optional_int(query, "index"))
            return
        self._send(request, "text/plain; charset=utf-8", b"not found", status=404)

    def _status_payload(self) -> dict[str, Any]:
        status = self.data.status()
        if hasattr(status, "as_dict"):
            status = status.as_dict()
        return {
            "url": self.url,
            "refresh_ms": self.refresh_ms,
            "status": _json_safe(status),
        }

    def _timeline_payload(self) -> dict[str, Any]:
        if hasattr(self.data, "timeline"):
            return _json_safe(self.data.timeline())
        return {"offline": False, "timestamps_ms": [], "streams": [], "events": [], "actions": {}}

    def _latest_payload(self, *, t_ms: int | None = None) -> dict[str, Any]:
        if hasattr(self.data, "visual_streams"):
            return {"streams": _json_safe(self.data.visual_streams(t_ms=t_ms))}
        counts = self.data.counts()
        now_ns = time.time_ns()
        streams = []
        for name, stream in self.data.streams.items():
            item: dict[str, Any] = {
                "name": name,
                "topic": stream.topic,
                "msg_type": stream.msg_type,
                "count": counts.get(name, 0),
                "image": False,
            }
            try:
                sample = self.data.latest([name])[name]
            except Exception:
                streams.append(item)
                continue
            item.update(
                {
                    "stamp_ns": sample.stamp_ns,
                    "recv_ns": sample.recv_ns,
                    "age_ms": (now_ns - sample.recv_ns) / 1_000_000.0,
                    "image": _looks_like_image(sample.msg),
                    "metadata": _json_safe(sample.metadata),
                }
            )
            streams.append(item)
        return {"streams": streams}

    def _send_image(
        self,
        request: BaseHTTPRequestHandler,
        stream: str,
        *,
        t_ms: int | None = None,
        index: int | None = None,
    ) -> None:
        if stream not in self.data.streams:
            self._send(request, "text/plain; charset=utf-8", b"unknown stream", status=404)
            return
        try:
            if hasattr(self.data, "image_png"):
                payload = self.data.image_png(stream, t_ms=t_ms, index=index)
            else:
                sample = self.data.latest([stream])[stream]
                image, encoding = image_to_array(sample.msg)
                if encoding == "bgr8":
                    image = image[:, :, ::-1]
                payload = encode_png(image)
        except Exception as exc:
            self._send(request, "text/plain; charset=utf-8", str(exc).encode(), status=415)
            return
        self._send(request, "image/png", payload)

    def _send_json(self, request: BaseHTTPRequestHandler, payload: Any) -> None:
        self._send(request, "application/json; charset=utf-8", json.dumps(payload, separators=(",", ":")).encode())

    def _send(self, request: BaseHTTPRequestHandler, content_type: str, body: bytes, *, status: int = 200) -> None:
        request.send_response(status)
        request.send_header("Content-Type", content_type)
        request.send_header("Content-Length", str(len(body)))
        request.send_header("Cache-Control", "no-store")
        request.end_headers()
        request.wfile.write(body)

    def _html(self) -> str:
        return HTML.replace("__REFRESH_MS__", str(self.refresh_ms))


def encode_png(image: np.ndarray) -> bytes:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        color_type = 0
        height, width = image.shape
        rows = [b"\x00" + np.ascontiguousarray(image[y]).tobytes() for y in range(height)]
    elif image.ndim == 3 and image.shape[2] in {3, 4}:
        color_type = 2 if image.shape[2] == 3 else 6
        height, width = image.shape[:2]
        rows = [b"\x00" + np.ascontiguousarray(image[y]).tobytes() for y in range(height)]
    else:
        raise ValueError(f"PNG image must be HxW, HxWx3, or HxWx4 uint8, got {image.shape}")

    raw = b"".join(rows)
    header = struct.pack("!IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(raw, level=1)),
            _png_chunk(b"IEND", b""),
        ]
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", binascii.crc32(kind + data) & 0xFFFFFFFF)


def _looks_like_image(msg: Any) -> bool:
    if isinstance(msg, np.ndarray):
        return msg.ndim in {2, 3}
    return all(hasattr(msg, name) for name in ("height", "width", "encoding", "data"))


def _optional_int(query: dict[str, list[str]], name: str) -> int | None:
    values = query.get(name)
    if not values or values[0] == "":
        return None
    return int(values[0])


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return str(value)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prometheus Data</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f8; color: #15171a; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 24px; border-bottom: 1px solid #d9dee3; background: #fff; }
    h1 { margin: 0; font-size: 20px; font-weight: 700; }
    .status { display: inline-flex; align-items: center; gap: 8px; padding: 7px 11px; border: 1px solid #cfd6dd; border-radius: 6px; background: #f9fafb; font-size: 13px; font-weight: 650; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #8a929b; }
    .ready .dot { background: #108548; }
    .failed .dot { background: #b42318; }
    main { padding: 20px 24px 28px; }
    .meta { display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; margin-bottom: 18px; }
    .metric { border: 1px solid #d9dee3; border-radius: 6px; padding: 12px; background: #fff; }
    .metric span { display: block; color: #66717d; font-size: 12px; }
    .metric strong { display: block; margin-top: 4px; font-size: 15px; overflow-wrap: anywhere; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
    .tile { border: 1px solid #d9dee3; border-radius: 6px; background: #fff; overflow: hidden; }
    .tile h2 { margin: 0; padding: 11px 12px; font-size: 14px; border-bottom: 1px solid #e5e8eb; display: flex; justify-content: space-between; gap: 10px; }
    .tile h2 small { color: #66717d; font-weight: 500; }
    .image { aspect-ratio: 16 / 9; display: grid; place-items: center; background: #101418; color: #d7dde3; }
    .image img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .empty { padding: 14px; color: #66717d; font-size: 13px; }
    .details { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: #e5e8eb; }
    .details div { background: #fff; padding: 9px 10px; font-size: 12px; }
    .details span { color: #66717d; display: block; margin-bottom: 2px; }
    @media (max-width: 760px) { header { align-items: flex-start; flex-direction: column; } .meta { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Prometheus Data</h1>
    <div id="status" class="status"><span class="dot"></span><span>connecting</span></div>
  </header>
  <main>
    <section class="meta">
      <div class="metric"><span>session</span><strong id="session">-</strong></div>
      <div class="metric"><span>message</span><strong id="message">-</strong></div>
      <div class="metric"><span>recording</span><strong id="recording">-</strong></div>
      <div class="metric"><span>episode</span><strong id="episode">-</strong></div>
    </section>
    <section id="timeline" class="metric" style="display:none; margin-bottom:18px;">
      <span>timeline</span>
      <input id="slider" type="range" min="0" max="0" value="0" style="width:100%; margin-top:10px;">
      <strong id="time-label">-</strong>
    </section>
    <section id="grid" class="grid"></section>
  </main>
  <script>
    const refreshMs = Number("__REFRESH_MS__");
    let timeline = {offline: false, timestamps_ms: []};
    let selectedMs = null;
    async function json(url) {
      const response = await fetch(url, {cache: "no-store"});
      if (!response.ok) throw new Error(`${response.status} ${url}`);
      return response.json();
    }
    function setStatus(payload) {
      const status = payload.status;
      const state = status.state || "unknown";
      const box = document.getElementById("status");
      box.className = `status ${state}`;
      box.lastElementChild.textContent = state;
      document.getElementById("session").textContent = status.name || "-";
      document.getElementById("message").textContent = status.message || "-";
      const details = status.details || {};
      document.getElementById("recording").textContent = details.recording ? "recording" : "idle";
      document.getElementById("episode").textContent = details.episode_dir || "-";
    }
    async function loadTimeline() {
      timeline = await json("/api/timeline");
      const section = document.getElementById("timeline");
      const slider = document.getElementById("slider");
      if (!timeline.offline || !timeline.timestamps_ms.length) {
        section.style.display = "none";
        return;
      }
      section.style.display = "block";
      slider.max = String(timeline.timestamps_ms.length - 1);
      slider.value = String(timeline.timestamps_ms.length - 1);
      selectedMs = timeline.timestamps_ms[Number(slider.value)];
      document.getElementById("time-label").textContent = `${selectedMs} ms`;
      slider.oninput = () => {
        selectedMs = timeline.timestamps_ms[Number(slider.value)];
        document.getElementById("time-label").textContent = `${selectedMs} ms`;
        refreshStreams();
      };
    }
    function renderStreams(payload) {
      const grid = document.getElementById("grid");
      grid.replaceChildren(...payload.streams.map(stream => {
        const tile = document.createElement("article");
        tile.className = "tile";
        const title = document.createElement("h2");
        title.innerHTML = `<span>${stream.name}</span><small>${stream.count || 0} frames</small>`;
        tile.appendChild(title);
        const image = document.createElement("div");
        image.className = "image";
        if (stream.image) {
          const img = document.createElement("img");
          const params = new URLSearchParams({stream: stream.name, _: String(Date.now())});
          if (selectedMs !== null) params.set("t_ms", String(selectedMs));
          img.src = `/api/image?${params.toString()}`;
          image.appendChild(img);
        } else {
          const empty = document.createElement("div");
          empty.className = "empty";
          empty.textContent = stream.count ? "non-image stream" : "waiting for data";
          image.appendChild(empty);
        }
        tile.appendChild(image);
        const details = document.createElement("div");
        details.className = "details";
        details.innerHTML = `
          <div><span>${stream.offline ? "frame" : "age"}</span>${stream.offline ? stream.index : (stream.age_ms === undefined ? "-" : stream.age_ms.toFixed(0) + " ms")}</div>
          <div><span>stamp</span>${stream.stamp_ms === undefined ? (stream.stamp_ns || "-") : stream.stamp_ms + " ms"}</div>
          <div><span>${stream.offline ? "codec" : "topic"}</span>${stream.offline ? (stream.codec || "-") : (stream.topic || "-")}</div>
        `;
        tile.appendChild(details);
        return tile;
      }));
    }
    async function refreshStreams() {
      const suffix = selectedMs === null ? "" : `?t_ms=${encodeURIComponent(selectedMs)}`;
      renderStreams(await json(`/api/latest${suffix}`));
    }
    async function tick() {
      try {
        setStatus(await json("/api/status"));
        await refreshStreams();
      } catch (error) {
        const box = document.getElementById("status");
        box.className = "status failed";
        box.lastElementChild.textContent = String(error);
      }
    }
    loadTimeline().then(() => {
      tick();
      if (!timeline.offline) setInterval(tick, refreshMs);
    });
  </script>
</body>
</html>
"""
