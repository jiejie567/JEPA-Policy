from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from prometheus.data.raw_episode import image_to_array
from prometheus.visualization.tactile import is_tactile_flow_stream, pointcloud_msg_to_dz_bgr


CAMERA_ORDER = (
    ("left_wrist_0_color", "left0 rgb"),
    ("right_wrist_0_color", "right0 rgb"),
    ("base_0_color", "base0 rgb"),
)
TACTILE_ORDER = (
    ("left_wrist_0_tactile_flow", "left0 tactile dz"),
    ("left_wrist_1_tactile_flow", "left1 tactile dz"),
    ("right_wrist_0_tactile_flow", "right0 tactile dz"),
    ("right_wrist_1_tactile_flow", "right1 tactile dz"),
)


@dataclass
class LocalDataPreview:
    data: Any
    window: str = "Prometheus Collection Preview"
    width: int = 1280
    height: int = 720
    fps: float = 20.0
    stale_timeout_s: float = 1.0
    max_poll_ms: float = 0.0
    ui_mode: str = "collection"
    show_tactile: bool = False

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.fps = float(self.fps)
        self.stale_timeout_s = float(self.stale_timeout_s)
        self.max_poll_ms = float(self.max_poll_ms)
        self.ui_mode = str(self.ui_mode).strip().lower()
        self.show_tactile = bool(self.show_tactile)
        self._started = False
        self._disabled = False
        self._next_draw = 0.0
        self._status = "IDLE"
        self._detail = ""
        self._status_at = 0.0
        self._dropped_frames = 0
        self._close_requested = False
        self._tk_root: Any | None = None
        self._tk_label: Any | None = None
        self._tk_photo: Any | None = None

    def start(self) -> None:
        if self._started or self._disabled:
            return
        if not os.environ.get("DISPLAY"):
            print("[preview] DISPLAY is not set; local preview window is disabled.", flush=True)
            self._disabled = True
            return
        try:
            import tkinter as tk

            root = tk.Tk()
            root.title(self.window)
            root.geometry(f"{self.width}x{self.height}")
            label = tk.Label(root)
            label.pack(fill=tk.BOTH, expand=True)
            root.bind("q", lambda _event: self._request_close())
            root.protocol("WM_DELETE_WINDOW", self._request_close)
            self._tk_root = root
            self._tk_label = label
            self._next_draw = 0.0
            self._started = True
            self.poll()
        except Exception as exc:
            print(f"[preview] disabled: {type(exc).__name__}: {exc}", flush=True)
            self._disabled = True
            self._cleanup_tk()

    def close(self) -> None:
        if not self._started and self._tk_root is None:
            return
        self._started = False
        self._cleanup_tk()

    def set_status(self, status: str, detail: str = "") -> None:
        self._status = str(status).upper()
        self._detail = str(detail)
        self._status_at = time.monotonic()

    @property
    def url(self) -> str:
        return f"local-window:{self.window}"

    def poll(self) -> None:
        if not self._started or self._tk_root is None or self._tk_label is None:
            return
        if self._close_requested:
            self.close()
            return

        period = 1.0 / max(1.0, self.fps)
        now = time.monotonic()
        if now >= self._next_draw:
            draw_start = time.monotonic()
            try:
                import cv2

                canvas = self._canvas(cv2, now)
                elapsed_ms = (time.monotonic() - draw_start) * 1000.0
                if self.max_poll_ms > 0.0 and elapsed_ms > self.max_poll_ms:
                    self._dropped_frames += 1
                else:
                    self._show_canvas(canvas)
            except Exception as exc:
                print(f"[preview] draw error: {type(exc).__name__}: {exc}", flush=True)
            self._next_draw = now + period

        try:
            self._tk_root.update_idletasks()
            self._tk_root.update()
        except Exception:
            self.close()

    def _request_close(self) -> None:
        self._close_requested = True

    def _cleanup_tk(self) -> None:
        root = self._tk_root
        self._tk_root = None
        self._tk_label = None
        self._tk_photo = None
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    def _show_canvas(self, canvas: np.ndarray) -> None:
        from PIL import Image, ImageTk

        image = Image.fromarray(np.ascontiguousarray(canvas[:, :, ::-1]))
        photo = ImageTk.PhotoImage(image=image)
        self._tk_photo = photo
        self._tk_label.configure(image=photo)

    def _canvas(self, cv2: Any, now: float) -> np.ndarray:
        status = self._status
        detail = self._detail
        age = now - self._status_at if self._status_at else 0.0
        rows = self._stream_rows()
        header_h = max(150, self.height // 5)
        canvas = np.full((self.height, self.width, 3), _status_color(status, ui_mode=self.ui_mode), dtype=np.uint8)
        title = _preview_title(status, ui_mode=self.ui_mode)
        cv2.putText(canvas, title, (34, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        lines = [
            f"status: {status}  age: {age:.1f}s",
            _preview_hint(ui_mode=self.ui_mode),
        ]
        if detail:
            lines.append(f"detail: {detail[:90]}")
        y = 112
        for line in lines:
            cv2.putText(canvas, line, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28

        area = canvas[header_h:, :]
        area[:] = 18
        for tile, x, y in self._tiles(cv2, rows, area.shape[1], area.shape[0], now):
            h, w = tile.shape[:2]
            area[y : y + h, x : x + w] = tile
        return canvas

    def _stream_rows(self) -> list[list[tuple[str, str]]]:
        available = set(self.data.streams)
        rows = []
        preview_rows = [CAMERA_ORDER]
        if self.show_tactile:
            preview_rows.append(TACTILE_ORDER)
        for row in preview_rows:
            selected = [(name, label) for name, label in row if name in available]
            if selected:
                rows.append(selected)
        return rows

    def _tiles(self, cv2: Any, rows: list[list[tuple[str, str]]], area_w: int, area_h: int, now: float):
        cols = max((len(row) for row in rows), default=1)
        row_count = max(1, len(rows))
        gap = 12
        tile_w = max(1, (area_w - gap * (cols + 1)) // cols)
        tile_h = max(1, (area_h - gap * (row_count + 1)) // row_count)
        counts = self.data.counts()
        tiles = []
        for row_idx, row in enumerate(rows):
            for col, (name, label) in enumerate(row):
                tile = np.full((tile_h, tile_w, 3), 26, dtype=np.uint8)
                sample = None
                try:
                    sample = self.data.latest([name])[name]
                    if is_tactile_flow_stream(name):
                        image = pointcloud_msg_to_dz_bgr(sample.msg)
                    else:
                        image, encoding = image_to_array(sample.msg)
                        image = _as_bgr(cv2, image, encoding)
                    tile = _letterbox(cv2, image, tile_w, tile_h)
                    age_s = max(0.0, (time.time_ns() - sample.recv_ns) / 1_000_000_000.0)
                except Exception:
                    age_s = now
                    _draw_center_text(cv2, tile, "waiting for frames")
                unhealthy = age_s >= self.stale_timeout_s
                if sample is not None and unhealthy:
                    overlay = tile.copy()
                    overlay[:] = (25, 25, 120)
                    tile = cv2.addWeighted(tile, 0.55, overlay, 0.45, 0.0)
                    _draw_center_text(cv2, tile, f"STALE {age_s:.1f}s", color=(255, 255, 255))
                label_text = f"{label}  frames:{counts.get(name, 0)} age:{age_s:.2f}s"
                cv2.rectangle(tile, (0, 0), (tile.shape[1], 42), (0, 0, 0), -1)
                cv2.putText(tile, label_text[:80], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                if unhealthy:
                    cv2.rectangle(tile, (1, 1), (tile.shape[1] - 2, tile.shape[0] - 2), (40, 40, 255), 5)
                tiles.append((tile, gap + col * (tile_w + gap), gap + row_idx * (tile_h + gap)))
        return tiles


def _preview_title(status: str, *, ui_mode: str) -> str:
    if ui_mode == "dagger":
        return "DAGGER PREVIEW"
    return "RECORDING" if status.upper() == "RUNNING" else "PROMETHEUS PREVIEW"


def _preview_hint(*, ui_mode: str) -> str:
    if ui_mode == "dagger":
        return "A: pause   B: teach   C: cycle   q: close preview window"
    return "aa: start    bb: stop/save    cc: mark    q: close preview window"


def _status_color(status: str, *, ui_mode: str = "collection") -> tuple[int, int, int]:
    status = status.upper()
    if ui_mode == "dagger":
        dagger_colors = {
            "IDLE": (120, 120, 120),
            "AUTONOMOUS": (45, 155, 55),
            "PAUSED": (30, 180, 220),
            "CORRECTING": (190, 70, 40),
            "CYCLE_WAIT": (30, 120, 220),
            "ERROR": (30, 35, 190),
        }
        if status in dagger_colors:
            return dagger_colors[status]
    if status == "RUNNING":
        return (45, 155, 55)
    if status == "ERROR":
        return (30, 35, 190)
    if status == "SAVED":
        return (170, 105, 30)
    return (205, 90, 25)


def _as_bgr(cv2: Any, image: np.ndarray, encoding: str) -> np.ndarray:
    encoding = encoding.lower()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if encoding in {"rgb8", "rgba8"}:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR if encoding == "rgb8" else cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3] if image.ndim == 3 and image.shape[2] > 3 else image


def _letterbox(cv2: Any, image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_w = max(1, int(round(image.shape[1] * scale)))
    resized_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    tile[y : y + resized_h, x : x + resized_w] = resized
    return tile


def _draw_center_text(
    cv2: Any,
    tile: np.ndarray,
    text: str,
    *,
    color: tuple[int, int, int] = (180, 180, 180),
) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
    x = max(12, (tile.shape[1] - text_w) // 2)
    y = max(48, (tile.shape[0] + text_h) // 2)
    cv2.putText(tile, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)