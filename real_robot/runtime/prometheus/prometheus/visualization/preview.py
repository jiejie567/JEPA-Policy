from __future__ import annotations

from pathlib import Path
from typing import Any

from prometheus.dagger.phase import DAggerPhase

DEFAULT_REALTIME_VIS_CFG = {
    "window": "DAgger Preview",
    "width": 1280,
    "height": 720,
    "fps": 12.0,
    "stale_timeout_s": 1.0,
    "max_poll_ms": 2.0,
}


def _preview_cfg(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def realtime_vis_enabled(workflow_cfg: dict[str, Any]) -> bool:
    return bool(workflow_cfg.get("realtime_vis", False))


def resolve_realtime_vis_cfg(workflow_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULT_REALTIME_VIS_CFG)
    cfg.update(_preview_cfg(workflow_cfg.get("realtime_vis_cfg")))
    return cfg


def start_realtime_vis(data_session: Any, workflow_cfg: dict[str, Any]) -> Any | None:
    if not realtime_vis_enabled(workflow_cfg):
        return None
    cfg = resolve_realtime_vis_cfg(workflow_cfg)
    try:
        from prometheus.visualization.local_preview import LocalDataPreview

        viewer = LocalDataPreview(
            data_session,
            window=str(cfg["window"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            fps=float(cfg["fps"]),
            stale_timeout_s=float(cfg["stale_timeout_s"]),
            max_poll_ms=float(cfg["max_poll_ms"]),
            ui_mode="dagger",
        )
        viewer.start()
        print(f"[realtime_vis] {viewer.url}", flush=True)
        return viewer
    except Exception as exc:
        print(f"[realtime_vis] disabled: {exc}", flush=True)
        return None


def start_collection_preview(data_session: Any, workflow_cfg: dict[str, Any]) -> Any | None:
    preview_cfg = _preview_cfg(workflow_cfg.get("preview"))
    if not bool(preview_cfg.get("enabled", False)):
        return None
    mode = str(preview_cfg.get("mode", "local")).strip().lower()
    if mode == "web":
        viewer = data_session.visualize(
            host=str(preview_cfg.get("host", "127.0.0.1")),
            port=int(preview_cfg.get("port", 7860)),
            refresh_ms=int(preview_cfg.get("refresh_ms", 100)),
        )
    else:
        from prometheus.visualization.local_preview import LocalDataPreview

        viewer = LocalDataPreview(
            data_session,
            window=str(preview_cfg.get("window", "Prometheus Collection Preview")),
            width=int(preview_cfg.get("width", 1280)),
            height=int(preview_cfg.get("height", 720)),
            fps=float(preview_cfg.get("fps", 20.0)),
            stale_timeout_s=float(preview_cfg.get("stale_timeout_s", 1.0)),
            ui_mode="collection",
        )
    viewer.start()
    print(f"[preview] {viewer.url}", flush=True)
    return viewer


def set_realtime_vis_status(preview: Any | None, status: str, detail: str = "") -> None:
    if preview is None:
        return
    set_status = getattr(preview, "set_status", None)
    if callable(set_status):
        set_status(status, detail)


def poll_realtime_vis(preview: Any | None) -> None:
    if preview is None:
        return
    try:
        poll = getattr(preview, "poll", None)
        if callable(poll):
            poll()
    except Exception as exc:
        print(f"[realtime_vis] poll error: {exc}", flush=True)


def close_realtime_vis(preview: Any | None) -> None:
    if preview is None:
        return
    try:
        close = getattr(preview, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        print(f"[realtime_vis] close error: {exc}", flush=True)


def build_dagger_preview_detail(controller: Any, data_session: Any) -> str:
    parts: list[str] = []
    recorder = getattr(data_session, "recorder", None)
    if recorder is not None:
        details_fn = getattr(recorder, "details", None)
        if callable(details_fn):
            episode_dir = details_fn().get("episode_dir")
            if episode_dir:
                parts.append(f"ep={Path(str(episode_dir)).name}")

    parts.append(f"event={controller.event_level}")
    phase = controller.phase
    if phase == DAggerPhase.AUTONOMOUS:
        parts.append("infer")
    elif phase == DAggerPhase.PAUSED:
        parts.append("A:hold")
    elif phase == DAggerPhase.CORRECTING:
        parts.append("B:teach")
    elif phase == DAggerPhase.CYCLE_WAIT:
        parts.append("C:wait")
    return " ".join(parts)


def sync_dagger_preview(preview: Any | None, controller: Any, data_session: Any) -> None:
    if preview is None or controller is None:
        return
    set_realtime_vis_status(
        preview,
        controller.phase.value.upper(),
        build_dagger_preview_detail(controller, data_session),
    )


# Back-compat aliases for record_drag.
set_preview_status = set_realtime_vis_status
poll_preview = poll_realtime_vis
close_preview = close_realtime_vis
