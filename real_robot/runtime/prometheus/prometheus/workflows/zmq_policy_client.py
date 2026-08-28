from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prometheus.data.numpy_buffer import NumpyFrame, NumpySample
from prometheus.data.types import DataStream
from prometheus.policy.scheduler import ActionChunk
from prometheus.utils.workflow import config_dict, instantiate_config, section
from prometheus_env.client import connect


@dataclass(frozen=True)
class ClientRuntimeConfig:
    addr: str
    max_steps: int
    print_every: int
    latency_log: Path | None
    obs_keys: tuple[str, ...]
    expect: dict[str, Any]


class ZmqPolicyData:
    """DataSession-shaped view backed by ZMQ env reset/step replies."""

    def __init__(self, *, obs: Mapping[str, Any], info: Mapping[str, Any]):
        self.update(obs=obs, info=info)

    def update(self, *, obs: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        self.obs = dict(obs)
        self.info = dict(info)

    def window_numpy(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        count: int,
        stride: int = 1,
        slop_ms: float,
        wait_latest: bool = False,
        timeout_ms: float = 0.0,
    ) -> list[NumpyFrame]:
        if int(stride) != 1:
            raise ValueError("ZMQ policy client currently expects stride=1; configure env history at policy rate")
        frames = _frames_from_info(self.info)
        if len(frames) != int(count):
            raise ValueError(f"env returned {len(frames)} history frames, policy requested {int(count)}")
        selected = tuple(str(name) for name in (names or frames[-1].samples))
        missing = [name for name in selected if name not in frames[-1].samples]
        if missing:
            raise KeyError(f"env history missing streams: {missing}")
        if str(anchor) not in frames[-1].samples:
            raise KeyError(f"env history missing anchor stream {anchor!r}")
        return [
            NumpyFrame(
                stamp_ns=frame.stamp_ns,
                samples={name: frame.samples[name] for name in selected},
                skew_ms={name: frame.skew_ms[name] for name in selected},
            )
            for frame in frames
        ]

    def latest_numpy(self, names: Sequence[str] | None = None) -> dict[str, NumpySample]:
        selected = tuple(str(name) for name in (names or self.obs))
        latest_stamps = _latest_history_stamps(self.info)
        out: dict[str, NumpySample] = {}
        for name in selected:
            if name not in self.obs:
                raise KeyError(f"env latest obs missing stream {name!r}")
            stamp_ns = int(latest_stamps.get(name, 0))
            out[name] = _sample(name, self.obs[name], stamp_ns=stamp_ns, frame_stamp_ns=stamp_ns)
        return out


def run_from_config(cfg: Any) -> int:
    data = config_dict(cfg, "zmq_policy_client")
    client_cfg = _runtime_config(data)
    policy = instantiate_config(section(data, "policy"), "policy")
    env = connect(client_cfg.addr)
    latency_file = _open_latency_log(client_cfg.latency_log)
    obs_keys = _obs_keys(policy, client_cfg.obs_keys)
    history = _history_request(policy)
    step_idx = 0
    try:
        obs, info = env.reset(obs_keys=obs_keys, expect=client_cfg.expect, **history)
        policy_data = ZmqPolicyData(obs=obs, info=info)
        print(
            f"[policy_zmq] connected addr={client_cfg.addr} obs={list(obs_keys)} "
            f"expect={client_cfg.expect}",
            flush=True,
        )
        while client_cfg.max_steps <= 0 or step_idx < client_cfg.max_steps:
            step_start = time.perf_counter()
            infer_start = time.perf_counter()
            chunk = _as_action_chunk(policy.infer(policy_data))
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            env_start = time.perf_counter()
            obs, _reward, terminated, truncated, info = env.step(chunk.actions, obs_keys=obs_keys, **history)
            env_step_ms = (time.perf_counter() - env_start) * 1000.0
            policy_data.update(obs=obs, info=info)
            step_idx += 1
            total_ms = (time.perf_counter() - step_start) * 1000.0
            _write_latency(
                latency_file,
                {
                    "step": step_idx,
                    "infer_ms": infer_ms,
                    "env_step_roundtrip_ms": env_step_ms,
                    "total_ms": total_ms,
                    "policy_latency_ms": dict(chunk.metadata.get("policy_latency_ms", {})),
                    "server_scheduler_wait_ms": info.get("scheduler_wait_ms"),
                    "server_obs_wait_ms": info.get("obs_wait_ms"),
                    "server_step_ms": info.get("server_step_ms"),
                    "sent_action_count": info.get("sent_action_count"),
                    "pending_count": info.get("pending_count"),
                },
            )
            if client_cfg.print_every > 0 and step_idx % client_cfg.print_every == 0:
                print(
                    f"[policy_zmq] step={step_idx} total_ms={total_ms:.2f} "
                    f"action_shape={tuple(chunk.actions.shape)}",
                    flush=True,
                )
            if terminated or truncated:
                obs, info = env.reset(obs_keys=obs_keys, expect=client_cfg.expect, **history)
                policy_data.update(obs=obs, info=info)
    finally:
        if latency_file is not None:
            latency_file.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()
        env.close()
    return 0


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="zmq_policy_client")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()


def _runtime_config(data: Mapping[str, Any]) -> ClientRuntimeConfig:
    client = section(data, "client")
    expect = dict(client.get("expect") or {})
    expect = {key: value for key, value in expect.items() if value is not None}
    obs_keys = client.get("obs_keys", "auto")
    if obs_keys == "auto":
        selected_obs_keys: tuple[str, ...] = ()
    elif isinstance(obs_keys, str):
        selected_obs_keys = (obs_keys,)
    else:
        selected_obs_keys = tuple(str(item) for item in obs_keys)
    latency_log = client.get("latency_log")
    return ClientRuntimeConfig(
        addr=str(client["addr"]),
        max_steps=int(client["max_steps"]),
        print_every=int(client.get("print_every", 1)),
        latency_log=None if latency_log in {None, ""} else Path(str(latency_log)).expanduser(),
        obs_keys=selected_obs_keys,
        expect=expect,
    )


def _obs_keys(policy: Any, configured: Sequence[str]) -> tuple[str, ...]:
    if configured:
        return tuple(str(item) for item in configured)
    active = getattr(policy, "_active_stream_names", None)
    if callable(active):
        return tuple(str(item) for item in active())
    names = getattr(policy, "stream_names", None)
    if names:
        return tuple(str(item) for item in names)
    return ("robot/qpos",)


def _history_request(policy: Any) -> dict[str, Any]:
    count = int(getattr(policy, "window_size", 1))
    if count <= 1:
        return {}
    return {
        "history_count": count,
        "history_anchor": str(getattr(policy, "anchor", "")),
        "history_slop_ms": float(getattr(policy, "slop_ms", 60.0)),
    }


def _frames_from_info(info: Mapping[str, Any]) -> list[NumpyFrame]:
    history = info.get("obs_history")
    sample_stamps = info.get("obs_history_timestamps")
    frame_stamps = info.get("obs_history_frame_timestamps")
    if not isinstance(history, Sequence) or not isinstance(sample_stamps, Sequence) or not isinstance(frame_stamps, Sequence):
        raise RuntimeError("env reply is missing obs_history; policy requested a window but env did not return one")
    if not (len(history) == len(sample_stamps) == len(frame_stamps)):
        raise RuntimeError("obs_history lengths do not match")
    frames: list[NumpyFrame] = []
    for frame_values, frame_sample_stamps, frame_stamp in zip(history, sample_stamps, frame_stamps):
        if not isinstance(frame_values, Mapping) or not isinstance(frame_sample_stamps, Mapping):
            raise RuntimeError("obs_history entries must be mappings")
        samples = {
            str(name): _sample(str(name), value, stamp_ns=int(frame_sample_stamps[str(name)]), frame_stamp_ns=int(frame_stamp))
            for name, value in frame_values.items()
        }
        frames.append(
            NumpyFrame(
                stamp_ns=int(frame_stamp),
                samples=samples,
                skew_ms={name: (sample.stamp_ns - int(frame_stamp)) / 1_000_000.0 for name, sample in samples.items()},
            )
        )
    return frames


def _latest_history_stamps(info: Mapping[str, Any]) -> dict[str, int]:
    stamps = info.get("obs_history_timestamps")
    if isinstance(stamps, Sequence) and stamps:
        latest = stamps[-1]
        if isinstance(latest, Mapping):
            return {str(key): int(value) for key, value in latest.items()}
    return {}


def _sample(name: str, value: Any, *, stamp_ns: int, frame_stamp_ns: int) -> NumpySample:
    return NumpySample(
        name=name,
        data=value,
        stamp_ns=int(stamp_ns),
        recv_ns=int(stamp_ns),
        stream=DataStream(name=name, topic=f"zmq://{name}", msg_type="pyobj"),
        encoding="",
        metadata={"source": "zmq_policy_client", "frame_stamp_ns": int(frame_stamp_ns)},
    )


def _as_action_chunk(value: Any) -> ActionChunk:
    if isinstance(value, ActionChunk):
        return value
    raise TypeError(f"policy.infer() must return ActionChunk, got {type(value).__name__}")


def _open_latency_log(path: Path | None) -> Any | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    print(f"[policy_zmq] latency_log={path}", flush=True)
    return handle


def _write_latency(handle: Any | None, payload: Mapping[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(_jsonable(dict(payload)), separators=(",", ":"), sort_keys=True) + "\n")
    handle.flush()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


if __name__ == "__main__":
    main()
