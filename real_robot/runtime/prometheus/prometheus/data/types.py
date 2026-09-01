from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


STREAM_REQUIRED_KEYS = frozenset({"topic", "msg_type"})
STREAM_ALLOWED_KEYS = frozenset({"topic", "msg_type", "domain", "ros_domain_id"})


@dataclass(frozen=True)
class DataStream:
    name: str
    topic: str
    msg_type: str
    domain: str | None = None
    ros_domain_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "topic": self.topic,
            "msg_type": self.msg_type,
            "domain": self.domain,
            "ros_domain_id": self.ros_domain_id,
        }


@dataclass(frozen=True)
class DataSample:
    name: str
    msg: Any
    stamp_ns: int
    recv_ns: int
    stream: DataStream
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def topic(self) -> DataStream:
        return self.stream


@dataclass(frozen=True)
class DataFrame:
    stamp_ns: int
    samples: dict[str, DataSample]
    skew_ms: dict[str, float]


def normalize_streams(raw: Mapping[str, Any]) -> dict[str, DataStream]:
    streams: dict[str, DataStream] = {}
    for name, spec in raw.items():
        if isinstance(spec, DataStream):
            streams[str(name)] = spec
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"data stream {name!r} must be a mapping")
        keys = set(spec)
        missing = STREAM_REQUIRED_KEYS.difference(keys)
        unknown = keys.difference(STREAM_ALLOWED_KEYS)
        if missing:
            raise ValueError(f"data stream {name!r} missing keys: {sorted(missing)}")
        if unknown:
            raise ValueError(f"data stream {name!r} has unknown keys: {sorted(unknown)}")
        if not spec["topic"]:
            raise ValueError(f"data stream {name!r} topic must be non-empty")
        if not spec["msg_type"]:
            raise ValueError(f"data stream {name!r} msg_type must be non-empty")
        ros_domain_id = spec.get("ros_domain_id")
        streams[str(name)] = DataStream(
            name=str(name),
            topic=str(spec["topic"]),
            msg_type=str(spec["msg_type"]),
            domain=None if spec.get("domain") is None else str(spec["domain"]),
            ros_domain_id=None if ros_domain_id is None else int(ros_domain_id),
        )
    return streams


def msg_stamp_ns(msg: Any) -> int:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
