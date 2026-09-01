from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from threading import RLock

from prometheus.data.types import DataFrame, DataSample, DataStream


class DataBuffer:
    def __init__(self, streams: dict[str, DataStream], *, history: int):
        if int(history) <= 0:
            raise ValueError("data history must be positive")
        self.history = int(history)
        self.streams = dict(streams)
        self.buffers = {name: deque(maxlen=self.history) for name in self.streams}
        self._lock = RLock()

    def append(self, sample: DataSample) -> None:
        with self._lock:
            self.buffers[self._stream_name(sample.name)].append(sample)

    def ready(self, names: Sequence[str] | None = None, *, count: int = 1) -> bool:
        if count <= 0:
            raise ValueError("count must be positive")
        with self._lock:
            return all(len(self.buffers[name]) >= count for name in self._names(names))

    def latest(self, names: Sequence[str] | None = None) -> dict[str, DataSample]:
        with self._lock:
            return {name: self._sample_at(name, -1) for name in self._names(names)}

    def frame(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        slop_ms: float,
        anchor_index: int = -1,
    ) -> DataFrame:
        with self._lock:
            anchor_sample = self._sample_at(anchor, anchor_index)
            return self._frame_at(anchor_sample.stamp_ns, names=self._names(names), slop_ms=slop_ms)

    def window(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        count: int,
        stride: int = 1,
        slop_ms: float,
    ) -> list[DataFrame]:
        if count <= 0:
            raise ValueError("count must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        with self._lock:
            anchor = self._stream_name(anchor)
            anchor_samples = list(self.buffers[anchor])
            needed = (count - 1) * stride + 1
            if len(anchor_samples) < needed:
                raise ValueError(f"anchor {anchor!r} needs {needed} samples, got {len(anchor_samples)}")
            selected = list(reversed(anchor_samples[-1 : -needed - 1 : -stride]))
            selected_names = self._names(names)
            return [
                self._frame_at(sample.stamp_ns, names=selected_names, slop_ms=slop_ms)
                for sample in selected
            ]

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {name: len(buffer) for name, buffer in self.buffers.items()}

    def _frame_at(self, stamp_ns: int, *, names: tuple[str, ...], slop_ms: float) -> DataFrame:
        if stamp_ns <= 0:
            raise ValueError("aligned data frame requires positive timestamps")
        slop_ns = int(float(slop_ms) * 1_000_000)
        if slop_ns < 0:
            raise ValueError("slop_ms must be non-negative")
        samples = {name: self._nearest(name, stamp_ns, slop_ns) for name in names}
        skew_ms = {name: (sample.stamp_ns - stamp_ns) / 1_000_000.0 for name, sample in samples.items()}
        return DataFrame(stamp_ns=stamp_ns, samples=samples, skew_ms=skew_ms)

    def _nearest(self, name: str, stamp_ns: int, slop_ns: int) -> DataSample:
        name = self._stream_name(name)
        candidates = [sample for sample in self.buffers[name] if sample.stamp_ns > 0]
        if not candidates:
            raise ValueError(f"stream {name!r} has no timestamped samples")
        sample = min(candidates, key=lambda item: abs(item.stamp_ns - stamp_ns))
        skew_ns = abs(sample.stamp_ns - stamp_ns)
        if skew_ns > slop_ns:
            raise ValueError(
                f"stream {name!r} nearest sample skew {skew_ns / 1_000_000.0:.3f} ms "
                f"exceeds slop {slop_ns / 1_000_000.0:.3f} ms"
            )
        return sample

    def _sample_at(self, name: str, index: int) -> DataSample:
        name = self._stream_name(name)
        if not self.buffers[name]:
            raise ValueError(f"stream {name!r} has no samples")
        try:
            return self.buffers[name][index]
        except IndexError as exc:
            raise ValueError(f"stream {name!r} has {len(self.buffers[name])} samples, cannot read index {index}") from exc

    def _stream_name(self, name: str) -> str:
        if name not in self.buffers:
            raise KeyError(f"unknown data stream name: {name!r}")
        return name

    def _names(self, names: Sequence[str] | None) -> tuple[str, ...]:
        if isinstance(names, str):
            raise TypeError("names must be an iterable of stream names, not a string")
        selected = tuple(self.buffers) if names is None else tuple(names)
        missing = [name for name in selected if name not in self.buffers]
        if missing:
            raise KeyError(f"unknown data stream names: {missing}")
        return selected
