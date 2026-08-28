from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np

from prometheus.policy.diffusion_policy import (
    DiffusionInferenceRuntime,
    DiffusionPolicy,
    _clip_state_to_training_support,
    _latency_compensated_actions,
)
from prometheus.policy.jepa_policy import (
    JEPAPolicy,
    MIPPolicy,
    _normalize_state_with_clip,
)
from prometheus.sessions.data import DataSession
from prometheus.workflows.rollout_sync import _wait_policy_inputs_ready


class FakeJEPARuntime:
    def __init__(self) -> None:
        self.observations = []

    def predict(self, obs):
        self.observations.append(obs)
        return np.ones((8, 14), dtype=np.float32)

    def status(self):
        return {"fake": True}


class FakeEpisodeRuntime(FakeJEPARuntime):
    def __init__(self) -> None:
        super().__init__()
        self.reset_count = 0

    def reset_episode(self) -> None:
        self.reset_count += 1


def test_state_normalization_clips_live_values_outside_training_support():
    value = np.asarray([[0.0, 0.08177163]], dtype=np.float32)
    minimum = np.asarray([-1.0, 0.08164790], dtype=np.float32)
    value_range = np.asarray([2.0, 0.000009514], dtype=np.float32)

    normalized, clipped_dimensions = _normalize_state_with_clip(
        value,
        minimum,
        value_range,
    )

    np.testing.assert_allclose(normalized[0, 0], 0.0)
    np.testing.assert_allclose(normalized[0, 1], 1.0)
    assert clipped_dimensions == (1,)


def test_diffusion_state_clip_uses_checkpoint_raw_bounds():
    value = np.asarray(
        [
            [0.0, 0.08177163, 0.005],
            [0.0, 0.0821, 0.02],
        ],
        dtype=np.float32,
    )
    minimum = np.asarray([-1.0, 0.0107, 0.01], dtype=np.float32)
    maximum = np.asarray([1.0, 0.082, 0.08], dtype=np.float32)

    clipped, clipped_dimensions = _clip_state_to_training_support(
        value,
        minimum,
        maximum,
    )

    np.testing.assert_allclose(clipped[:, 0], value[:, 0])
    np.testing.assert_allclose(clipped[:, 1], [0.08177163, 0.082])
    np.testing.assert_allclose(clipped[:, 2], [0.01, 0.02])
    assert clipped_dimensions == (1, 2)


def test_diffusion_runtime_reuses_seed_within_episode_and_increments_between_episodes():
    runtime = DiffusionInferenceRuntime.__new__(DiffusionInferenceRuntime)
    runtime.seed = 41
    runtime.episode_index = -1
    runtime.episode_seed = runtime.seed
    runtime.episode_inference_index = 9

    runtime.reset_episode()
    assert runtime.episode_index == 0
    assert runtime.episode_seed == 41
    assert runtime.episode_inference_index == 0

    runtime.episode_inference_index = 12
    runtime.reset_episode()
    assert runtime.episode_index == 1
    assert runtime.episode_seed == 42
    assert runtime.episode_inference_index == 0


def test_diffusion_runtime_latency_profiles_fit_checkpoint_horizon():
    full = np.arange(10 * 14, dtype=np.float32).reshape(10, 14)

    dp16 = _latency_compensated_actions(
        full,
        n_obs_steps=2,
        n_action_steps=8,
        latency_compensation_steps=1,
    )
    np.testing.assert_array_equal(dp16, full[2:10])

    dp100 = _latency_compensated_actions(
        full,
        n_obs_steps=2,
        n_action_steps=8,
        latency_compensation_steps=3,
    )
    np.testing.assert_array_equal(dp100, full[4:10])

def test_diffusion_policy_propagates_episode_reset_to_runtime():
    runtime = FakeEpisodeRuntime()
    policy = DiffusionPolicy(
        inputs={"window_size": 2, "stride": 3, "wait_latest": False},
        runtime=runtime,
    )

    policy.reset_episode()

    assert runtime.reset_count == 1


def test_jepa_policy_builds_training_layout_and_abs_qpos_chunk():
    runtime = FakeJEPARuntime()
    policy = JEPAPolicy(
        inputs={"window_size": 2, "stride": 3, "wait_latest": False},
        runtime=runtime,
    )
    data = _data_session(frame_count=4)

    chunk = policy.infer(data)

    assert chunk.action_space == "abs_qpos"
    assert chunk.hz == 10.0
    assert chunk.actions.shape == (8, 14)
    obs = runtime.observations[0]
    assert obs["base_image"].shape == (2, 3, 4, 5)
    assert obs["left_wrist_image"].shape == (2, 3, 4, 5)
    assert obs["right_wrist_image"].shape == (2, 3, 4, 5)
    assert obs["state"].shape == (2, 14)
    np.testing.assert_array_equal(chunk.metadata["current_action"], np.arange(14) + 3)
    np.testing.assert_array_equal(chunk.metadata["current_velocity"], np.zeros(14))


def test_jepa_policy_requires_a_decoded_frame_after_chunk_boundary():
    runtime = FakeJEPARuntime()
    policy = JEPAPolicy(
        inputs={
            "window_size": 2,
            "stride": 3,
            "wait_latest": False,
            "observation_barrier_timeout_ms": 100.0,
        },
        runtime=runtime,
    )
    data = _data_session(frame_count=4)
    original_window_numpy = data.window_numpy
    calls = []

    def capture_window(**kwargs):
        calls.append(dict(kwargs))
        return original_window_numpy(**kwargs)

    data.window_numpy = capture_window
    target_stamp_ns = 4 * 33_000_000
    policy.require_observation_after(target_stamp_ns)

    chunk = policy.infer(data)

    assert calls[0]["min_anchor_stamp_ns"] == target_stamp_ns
    assert calls[0]["timeout_ms"] == 100.0
    assert chunk.metadata["frame_stamp_ns"] >= target_stamp_ns
    assert chunk.metadata["observation_barrier_stamp_ns"] == target_stamp_ns
    assert policy.status()["pending_observation_barrier_stamp_ns"] is None
    assert policy.status()["last_observation_barrier_stamp_ns"] == target_stamp_ns


def test_mip_and_diffusion_adapters_share_the_robot_contract():
    data = _data_session(frame_count=4)
    for policy_class, expected_name in (
        (MIPPolicy, "mip"),
        (DiffusionPolicy, "diffusion_policy"),
    ):
        policy = policy_class(
            inputs={"window_size": 2, "stride": 3, "wait_latest": False},
            runtime=FakeJEPARuntime(),
        )
        chunk = policy.infer(data)
        assert chunk.action_space == "abs_qpos"
        assert chunk.actions.shape == (8, 14)
        assert chunk.metadata["model"] == expected_name


def test_jepa_policy_waits_for_each_camera_warmup_before_first_inference():
    policy = JEPAPolicy(
        inputs={
            "window_size": 2,
            "stride": 3,
            "startup_camera_frames": 60,
            "wait_latest": False,
        },
        runtime=FakeJEPARuntime(),
    )
    data = _WarmupDataSession(
        {
            "base_0_color": 60,
            "left_wrist_0_color": 60,
            "right_wrist_0_color": 60,
            # Robot state can publish at a different rate; it is not part of
            # the per-camera warmup threshold.
            "robot_state": 1,
        }
    )

    _wait_policy_inputs_ready(data, policy, timeout_s=0.1)

    assert (
        (
            "base_0_color",
            "left_wrist_0_color",
            "right_wrist_0_color",
        ),
        60,
    ) in data.ready_calls
    assert (policy.stream_names, 1) in data.ready_calls
    assert data.window_calls == 1


class _WarmupDataSession:
    def __init__(self, counts):
        self._counts = dict(counts)
        self.ready_calls = []
        self.window_calls = 0

    def ready(self, names, *, count=1):
        selected = tuple(names)
        self.ready_calls.append((selected, int(count)))
        return all(self._counts.get(name, 0) >= count for name in selected)

    def numpy_counts(self):
        return dict(self._counts)

    def counts(self):
        return dict(self._counts)

    def window_numpy(self, **_kwargs):
        self.window_calls += 1
        return [object(), object()]


def _data_session(*, frame_count: int) -> DataSession:
    names = (
        "base_0_color",
        "left_wrist_0_color",
        "right_wrist_0_color",
        "robot_state",
    )
    topics = {
        name: {"topic": f"/{name}", "msg_type": "test_msgs/Fake"}
        for name in names
    }
    data = DataSession(
        topics,
        history=16,
        bridge={"enabled": False},
        numpy={"enabled": True},
    )
    data.start()
    for index in range(frame_count):
        stamp_ns = (index + 1) * 33_000_000
        data.ingest(
            "robot_state",
            SimpleNamespace(
                name=[f"joint_{item}" for item in range(14)],
                position=(np.arange(14) + index).tolist(),
                velocity=np.zeros(14).tolist(),
                effort=np.zeros(14).tolist(),
            ),
            stamp_ns=stamp_ns,
        )
        for camera in names[:3]:
            data.ingest(
                camera,
                _image_msg(np.full((4, 5, 3), index, dtype=np.uint8)),
                stamp_ns=stamp_ns,
            )
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if all(data.numpy_counts().get(name, 0) >= frame_count for name in names):
            return data
        time.sleep(0.01)
    raise AssertionError(f"numpy decoding timed out: {data.numpy_counts()}")


def _image_msg(array: np.ndarray):
    array = np.ascontiguousarray(array)
    return SimpleNamespace(
        height=array.shape[0],
        width=array.shape[1],
        encoding="rgb8",
        step=array.strides[0],
        data=array.tobytes(),
    )
