from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.train_robomimic import parallel_image_eval
from mip.envs.persistent_image_rollout import PersistentImageRolloutPool


class _Connection:
    def __init__(self, value):
        self.value = value
        self.sent = []

    def send(self, value):
        self.sent.append(value)


def test_pool_step_can_address_only_active_workers():
    pool = object.__new__(PersistentImageRolloutPool)
    pool.num_workers = 3
    pool.started = True
    pool.connections = [_Connection(i) for i in range(3)]
    pool.start = lambda: None
    pool._recv = lambda connection, worker_id: (
        "ok",
        (worker_id, connection.value),
    )

    result = pool.step(["left", "right"], worker_indices=[0, 2])

    assert result == [(0, 0), (2, 2)]
    assert pool.connections[0].sent == [("step", {"action": "left"})]
    assert pool.connections[1].sent == []
    assert pool.connections[2].sent == [("step", {"action": "right"})]
    with pytest.raises(ValueError, match="unique valid workers"):
        pool.step(["a", "b"], worker_indices=[1, 1])


def test_pool_reset_can_resume_only_missing_episode_workers():
    pool = object.__new__(PersistentImageRolloutPool)
    pool.num_workers = 3
    pool.started = True
    pool.connections = [_Connection(i) for i in range(3)]
    pool.start = lambda: None
    pool._recv = lambda connection, worker_id: ("ok", worker_id)

    assert pool.reset([101, 303], worker_indices=[0, 2]) == [0, 2]
    assert pool.connections[0].sent == [("reset", {"seed": 101})]
    assert pool.connections[1].sent == []
    assert pool.connections[2].sent == [("reset", {"seed": 303})]


class _IdentityNormalizer:
    def normalize(self, value):
        return value

    def unnormalize(self, value):
        return value


class _Agent:
    def sample(self, act_0, **kwargs):
        return torch.zeros_like(act_0)


class _Pool:
    num_workers = 2

    def __init__(self):
        self.calls = []

    @staticmethod
    def _observation():
        return {"state": np.zeros((1, 2, 1), dtype=np.float32)}

    def reset(self, seeds):
        return [(self._observation(), {}) for _ in seeds]

    def step(self, actions, worker_indices=None):
        self.calls.append(list(worker_indices))
        results = []
        for worker_index in worker_indices:
            success = worker_index == 0 and len(self.calls) == 1
            finished = worker_index == 1 and len(self.calls) == 2
            results.append(
                (
                    self._observation(),
                    np.asarray([float(success)], dtype=np.float32),
                    np.asarray([finished]),
                    np.asarray([False]),
                    {"success": np.asarray([success])},
                )
            )
        return results


def test_parallel_eval_stops_successful_workers_early():
    config = SimpleNamespace(
        task=SimpleNamespace(
            obs_type="image",
            save_video=False,
            max_episode_steps=16,
            obs_steps=2,
            horizon=4,
            act_dim=1,
            act_steps=8,
            abs_action=False,
            env_name="fake",
        ),
        log=SimpleNamespace(save_video=False, eval_episodes=2),
        optimization=SimpleNamespace(
            device="cpu",
            future_joint_mode=False,
        ),
        eval=SimpleNamespace(rollout_seed=12345),
    )
    dataset = SimpleNamespace(
        normalizer={
            "obs": {"state": _IdentityNormalizer()},
            "action": _IdentityNormalizer(),
        }
    )
    pool = _Pool()

    metrics = parallel_image_eval(
        config, pool, dataset, _Agent(), num_steps=1
    )

    assert pool.calls == [[0, 1], [1]]
    assert metrics["mean_success_1"] == pytest.approx(0.5)
    assert metrics["mean_step_1"] == pytest.approx(12.0)


class _RewardOnlyPool(_Pool):
    def step(self, actions, worker_indices=None):
        self.calls.append(list(worker_indices))
        return [
            (
                self._observation(),
                np.asarray([1.0], dtype=np.float32),
                np.asarray([False]),
                np.asarray([False]),
                {"success": np.asarray([False])},
            )
            for _ in worker_indices
        ]


def test_parallel_eval_does_not_treat_positive_reward_as_success():
    config = SimpleNamespace(
        task=SimpleNamespace(
            obs_type="image", save_video=False, max_episode_steps=8,
            obs_steps=2, horizon=4, act_dim=1, act_steps=8,
            abs_action=False, env_name="fake",
        ),
        log=SimpleNamespace(save_video=False, eval_episodes=2),
        optimization=SimpleNamespace(device="cpu", future_joint_mode=False),
        eval=SimpleNamespace(rollout_seed=12345),
    )
    dataset = SimpleNamespace(
        normalizer={
            "obs": {"state": _IdentityNormalizer()},
            "action": _IdentityNormalizer(),
        }
    )

    metrics = parallel_image_eval(
        config, _RewardOnlyPool(), dataset, _Agent(), num_steps=1
    )

    assert metrics["mean_reward_1"] == pytest.approx(1.0)
    assert metrics["mean_success_1"] == pytest.approx(0.0)


def test_parallel_eval_uses_positive_reward_as_libero_success():
    config = SimpleNamespace(
        task=SimpleNamespace(
            obs_type="image", save_video=False, max_episode_steps=8,
            obs_steps=2, horizon=4, act_dim=1, act_steps=8,
            abs_action=False, env_name="mug_mug", env_type="libero",
        ),
        log=SimpleNamespace(save_video=False, eval_episodes=2),
        optimization=SimpleNamespace(device="cpu", future_joint_mode=False),
        eval=SimpleNamespace(rollout_seed=12345),
    )
    dataset = SimpleNamespace(
        normalizer={
            "obs": {"state": _IdentityNormalizer()},
            "action": _IdentityNormalizer(),
        }
    )

    metrics = parallel_image_eval(
        config, _RewardOnlyPool(), dataset, _Agent(), num_steps=1
    )

    assert metrics["mean_reward_1"] == pytest.approx(1.0)
    assert metrics["mean_success_1"] == pytest.approx(1.0)
    assert metrics["reward_success_disagreements_1"] == 0
