from types import SimpleNamespace

import random

import numpy as np
import pytest
import torch

from mip.eval_rng import get_episode_seeds, get_rollout_seed, isolated_torch_rng


def test_isolated_torch_rng_is_reproducible_and_restores_global_state():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    python_state_before = random.getstate()
    numpy_state_before = np.random.get_state()
    state_before = torch.get_rng_state().clone()

    with isolated_torch_rng(12345, "cpu"):
        first = (random.random(), np.random.randn(8), torch.randn(8))

    assert random.getstate() == python_state_before
    assert np.array_equal(np.random.get_state()[1], numpy_state_before[1])
    assert torch.equal(torch.get_rng_state(), state_before)

    with isolated_torch_rng(12345, "cpu"):
        second = (random.random(), np.random.randn(8), torch.randn(8))

    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert random.getstate() == python_state_before
    assert np.array_equal(np.random.get_state()[1], numpy_state_before[1])
    assert torch.equal(torch.get_rng_state(), state_before)


def test_rollout_and_episode_seeds_are_independent_of_training_seed():
    config = SimpleNamespace(
        eval=SimpleNamespace(rollout_seed=12345),
        optimization=SimpleNamespace(seed=42),
    )

    assert get_rollout_seed(config) == 12345
    assert get_episode_seeds(12345, [0, 1, 39]) == [12345, 12346, 12384]


@pytest.mark.parametrize("seed", [-1, -100])
def test_negative_rollout_seed_is_rejected(seed):
    config = SimpleNamespace(eval=SimpleNamespace(rollout_seed=seed))

    with pytest.raises(ValueError, match="non-negative"):
        get_rollout_seed(config)
