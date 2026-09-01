"""Tests for transferring the learned Header balance controller."""

from copy import deepcopy

import torch

from mjlab_microduck.beak_bootstrap import (
    ACTOR_PHASE_SLICE,
    CRITIC_PHASE_SLICE,
    prepare_beak_bootstrap,
)


def _model_state(obs_dim: int, action_std: bool = False):
    state = {
        "mlp.0.weight": torch.randn(8, obs_dim),
        "obs_normalizer._mean": torch.randn(1, obs_dim),
        "obs_normalizer._var": torch.rand(1, obs_dim),
        "obs_normalizer._std": torch.rand(1, obs_dim),
    }
    if action_std:
        state["distribution.std_param"] = torch.rand(14)
    return state


def _checkpoint():
    return {
        "actor_state_dict": _model_state(61, action_std=True),
        "critic_state_dict": _model_state(80),
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(9)}}, "param_groups": []},
        "iter": 2999,
        "infos": {"env_state": {"common_step_counter": 72_000}},
    }


def test_transfer_keeps_balance_weights_except_new_phase_columns():
    source = _checkpoint()
    original = deepcopy(source)
    result = prepare_beak_bootstrap(source)

    assert torch.equal(source["actor_state_dict"]["mlp.0.weight"], original["actor_state_dict"]["mlp.0.weight"])
    for name, phase_slice in (
        ("actor_state_dict", ACTOR_PHASE_SLICE),
        ("critic_state_dict", CRITIC_PHASE_SLICE),
    ):
        before = original[name]["mlp.0.weight"]
        after = result[name]["mlp.0.weight"]
        keep = torch.ones(before.shape[1], dtype=torch.bool)
        keep[phase_slice] = False
        assert torch.equal(after[:, keep], before[:, keep])
        assert torch.count_nonzero(after[:, phase_slice]) == 0
        assert torch.count_nonzero(result[name]["obs_normalizer._mean"][:, phase_slice]) == 0
        assert torch.all(result[name]["obs_normalizer._var"][:, phase_slice] == 1)
        assert torch.all(result[name]["obs_normalizer._std"][:, phase_slice] == 1)


def test_transfer_restarts_optimizer_iteration_and_exploration():
    result = prepare_beak_bootstrap(_checkpoint(), exploration_std=0.27)
    assert result["optimizer_state_dict"]["state"] == {}
    assert result["iter"] == 0
    assert result["infos"]["env_state"]["common_step_counter"] == 0
    assert torch.all(result["actor_state_dict"]["distribution.std_param"] == 0.27)
