"""Checkpoint surgery for starting BeakThrow from a stable Header policy.

Header and BeakThrow deliberately share the 61D actor / 80D critic layouts.
The three command slots changed meaning, however: Header observed a near-zero
twist command while BeakThrow stores ``(cos(phase), sin(phase), phase)`` there.
Loading Header verbatim therefore feeds several-standard-deviation inputs into
the old balance network.  This module makes the transfer safe by neutralizing
only those three inputs while retaining the learned standing controller.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


ACTOR_OBS_DIM = 61
CRITIC_OBS_DIM = 80
ACTION_DIM = 14

# Actor order: ang_vel(3), gravity(3), joint_pos(14), joint_vel(14),
# last_action(14), twist/phase(3), feet(6), ground(4).
ACTOR_PHASE_SLICE = slice(48, 51)

# Critic prepends base_lin_vel(3) to the actor's first five terms, then stores
# twist/phase immediately after last_action.
CRITIC_PHASE_SLICE = slice(51, 54)


def _neutralize_observation_inputs(
    state: dict[str, torch.Tensor], expected_dim: int, phase_slice: slice
) -> None:
    first_weight = state["mlp.0.weight"]
    if tuple(first_weight.shape)[1] != expected_dim:
        raise ValueError(
            f"checkpoint first layer expects {first_weight.shape[1]} observations; "
            f"expected {expected_dim}"
        )

    for name in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        tensor = state[name]
        if tuple(tensor.shape) != (1, expected_dim):
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected (1, {expected_dim})")

    # Exact neutral transfer: the new phase signal cannot alter the old policy
    # on iteration zero, while gradients can immediately grow these columns.
    first_weight[:, phase_slice] = 0.0
    state["obs_normalizer._mean"][:, phase_slice] = 0.0
    state["obs_normalizer._var"][:, phase_slice] = 1.0
    state["obs_normalizer._std"][:, phase_slice] = 1.0


def prepare_beak_bootstrap(
    checkpoint: dict[str, Any], *, exploration_std: float = 0.35
) -> dict[str, Any]:
    """Return a BeakThrow-safe copy of a Header checkpoint."""
    if exploration_std <= 0:
        raise ValueError("exploration_std must be positive")
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")

    result = deepcopy(checkpoint)
    actor = result["actor_state_dict"]
    critic = result["critic_state_dict"]
    _neutralize_observation_inputs(actor, ACTOR_OBS_DIM, ACTOR_PHASE_SLICE)
    _neutralize_observation_inputs(critic, CRITIC_OBS_DIM, CRITIC_PHASE_SLICE)

    std = actor.get("distribution.std_param")
    if std is None or std.numel() not in (1, ACTION_DIM):
        raise ValueError("actor distribution.std_param is not scalar or 14D")
    std.fill_(exploration_std)

    # This is a new optimization problem, not a continuation of Header. Keep
    # the optimizer parameter groups (needed for strict loading) but discard
    # stale Adam moments, learning iteration and curriculum clock.
    result["optimizer_state_dict"]["state"] = {}
    result["iter"] = 0
    infos = result.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        result["infos"] = infos
    env_state = infos.setdefault("env_state", {})
    env_state["common_step_counter"] = 0
    return result


def prepare_beak_bootstrap_file(
    input_path: Path | str,
    output_path: Path | str,
    *,
    exploration_std: float = 0.35,
) -> Path:
    """Load, patch, and save a Header checkpoint for BeakThrow."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    prepared = prepare_beak_bootstrap(checkpoint, exploration_std=exploration_std)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prepared, output_path)
    return output_path
