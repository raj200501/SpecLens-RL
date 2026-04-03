"""PPO loss helpers."""

from __future__ import annotations

import torch


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy: torch.Tensor,
    entropy_coef: float,
):
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = 0.5 * (returns - values).pow(2).mean()
    entropy_loss = -entropy.mean()
    total = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total, policy_loss, value_loss, entropy_loss

