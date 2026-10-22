"""Policy evaluation on fresh, seeded layouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from speclens.envs import make_env
from speclens.rl.networks import ActorCritic


def rollout(
    model: ActorCritic,
    env_name: str,
    layout_seeds: Sequence[int],
    split: str = "id",
    deterministic: bool = False,
    channel_scale: float = 1.0,
    record_trajectories: bool = False,
    action_seed: int = 0,
) -> list[dict]:
    """Run one episode per layout seed, all environments stepped in a batch.

    Each returned record holds the episode statistics, the split and layout
    seed, and (optionally) the trajectory a monitor would observe. Actions are
    sampled from the policy (the behavior PPO was trained to produce) using a
    generator seeded by ``action_seed``, so evaluations are reproducible;
    ``deterministic=True`` takes the most likely action instead.
    """
    envs = [
        make_env(env_name, split=split, channel_scale=channel_scale, record_trajectory=record_trajectories)
        for _ in layout_seeds
    ]
    obs = np.stack([env.reset(seed=int(seed))[0] for env, seed in zip(envs, layout_seeds)])
    generator = torch.Generator().manual_seed(action_seed)
    records: list[dict | None] = [None] * len(envs)
    active = list(range(len(envs)))
    while active:
        actions = model.act(obs[active], deterministic=deterministic, generator=generator)
        still_active = []
        for i, action in zip(active, actions):
            obs[i], _, terminated, truncated, info = envs[i].step(int(action))
            if terminated or truncated:
                record = {"split": split, "layout_seed": int(layout_seeds[i]), **info["episode"]}
                if record_trajectories:
                    record["trajectory"] = envs[i].trajectory()
                records[i] = record
            else:
                still_active.append(i)
        active = still_active
    return [r for r in records if r is not None]
