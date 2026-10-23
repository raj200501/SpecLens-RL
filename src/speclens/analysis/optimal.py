"""Exact proxy-optimal behavior, as a reference for what Proxy PPO converges to.

For one layout, every state reachable from the start is enumerated by stepping
the real environment, and the finite-horizon discounted MDP is then solved by
backward induction. Transitions are recorded with the channel at scale 1, so a
single enumeration serves every channel scale.

This answers a question the training runs alone cannot: at a given channel
scale, is hacking what the proxy reward actually asks for, or does PPO miss (or
find) it for reasons of exploration?
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd

from speclens.envs import make_env
from speclens.envs.base import ACTION_NAMES, GridTask

# Task attributes that, with the agent cell and time, make up the Markov state.
# ``ready_at`` is stored relative to the current time step.
TASK_STATE = {
    "ProxyFarming": (),
    "SensorTamper": ("carrying", "delivered", "ready_at"),
    "GemTimeout": ("collected", "reached_at"),
}

State = tuple
Graph = dict[State, list[tuple[float, float, State | None]]]


def _get(env: GridTask) -> State:
    task = []
    for name in TASK_STATE[env.name]:
        value = getattr(env, name)
        if name == "ready_at":
            value = max(0, value - env.t)
        task.append(frozenset(value) if isinstance(value, set) else value)
    return (env.agent, env.t, tuple(task))


def _set(env: GridTask, state: State) -> None:
    env.agent, env.t, task = state
    for name, value in zip(TASK_STATE[env.name], task, strict=True):
        if name == "ready_at":
            value = env.t + value
        setattr(env, name, set(value) if isinstance(value, frozenset) else value)


def enumerate_transitions(env: GridTask) -> tuple[State, Graph]:
    """All states reachable from the current (just reset) state of ``env``, which must have channel scale 1.

    Each state maps to one ``(reward_without_channel, channel, next_state)`` per action, with
    ``next_state=None`` when the episode ends.
    """
    if env.channel_scale != 1.0:
        raise ValueError("enumerate transitions with channel_scale=1 and rescale the channel when solving")
    start, graph, frontier = _get(env), {}, [_get(env)]
    while frontier:
        state = frontier.pop()
        if state in graph:
            continue
        edges = []
        for action in range(len(ACTION_NAMES)):
            _set(env, state)
            _, proxy, terminated, truncated, info = env.step(action)
            nxt = None if terminated or truncated else _get(env)
            edges.append((proxy - info["channel_reward"], info["channel_reward"], nxt))
            if nxt is not None and nxt not in graph:
                frontier.append(nxt)
        graph[state] = edges
    return start, graph


def solve(graph: Graph, scale: float, gamma: float) -> dict[State, float]:
    """Optimal discounted proxy value of every state at channel scale ``scale``."""
    value: dict[State, float] = {}
    for state in sorted(graph, key=lambda s: -s[1]):  # every transition increases t
        value[state] = max(_q(edge, scale, gamma, value) for edge in graph[state])
    return value


def _q(edge, scale: float, gamma: float, value: dict[State, float]) -> float:
    base, channel, nxt = edge
    return base + scale * channel + (gamma * value[nxt] if nxt is not None else 0.0)


def optimal_episode(name: str, split: str, seed: int, scales: Iterable[float], gamma: float) -> list[dict]:
    """Roll out a proxy-optimal policy on one layout for each channel scale.

    Ties between actions are broken by action order. Returns the episode
    statistics of each rollout plus the optimal value of the start state.
    """
    env = make_env(name, split=split)
    env.reset(seed=seed)
    start, graph = enumerate_transitions(env)
    rows = []
    for scale in scales:
        value = solve(graph, scale, gamma)
        env = make_env(name, split=split, channel_scale=scale)
        env.reset(seed=seed)
        discounted, done, k = 0.0, False, 0
        while not done:
            qs = [_q(edge, scale, gamma, value) for edge in graph[_get(env)]]
            _, reward, terminated, truncated, info = env.step(qs.index(max(qs)))
            discounted += gamma**k * reward
            done, k = terminated or truncated, k + 1
        rows.append(
            {"env": name, "split": split, "channel_scale": scale, "layout_seed": seed,
             "optimal_value": value[start], "discounted_proxy_return": discounted, **info["episode"]}
        )  # fmt: skip
    return rows


def _solve_layout(job: tuple[str, int], split: str, scales: list[float], gamma: float) -> list[dict]:
    return optimal_episode(job[0], split, job[1], scales, gamma)


def optimal_behavior(
    envs: Iterable[str],
    scales: Iterable[float],
    gamma: float,
    seeds: Iterable[int],
    split: str = "id",
    workers: int | None = None,
) -> pd.DataFrame:
    """Proxy-optimal episodes for every task, channel scale and layout seed."""
    jobs = [(name, seed) for name in envs for seed in seeds]
    solve_layout = partial(_solve_layout, split=split, scales=list(scales), gamma=gamma)
    with ProcessPoolExecutor(max_workers=workers or min(4, os.cpu_count() or 1)) as pool:
        rows = [row for layout in pool.map(solve_layout, jobs, chunksize=8) for row in layout]
    return pd.DataFrame(rows)
