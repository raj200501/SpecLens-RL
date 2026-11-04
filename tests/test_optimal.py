import pytest

from speclens.analysis.optimal import enumerate_transitions, optimal_episode, solve
from speclens.envs import ENVS, make_env

ENV_NAMES = sorted(ENVS)
SCALES = [0.0, 0.25, 1.0, 2.0]


@pytest.mark.parametrize("name", ENV_NAMES)
@pytest.mark.parametrize("split", ["id", "ood"])
def test_solver_value_matches_a_replay_in_the_real_environment(name, split):
    """If the solver's state missed a piece of task state, the replayed return would differ from its value."""
    for seed in range(2):
        for row in optimal_episode(name, split, seed, SCALES, gamma=0.99):
            assert row["discounted_proxy_return"] == pytest.approx(row["optimal_value"], abs=1e-9)


@pytest.mark.parametrize("name", ENV_NAMES)
def test_without_the_channel_the_optimum_is_the_intended_behavior(name):
    for seed in range(5):
        (row,) = optimal_episode(name, "id", seed, [0.0], gamma=0.99)
        assert row["success"] and not row["hack"]


@pytest.mark.parametrize("name", ENV_NAMES)
def test_optimal_value_grows_with_the_channel(name):
    env = make_env(name)
    env.reset(seed=0)
    start, graph = enumerate_transitions(env)
    values = [solve(graph, scale, gamma=0.99)[start] for scale in SCALES]
    assert values == sorted(values) and values[-1] > values[0]


def test_enumeration_requires_the_unscaled_channel():
    env = make_env("ProxyFarming", channel_scale=0.5)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        enumerate_transitions(env)


def _brute_force_values(env, scales, gamma: float, horizon: int) -> list[float]:
    """Best discounted proxy return at each channel scale, over every action sequence of up to ``horizon`` steps.

    The search steps deep copies of the environment, so it carries the environment's whole state and does
    not depend on the solver's state definition. A sequence's return is linear in the channel scale, so one
    search serves every scale.
    """
    import copy

    leaves = []  # (discounted reward without the channel, discounted channel reward)
    stack = [(env, 0.0, 0.0, 0)]
    while stack:
        node, base, channel, k = stack.pop()
        for action in range(5):
            child = copy.deepcopy(node)
            _, proxy, terminated, truncated, info = child.step(action)
            b = base + gamma**k * (proxy - info["channel_reward"])
            c = channel + gamma**k * info["channel_reward"]
            if terminated or truncated or k + 1 == horizon:
                leaves.append((b, c))
            else:
                stack.append((child, b, c, k + 1))
    return [max(b + scale * c for b, c in leaves) for scale in scales]


HORIZON = 5
BRUTE_SCALES = [0.0, 1.0, 3.0]


@pytest.mark.parametrize("name", ENV_NAMES)
def test_solver_matches_brute_force_search_on_short_horizons(name):
    """Exhaustive search over all 5^5 action sequences must find exactly the solver's optimum."""
    for seed in range(3):
        env = make_env(name, max_steps=HORIZON)
        env.reset(seed=seed)
        start, graph = enumerate_transitions(env)
        searched = make_env(name, max_steps=HORIZON)
        searched.reset(seed=seed)
        brute = _brute_force_values(searched, BRUTE_SCALES, 0.99, HORIZON)
        for scale, best in zip(BRUTE_SCALES, brute):
            assert solve(graph, scale, gamma=0.99)[start] == pytest.approx(best, abs=1e-9)


@pytest.mark.parametrize(
    ("name", "reduced_state"),
    [("SensorTamper", ("carrying", "delivered")), ("GemTimeout", ("reached_at",))],  # no cooldown / no gems
)
def test_brute_force_check_catches_missing_task_state(monkeypatch, name, reduced_state):
    """Deleting part of the solver's state must make it disagree with exhaustive search on some layouts."""
    from speclens.analysis import optimal

    monkeypatch.setitem(optimal.TASK_STATE, name, reduced_state)
    mismatches = 0
    for seed in range(10):
        env = make_env(name, max_steps=HORIZON)
        env.reset(seed=seed)
        start, graph = optimal.enumerate_transitions(env)
        searched = make_env(name, max_steps=HORIZON)
        searched.reset(seed=seed)
        best = _brute_force_values(searched, [1.0], 0.99, HORIZON)[0]
        mismatches += abs(solve(graph, 1.0, gamma=0.99)[start] - best) > 1e-9
    assert mismatches > 0
