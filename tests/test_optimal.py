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
