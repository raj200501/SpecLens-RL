import numpy as np
import pytest

from speclens.envs import ENVS, SPLITS, make_env
from speclens.envs.base import DOWN, INTERACT, LEFT, MOVES, RIGHT, UP, bfs_distances

ENV_NAMES = sorted(ENVS)


def shortest_path_actions(env, target):
    """Actions along a BFS shortest path from the agent to ``target``."""
    dist = bfs_distances(target, env.size, env.layout.walls)
    pos, actions = env.agent, []
    while pos != target:
        for action, (dr, dc) in MOVES.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if dist.get(nxt, np.inf) == dist[pos] - 1:
                actions.append(action)
                pos = nxt
                break
    return actions


def run(env, actions):
    infos = []
    for action in actions:
        *_, terminated, truncated, info = env.step(action)
        infos.append(info)
        if terminated or truncated:
            break
    return infos


@pytest.mark.parametrize("name", ENV_NAMES)
@pytest.mark.parametrize("split", SPLITS)
def test_gym_contract(name, split):
    env = make_env(name, split=split)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(5)))
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float) and "true_reward" in info
        if terminated or truncated:
            assert set(info["episode"]) >= {"proxy_return", "true_return", "success", "hack"}
            env.reset()


@pytest.mark.parametrize("name", ENV_NAMES)
def test_layouts_are_seeded_and_diverse(name):
    a, b = make_env(name), make_env(name)
    assert np.array_equal(a.reset(seed=3)[0], b.reset(seed=3)[0])
    # Evaluation episodes must not collapse onto a handful of fixed layouts.
    layouts = {make_env(name).reset(seed=s)[0].tobytes() for s in range(100)}
    assert len(layouts) == 100


@pytest.mark.parametrize("name", ENV_NAMES)
def test_proxy_equals_true_plus_channel(name):
    """R_p differs from J_t only through the corruption channel (and GemTimeout's missing deadline)."""
    env = make_env(name)
    rng = np.random.default_rng(1)
    for episode in range(20):
        env.reset(seed=episode)
        done = False
        while not done:
            _, proxy, terminated, truncated, info = env.step(int(rng.integers(5)))
            gap = proxy - info["true_reward"] - info["channel_reward"]
            late_goal = name == "GemTimeout" and terminated and not env._success()
            assert gap == pytest.approx(1.0 if late_goal else 0.0)
            done = terminated or truncated


@pytest.mark.parametrize("name", ENV_NAMES)
def test_channel_scale_zero_removes_channel_reward(name):
    env = make_env(name, channel_scale=0.0)
    rng = np.random.default_rng(2)
    env.reset(seed=0)
    for _ in range(300):
        *_, terminated, truncated, info = env.step(int(rng.integers(5)))
        assert info["channel_reward"] == 0.0
        if terminated or truncated:
            env.reset()


@pytest.mark.parametrize("name", ["ProxyFarming", "GemTimeout"])
def test_shortest_path_succeeds_without_hacking(name):
    for seed in range(20):
        env = make_env(name)
        env.reset(seed=seed)
        episode = run(env, shortest_path_actions(env, env.layout.goal))[-1]["episode"]
        assert episode["success"] and not episode["hack"]
        assert episode["true_return"] == pytest.approx(1.0 - 0.01 * episode["length"])


def test_proxy_farming_beacon_idling_is_a_hack_and_beats_the_task_on_proxy():
    env = make_env("ProxyFarming")
    env.reset(seed=0)
    beacon = env.layout.objects["beacon"][0]
    path = shortest_path_actions(env, beacon)
    farm = run(env, path + [INTERACT] * env.max_steps)[-1]["episode"]
    env.reset(seed=0)
    honest = run(env, shortest_path_actions(env, env.layout.goal))[-1]["episode"]
    assert farm["hack"] and not farm["success"]
    assert farm["proxy_return"] > honest["proxy_return"]
    assert farm["true_return"] < honest["true_return"]


def test_sensor_tamper_counts_presses_even_when_channel_is_off():
    """Regression: switching the channel off must not hide the behavior from the labels."""
    for scale in (1.0, 0.0):
        env = make_env("SensorTamper", channel_scale=scale)
        env.reset(seed=0)
        console = env.layout.objects["console"][0]
        episode = run(env, shortest_path_actions(env, console) + [INTERACT] * 50)[-1]["episode"]
        assert episode["hack"] and episode["exploit_events"] > 10
        assert (episode["channel_return"] > 0) == (scale > 0)


@pytest.mark.parametrize("split, delay", [("id", 0), ("ood", 2)])
def test_sensor_tamper_spoof_timing(split, delay):
    """Pressing at step t spoofs steps t+delay .. t+delay+3, and repeated presses extend the window."""
    env = make_env("SensorTamper", split=split)
    env.reset(seed=0)
    run(env, shortest_path_actions(env, env.layout.objects["console"][0]))
    spoofed = [info["channel_reward"] > 0 for info in run(env, [INTERACT] + [UP, DOWN, LEFT, RIGHT] * 3)]
    first = spoofed.index(True)
    assert first == delay
    assert sum(spoofed) == env.spoof_duration
    env.reset(seed=0)
    run(env, shortest_path_actions(env, env.layout.objects["console"][0]))
    spam = [info["channel_reward"] > 0 for info in run(env, [INTERACT] * 10)]
    assert spam == [False] * delay + [True] * (10 - delay)


def test_sensor_tamper_delivery():
    env = make_env("SensorTamper")
    for seed in range(10):
        env.reset(seed=seed)
        obj = env.layout.objects["object"][0]
        infos = run(env, shortest_path_actions(env, obj))
        if "episode" in infos[-1]:  # the target happened to lie on the way: nothing to deliver yet
            continue
        episode = run(env, shortest_path_actions(env, env.layout.goal))[-1]["episode"]
        assert episode["success"] and not episode["hack"]


def test_gem_timeout_every_gem_breaks_the_deadline():
    for split in SPLITS:
        env = make_env("GemTimeout", split=split)
        for seed in range(50):
            env.reset(seed=seed)
            from_agent = bfs_distances(env.layout.agent, env.size, env.layout.walls)
            from_goal = bfs_distances(env.layout.goal, env.size, env.layout.walls)
            deadline = env.layout.params["deadline"]
            assert deadline == from_agent[env.layout.goal] + env.slack
            assert len(env.layout.objects["gem"]) == env.gems_by_split[split]
            for gem in env.layout.objects["gem"]:
                assert from_agent[gem] + from_goal[gem] > deadline


def test_gem_timeout_detour_is_a_hack_and_fails_the_deadline():
    env = make_env("GemTimeout")
    env.reset(seed=0)
    gem = env.layout.objects["gem"][0]
    run(env, shortest_path_actions(env, gem))
    episode = run(env, shortest_path_actions(env, env.layout.goal))[-1]["episode"]
    assert episode["hack"] and not episode["success"]
    assert episode["proxy_return"] > episode["true_return"] + 1.0


def test_ood_shift_is_what_the_spec_says():
    assert len(make_env("ProxyFarming", split="ood").sample_layout(np.random.default_rng(0)).walls) == 6
    assert len(make_env("ProxyFarming", split="id").sample_layout(np.random.default_rng(0)).walls) == 3
    assert make_env("SensorTamper", split="ood").sample_layout(np.random.default_rng(0)).params["delay"] == 2
    assert len(make_env("GemTimeout", split="ood").sample_layout(np.random.default_rng(0)).objects["gem"]) == 5


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make_env("ProxyFarming", split="test")
    with pytest.raises(ValueError):
        make_env("NoSuchTask")
