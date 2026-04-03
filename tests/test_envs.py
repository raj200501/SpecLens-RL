import numpy as np
import pytest

from src.envs import make_env


@pytest.mark.parametrize("env_name", ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"])
def test_env_reset_step_contract(env_name):
    env = make_env(env_name, split="id")
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert "episode_summary" in info
    obs2, reward, terminated, truncated, info = env.step(4)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "episode_summary" in info
    assert "true_score" in info["episode_summary"]


@pytest.mark.parametrize("env_name", ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"])
def test_env_seed_determinism(env_name):
    env_a = make_env(env_name, split="id")
    env_b = make_env(env_name, split="id")
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    assert np.allclose(obs_a, obs_b)


def test_proxy_farming_reward_hack_label():
    env = make_env("ProxyFarming-v0", split="id")
    env.reset(seed=0)
    env.step(1)
    env.step(4)
    env.step(4)
    _, _, _, _, info = env.step(4)
    assert info["episode_summary"]["reward_hack"] is True
    assert info["episode_summary"]["beacon_interaction_count"] >= 3


def test_sensor_tamper_reward_hack_label():
    env = make_env("SensorTamper-v0", split="id")
    env.reset(seed=0)
    env.step(3)
    _, _, _, _, info = env.step(4)
    assert info["episode_summary"]["tamper_used"] is True
    assert info["episode_summary"]["reward_hack"] is True


def test_true_score_progress_increases_toward_goal():
    env = make_env("GemTimeout-v0", split="id")
    env.reset(seed=0)
    baseline = env.get_episode_summary()["true_score"]
    _, _, _, _, info = env.step(1)
    assert info["episode_summary"]["true_score"] >= baseline

