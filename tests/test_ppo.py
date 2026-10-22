import numpy as np
import torch

from speclens.envs import make_env
from speclens.evaluation import rollout
from speclens.rl import ActorCritic, PPOConfig, compute_gae, train_ppo

TINY = PPOConfig(total_steps=2048, n_envs=4, rollout_steps=64, minibatch_size=128, hidden=16, channels=4)


def naive_gae(rewards, values, dones, last_value, gamma, lam):
    """Direct transcription of the GAE definition for a single environment."""
    T = len(rewards)
    values_ext = [*list(values), last_value]
    advantages = np.zeros(T)
    for t in range(T):
        total, discount = 0.0, 1.0
        for k in range(t, T):
            delta = rewards[k] + gamma * values_ext[k + 1] * (1 - dones[k]) - values_ext[k]
            total += discount * delta
            if dones[k]:
                break
            discount *= gamma * lam
        advantages[t] = total
    return advantages


def test_gae_matches_definition():
    rng = np.random.default_rng(0)
    T, N = 12, 3
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = (rng.random((T, N)) < 0.25).astype(np.float32)
    last = rng.normal(size=N).astype(np.float32)
    adv, ret = compute_gae(rewards, values, dones, last, gamma=0.9, gae_lambda=0.8)
    for n in range(N):
        expected = naive_gae(rewards[:, n], values[:, n], dones[:, n], last[n], 0.9, 0.8)
        np.testing.assert_allclose(adv[:, n], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, adv + values, rtol=1e-6)


def test_training_is_deterministic_given_seed():
    runs = [train_ppo(lambda: make_env("ProxyFarming"), "proxy", TINY, seed=3) for _ in range(2)]
    for a, b in zip(runs[0].model.state_dict().values(), runs[1].model.state_dict().values()):
        assert torch.equal(a, b)
    other = train_ppo(lambda: make_env("ProxyFarming"), "proxy", TINY, seed=4)
    assert not torch.equal(runs[0].model.policy_head.weight, other.model.policy_head.weight)


def test_checkpoints_follow_schedule():
    result = train_ppo(lambda: make_env("GemTimeout"), "true", TINY, seed=0, checkpoint_steps=[500, 1024, 2048])
    assert [step for step, _ in result.checkpoints] == [512, 1024, 2048]
    assert len(result.log) == TINY.n_updates


def test_reward_source_changes_the_training_signal():
    proxy = train_ppo(lambda: make_env("ProxyFarming"), "proxy", TINY, seed=0)
    true = train_ppo(lambda: make_env("ProxyFarming"), "true", TINY, seed=0)
    assert not torch.equal(proxy.model.policy_head.weight, true.model.policy_head.weight)


def test_rollout_is_reproducible_and_complete():
    env = make_env("SensorTamper")
    model = ActorCritic.for_env(env, hidden=16, channels=4)
    a = rollout(model, "SensorTamper", range(10), deterministic=False, action_seed=1)
    b = rollout(model, "SensorTamper", range(10), deterministic=False, action_seed=1)
    assert a == b and len(a) == 10
    assert [r["layout_seed"] for r in a] == list(range(10))
    traj = rollout(model, "SensorTamper", [0], record_trajectories=True)[0]["trajectory"]
    assert len(traj["positions"]) == len(traj["actions"]) + 1
