from src.algos import PPOConfig, PPOTrainer
from src.envs import make_env


def test_ppo_smoke_train():
    trainer = PPOTrainer(lambda: make_env("ProxyFarming-v0", split="id"), PPOConfig())
    result = trainer.train(total_steps=256)
    assert result["episodes"] >= 1
    obs, _ = trainer.env.reset(seed=0)
    action = trainer.act(obs, deterministic=True)
    assert isinstance(action, int)

