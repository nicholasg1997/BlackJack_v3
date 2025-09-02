from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO
import numpy as np

from blackjack.env.game import BlackJack
from blackjack.utils import wrappers
from blackjack.custom_policies.CustomAC import CustomAC

def mask_fn(env, bet_bins=18):
    mask_parts = env.get_action_mask()
    if isinstance(mask_parts, list) and len(mask_parts) == 2:
        action_mask, bet_mask = mask_parts
        return np.concatenate([action_mask, bet_mask])
    else:
        action_mask = mask_parts
        bet_mask = np.ones(bet_bins, dtype=np.float32)
        return np.concatenate([action_mask, bet_mask])


def make_env():
    env = wrappers.FlattenObsWrapper(BlackJack())
    env = wrappers.MaskWrapper(env, bet_bins=18)
    env = ActionMasker(env, mask_fn)
    return env


def main():
    vec_env = SubprocVecEnv([make_env for _ in range(8)])

    model = MaskablePPO(
        policy=CustomAC,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/",
    )

    model.learn(total_timesteps=100_000, use_masking=True)


if __name__ == "__main__":
    main()

