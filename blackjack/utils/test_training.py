from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from blackjack.env.game import BlackJack
from blackjack.utils.logging import BlackjackMetricsCallback
from blackjack.callbacks.callbacks import *

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import math
import os

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy

def mask_fn(env: gym.Env):
    return env.get_action_mask()

def make_blackjack_env(num_decks=1):
    def _init():
        env = BlackJack(num_decks=num_decks)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init

def main():
    NUM_ENVS = 64
    TOTAL_TIMESTEPS = 40_000_000
    initial_ent_coef = 0.5
    final_ent_coef = 0.005

    NUM_DECKS= 2

    POLICY_KWARGS = dict(
        net_arch=dict(pi=[256, 128], vf=[512, 256, 128], ortho_init=True), # try bigger network next pi=[512, 256, 128], vf=[1024, 512, 256, 128]
    )

    EVAL_FREQ_TOTAL_STEPS = 100_000
    EVAL_FREQ_PER_ENV = EVAL_FREQ_TOTAL_STEPS // NUM_ENVS

    print("--- Setting up environment for training ---")
    vec_env = SubprocVecEnv([make_blackjack_env(num_decks=NUM_DECKS) for _ in range(NUM_ENVS)])
    vec_env = VecNormalize(vec_env, norm_reward=True, norm_obs=True, gamma=0.99)

    eval_env = SubprocVecEnv([make_blackjack_env(num_decks=NUM_DECKS) for _ in range(128)])
    eval_env = VecNormalize(eval_env, norm_reward=True,
                            norm_obs=True, gamma=0.99, training=False)
    eval_env.seed(42)

    metrics_callback = ReturnMetricsCallback(verbose=0)
    save_on_best_eval_cb = SaveOnBestEV(metrics_callback, save_path=f"./logs/best_model_ev/ppobestmodel_decks_{NUM_DECKS}", verbose=1)

    eval_callback = CustomMaskableEvalCallback(
        metrics_callback,
        eval_env,
        best_model_save_path=f"./logs/best_model/ppomodel_decks_{NUM_DECKS}",
        log_path="./logs/eval_logs/",
        eval_freq=EVAL_FREQ_PER_ENV,
        n_eval_episodes=512,
        deterministic=True,
        render=False,
        callback_on_new_best=save_on_best_eval_cb,
        verbose=1,
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=exponential_decay_schedule(initial_value=3e-4, final_value=1e-5, decay_rate=5.0),
        n_steps=2048,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        vf_coef=0.75,
        ent_coef=initial_ent_coef,
        verbose=1,
        tensorboard_log="./logs/blackjack_ppo_shaped/",
        policy_kwargs=POLICY_KWARGS,
    )

    print(f"--- Starting end-to-end training for {TOTAL_TIMESTEPS} timesteps ---")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        use_masking=True,
        callback=[BlackjackMetricsCallback(),
                  metrics_callback,
                  EntropyScheduleCallback(initial_ent_coef, final_ent_coef, int(TOTAL_TIMESTEPS*0.5)),
                  eval_callback],
    )

    print("--- Training complete ---")
    model.save(f"blackjack_agent_shaped_reward_{NUM_DECKS}decks.zip")
    vec_env.save(f"vec_normalize_stats_shaped_{NUM_DECKS}decks.pkl")
    vec_env.close()


if __name__ == "__main__":
    main()