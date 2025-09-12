from sb3_contrib.common.wrappers import ActionMasker

from blackjack.env.game import BlackJack
from blackjack.utils.logging import BlackjackMetricsCallback, ReturnMetricsCallback
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import math

from stable_baselines3.common.callbacks import EvalCallback

def exponential_decay_schedule(initial_value: float, final_value: float, decay_rate: float):
    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * math.exp(-decay_rate * (1 - progress_remaining))
    return schedule

class EntropyScheduleCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.1, final_ent_coef=0.001, decay_steps=10_000_000):
        super().__init__()
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.decay_steps = decay_steps

    def _on_step(self):
        progress = 1.0 - (self.num_timesteps / self.decay_steps)
        progress = max(0.0, min(1.0, progress))
        ent_coef = self.initial_ent_coef * progress + self.final_ent_coef * (1 - progress)
        self.model.ent_coef = ent_coef
        self.logger.record("entropy/ent_coef", ent_coef)
        return True

def mask_fn(env: gym.Env):
    return env.get_action_mask()


def make_blackjack_env():
    def _init():
        env = BlackJack(num_decks=6)
        env = ActionMasker(env, mask_fn)
        return env

    return _init


def main():
    NUM_ENVS = 64
    TOTAL_TIMESTEPS = 40_000_000
    initial_ent_coef = 0.5
    final_ent_coef = 0.005

    POLICY_KWARGS = dict(
        net_arch=dict(pi=[256, 128], vf=[512, 256, 128], ortho_init=True), # try bigger network next pi=[512, 256, 128], vf=[1024, 512, 256, 128]
    )

    print("--- Setting up environment for training ---")
    vec_env = SubprocVecEnv([make_blackjack_env() for _ in range(NUM_ENVS)])
    vec_env = VecNormalize(vec_env, norm_reward=True, norm_obs=True, gamma=0.99)

    eval_env = SubprocVecEnv([make_blackjack_env() for _ in range(128)])
    eval_env = VecNormalize(eval_env, norm_reward=True, norm_obs=True, gamma=0.99)
    eval_env.seed(42)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval_logs/",
        eval_freq=100_000,
        n_eval_episodes=128,
        deterministic=True,
        render=False,
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
        callback=[BlackjackMetricsCallback(), ReturnMetricsCallback(),
                  EntropyScheduleCallback(initial_ent_coef, final_ent_coef, int(TOTAL_TIMESTEPS*0.5)),
                  eval_callback],
    )

    print("--- Training complete ---")
    model.save("blackjack_agent_shaped_reward.zip")
    vec_env.save("vec_normalize_stats_shaped.pkl")
    vec_env.close()


if __name__ == "__main__":
    main()