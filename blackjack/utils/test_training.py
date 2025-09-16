from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from blackjack.env.game import BlackJack
from blackjack.utils.logging import BlackjackMetricsCallback, ReturnMetricsCallback
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import math
import os

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy

import numpy as np


class DeckScheduleCallback(BaseCallback):
    def __init__(self, total_steps: int, update_freq: int = 20480, switch_progress: float = 0.7, verbose: int = 1):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.update_freq = update_freq
        self.switch_progress = switch_progress # Progress at which to switch to uniform distribution
        self.deck_options = [2, 3, 4, 5, 6]

    def _on_step(self) -> bool:
        # Update probabilities periodically
        if self.n_calls % self.update_freq == 0:
            # Calculate current progress in the curriculum phase
            progress = self.num_timesteps / (self.total_steps * self.switch_progress)
            progress = min(progress, 1.0) # Cap progress at 1.0

            # Linearly interpolate between the initial and final probabilities
            # Initial: 50% for 2 decks, 50% for 3 decks
            initial_probs = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
            # Final: 20% for each deck size (uniform)
            final_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

            current_probs = initial_probs * (1 - progress) + final_probs * progress
            current_probs /= current_probs.sum() # Ensure it sums to 1

            # Use env_method to call the update function on each parallel environment
            self.training_env.env_method("update_deck_probs", current_probs)

            if self.verbose > 0:
                self.logger.record("deck/progress", progress)
                # Log the probability of choosing 6 decks to see the change
                self.logger.record("deck/prob_6_decks", current_probs[-1])
        return True

class CustomMaskableEvalCallback(MaskableEvalCallback):
    def __init__(self, metrics_callback: ReturnMetricsCallback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_callback = metrics_callback

    def _evaluate_policy(self, model, eval_env, n_eval_episodes: int):
        self.metrics_callback.reset()
        mean_reward, std_reward = maskable_evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=self.deterministic,
            callback=self.metrics_callback
        )
        if self.metrics_callback.num_episodes > 0:
            self.metrics_callback.ev_per_hand = self.metrics_callback.total_winnings / self.metrics_callback.num_episodes
            if self.metrics_callback.total_wagered > 0:
                self.metrics_callback.roi = self.metrics_callback.total_winnings / self.metrics_callback.total_wagered
            else:
                self.metrics_callback.roi = 0.0
        else:
            self.metrics_callback.ev_per_hand = -float('inf')
            self.metrics_callback.roi = 0.0

        self.logger.record("eval/ev_per_hand", self.metrics_callback.ev_per_hand)
        self.logger.record("eval/roi", self.metrics_callback.roi)

        return mean_reward, std_reward

class SaveOnBestEV(BaseCallback):
    def __init__(self, metrics_callback: ReturnMetricsCallback, save_path: str, verbose=1):
        super().__init__(verbose)
        self.metrics_callback = metrics_callback
        self.save_path = save_path
        self.best_ev = -float('inf')
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        current_ev = self.metrics_callback.ev_per_hand
        if self.verbose > 0:
            print(f"Current EV: {current_ev:.4f}, Best EV: {self.best_ev:.4f}")
        if current_ev == 0:
            print("Warning: Current EV is exactly zero, which may indicate an issue.")
            current_ev = -float('inf')
        if current_ev > self.best_ev:
            self.best_ev = current_ev
            full_path = os.path.join(self.save_path, f"best_model_ev.zip")
            self.model.save(full_path)
            if self.verbose > 0:
                print(f"New best EV: {self.best_ev:.4f} at step {self.num_timesteps} (saved to {full_path})")
        return True

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

def make_blackjack_env(num_decks=1):
    def _init():
        env = BlackJack(num_decks=num_decks)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def main():
    NUM_ENVS = 64
    NUM_DECKS = 2
    TOTAL_TIMESTEPS = 60_000_000
    initial_ent_coef = 0.5
    final_ent_coef = 0.005

    POLICY_KWARGS = dict(
        net_arch=dict(pi=[256, 128], vf=[512, 256, 128], ortho_init=True), # try bigger network next pi=[512, 256, 128], vf=[1024, 512, 256, 128]
    )

    EVAL_FREQ_TOTAL_STEPS = 100_000
    EVAL_FREQ_PER_ENV = EVAL_FREQ_TOTAL_STEPS // NUM_ENVS

    print("--- Setting up environment for training ---")
    vec_env = SubprocVecEnv([make_blackjack_env(num_decks=NUM_DECKS) for _ in range(NUM_ENVS)])
    vec_env = VecNormalize(vec_env, norm_reward=True, norm_obs=True, gamma=0.99)

    eval_env = SubprocVecEnv([make_blackjack_env(num_decks=6) for _ in range(128)])
    eval_env = VecNormalize(eval_env, norm_reward=True,
                            norm_obs=True, gamma=0.99, training=False)
    eval_env.seed(42)

    metrics_callback = ReturnMetricsCallback(verbose=0)
    save_on_best_eval_cb = SaveOnBestEV(metrics_callback, save_path="./logs/multideck/best_model_ev/", verbose=1)

    deck_schedule_cb = DeckScheduleCallback(total_steps=TOTAL_TIMESTEPS, update_freq=EVAL_FREQ_PER_ENV, verbose=1)

    eval_callback = CustomMaskableEvalCallback(
        metrics_callback,
        eval_env,
        best_model_save_path="./logs/multideck/best_model/",
        log_path="./logs/multideck/eval_logs/",
        eval_freq=EVAL_FREQ_PER_ENV,
        n_eval_episodes=1024,
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
                  eval_callback,
                  deck_schedule_cb],
    )

    print("--- Training complete ---")
    model.save("blackjack_agent_multideck.zip")
    vec_env.save("vec_normalize_stats_multideck.pkl")
    vec_env.close()


if __name__ == "__main__":
    main()