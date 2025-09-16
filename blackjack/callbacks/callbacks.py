from blackjack.utils.logging import ReturnMetricsCallback
from stable_baselines3.common.callbacks import BaseCallback
import math
import os

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy

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