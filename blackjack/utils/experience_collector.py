from math import gamma

import torch
import gymnasium as gym
import numpy as np

class RolloutBuffer:
    def __init__(self, buffer_size: int, observation_space: gym.spaces.Dict, action_space: gym.spaces.Dict,
                 gamma: float = 0.99, gae_lambda: float = 0.95, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda =gae_lambda

        self.obs = {key: np.zeros((buffer_size,) + space.shape, dtype=space.dtype)
                    for key, space in observation_space.spaces.items()}

        self.actions = {key: np.zeros((buffer_size,) + space.shape, dtype=space.dtype)
                        for key, space in action_space.spaces.items()}

        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs: dict, action: dict, reward: float, value: float, log_prob: float, done: bool) -> None:
        if self.pos >= self.buffer_size:
            raise IndexError("RolloutBuffer is full. Call 'compute_returns_and_advantages' before adding more data.")

        for key in self.obs:
            self.obs[key][self.pos] = obs[key]

        for key in self.actions:
            self.actions[key][self.pos] = action[key]

        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value:float) -> None:
        advantages = np.zeros(self.buffer_size, dtype=np.float32)
        returns = np.zeros(self.buffer_size, dtype=np.float32)
        last_gae_lam = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
            returns[step] = advantages[step] + self.values[step]

        self.advantages[:] = advantages
        self.returns[:] = returns

    def get_batches(self, batch_size: int):
        indices = np.arange(self.buffer_size if self.full else self.pos)
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = start+batch_size
            batch_indices = indices[start:end]

            batch = {
                'obs': {key: torch.tensor(self.obs[key][batch_indices], dtype=torch.float32).to(self.device)
                        for key in self.obs},
                'actions': {key: torch.tensor(self.actions[key][batch_indices], dtype=torch.float32).to(self.device)
                            for key in self.actions},
                'rewards': torch.tensor(self.rewards[batch_indices], dtype=torch.float32).to(self.device),
                'values': torch.tensor(self.values[batch_indices], dtype=torch.float32).to(self.device),
                'log_probs': torch.tensor(self.log_probs[batch_indices], dtype=torch.float32).to(self.device),
                'dones': torch.tensor(self.dones[batch_indices], dtype=torch.float32).to(self.device),
                'advantages': torch.tensor(self.advantages[batch_indices], dtype=torch.float32).to(self.device),
                'returns': torch.tensor(self.returns[batch_indices], dtype=torch.float32).to(self.device),
            }
            yield batch

    def reset(self) -> None:
        self.pos = 0
        self.full = False