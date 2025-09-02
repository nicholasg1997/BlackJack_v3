import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, WrapperObsType, ActType, WrapperActType


class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        flat_dim = sum(np.prod(space.shape) for space in env.observation_space.spaces.values())
        flat_dim = int(flat_dim)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(flat_dim,), dtype=np.float32)

    def observation(self, obs: ObsType) -> WrapperObsType:
        return np.concatenate([obs[key].flatten() for key in sorted(obs)])


class MaskWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, bet_bins: int = 18):
        super().__init__(env)
        self.bet_bins = bet_bins

    def _get_base_env(self):
        env = self.env
        while hasattr(env, "env"):
            if hasattr(env, "is_betting_phase"):
                return env
            env = env.env
        return env

    def get_action_mask(self) -> np.ndarray:
        base_env = self._get_base_env()
        if base_env.is_betting_phase:
            return [np.zeros(base_env.action_space.nvec[0], dtype=np.float32),
                    np.ones(base_env.action_space.nvec[1], dtype=np.float32)]
        legal_mask = np.zeros(base_env.action_space.nvec[0], dtype=np.float32)
        legal_moves = base_env.get_legal_moves()
        for move in legal_moves:
            legal_mask[move.value] = 1.0
        return [legal_mask, np.ones(base_env.action_space.nvec[1], dtype=np.float32)]
