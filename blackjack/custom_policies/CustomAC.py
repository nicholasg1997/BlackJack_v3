from typing import Optional

import torch as th
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch.nn as nn
import torch
from stable_baselines3.common.type_aliases import PyTorchObs
from torch.distributions import Categorical, Normal
from gymnasium import spaces


class CustomAC(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs,
                         net_arch=dict(pi=[64, 64], vf=[64, 64]))

    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()

    def forward(self, obs, deterministic=False, action_masks=None):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.policy_net(features)
        latent_vf = self.mlp_extractor.value_net(features)

        num_game_actions = self.action_space.nvec[0]
        num_bet_actions = self.action_space.nvec[1]

        action_logits = nn.Linear(self.mlp_extractor.latent_dim_pi, num_game_actions)(latent_pi)

        bet_logits = nn.Linear(self.mlp_extractor.latent_dim_pi, num_bet_actions)(latent_pi)

        if action_masks is not None:
            # Convert numpy array to PyTorch tensor if needed
            if isinstance(action_masks, th.Tensor):
                action_masks_tensor = action_masks
            else:
                action_masks_tensor = th.from_numpy(action_masks).to(latent_pi.device)

            game_mask = action_masks_tensor[:, :num_game_actions].bool()
            bet_mask = action_masks_tensor[:, num_game_actions:].bool()
            action_logits[~game_mask] = -1e8
            bet_logits[~bet_mask] = -1e8

        game_distribution = Categorical(logits=action_logits)
        bet_distribution = Categorical(logits=bet_logits)

        game_action = game_distribution.mode if deterministic else game_distribution.sample()
        bet_action = bet_distribution.mode if deterministic else bet_distribution.sample()

        actions = torch.stack([game_action, bet_action], dim=1)
        log_prob = game_distribution.log_prob(game_action) + bet_distribution.log_prob(bet_action)

        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def get_distribution(self, obs, action_masks=None):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.policy_net(features)

        num_game_actions = self.action_space.nvec[0]
        num_bet_actions = self.action_space.nvec[1]

        action_logits = nn.Linear(self.mlp_extractor.latent_dim_pi, num_game_actions)(latent_pi)
        bet_logits = nn.Linear(self.mlp_extractor.latent_dim_pi, num_bet_actions)(latent_pi)

        if action_masks is not None:
            # Convert numpy array to PyTorch tensor if needed
            if isinstance(action_masks, th.Tensor):
                action_masks_tensor = action_masks
            else:
                action_masks_tensor = th.from_numpy(action_masks).to(latent_pi.device)

            game_mask = action_masks_tensor[:, :num_game_actions].bool()
            bet_mask = action_masks_tensor[:, num_game_actions:].bool()
            action_logits[~game_mask] = -1e8
            bet_logits[~bet_mask] = -1e8

        game_distribution = Categorical(logits=action_logits)
        bet_distribution = Categorical(logits=bet_logits)

        class HybridDist:
            def log_prob(self, actions):
                game_actions = actions[:, 0]
                bet_actions = actions[:, 1]
                return game_distribution.log_prob(game_actions) + bet_distribution.log_prob(bet_actions)

            def entropy(self):
                return game_distribution.entropy() + bet_distribution.entropy()

            def mode(self):
                return torch.stack([game_distribution.mode, bet_distribution.mode], dim=1)

            def sample(self):
                return torch.stack([game_distribution.sample(), bet_distribution.sample()], dim=1)

        return HybridDist()

    def evaluate_actions(self, obs, actions, action_masks=None):
        distribution = self.get_distribution(obs, action_masks)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        features = self.extract_features(obs)
        values = self.value_net(self.mlp_extractor.value_net(features))
        return values, log_prob, entropy

