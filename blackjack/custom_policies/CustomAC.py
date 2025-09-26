import torch
import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from torch.distributions.categorical import Categorical


class CustomAC(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)

        self.num_game_actions = self.action_space.nvec[0]
        self.num_bet_actions = self.action_space.nvec[1]
        self.action_head = nn.Linear(self.mlp_extractor.latent_dim_pi, self.num_game_actions)
        self.bet_head = nn.Linear(self.mlp_extractor.latent_dim_pi, self.num_bet_actions)

    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()

    def forward(self, obs, deterministic=False, action_masks=None):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.policy_net(features)
        latent_vf = self.mlp_extractor.value_net(features)

        action_logits = self.action_head(latent_pi)
        bet_logits = self.bet_head(latent_pi)

        if action_masks is not None:
            # Convert numpy array to PyTorch tensor if needed
            if isinstance(action_masks, th.Tensor):
                action_masks_tensor = action_masks
            else:
                action_masks_tensor = th.from_numpy(action_masks).to(latent_pi.device)

            game_mask = action_masks_tensor[:, :self.num_game_actions].bool()
            bet_mask = action_masks_tensor[:, self.num_game_actions:].bool()
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

        action_logits = self.action_head(latent_pi)
        bet_logits = self.bet_head(latent_pi)

        if action_masks is not None:
            if not isinstance(action_masks, torch.Tensor):
                action_masks = torch.from_numpy(action_masks).to(latent_pi.device)

            game_mask = action_masks[:, :self.num_game_actions].bool()
            bet_mask = action_masks[:, self.num_game_actions:].bool()

            action_logits = action_logits.masked_fill(~game_mask, -1e8)
            bet_logits = bet_logits.masked_fill(~bet_mask, -1e8)

        game_distribution = Categorical(logits=action_logits)
        bet_distribution = Categorical(logits=bet_logits)

        class HybridDist:
            def log_prob(self, actions):
                return (game_distribution.log_prob(actions[:, 0]) +
                        bet_distribution.log_prob(actions[:, 1]))

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

class SplitHeadAC(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs,)

        self.num_game_actions = self.action_space.nvec[0]
        self.num_bet_actions = self.action_space.nvec[1]

        latent_dim = self.mlp_extractor.latent_dim_pi

        self.game_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_game_actions)
        )
        self.bet_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_bet_actions)
        )

    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()

    def forward(self, obs, deterministic=False, action_masks=None):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.policy_net(features)
        latent_vf = self.mlp_extractor.value_net(features)

        action_logits = self.game_head(latent_pi)
        bet_logits = self.bet_head(latent_pi)

        if action_masks is not None:
            # Convert numpy array to PyTorch tensor if needed
            if isinstance(action_masks, th.Tensor):
                action_masks_tensor = action_masks
            else:
                action_masks_tensor = th.from_numpy(action_masks).to(latent_pi.device)

            game_mask = action_masks_tensor[:, :self.num_game_actions].bool()
            bet_mask = action_masks_tensor[:, self.num_game_actions:].bool()
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

        action_logits = self.game_head(latent_pi)
        bet_logits = self.bet_head(latent_pi)

        if action_masks is not None:
            if not isinstance(action_masks, torch.Tensor):
                action_masks = torch.from_numpy(action_masks).to(latent_pi.device)

            game_mask = action_masks[:, :self.num_game_actions].bool()
            bet_mask = action_masks[:, self.num_game_actions:].bool()

            action_logits = action_logits.masked_fill(~game_mask, -1e8)
            bet_logits = bet_logits.masked_fill(~bet_mask, -1e8)

        game_distribution = Categorical(logits=action_logits)
        bet_distribution = Categorical(logits=bet_logits)

        class HybridDist:
            def log_prob(self, actions):
                return (game_distribution.log_prob(actions[:, 0]) +
                        bet_distribution.log_prob(actions[:, 1]))

            def entropy(self):
                return game_distribution.entropy() + bet_distribution.entropy()

            def mode(self):
                return torch.stack([game_distribution.mode, bet_distribution.mode], dim=1)

            def sample(self):
                return torch.stack([game_distribution.sample(), bet_distribution.sample()], dim=1)

        return HybridDist()

    def evaluate_actions(self, obs, actions, action_masks=None):
        try:
            # Ensure we have valid observations
            if obs is None:
                raise ValueError("Observations cannot be None")

            # Get the action distribution
            distribution = self.get_distribution(obs, action_masks)
            if distribution is None:
                raise ValueError("Distribution cannot be None")

            # Calculate log probabilities and entropy
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            # Extract features and calculate values
            features = self.extract_features(obs)
            if features is None:
                raise ValueError("Features cannot be None")

            # Ensure mlp_extractor and value_net are properly initialized
            if not hasattr(self, 'mlp_extractor') or self.mlp_extractor is None:
                raise ValueError("mlp_extractor is not initialized")
            if not hasattr(self, 'value_net') or self.value_net is None:
                raise ValueError("value_net is not initialized")

            value_features = self.mlp_extractor.value_net(features)
            values = self.value_net(value_features)

            return values, log_prob, entropy

        except Exception as e:
            print(f"Error in evaluate_actions: {e}")
            # Return default values to prevent None unpacking
            batch_size = obs.shape[0] if hasattr(obs, 'shape') else 1
            device = obs.device if hasattr(obs, 'device') else torch.device('cpu')

            values = torch.zeros((batch_size, 1), device=device)
            log_prob = torch.zeros((batch_size,), device=device)
            entropy = torch.zeros((batch_size,), device=device)

            return values, log_prob, entropy


class FullySeparatedAC(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, learn_count: bool = False, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        self.num_game_actions = self.action_space.nvec[0]
        self.num_bet_actions = self.action_space.nvec[1]
        self.learn_count = learn_count

        # Card counting weights (10 values for cards 2–10, Ace)
        if self.learn_count:
            self.card_weights = nn.Parameter(th.zeros(10, dtype=th.float32))
            nn.init.uniform_(self.card_weights, -0.1, 0.1)  # Initialize near Hi-Lo
        else:
            self.card_weights = th.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=th.float32)

        # Game and bet heads
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.game_head = nn.Sequential(
            nn.Linear(latent_dim_pi, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.num_game_actions)
        )
        self.bet_head = nn.Sequential(
            nn.Linear(latent_dim_pi, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.num_bet_actions)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.game_head, self.bet_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        if self.learn_count:
            # Initialize card_weights close to Hi-Lo for better convergence
            nn.init.constant_(self.card_weights[0:5], 1.0)  # 2–6
            nn.init.constant_(self.card_weights[5:8], 0.0)  # 7–9
            nn.init.constant_(self.card_weights[8:10], -1.0)  # 10, Ace

    def _check_for_nan(self, tensor, name="tensor"):
        if th.isnan(tensor).any():
            print(f"Warning: NaN detected in {name}, replacing with zeros")
            return th.where(th.isnan(tensor), th.zeros_like(tensor), tensor)
        return tensor

    def forward(self, obs, deterministic=False, action_masks=None):
        features = self.extract_features(obs)
        features = self._check_for_nan(features, "features")

        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi = self._check_for_nan(latent_pi, "latent_pi")
        latent_vf = self._check_for_nan(latent_vf, "latent_vf")

        values = self.value_net(latent_vf)
        action_logits = self.game_head(latent_pi)
        bet_logits = self.bet_head(latent_pi)

        action_logits = self._check_for_nan(action_logits, "action_logits")
        bet_logits = self._check_for_nan(bet_logits, "bet_logits")

        if action_masks is not None:
            if not isinstance(action_masks, th.Tensor):
                action_masks = th.from_numpy(action_masks).to(features.device)
            game_mask = action_masks[:, :self.num_game_actions].bool()
            bet_mask = action_masks[:, self.num_game_actions:].bool()
            action_logits = action_logits.masked_fill(~game_mask, -1e8)
            bet_logits = bet_logits.masked_fill(~bet_mask, -1e8)

        action_logits = self._check_for_nan(action_logits, "final_action_logits")
        bet_logits = self._check_for_nan(bet_logits, "final_bet_logits")

        game_dist = Categorical(logits=action_logits)
        bet_dist = Categorical(logits=bet_logits)

        game_action = game_dist.sample()
        bet_action = bet_dist.sample()

        actions = th.stack([game_action, bet_action], dim=1)
        log_prob = game_dist.log_prob(game_action) + bet_dist.log_prob(bet_action)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        features = self.extract_features(obs)
        features = self._check_for_nan(features, "eval_features")

        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi = self._check_for_nan(latent_pi, "eval_latent_pi")
        latent_vf = self._check_for_nan(latent_vf, "eval_latent_vf")

        values = self.value_net(latent_vf)
        action_logits = self.game_head(latent_pi)
        bet_logits = self.bet_head(latent_pi)

        action_logits = self._check_for_nan(action_logits, "eval_action_logits")
        bet_logits = self._check_for_nan(bet_logits, "eval_bet_logits")

        if action_masks is not None:
            if not isinstance(action_masks, th.Tensor):
                action_masks = th.from_numpy(action_masks).to(features.device)
            game_mask = action_masks[:, :self.num_game_actions].bool()
            bet_mask = action_masks[:, self.num_game_actions:].bool()
            action_logits = action_logits.masked_fill(~game_mask, -1e8)
            bet_logits = bet_logits.masked_fill(~bet_mask, -1e8)

        action_logits = self._check_for_nan(action_logits, "eval_final_action_logits")
        bet_logits = self._check_for_nan(bet_logits, "eval_final_bet_logits")

        game_dist = Categorical(logits=action_logits)
        bet_dist = Categorical(logits=bet_logits)

        game_actions, bet_actions = actions[:, 0], actions[:, 1]
        log_prob = game_dist.log_prob(game_actions) + bet_dist.log_prob(bet_actions)
        entropy = game_dist.entropy() + bet_dist.entropy()

        return values, log_prob, entropy




class CustomBlackjackPolicy(MaskableActorCriticPolicy):
    """
    Custom actor-critic policy for Blackjack with trainable card counting weights.
    Features multiple action heads with layer normalization and true count calculation.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.MultiDiscrete,
        lr_schedule,
        num_decks: int = 6,
        card_weights=None,
        **kwargs,
    ):
        self.num_decks = num_decks
        self.card_weights = card_weights
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        """
        Build the network architecture.
        """
        self.features_extractor = CombinedExtractor(self.observation_space)
        self.features_dim = self.features_extractor.features_dim

        self.num_game_actions = self.action_space.nvec[0]
        self.num_bet_actions = self.action_space.nvec[1]

        if self.card_weights is None:
            self.card_weights = nn.Parameter(th.zeros(10, dtype=th.float32), requires_grad=True)
        else:
            self.card_weights = nn.Parameter(th.tensor(self.card_weights, dtype=th.float32), requires_grad=False)

        shared_net_input_dim = self.features_dim + 1
        shared_net_output_dim = 256

        self.shared_net = nn.Sequential(
            nn.Linear(shared_net_input_dim, 128),
            #nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(128, shared_net_output_dim),
            #nn.LayerNorm(256),
            nn.ReLU(),
        )

        self.game_head = nn.Sequential(
            nn.Linear(shared_net_output_dim, 64),
            #nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            #nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.num_game_actions)
        )
        self.bet_head = nn.Sequential(
            nn.Linear(shared_net_output_dim, 64),
            #nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            #nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.num_bet_actions)
        )
        self.value_net = nn.Sequential(
            nn.Linear(shared_net_output_dim, 64),
            #nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        for param_group in self.optimizer.param_groups:
            param_group['max_grad_norm'] = 0.5

    def _initialize_weights(self) -> None:
        """Initialize network weights and card_weights."""
        for module in [self.shared_net, self.game_head, self.bet_head, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu') if 'ReLU' in str(module) else 1.0)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

        with th.no_grad():
            self.card_weights[0:5] = 1.0  # 2-6
            self.card_weights[5:8] = 0.0  # 7-9
            self.card_weights[8:10] = -1.0  # 10-A
            self.card_weights += 0.1 * th.randn_like(self.card_weights)

    def _get_latent(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute latent features including true count."""
        base_features = self.extract_features(obs)

        seen_cards = obs["seen_card_counts"].to(base_features.dtype)
        max_count = 4.0 * float(self.num_decks)
        seen_cards_norm = seen_cards / (max_count + 1e-8)
        deck_remaining_fraction = obs["deck_remaining"].to(base_features.dtype).clamp(min=0.0, max=1.0)
        decks_left = (deck_remaining_fraction * float(self.num_decks)).clamp(min=1e-2)

        clamped_weights = th.clamp(self.card_weights, -2.0, 2.0)
        running_count = th.sum(seen_cards_norm * clamped_weights, dim=1, keepdim=True)
        true_count = running_count / decks_left
        true_count = th.tanh(true_count / 4.0)  # Scale to [-1, 1]


        combined_features = th.cat([base_features, true_count], dim=1)
        return self.shared_net(combined_features)

    def forward(self, obs: Dict[str, th.Tensor], deterministic: bool = False, action_masks: Optional[np.ndarray] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass with action masking."""
        latent_features = self._get_latent(obs)

        values = self.value_net(latent_features)
        game_logits = self.game_head(latent_features)
        bet_logits = self.bet_head(latent_features)

        if action_masks is not None:
            action_masks = th.as_tensor(action_masks, dtype=th.bool, device=game_logits.device)
            game_mask = action_masks[:, :self.num_game_actions]
            bet_mask = action_masks[:, self.num_game_actions:]
            game_logits = th.where(game_mask, game_logits, th.tensor(-1e8, device=game_logits.device))
            bet_logits = th.where(bet_mask, bet_logits, th.tensor(-1e8, device=bet_logits.device))

        game_dist = Categorical(logits=game_logits)
        bet_dist = Categorical(logits=bet_logits)

        if deterministic:
            game_action = th.argmax(game_logits, dim=1)
            bet_action = th.argmax(bet_logits, dim=1)
        else:
            game_action = game_dist.sample()
            bet_action = bet_dist.sample()

        actions = th.stack([game_action, bet_action], dim=1)
        log_prob = game_dist.log_prob(game_action) + bet_dist.log_prob(bet_action)

        return actions, values, log_prob

    def predict_values(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute values using custom latent features."""
        latent = self._get_latent(obs)
        return self.value_net(latent)

    def evaluate_actions(self, obs: Dict[str, th.Tensor], actions: th.Tensor, action_masks: Optional[np.ndarray] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Evaluate actions for PPO updates (ignore action_masks)."""
        latent_features = self._get_latent(obs)

        values = self.value_net(latent_features)
        game_logits = self.game_head(latent_features)
        bet_logits = self.bet_head(latent_features)

        game_dist = Categorical(logits=game_logits)
        bet_dist = Categorical(logits=bet_logits)

        game_actions, bet_actions = actions[:, 0], actions[:, 1]
        log_prob = game_dist.log_prob(game_actions) + bet_dist.log_prob(bet_actions)
        entropy = game_dist.entropy() + bet_dist.entropy()

        return values, log_prob, entropy