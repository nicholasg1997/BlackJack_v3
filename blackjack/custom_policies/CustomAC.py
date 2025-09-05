import torch as th
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import MlpExtractor
from torch.distributions import Categorical


class CustomAC(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs,
                         net_arch=dict(pi=[64, 64], vf=[128, 128]))

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
        super().__init__(observation_space, action_space, *args, **kwargs,
                         net_arch=dict(pi=[64, 64], vf=[128, 128]))

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
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # The 'net_arch' and 'features_extractor' will be passed in via kwargs from the PPO constructor
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        self.num_game_actions = self.action_space.nvec[0]
        self.num_bet_actions = self.action_space.nvec[1]

        # Get the latent dimension from the MlpExtractor created by the super() call
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Define your custom heads that will be placed on top of the base policy network
        self.game_head = nn.Sequential(
            nn.Linear(latent_dim_pi, 128),
            nn.LayerNorm(128),  # Swapped BatchNorm for LayerNorm, often more stable in RL
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.num_game_actions)
        )
        self.bet_head = nn.Sequential(
            nn.Linear(latent_dim_pi, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.num_bet_actions)
        )

        # Initialize weights properly to prevent NaN
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization to prevent NaN"""
        for module in [self.game_head, self.bet_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _check_for_nan(self, tensor, name="tensor"):
        """Check for NaN values and replace them with zeros"""
        if th.isnan(tensor).any():
            print(f"Warning: NaN detected in {name}, replacing with zeros")
            return th.where(th.isnan(tensor), th.zeros_like(tensor), tensor)
        return tensor

    def forward(self, obs, deterministic=False, action_masks=None):
        # Let the base class handle feature extraction and the main MLP
        features = self.extract_features(obs)
        features = self._check_for_nan(features, "features")

        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi = self._check_for_nan(latent_pi, "latent_pi")
        latent_vf = self._check_for_nan(latent_vf, "latent_vf")

        values = self.value_net(latent_vf)  # Get value from the base value network

        # Pass the policy features through your custom heads
        action_logits = self.game_head(latent_pi)
        bet_logits = self.bet_head(latent_pi)

        # Check for NaN in logits
        action_logits = self._check_for_nan(action_logits, "action_logits")
        bet_logits = self._check_for_nan(bet_logits, "bet_logits")

        # Masking and action sampling logic remains the same
        if action_masks is not None:
            if not isinstance(action_masks, th.Tensor):
                action_masks = th.from_numpy(action_masks).to(features.device)
            game_mask = action_masks[:, :self.num_game_actions].bool()
            bet_mask = action_masks[:, self.num_game_actions:].bool()
            # Use -1e8 instead of -float('inf') for numerical stability
            action_logits = action_logits.masked_fill(~game_mask, -1e8)
            bet_logits = bet_logits.masked_fill(~bet_mask, -1e8)

        # Final check before creating distributions
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

        # Check for NaN in logits
        action_logits = self._check_for_nan(action_logits, "eval_action_logits")
        bet_logits = self._check_for_nan(bet_logits, "eval_bet_logits")

        if action_masks is not None:
            if not isinstance(action_masks, th.Tensor):
                action_masks = th.from_numpy(action_masks).to(features.device)
            game_mask = action_masks[:, :self.num_game_actions].bool()
            bet_mask = action_masks[:, self.num_game_actions:].bool()
            action_logits = action_logits.masked_fill(~game_mask, -1e8)
            bet_logits = bet_logits.masked_fill(~bet_mask, -1e8)

        # Final check before creating distributions
        action_logits = self._check_for_nan(action_logits, "eval_final_action_logits")
        bet_logits = self._check_for_nan(bet_logits, "eval_final_bet_logits")

        game_dist = Categorical(logits=action_logits)
        bet_dist = Categorical(logits=bet_logits)

        game_actions, bet_actions = actions[:, 0], actions[:, 1]
        log_prob = game_dist.log_prob(game_actions) + bet_dist.log_prob(bet_actions)
        entropy = game_dist.entropy() + bet_dist.entropy()

        return values, log_prob, entropy


