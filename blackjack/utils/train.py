from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
import numpy as np

from blackjack.env.game import BlackJack
from blackjack.utils import wrappers
from blackjack.custom_policies.CustomAC import CustomAC, SplitHeadAC, FullySeparatedAC
from blackjack.utils.logging import BlackjackMetricsCallback, ReturnMetricsCallback
from stable_baselines3.common.torch_layers import CombinedExtractor, BaseFeaturesExtractor
import torch.nn as nn
import torch as th


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)

        if hasattr(observation_space, 'shape'):
            input_dim = observation_space.shape[0]
            self.extractor = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            )
            self._features_dim = 64
        else:
            self.extractors = nn.ModuleDict({
                "player_total": nn.Sequential(nn.Linear(23, 32), nn.LayerNorm(32), nn.ReLU()),
                "player_has_blackjack": nn.Sequential(nn.Linear(1, 16), nn.LayerNorm(16), nn.ReLU()),
                "player_is_soft": nn.Sequential(nn.Linear(1, 16), nn.LayerNorm(16), nn.ReLU()),
                "dealer_showing": nn.Sequential(nn.Linear(11, 32), nn.LayerNorm(32), nn.ReLU()),
                "deck_remaining": nn.Sequential(nn.Linear(1, 16), nn.LayerNorm(16), nn.ReLU()),
                "deck_draw_probs": nn.Sequential(nn.Linear(10, 32), nn.LayerNorm(32), nn.ReLU()),
                "is_betting_phase": nn.Sequential(nn.Linear(1, 16), nn.LayerNorm(16), nn.ReLU()),
                "true_count": nn.Sequential(nn.Linear(1, 16), nn.LayerNorm(16), nn.ReLU()),
            })

            with th.no_grad():
                sample = observation_space.sample()
                if isinstance(sample, dict):
                    sample = {k: th.as_tensor(v[None]).float() for k, v in sample.items()}
                    out = []
                    for k, extractor in self.extractors.items():
                        out.append(extractor(sample[k]))
                    total_dim = sum([x.shape[1] for x in out])
                    self._features_dim = total_dim

    def forward(self, obs):
        if hasattr(self, 'extractor'):
            result = self.extractor(obs)
            if th.isnan(result).any():
                print("Warning: NaN detected in feature extractor output")
                result = th.where(th.isnan(result), th.zeros_like(result), result)
            return result
        else:
            outs = []
            for k, extractor in self.extractors.items():
                out = extractor(obs[k])
                if th.isnan(out).any():
                    print(f"Warning: NaN detected in extractor {k}")
                    out = th.where(th.isnan(out), th.zeros_like(out), out)
                outs.append(out)
            return th.cat(outs, dim=1)



def mask_fn(env, bet_bins=18):
    mask_parts = env.get_action_mask()
    if isinstance(mask_parts, list) and len(mask_parts) == 2:
        action_mask, bet_mask = mask_parts
        return np.concatenate([action_mask, bet_mask])
    else:
        action_mask = mask_parts
        bet_mask = np.ones(bet_bins, dtype=np.float32)
        return np.concatenate([action_mask, bet_mask])


def make_env(fixed_bet=None, learn_count=False):
    env = wrappers.FlattenObsWrapper(BlackJack(num_decks=3, fixed_bet=fixed_bet, learn_count=learn_count))
    env = wrappers.MaskWrapper(env, bet_bins=18)
    env = ActionMasker(env, mask_fn)
    return env


def get_base_env_parameters(vec_env):
    """
    Helper function to safely extract parameters from vectorized environments
    """
    try:
        if hasattr(vec_env, 'get_attr'):
            env_params = []
            try:
                params_list = vec_env.get_attr('get_parameters')
                for get_params_fn in params_list:
                    if callable(get_params_fn):
                        params = get_params_fn()
                        env_params.append(params)
                    else:
                        env_params.append(None)
            except (AttributeError, Exception) as e:
                print(f"Could not retrieve environment parameters: {e}")
                return []

            return env_params
        else:
            print("Vector environment does not support get_attr method")
            return []
    except Exception as e:
        print(f"Error getting environment parameters: {e}")
        return []


def main():
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    vec_env = SubprocVecEnv([lambda: make_env(fixed_bet=10, learn_count=False) for _ in range(64)])
    vec_env = VecNormalize(vec_env, norm_reward=False, norm_obs=True, gamma=0.99)

    model = MaskablePPO(
        policy=FullySeparatedAC,
        env=vec_env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        target_kl=0.02,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
    )

    # Phase 1: Train with fixed bets, Hi-Lo
    model.learn(total_timesteps=30_000_000, use_masking=True, callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()])

    # Save model
    model.save("fixed_bet_hilo.zip")

    # Phase 2: Variable bets
    vec_env = SubprocVecEnv([lambda: make_env(fixed_bet=None, learn_count=False) for _ in range(64)])
    vec_env = VecNormalize(vec_env, norm_reward=False, norm_obs=True)
    model = MaskablePPO.load("fixed_bet_hilo.zip", env=vec_env)

    env_params = get_base_env_parameters(vec_env)
    if env_params and any(p for p in env_params):
        valid_params = [p for p in env_params if p and "card_weights" in p]
        if valid_params:
            card_weights = [p["card_weights"] for p in valid_params]
            all_params = list(model.policy.parameters()) + card_weights
            model.optimizer = th.optim.Adam(all_params, lr=1e-4, weight_decay=1e-5)

    model.learn(total_timesteps=40_000_000, use_masking=True,
                callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()])



    # Phase 3: Fine-tune with learned weights
    vec_env = SubprocVecEnv([lambda: make_env(fixed_bet=None, learn_count=True) for _ in range(64)])
    vec_env = VecNormalize(vec_env, norm_reward=False, norm_obs=True)
    model = MaskablePPO.load("fixed_bet_hilo.zip", env=vec_env)

    env_params = get_base_env_parameters(vec_env)

    if env_params and any(p for p in env_params):
        valid_params = [p for p in env_params if p and "card_weights" in p]
        if valid_params:
            card_weights = [p["card_weights"] for p in valid_params]
            all_params = list(model.policy.parameters()) + card_weights
            model.optimizer = th.optim.Adam(all_params, lr=1e-4, weight_decay=1e-5)
        else:
            print("No valid environment parameters found, using default optimizer")
    else:
        print("Could not retrieve environment parameters, using default optimizer")

    model.learn(total_timesteps=50_000_000, use_masking=True, callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()])

    if env_params and env_params[0]:
        print("Learned card weights (2-11):", env_params[0]["card_weights"].detach().numpy())


if __name__ == "__main__":
    main()

