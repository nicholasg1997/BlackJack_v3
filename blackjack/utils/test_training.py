from sb3_contrib.common.wrappers import ActionMasker

from blackjack.env.game import BlackJack
from blackjack.utils.logging import BlackjackMetricsCallback, ReturnMetricsCallback
from blackjack.custom_policies.CustomAC import CustomBlackjackPolicy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import LinearSchedule
import logging
import torch as th

def mask_fn(env: BlackJack):
    mask = env.get_action_mask()
    logging.debug(f"Action mask: {mask}")  # Debug: Shape (22,)
    return mask

def make_blackjack_env(fixed_bet=None, num_decks=6):
    def _init():
        env = BlackJack(fixed_bet=fixed_bet, num_decks=num_decks)
        env = ActionMasker(env, mask_fn)
        obs, _ = env.reset()
        logging.debug(f"Observation keys: {list(obs.keys())}")
        return env
    return _init

def main():
    NUM_ENVS = 64
    POLICY_KWARGS = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        optimizer_class=th.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-5)
    )

    logging.basicConfig(level=logging.INFO)

    # Phase 1: Basic Strategy (Game Actions)
    print("--- Starting Phase 1: Basic Strategy ---")
    vec_env_p1 = SubprocVecEnv([make_blackjack_env(fixed_bet=11) for _ in range(NUM_ENVS)])
    vec_env_p1 = VecNormalize(vec_env_p1, norm_reward=True, norm_obs=True, gamma=0.99)

    model = MaskablePPO(
        CustomBlackjackPolicy,
        vec_env_p1,
        learning_rate=LinearSchedule(3e-4, 1e-5, 1.0),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        ent_coef=0.05,
        verbose=1,
        tensorboard_log="./logs/",
        policy_kwargs=POLICY_KWARGS,
    )
    # Freeze card_weights and bet_head
    model.policy.card_weights.requires_grad = False

    print("Starting Phase 1 training...")
    model.learn(
        total_timesteps=8_000_000,
        use_masking=True,
        callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()]
    )

    model.save("phase1_basic_strategy.zip")
    vec_env_p1.save("vec_normalize_stats.pkl")
    vec_env_p1.close()

    # Phase 2: Card Counting and Betting
    print("\n--- Starting Phase 2: Card Counting and Betting ---")
    vec_env_p2 = SubprocVecEnv([make_blackjack_env(fixed_bet=None) for _ in range(NUM_ENVS)])
    vec_env_p2 = VecNormalize.load("vec_normalize_stats.pkl", vec_env_p2)
    vec_env_p2.norm_reward = True  # Re-enable for stability
    vec_env_p2.norm_obs = True
    vec_env_p2.training = True

    model = MaskablePPO.load("phase1_basic_strategy.zip", env=vec_env_p2)
    # Freeze game_head, unfreeze card_weights and bet_head
    for param in model.policy.game_head.parameters():
        param.requires_grad = False
    model.policy.card_weights.requires_grad = True
    for param in model.policy.bet_head.parameters():
        param.requires_grad = True
    model.learning_rate = LinearSchedule(1e-4, 1e-5, 1.0)
    model.ent_coef = 0.15

    print("Starting Phase 2 training...")
    model.learn(
        total_timesteps=5_000_000,
        use_masking=True,
        callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()]
    )

    model.save("phase2_betting.zip")
    vec_env_p2.save("vec_normalize_stats.pkl")
    vec_env_p2.close()

    # Phase 3: Fine-Tuning
    print("\n--- Starting Phase 3: Fine-Tuning ---")
    vec_env_p3 = SubprocVecEnv([make_blackjack_env(fixed_bet=None) for _ in range(NUM_ENVS)])
    vec_env_p3 = VecNormalize.load("vec_normalize_stats.pkl", vec_env_p3)
    vec_env_p3.norm_reward = True
    vec_env_p3.norm_obs = True
    vec_env_p3.training = True

    model = MaskablePPO.load("phase2_betting.zip", env=vec_env_p3)
    # Unfreeze all parameters
    for param in model.policy.parameters():
        param.requires_grad = True
    model.learning_rate = LinearSchedule(1e-5, 1e-6, 1.0)
    model.ent_coef = 0.15

    print("Starting Phase 3 training...")
    model.learn(
        total_timesteps=5_000_000,
        use_masking=True,
        callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()]
    )

    model.save("phase3_finetune.zip")
    vec_env_p3.save("vec_normalize_stats.pkl")
    vec_env_p3.close()

    print("Final card weights:", model.policy.card_weights.detach().cpu().numpy())

if __name__ == "__main__":
    main()