import torch as th
from torch.optim import Adam
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

from blackjack.env.game import BlackJack
from blackjack.utils.logging import BlackjackMetricsCallback, ReturnMetricsCallback


def mask_fn(env: BlackJack):
    return env.get_action_mask()


def make_blackjack_env(fixed_bet=None, learn_count=False, num_decks=5, card_weights=None):
    def _init():
        env = BlackJack(fixed_bet=fixed_bet, learn_count=learn_count, num_decks=num_decks, card_weights=card_weights)
        env = ActionMasker(env, mask_fn)
        return env

    return _init


def main():
    NUM_ENVS = 64

    print("--- Starting Phase 1: Basic Strategy ---")

    vec_env_p1 = SubprocVecEnv([make_blackjack_env(fixed_bet=10, learn_count=False) for _ in range(NUM_ENVS)])
    vec_env_p1 = VecNormalize(vec_env_p1, norm_reward=True, norm_obs=True, gamma=0.99)

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env_p1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/",
    )

    print("Starting training...")
    model.learn(total_timesteps=3_000_000, use_masking=True, callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()])

    model.save("phase1_basic_strategy.zip")
    vec_env_p1.save("vec_normalize_stats.pkl")
    vec_env_p1.close()

    print("--- Phase 1 Complete ---")
    print("\n--- Starting Phase 2: Betting Strategy ---")
    vec_env_p2 = SubprocVecEnv([make_blackjack_env(fixed_bet=None, learn_count=False) for _ in range(NUM_ENVS)])
    vec_env_p2 = VecNormalize.load("vec_normalize_stats.pkl", vec_env_p2)

    model = MaskablePPO.load("phase1_basic_strategy.zip", env=vec_env_p2)
    model.learn(total_timesteps=3_000_000, use_masking=True, callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()])

    model.save("phase2_betting.zip")
    vec_env_p2.save("vec_normalize_stats.pkl")
    vec_env_p2.close()

    # --- Phase 3: Fine-Tune Card Counting ---
    print("\n--- Starting Phase 3: Learn Card Weights ---")
    initial_weights = th.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=th.float32)
    card_weights_tensor = th.nn.Parameter(initial_weights)

    env_fns = [
        make_blackjack_env(fixed_bet=None, learn_count=True, card_weights=card_weights_tensor)
        for _ in range(NUM_ENVS)
    ]

    vec_env_p3 = SubprocVecEnv(env_fns)
    vec_env_p3 = VecNormalize.load("vec_normalize_stats.pkl", vec_env_p3)

    model = MaskablePPO.load("phase2_betting.zip", env=vec_env_p3)

    all_params = list(model.policy.parameters()) + [card_weights_tensor]
    model.policy.optimizer = Adam(all_params, lr=model.learning_rate)

    model.learn(total_timesteps=5_000_000, use_masking=True, callback=[BlackjackMetricsCallback(), ReturnMetricsCallback()])
    model.save("phase3_final_agent.zip")

    print("\n--- Training Complete ---")
    print("Final learned card weights (for cards 2 through 11):")
    print(card_weights_tensor.detach().numpy())
    vec_env_p3.close()


if __name__ == "__main__":
    main()
