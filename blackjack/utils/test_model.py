import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import MaskablePPO

from blackjack.env.game import BlackJack
from stable_baselines3.common.vec_env import DummyVecEnv

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from blackjack.env.rules import BlackJackRules

#MODEL_PATH = "logs/best_model_ev/ppobestmodel_decks_6/best_model_ev.zip"
#VEC_NORMALIZE_PATH = "logs/best_model_ev/ppobestmodel_decks_6/best_vec_normalize.pkl"
MODEL_PATH = "blackjack_agent_shaped_reward_6decks.zip"
VEC_NORMALIZE_PATH = "vec_normalize_stats_shaped_6decks.pkl"

num_decks = 6
rules = BlackJackRules(
    num_decks=num_decks,
    min_bet=2,
    max_bet=100,
    allow_split=True
)

def mask_fn(env: gym.Env):
    return env.get_action_mask()

def make_blackjack_env(bj_rules=rules):
    def _init():
        env = BlackJack(rules=bj_rules, model="PPO")
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def evaluate(num_episodes=30_000):
    # Create environment with same rules as training
    venv = DummyVecEnv([make_blackjack_env(bj_rules=rules)])
    venv = VecNormalize.load(VEC_NORMALIZE_PATH, venv)
    venv.training = False
    venv.norm_reward = False

    model = MaskablePPO.load(MODEL_PATH, env=venv)

    total_balance = []
    all_episode_rewards = []
    wins, losses, draws = [], [], []
    placed_bets = []
    total_hands = 0

    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for ep in range(num_episodes):
            obs = venv.reset()
            done = False
            episode_winnings = 0.0
            episode_bets = 0.0
            episode_wins = 0
            episode_losses = 0
            episode_draws = 0

            while not done:
                action, _states = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=venv.get_attr("get_action_mask")[0]()
                )
                obs, reward, done, info = venv.step(action)
                if done and "hands" in info[0]:
                    # Sum winnings and bets, count outcomes per hand
                    for hand_info in info[0]["hands"]:
                        episode_winnings += hand_info.get("winnings", 0.0)
                        episode_bets += hand_info.get("bet", 0.0)
                        if hand_info.get("win", False):
                            episode_wins += 1
                        elif hand_info.get("loss", False):
                            episode_losses += 1
                        elif hand_info.get("push", False):
                            episode_draws += 1
                    total_hands += len(info[0]["hands"])

            all_episode_rewards.append(episode_winnings)
            placed_bets.append(episode_bets)
            if episode_wins > 0:
                wins.append(episode_winnings)
            if episode_losses > 0:
                losses.append(episode_winnings)
            if episode_draws > 0:
                draws.append(episode_winnings)
            total_balance.append(info[0]["balance"])

            pbar.set_postfix({
                "EpReward": f"{episode_winnings:.2f}",
                "AvgReward": f"{np.mean(all_episode_rewards):.2f}",
                "WinRate": f"{len(wins) / (ep + 1) * 100:.2f}%",
                "HandWinRate": f"{sum(len(info[0]['hands']) for w in wins) / total_hands * 100:.2f}%",
            })
            pbar.update(1)

    avg_reward = np.mean(all_episode_rewards)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    avg_bet = np.mean(placed_bets) if placed_bets else 0
    hand_win_rate = sum(len(info[0]["hands"]) for w in wins) / total_hands if total_hands > 0 else 0
    hand_loss_rate = sum(len(info[0]["hands"]) for l in losses) / total_hands if total_hands > 0 else 0
    hand_draw_rate = sum(len(info[0]["hands"]) for d in draws) / total_hands if total_hands > 0 else 0

    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"average reward over {total_hands} hands: {sum(all_episode_rewards) / total_hands:.2f}")
    print(f"Average Win: {avg_win:.2f}, Average Loss: {avg_loss:.2f}")
    print(f"Wins: {len(wins)}, Losses: {len(losses)}, Draws: {len(draws)}")
    print(f"Per-Episode Win Rate: {len(wins) / num_episodes * 100:.2f}%, Loss Rate: {len(losses) / num_episodes * 100:.2f}%, Draw Rate: {len(draws) / num_episodes * 100:.2f}%")
    print(f"Per-Hand Win Rate: {hand_win_rate * 100:.2f}%, Loss Rate: {hand_loss_rate * 100:.2f}%, Draw Rate: {hand_draw_rate * 100:.2f}%")
    print(f"Total profit over {num_episodes} episodes: {sum(all_episode_rewards):.2f}")
    print(f"Max Win: {max(all_episode_rewards) if all_episode_rewards else 0:.2f}, Max Loss: {min(all_episode_rewards) if all_episode_rewards else 0:.2f}")
    print(f"Frequency of max win: {all_episode_rewards.count(max(all_episode_rewards)) if all_episode_rewards else 0}, Frequency of max loss: {all_episode_rewards.count(min(all_episode_rewards)) if all_episode_rewards else 0}")
    print(f"Average bet placed: {avg_bet:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(total_balance)
    plt.xlabel('Episode')
    plt.ylabel('Balance')
    plt.title(f'Balance Over Episodes ({num_decks} decks, Avg Reward: {avg_reward:.2f})')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    evaluate(num_episodes=50_000)