import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import MaskablePPO

from blackjack.env.game import BlackJack
from stable_baselines3.common.vec_env import DummyVecEnv

from tqdm import tqdm
import matplotlib.pyplot as plt

from blackjack.env.rules import BlackJackRules

#MODEL_PATH = "logs/best_model_ev/ppobestmodel_decks_4/best_model_ev.zip"
#VEC_NORMALIZE_PATH = "logs/best_model_ev/ppobestmodel_decks_4/best_vec_normalize.pkl"
MODEL_PATH = "blackjack_agent_shaped_reward_4decks.zip"
VEC_NORMALIZE_PATH = "vec_normalize_stats_shaped_4decks.pkl"

rules = BlackJackRules(
    num_decks=4,
    min_bet=2,
    max_bet=100,
)


def mask_fn(env: gym.Env):
    return env.get_action_mask()


def make_blackjack_env(bj_rules=rules):
    def _init():
        env = BlackJack(rules=bj_rules)
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def evaluate(num_episodes=1_000):
    venv = DummyVecEnv([lambda: BlackJack()])
    venv = VecNormalize.load(VEC_NORMALIZE_PATH, venv)
    venv.training = False
    venv.norm_reward = False

    model = MaskablePPO.load(MODEL_PATH, env=venv)

    total_balance = []
    all_episode_rewards = []
    wins, losses = [], []
    placed_bets = []
    draws = 0

    #obs = venv.reset()
    with tqdm(total=num_episodes) as pbar:
        for ep in range(num_episodes):
            obs = venv.reset()
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True,
                                                action_masks=venv.get_attr("get_action_mask")[0]())
                obs, reward, done, info = venv.step(action)
                if info[0].get('bet_placed', False):
                    placed_bets.append(info[0]['bet_placed'])
            winning = info[0]['winnings']
            all_episode_rewards.append(winning)
            if winning > 0:
                wins.append(winning)
            elif winning < 0:
                losses.append(winning)
            else:
                draws += 1
            total_balance.append(info[0]['balance'])

            pbar.set_postfix({
                "EpReward": f"{winning:.2f}",
                "AvgReward": f"{sum(all_episode_rewards) / len(all_episode_rewards):.2f}",
                "WinRate": f"{len(wins) / (ep + 1) * 100:.2f}%",
            })
            pbar.update(1)


    avg_reward = sum(all_episode_rewards) / len(all_episode_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    print(f"Average Win: {avg_win}, Average Loss: {avg_loss}, Wins: {len(wins)}, Losses: {len(losses)}, Draws: {draws}")
    print(f"Win Rate: {len(wins) / num_episodes * 100:.2f}%, Loss Rate: {len(losses) / num_episodes * 100:.2f}%, Draw Rate: {draws / num_episodes * 100:.2f}%")
    print(f"Total profit over {num_episodes} episodes: {sum(all_episode_rewards)}")
    print(f"Max Win: {max(all_episode_rewards)}, Max Loss: {min(all_episode_rewards)}")
    print(f"frequency of max win: {all_episode_rewards.count(max(all_episode_rewards))}, frequency of max loss: {all_episode_rewards.count(min(all_episode_rewards))}")
    print(f"average bet placed: {sum(placed_bets) / len(placed_bets) if placed_bets else 0}")
    #print(placed_bets)
    plt.figure(figsize=(12, 6))
    plt.plot(total_balance)
    plt.xlabel('Episode')
    plt.ylabel('Balance')
    plt.title('Balance Over Episodes')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    evaluate(num_episodes=100_000)