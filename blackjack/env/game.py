from enum import Enum
import gymnasium as gym
import numpy as np

from typing import Optional

from blackjack.deck.deck import Deck
from blackjack.player.player import Player


class Action(Enum):
    HIT = 0
    STAND = 1
    DOUBLE = 2
    BET = 3
    #SPLIT = 3 # Splitting is not yet supported.


class BlackJack(gym.Env):
    def __init__(self, num_decks: int = 6, starting_balance: int = 1000,
                 dealer_hits_soft: bool = True, dealer_stay_value: int = 17,
                 min_bet: int = 2, max_bet: int = 20, reshuffle_threshold: float = 0.25, bet_bins: int = 18,
                 fixed_bet: Optional[int] = None, learn_count: bool = False):
        super().__init__()
        self.num_decks = num_decks
        self.reshuffle_threshold = reshuffle_threshold
        self.starting_balance = starting_balance
        self.dealer_hits_soft_17 = dealer_hits_soft
        self.dealer_stay_value = dealer_stay_value
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bet_bins = bet_bins
        self.fixed_bet = fixed_bet

        self.deck = Deck(num_decks=self.num_decks, learn_count=learn_count)
        self.player = Player(starting_balance=self.starting_balance)
        self.dealer = Player(dealer=True)

        self.is_betting_phase = True
        self.round_over = False
        self.reset()

        self.observation_space = gym.spaces.Dict(
            {
                "player_total": gym.spaces.Box(low=0, high=1, shape=(23,), dtype=np.float32),  # player stat (one hot encoded)
                "player_has_blackjack": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # player stat (one hot encoded)
                "player_is_soft": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # player stat (need a better name for this) (one hot encoded)
                "dealer_showing": gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32),  # dealer stat (one hot encoded)
                "deck_remaining": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # game stat
                #"deck_card_probs": gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),  # game stat
                "deck_draw_probs": gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),  # game stat
                "is_betting_phase": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # game stat
                "true_count": gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32), # game stat
            }
        )

        self.action_space = gym.spaces.MultiDiscrete([4, self.bet_bins])

    @property
    def dealer_showing(self):
        if len(self.dealer.hand) > 0:
            return self.dealer.hand[0]
        return None

    def set_bet(self, bet: float):
        if self.fixed_bet:
            bet_amount = self.fixed_bet
        else:
            bet_amount = int(bet * (self.max_bet - self.min_bet) + self.min_bet)
        self.player.bet = bet_amount

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.deck.deck_remaining < self.reshuffle_threshold:
            self.deck.reset()
        self.dealer.reset()
        self.player.reset()
        self.round_over = False
        self.is_betting_phase = True

        return self._get_obs(), {}

    def deal_starting_hands(self):
        for _ in range(2):
            self.player.add_card(self.deck.draw_card())
            self.dealer.add_card(self.deck.draw_card())

    def dealer_autoplay(self):
        while True:
            if self.dealer.hand_total < self.dealer_stay_value:
                self.dealer.add_card(self.deck.draw_card())
            elif self.dealer.hand_total == self.dealer_stay_value and self.dealer.has_soft_ace and self.dealer_hits_soft_17:
                self.dealer.add_card(self.deck.draw_card())
            else:
                break

    def step(self, action: dict):
        action_idx, bet_idx = action
        action = Action(action_idx)
        bet = bet_idx / (self.bet_bins - 1)
        info = {"balance": self.player.balance, "is_betting_phase": self.is_betting_phase}

        if self.is_betting_phase:
            self.set_bet(bet)
            self.deal_starting_hands()
            self.is_betting_phase = False
            info["bet_placed"] = self.player.bet
            return self._get_obs(), 0, False, False, info

        player_done = False

        if action == Action.HIT:
            player_done = self._hit()
        elif action == Action.STAND:
            player_done = True
        elif action == Action.DOUBLE:
            player_done = self._double()
        else:
            raise ValueError(f"Invalid action: {action}")

        info["is_bust"] = self.player.is_bust

        if player_done:
            self.dealer_autoplay()
            reward = self._get_reward(info)
            done = True
        else:
            reward = 0
            done = False

        info["balance"] = self.player.balance
        return self._get_obs(), reward, done, False, info

    def _hit(self):
        self.player.add_card(self.deck.draw_card())
        return self.player.is_bust

    def _double(self):
        assert len(self.player.hand) == 2, "Player must have 2 cards to double down."
        self.player.bet *= 2
        self.player.add_card(self.deck.draw_card())
        return True

    def _split(self):
        pass

    def _get_reward(self, info: dict):
        dealer_total = self.dealer.hand_total
        player = self.player
        if player.has_blackjack and not self.dealer.has_blackjack:
            result = 1.5
            info["win"] = True
            info["loss"] = False
            info["push"] = False
        elif player.is_bust:
            result = -1
            info["win"] = False
            info["loss"] = True
            info["push"] = False
        elif self.dealer.is_bust:
            result = 1
            info["win"] = True
            info["loss"] = False
            info["push"] = False
        elif player.hand_total > dealer_total:
            result = 1
            info["win"] = True
            info["loss"] = False
            info["push"] = False
        elif player.hand_total < dealer_total:
            result = -1
            info["win"] = False
            info["loss"] = True
            info["push"] = False
        else:
            result = 0
            info["win"] = False
            info["loss"] = False
            info["push"] = True

        winnings = self._calculate_reward(result)
        player.balance += winnings
        return winnings

    def _calculate_reward(self, result: float) -> float:
        return result * self.player.bet

    def _get_obs(self):
        player = self.player
        dealer_one_hot_showing = [0] * 11
        if self.dealer_showing is not None:
            dealer_one_hot_showing[self.dealer_showing - 1] = 1
        return {
            "player_total": np.array(player.one_hot_total, dtype=np.float32),
            "player_has_blackjack": np.array([int(player.has_blackjack)], dtype=np.float32),
            "player_is_soft": np.array([int(player.has_soft_ace)], dtype=np.float32),
            "dealer_showing": np.array(dealer_one_hot_showing, dtype=np.float32),
            "deck_remaining": np.array(self.deck.deck_remaining, dtype=np.float32),
            #"deck_card_probs": np.array(self.deck.get_percentage(), dtype=np.float32),
            "deck_draw_probs": np.array(self.deck.probability_of_drawing(), dtype=np.float32),
            "is_betting_phase": np.array([int(self.is_betting_phase)], dtype=np.float32),
            "true_count": np.array([np.clip(self.deck.true_count, -10, 10)], dtype=np.float32),
        }

    def get_parameters(self):
        return self.deck.get_parameters()

    def get_legal_moves(self):
        if self.is_betting_phase:
            return [Action.BET]
        legal_moves = [Action.HIT, Action.STAND]
        if len(self.player.hand) == 2:
            legal_moves.append(Action.DOUBLE)
        return legal_moves

    def get_action_mask(self):
        game_mask = np.zeros(self.action_space.nvec[0], dtype=bool)
        legal_moves = self.get_legal_moves()
        for action in legal_moves:
            game_mask[action.value] = True

        if self.is_betting_phase:
            bet_mask = np.ones(self.action_space.nvec[1], dtype=bool)
        else:
            bet_mask = np.zeros(self.action_space.nvec[1], dtype=bool)
            bet_mask[0] = True

        full_mask = np.concatenate([game_mask, bet_mask])
        return full_mask

    def __repr__(self):
        return f"BlackJack(num_decks={self.num_decks}, dealer={self.dealer}, deck_remaining={len(self.deck)})"


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    import gymnasium as gym
    env = BlackJack()
    check_env(env)

