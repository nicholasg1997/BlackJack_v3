from enum import Enum
import gymnasium as gym
import numpy as np

from typing import Optional

from blackjack.deck.deck import Deck
from blackjack.player.player import Player


class Action(Enum):
    BET = 0
    HIT = 1
    STAND = 2
    DOUBLE = 3
    #SPLIT = 3 # Splitting is not yet supported.


class BlackJack(gym.Env):
    def __init__(self, num_decks: int = 6, starting_balance: int = 1000,
                 dealer_hits_soft: bool = True, dealer_stay_value: int = 17,
                 min_bet: int = 2, max_bet: int = 20):
        super().__init__()
        self.num_decks = num_decks
        self.starting_balance = starting_balance
        self.dealer_hits_soft_17 = dealer_hits_soft
        self.dealer_stay_value = dealer_stay_value
        self.min_bet = min_bet
        self.max_bet = max_bet

        self.deck = Deck(num_decks=self.num_decks)
        self.player = Player(starting_balance=self.starting_balance)
        self.dealer = Player(dealer=True)

        self.is_bet_turn = True
        self.round_over = False
        self.reset()

        self.observation_space = gym.spaces.Dict(
            {
                "player_total": gym.spaces.Box(low=0, high=1, shape=(23,), dtype=np.float32),  # player stat (one hot encoded)
                "player_has_blackjack": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # player stat (one hot encoded)
                "player_is_soft": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # player stat (need a better name for this) (one hot encoded)
                "dealer_showing": gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32),  # dealer stat (one hot encoded)
                "deck_remaining": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),  # game stat
                "deck_card_probs": gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),  # game stat
                "deck_draw_probs": gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),  # game stat
                "is_bet_turn": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # game stat
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                "action": gym.spaces.Discrete(len(Action)),
                "bet": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )

    @property
    def dealer_showing(self):
        if len(self.dealer.hand) > 0:
            return self.dealer.hand[0]
        return None

    def set_bet(self, bet: float):
        bet_amount = int(bet * (self.max_bet - self.min_bet) + self.min_bet)
        if bet_amount > self.player.balance:
            print(f"Bet amount is below minimum. Player has run out of money.")
        self.player.bet = bet_amount

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.deck.deck_remaining < 0.25:
            self.deck.reset()
        self.dealer.reset()
        self.player.reset()
        self.is_bet_turn = True
        self.round_over = False
        #self.deal_starting_hands()

        return self._get_obs(), {}

    def deal_starting_hands(self):
        for _ in range(2):
            self.player.add_card(self.deck.draw_card())
            self.dealer.add_card(self.deck.draw_card())

    def dealer_autoplay(self):
        while True:
            print(self.dealer)
            if self.dealer.hand_total < self.dealer_stay_value:
                self.dealer.add_card(self.deck.draw_card())
            elif self.dealer.hand_total == self.dealer_stay_value and self.dealer.has_soft_ace and self.dealer_hits_soft_17:
                self.dealer.add_card(self.deck.draw_card())
            else:
                break

    def step(self, action: dict):
        # I need to figure out my order of operations more before I do this.
        # I need each player to do their turn and then once were done have the dealer go,
        # then check if each player beat the dealer and get a reward for each player

        action = Action(action["action"])
        player_done = False

        if action == Action.BET:
            self.set_bet(action["bet"])
            self.deal_starting_hands()
            self.is_bet_turn = False
        elif action == Action.HIT.value:
            player_done = self._hit()
        elif action == Action.STAND.value:
            player_done = True
        elif action == Action.DOUBLE.value:
            player_done = self._double()

        if player_done:
            self.round_over = True
        if self.round_over:
            self.dealer_autoplay()
            reward = self._get_reward()
            done = True
            obs = self._get_obs()
        else:
            reward = 0
            obs = self._get_obs()
            done = False
        return obs, reward, done, False, {}

    def _hit(self):
        self.player.add_card(self.deck.draw_card())
        return self.player.is_bust

    def _double(self):
        assert len(self.player.hand) == 2, "Player must have 2 cards to double down."
        assert self.player.balance >= self.player.bet * 2, "Player does not have enough balance to double down."
        self.player.bet *= 2
        self.player.add_card(self.deck.draw_card())
        return True

    def _split(self):
        pass

    def _get_reward(self):
        dealer_total = self.dealer.hand_total
        player = self.player
        if player.has_blackjack and not self.dealer.has_blackjack:
            player.balance += int(1.5 * player.bet)
            reward = self._calculate_reward(1.5)
        elif player.is_bust:
            player.balance -= player.bet
            reward = self._calculate_reward(-1)
        elif self.dealer.is_bust:
            player.balance += player.bet
            reward = self._calculate_reward(1)
        elif player.hand_total > dealer_total:
            player.balance += player.bet
            reward = self._calculate_reward(1)
        elif player.hand_total < dealer_total:
            player.balance -= player.bet
            reward = self._calculate_reward(-1)
        else:
            reward = 0
        return reward

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
            "deck_card_probs": np.array(self.deck.get_percentage(), dtype=np.float32),
            "deck_draw_probs": np.array(self.deck.probability_of_drawing(), dtype=np.float32),
            "is_bet_turn": np.array([int(self.is_bet_turn)], dtype=np.float32),
        }

    def get_legal_moves(self):
        if self.is_bet_turn:
            return [Action.BET]

        legal_moves = [Action.HIT, Action.STAND]
        if len(self.player.hand) == 2 and self.player.balance >= self.player.bet * 2:
            legal_moves.append(Action.DOUBLE)
        return legal_moves

    def __repr__(self):
        return f"BlackJack(num_decks={self.num_decks}, dealer={self.dealer}, deck_remaining={len(self.deck)})"




if __name__ == "__main__":
    game = BlackJack()

