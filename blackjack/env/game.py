from enum import Enum
import gymnasium as gym
import numpy as np

from typing import Optional

from blackjack.deck.deck import Deck
from blackjack.player.player import Player
from blackjack.card_counting.BJCounter import CardCounter
from dataclasses import dataclass, field

@dataclass(frozen=True)
class BlackJackRules:
    # deck rules and parameters
    num_decks: int = 6
    reshuffle_threshold: float = 0.25

    #dealer rules
    dealer_stay_value: int = 17
    dealer_hits_soft: bool = True

    # player allowed moves
    allow_double: bool = True
    allow_surrender: bool = False # Not yet implemented
    allow_split: bool = False     # Not yet implemented
    max_splits: int = 3           # Not yet implemented
    double_after_split: bool = False # Not yet implemented
    allow_insurance: bool = False # Not yet implemented
    resplit_aces: bool = False   # Not yet implemented

    # betting rules
    min_bet: int = 2
    max_bet: int = 10

    # other
    blackjack_reward: float = 1.5
    starting_balance: int = 1000


#TODO: Add splitting functionality.
class Action(Enum):
    HIT = 0
    STAND = 1
    DOUBLE = 2
    #SPLIT = 3 # Splitting is not yet supported.

#TODO: improve reward shaping.
class BlackJack(gym.Env):
    def __init__(self, num_decks: int = 0, starting_balance: int = 1000,
                 dealer_hits_soft: bool = True, dealer_stay_value: int = 17,
                 min_bet: int = 2, max_bet: int = 10, reshuffle_threshold: float = 0.25,
                 fixed_bet: Optional[int] = None):
        super().__init__()
        self.num_decks = np.random.randint(2, 7) if num_decks == 0 else num_decks
        self.reshuffle_threshold = reshuffle_threshold
        self.starting_balance = starting_balance
        self.dealer_hits_soft_17 = dealer_hits_soft
        self.dealer_stay_value = dealer_stay_value
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bet_bins = (self.max_bet - self.min_bet) + 1
        self.fixed_bet = fixed_bet

        self.deck = Deck(num_decks=self.num_decks)
        self.card_counter = CardCounter()
        self.player = Player(starting_balance=self.starting_balance)
        self.dealer = Player(dealer=True)

        self.is_betting_phase = True
        self.round_over = False

        self.observation_space = gym.spaces.Dict(
            {
                "player_total": gym.spaces.Box(low=0, high=1, shape=(23,), dtype=np.float32),
                "player_has_blackjack": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "player_is_soft": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "dealer_showing": gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32),
                "deck_remaining": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "deck_draw_probs": gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),
                "is_betting_phase": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "seen_card_counts": gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
                "can_double": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "true_count": gym.spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(len(Action) + self.bet_bins)
        self.reset()

    @property
    def dealer_showing(self):
        if len(self.dealer.hand) > 0:
            return self.dealer.hand[0]
        return None

    def draw_card(self):
        card = self.deck.draw_card()
        self.card_counter.update_count(card)
        return card

    def deal_starting_hand(self):
        for _ in range(2):
            self.player.add_card(self.draw_card())
            self.dealer.add_card(self.draw_card())

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
            self.card_counter.reset()
        self.dealer.reset()
        self.player.reset()
        self.round_over = False
        self.is_betting_phase = True
        return self._get_obs(), {}

    def dealer_autoplay(self):
        while True:
            if self.dealer.hand_total < self.dealer_stay_value:
                self.dealer.add_card(self.draw_card())
            elif self.dealer.hand_total == self.dealer_stay_value and self.dealer.has_soft_ace and self.dealer_hits_soft_17:
                self.dealer.add_card(self.draw_card())
            else:
                break

    def step(self, action: int):
        info = {}
        reward = 0

        if self.is_betting_phase:
            if action < len(Action):
                raise ValueError(f"Must place a bet during the betting phase. invalid action: {action}")
            bet_index = action - len(Action)
            bet_amount = bet_index + self.min_bet
            self.player.bet = bet_amount
            self.deal_starting_hand()
            self.is_betting_phase = False
            info["bet_placed"] = self.player.bet
            true_count = self.card_counter.true_count(self.deck.deck_remaining, self.num_decks)
            normalized_bet = (bet_amount - self.min_bet) / (self.max_bet - self.min_bet)
            alpha = 0.05
            shaped_reward = alpha * true_count * normalized_bet
            return self._get_obs(), shaped_reward, False, False, info

        if action >= len(Action):
            raise ValueError(f"Invalid action during play phase: {action}")

        action = Action(action)

        if action == Action.HIT:
            old_total = self.player.hand_total
            player_done = self._hit()
            info['action'] = Action.HIT.value
            new_total = self.player.hand_total
            delta_total = new_total - old_total
            if player_done:  # busted
                hit_reward = -0.2
            elif delta_total > 0:
                hit_reward = np.clip(delta_total / 21.0, 0, 0.1)
            else:  # something unexpected
                hit_reward = 0
            reward = hit_reward

        elif action == Action.STAND:
            reward = 0.005
            player_done = True
            info['action'] = Action.STAND.value

        elif action == Action.DOUBLE:
            player_done = self._double()
            info['action'] = Action.DOUBLE.value
        else:
            raise ValueError(f"Invalid action: {action}")

        info["is_bust"] = self.player.is_bust

        if player_done:
            self.dealer_autoplay()
            reward = self._get_reward(info)
            done = True
        else:
            done = False

        info["balance"] = self.player.balance
        return self._get_obs(), reward, done, False, info

    def _hit(self):
        self.player.add_card(self.draw_card())
        return self.player.is_bust

    def _double(self):
        assert len(self.player.hand) == 2, "Player must have 2 cards to double down."
        self.player.bet *= 2
        self.player.add_card(self.draw_card())
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
        info["winnings"] = winnings
        player.balance += winnings

        const = 0.75
        if result > 0:
            penalty = (self.max_bet - player.bet) * const
        elif result < 0:
            penalty = (player.bet - self.min_bet) * const
        else:
            penalty = 0

        norm_winnings = winnings / self.max_bet
        norm_penalty = penalty / (self.max_bet - self.min_bet)
        return norm_winnings - norm_penalty

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
            "deck_remaining": np.array([self.deck.deck_remaining], dtype=np.float32),
            "deck_draw_probs": np.array(self.deck.probability_of_drawing(), dtype=np.float32),
            "is_betting_phase": np.array([int(self.is_betting_phase)], dtype=np.float32),
            "seen_card_counts": np.array(self.card_counter.normalized_seen_card_counts(num_decks=self.num_decks).copy(),
                                         dtype=np.float32),
            "can_double": np.array([int(len(player.hand) == 2)], dtype=np.float32),
            "true_count": np.array([self.card_counter.true_count(self.deck.deck_remaining, self.num_decks)],
                                   dtype=np.float32),
        }

    def get_parameters(self):
        return self.card_counter.get_parameters()

    def get_legal_moves(self):
        if self.is_betting_phase:
            return list(range(3, 3 + self.bet_bins))
        legal_moves = [Action.HIT.value, Action.STAND.value]
        if len(self.player.hand) == 2:
            legal_moves.append(Action.DOUBLE.value)
        return legal_moves

    def get_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.int32)
        if self.is_betting_phase:
            mask[len(Action):] = 1
        else:
            mask[Action.HIT.value] = 1
            mask[Action.STAND.value] = 1
            if len(self.player.hand) == 2:
                mask[Action.DOUBLE.value] = 1
        assert mask.sum() > 0, "No legal moves available!"
        return mask

    def __repr__(self):
        return f"BlackJack(num_decks={self.num_decks}, dealer={self.dealer}, deck_remaining={len(self.deck)})"

