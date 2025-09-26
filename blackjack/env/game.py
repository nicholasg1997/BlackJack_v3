import gymnasium as gym
import numpy as np

from typing import Optional

from blackjack.deck.deck import Deck
from blackjack.player.player import Player
from blackjack.card_counting.BJCounter import CardCounter
from blackjack.env.rules import BlackJackRules
from blackjack.env.actions import Action

class BlackJack(gym.Env):
    def __init__(self, rules: BlackJackRules = BlackJackRules(), model: str = "PPO"):
        super().__init__()
        self.rules = rules
        self.model_type = model
        self.bet_bins = (self.rules.max_bet - self.rules.min_bet) + 1

        self.deck = Deck(num_decks=self.rules.num_decks)
        self.card_counter = CardCounter()
        self.player = Player(starting_balance=self.rules.starting_balance)
        self.dealer = Player(dealer=True)

        self.is_betting_phase = True
        self.round_over = False

        self._action_mapper = {
            Action.HIT: self._hit,
            Action.STAND: self._stand,
            Action.DOUBLE: self._double,
        }

        self.observation_space = self._define_obs_space()
        self.action_space = gym.spaces.Discrete(len(Action) + self.bet_bins)
        self.reset()

    def _define_obs_space(self):
        obs_space = {
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
        if self.model_type == "DQN":
            obs_space["action_mask"] = gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.bool_)
        return gym.spaces.Dict(obs_space)

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
        if self.rules.fixed_bet:
            bet_amount = self.rules.fixed_bet
        else:
            bet_amount = int(bet * (self.rules.max_bet - self.rules.min_bet) + self.rules.min_bet)
        self.player.bet = bet_amount

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.deck.deck_remaining < self.rules.reshuffle_threshold:
            self.deck.reset()
            self.card_counter.reset()
        self.dealer.reset()
        self.player.reset()
        self.round_over = False
        self.is_betting_phase = True
        return self._get_obs(), {}

    def dealer_autoplay(self):
        while True:
            if self.dealer.hand_total < self.rules.dealer_stay_value:
                self.dealer.add_card(self.draw_card())
            elif self.dealer.hand_total == self.rules.dealer_stay_value and self.dealer.has_soft_ace and self.rules.dealer_hits_soft:
                self.dealer.add_card(self.draw_card())
            else:
                break

    def step(self, action: int):
        if self.is_betting_phase:
            return self._handle_betting_phase(action)
        else:
            return self._handle_play_phase(action)

    def _handle_betting_phase(self, action: int):
        info = {}
        if action < len(Action):
            raise ValueError(f"Must place a bet during the betting phase. invalid action: {action}")
        bet_index = action - len(Action)
        bet_amount = bet_index + self.rules.min_bet
        self.player.bet = bet_amount
        info["bet_placed"] = self.player.bet

        self.deal_starting_hand()
        self.is_betting_phase = False

        reward = 0.0
        terminated = False
        return self._get_obs(), reward, terminated, False, info


    def _handle_play_phase(self, action: int):
        info = {}
        if action >= len(Action):
            raise ValueError(f"Invalid action during play phase: {action}")

        action = Action(action)
        info['action'] = action.value
        reward = 0.0

        turn_over, bonus = self._action_mapper[action]()
        terminated = False

        if turn_over:
            self.dealer_autoplay()
            reward = self._get_reward(info)
            terminated = True

        reward += bonus
        info["balance"] = self.player.balance
        return self._get_obs(), reward, terminated, False, info

    def _hit(self):
        self.player.add_card(self.draw_card())
        score_modifier = 0.0 # extra reward shaping can be calculated and done here if I need
        return self.player.is_bust, score_modifier

    def _stand(self):
        score_modifier = 0.0
        return True, score_modifier

    def _double(self):
        assert len(self.player.hand) == 2, "Player must have 2 cards to double down."
        self.player.bet *= 2
        self.player.add_card(self.draw_card())

        score_modifier = 0.0
        return True, score_modifier

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

        alpha = 0.1
        if result > 0:
            penalty = (self.rules.max_bet - player.bet) * alpha
        elif result < 0:
            penalty = (player.bet - self.rules.min_bet) * alpha
        else:
            penalty = 0

        norm_winnings = winnings / self.rules.max_bet
        norm_penalty = penalty / (self.rules.max_bet - self.rules.min_bet)
        return norm_winnings - norm_penalty

    def _calculate_reward(self, result: float) -> float:
        return result * self.player.bet

    def _get_obs(self):
        player = self.player
        dealer_one_hot_showing = [0] * 11
        if self.dealer_showing is not None:
            dealer_one_hot_showing[self.dealer_showing - 1] = 1

        obs =  {
            "player_total": np.array(player.one_hot_total, dtype=np.float32),
            "player_has_blackjack": np.array([int(player.has_blackjack)], dtype=np.float32),
            "player_is_soft": np.array([int(player.has_soft_ace)], dtype=np.float32),
            "dealer_showing": np.array(dealer_one_hot_showing, dtype=np.float32),
            "deck_remaining": np.array([self.deck.deck_remaining], dtype=np.float32),
            "deck_draw_probs": np.array(self.deck.probability_of_drawing(), dtype=np.float32),
            "is_betting_phase": np.array([int(self.is_betting_phase)], dtype=np.float32),
            "seen_card_counts": np.array(self.card_counter.normalized_seen_card_counts(num_decks=self.rules.num_decks).copy(),
                                         dtype=np.float32),
            "can_double": np.array([int(len(player.hand) == 2)], dtype=np.float32),
            "true_count": np.array([self.card_counter.true_count(self.deck.deck_remaining, self.rules.num_decks)],
                                   dtype=np.float32),

        }
        if self.model_type == "DQN":
            obs["action_mask"] = self.get_action_mask()
        return obs

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
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        if self.is_betting_phase:
            mask[len(Action):] = 1
        else:
            mask[Action.HIT.value] = 1
            mask[Action.STAND.value] = 1
            if len(self.player.hand) == 2 and self.rules.allow_double:
                mask[Action.DOUBLE.value] = 1
        assert mask.sum() > 0, "No legal moves available!"
        return mask

    def __repr__(self):
        return f"BlackJack(num_decks={self.rules.num_decks}, dealer={self.dealer}, deck_remaining={len(self.deck)})"

