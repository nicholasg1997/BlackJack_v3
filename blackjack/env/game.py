import gymnasium as gym
import numpy as np

from typing import Optional

from blackjack.deck.deck import Deck
from blackjack.player.player import Player, Hand
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
            Action.SPLIT: self._split,
        }


        self.action_space = gym.spaces.Discrete(len(Action) + self.bet_bins)
        self.observation_space = self._define_obs_space()
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
                "can_split": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "true_count": gym.spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32),
            }
        if self.model_type == "DQN":
            obs_space["action_mask"] = gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.bool_)
        return gym.spaces.Dict(obs_space)

    @property
    def dealer_showing(self):
        if self.dealer.hands and len(self.dealer.current_hand.cards) > 0:
            return self.dealer.current_hand.cards[0]
        return None

    def draw_card(self):
        card = self.deck.draw_card()
        self.card_counter.update_count(card)
        return card

    def deal_starting_hand(self):
        player_hand = Hand(bet=self.player.bet)
        self.player.add_hand(player_hand)
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
            if self.dealer.current_hand.total < self.rules.dealer_stay_value:
                self.dealer.current_hand.add_card(self.draw_card())
            elif self.dealer.current_hand.total == self.rules.dealer_stay_value and self.dealer.current_hand.has_soft_ace and self.rules.dealer_hits_soft:
                self.dealer.current_hand.add_card(self.draw_card())
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
        terminated = False

        turn_over, bonus = self._action_mapper[action]()
        reward += bonus

        if self.player.current_hand.is_bust:
            info["is_bust"] = True
            turn_over = True

        if turn_over:
            self.player.current_hand.done = True
            next_player_idx = self.player.current_hand_index + 1
            while next_player_idx < len(self.player.hands) and self.player.hands[next_player_idx].done:
                next_player_idx += 1
            if next_player_idx < len(self.player.hands):
                self.player.current_hand_index = next_player_idx
            else:
                self.dealer_autoplay()
                reward = self._get_total_reward(info)
                terminated = True

        info["balance"] = self.player.balance
        info["bet_placed"] = self.player.bet
        return self._get_obs(), reward, terminated, False, info

    def _hit(self):
        self.player.current_hand.add_card(self.draw_card())
        score_modifier = 0.0 # extra reward shaping can be calculated and done here if I need
        return self.player.current_hand.is_bust, score_modifier

    def _stand(self):
        score_modifier = 0.0
        return True, score_modifier

    def _double(self):
        assert len(self.player.current_hand.cards) == 2, "Player must have 2 cards to double down."
        self.player.current_hand.bet *= 2
        self.player.current_hand.add_card(self.draw_card())

        score_modifier = 0.0
        return True, score_modifier

    def _split(self):
        current_hand = self.player.current_hand
        assert len(current_hand.cards) == 2 and current_hand.cards[0] == current_hand.cards[1]

        card1, card2 = current_hand.cards

        current_hand.cards = []
        current_hand.aces = 0
        current_hand.add_card(card1)

        new_hand = Hand(bet=current_hand.bet)
        new_hand.add_card(card2)

        self.player.hands.insert(self.player.current_hand_index + 1, new_hand)

        current_hand.add_card(self.draw_card())
        new_hand.add_card(self.draw_card())

        score_modifier = 0.0
        return False, score_modifier

    def _get_reward_for_hand(self, hand: Hand, dealer_total: int, hand_info: dict) -> float:
        hand_info["bet"] = hand.bet
        if hand.has_blackjack and not self.dealer.has_blackjack:
            result = 1.5
            hand_info["win"] = True
            hand_info["loss"] = False
            hand_info["push"] = False
        elif hand.is_bust:
            result = -1
            hand_info["win"] = False
            hand_info["loss"] = True
            hand_info["push"] = False
        elif self.dealer.is_bust:
            result = 1
            hand_info["win"] = True
            hand_info["loss"] = False
            hand_info["push"] = False
        elif hand.total > dealer_total:
            result = 1
            hand_info["win"] = True
            hand_info["loss"] = False
            hand_info["push"] = False
        elif hand.total < dealer_total:
            result = -1
            hand_info["win"] = False
            hand_info["loss"] = True
            hand_info["push"] = False
        else:
            result = 0
            hand_info["win"] = False
            hand_info["loss"] = False
            hand_info["push"] = True

        winnings = result * hand.bet
        hand_info["winnings"] = winnings
        self.player.balance += winnings

        alpha = 0.05
        if result > 0:
            penalty = (self.rules.max_bet - hand.bet) * alpha
        elif result < 0:
            penalty = (hand.bet - self.rules.min_bet) * alpha
        else:
            penalty = 0

        norm_winnings = winnings / self.rules.max_bet
        norm_penalty = penalty / (self.rules.max_bet - self.rules.min_bet)
        return norm_winnings - norm_penalty

    def _get_total_reward(self, info: dict) -> float:
        dealer_total = self.dealer.hand_total
        total_reward = 0.0
        info["hands"] = []
        for hand in self.player.hands:
            hand_info = {}
            total_reward += self._get_reward_for_hand(hand, dealer_total, hand_info)
            info["hands"].append(hand_info)
        return total_reward

    def _get_obs(self):
        if self.is_betting_phase or len(self.player.hands) == 0:
            dealer_one_hot_showing = [0] * 11
            obs = {
                "player_total": np.array([0] * 23, dtype=np.float32),
                "player_has_blackjack": np.array([0], dtype=np.float32),
                "player_is_soft": np.array([0], dtype=np.float32),
                "dealer_showing": np.array(dealer_one_hot_showing, dtype=np.float32),
                "deck_remaining": np.array([self.deck.deck_remaining], dtype=np.float32),
                "deck_draw_probs": np.array(self.deck.probability_of_drawing(), dtype=np.float32),
                "is_betting_phase": np.array([1], dtype=np.float32),
                "seen_card_counts": np.array(
                    self.card_counter.normalized_seen_card_counts(num_decks=self.rules.num_decks).copy(),
                    dtype=np.float32),
                "can_double": np.array([0], dtype=np.float32),
                "can_split": np.array([0], dtype=np.float32),
                "true_count": np.array([self.card_counter.true_count(self.deck.deck_remaining, self.rules.num_decks)],
                                       dtype=np.float32),
            }
        else:
            player = self.player
            dealer_one_hot_showing = [0] * 11
            if self.dealer_showing is not None:
                dealer_one_hot_showing[self.dealer_showing - 1] = 1
            obs = {
                "player_total": np.array(player.one_hot_total, dtype=np.float32),
                "player_has_blackjack": np.array([int(player.has_blackjack)], dtype=np.float32),
                "player_is_soft": np.array([int(player.has_soft_ace)], dtype=np.float32),
                "dealer_showing": np.array(dealer_one_hot_showing, dtype=np.float32),
                "deck_remaining": np.array([self.deck.deck_remaining], dtype=np.float32),
                "deck_draw_probs": np.array(self.deck.probability_of_drawing(), dtype=np.float32),
                "is_betting_phase": np.array([int(self.is_betting_phase)], dtype=np.float32),
                "seen_card_counts": np.array(
                    self.card_counter.normalized_seen_card_counts(num_decks=self.rules.num_decks).copy(),
                    dtype=np.float32),
                "can_double": np.array([int(len(player.current_hand.cards) == 2)], dtype=np.float32),
                "can_split": np.array([int(self.rules.allow_split and len(player.current_hand.cards) == 2
                                           and player.current_hand.cards[0] == player.current_hand.cards[1])],
                                      dtype=np.float32),
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
        if len(self.player.current_hand.cards) == 2:
            if self.rules.allow_double:
                legal_moves.append(Action.DOUBLE.value)
            if (self.rules.allow_split and self.player.current_hand.cards[0] == self.player.current_hand.cards[1]
                    and len(self.player.hands) < self.rules.max_splits):
                legal_moves.append(Action.SPLIT.value)
        assert len(legal_moves) > 0, "No legal moves available!"
        return legal_moves

    def get_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        if self.is_betting_phase:
            mask[len(Action):] = 1
        else:
            mask[Action.HIT.value] = 1
            mask[Action.STAND.value] = 1
            if len(self.player.current_hand.cards) == 2 and self.rules.allow_double:
                mask[Action.DOUBLE.value] = 1
            if (len(self.player.current_hand.cards) == 2 and self.rules.allow_split
                and self.player.current_hand.cards[0] == self.player.current_hand.cards[1]
                    and (len(self.player.hands) < self.rules.max_splits)):
                mask[Action.SPLIT.value] = 1
        assert mask.sum() > 0, "No legal moves available!"
        return mask

    def __repr__(self):
        return f"BlackJack(num_decks={self.rules.num_decks}, dealer={self.dealer}, deck_remaining={len(self.deck)})"


if __name__ == "__main__":
    env = BlackJack()
    obs, _ = env.reset()
    env.set_bet(10)
    env.is_betting_phase = False
    env.deal_starting_hand()
    env.player.current_hand.cards = [8, 8]
    print(env.get_legal_moves())
    obs, reward, done, _, info = env.step(Action.SPLIT.value)
    print([h.cards for h in env.player.hands])