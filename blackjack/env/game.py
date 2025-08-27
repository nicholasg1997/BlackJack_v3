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
    #SPLIT = 3 # Splitting is not yet supported.


class BlackJack(gym.Env):
    def __init__(self, num_decks: int = 6, starting_balance: int = 1000,
                 dealer_hits_soft: bool = True, dealer_stay_value: int = 17,
                 min_bet: int = 2, max_bet: int = 20, num_players: int = 1):
        super().__init__()
        self.num_decks = num_decks
        self.starting_balance = starting_balance
        self.dealer_hits_soft_17 = dealer_hits_soft
        self.dealer_stay_value = dealer_stay_value
        self.min_bet = min_bet
        self.max_bet = max_bet

        self.deck = Deck(num_decks=self.num_decks)
        self.players = []
        for _ in range(num_players):
            self.players.append(Player(starting_balance=self.starting_balance))
        self.dealer = Player(dealer=True)

        self.current_player_index = 0
        self.round_over = False
        self.reset()

        self.observation_space = gym.spaces.Dict(
            {
                "player_total": gym.spaces.Box(low=0, high=1, shape=(23,), dtype=np.float32),  # player stat (one hot encoded)
                "player_has_blackjack": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32), # player stat (one hot encoded)
                "player_is_soft": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32), # player stat (need a better name for this) (one hot encoded)
                "dealer_showing": gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32),  # dealer stat (one hot encoded)
                "deck_remaining": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),  # game stat
                "deck_card_probs": gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),  # game stat
                "deck_draw_probs": gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),  # game stat
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

    @property
    def current_player(self):
        return self.players[self.current_player_index % len(self.players)]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.deck.reset_deck()
        self.dealer.reset()
        for player in self.players:
            player.reset()
        self.current_player_index = 0
        self.round_over = False
        self.deal_starting_hands()

        return self._get_obs(), {}

    def add_player(self, player):
        self.players.append(player)

    def add_players(self, players):
        for player in players:
            self.add_player(player)

    def remove_player(self, player):
        self.players.remove(player)

    def deal_starting_hands(self):
        for _ in range(2):
            for player in self.players:
                player.add_card(self.deck.draw_card())
            self.dealer.add_card(self.deck.draw_card())

    def set_bets(self, bets: list[float]):
        for i, player in enumerate(self.players):
            player.bet = bets[i]


    def dealer_autoplay(self):
        while True:
            print(self.dealer)
            if self.dealer.hand_total < self.dealer_stay_value:
                self.dealer.add_card(self.deck.draw_card())
            elif self.dealer.hand_total == self.dealer_stay_value and self.dealer.has_soft_ace and self.dealer_hits_soft_17:
                self.dealer.add_card(self.deck.draw_card())
            else:
                break

    def step(self, action: Action):
        # I need to figure out my order of operations more before I do this.
        # I need each player to do their turn and then once were done have the dealer go,
        # then check if each player beat the dealer and get a reward for each player

        action = Action(action["action"])
        current_player = self.current_player
        player_done = False

        if action == Action.HIT.value:
            player_done = self._hit(current_player)
        elif action == Action.STAND.value:
            player_done = True
        elif action == Action.DOUBLE.value:
            player_done = self._double(current_player)

        if player_done:
            self.current_player_index += 1

        self.round_over = self.current_player_index >= len(self.players)
        if self.round_over:
            self.dealer_autoplay()
            rewards = self._get_reward()
            done = True
            obs = self._get_obs(player_index=0)
        else:
            rewards = [0] * len(self.players)
            obs = self._get_obs()
            done = False
        return obs, rewards, done, False, {}

    def _hit(self, current_player):
        current_player.add_card(self.deck.draw_card())
        return current_player.is_bust

    def _double(self, current_player):
        assert len(current_player.hand) == 2, "Player must have 2 cards to double down."
        assert current_player.balance >= current_player.bet * 2, "Player does not have enough balance to double down."
        current_player.add_card(self.deck.draw_card())
        return True

    def _split(self):
        pass

    def _get_reward(self):
        # need to get reward for each player. maybe players should have a reward variable?
        # maybe i will pass the agent instead of player which will have reward variable.
        # maybe ill just have a list or rewards
        dealer_total = self.dealer.hand_total
        rewards = []

        for player in self.players:
            if player.has_blackjack and not self.dealer.has_blackjack:
                player.balance += int(1.5 * player.bet)
                reward = 1.5 * player.bet
            elif player.is_bust:
                player.balance -= player.bet
                reward = -1 * player.bet
            elif self.dealer.is_bust:
                player.balance += player.bet
                reward = 1 * player.bet
            elif player.hand_total > dealer_total:
                player.balance += player.bet
                reward = 1 * player.bet
            elif player.hand_total < dealer_total:
                player.balance -= player.bet
                reward = -1 * player.bet
            else:
                reward = 0
            rewards.append(reward)
        return rewards

    def _get_obs(self, player_index: Optional[int] = None):
        if player_index is None:
            player_index = self.current_player_index
        if player_index >= len(self.players):
            player_index = 0

        player = self.players[player_index]
        dealer_one_hot_showing = [0] * 11
        if self.dealer_showing is not None:
            dealer_one_hot_showing[self.dealer_showing - 1] = 1
        deck_remaining = len(self.deck) / (52 * self.num_decks)
        return {
            "player_total": np.array(player.one_hot_total, dtype=np.float32),
            "player_has_blackjack": np.array([int(player.has_blackjack)], dtype=np.float32),
            "player_is_soft": np.array([int(player.has_soft_ace)], dtype=np.float32),
            "dealer_showing": np.array(dealer_one_hot_showing, dtype=np.float32),
            "deck_remaining": np.array(deck_remaining, dtype=np.float32),
            "deck_card_probs": np.array(self.deck.get_percentage(), dtype=np.float32),
            "deck_draw_probs": np.array(self.deck.get_percentage(), dtype=np.float32),
        }

    def get_legal_moves(self):
        current_player = self.current_player
        legal_moves = [Action.HIT, Action.STAND]
        if len(current_player.hand) == 2 and current_player.balance >= current_player.bet * 2:
            legal_moves.append(Action.DOUBLE)
        #if len(current_player.hand) == 2 and current_player.hand[0] == current_player.hand[1]:
        #    legal_moves.append(Action.SPLIT)
        return legal_moves

    def __repr__(self):
        return f"BlackJack(num_decks={self.num_decks}, players={self.players}, dealer={self.dealer}, deck_remaining={len(self.deck)})"




if __name__ == "__main__":
    game = BlackJack()
    print(game)
    print(game.deck)
    game.dealer_autoplay()
    print(game.current_player)
    print(game._get_obs())

