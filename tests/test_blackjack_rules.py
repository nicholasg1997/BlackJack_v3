"""Integration tests: blackjack rules enforced by :class:`~blackjack.env.game.BlackJack`."""

from __future__ import annotations

from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest

from blackjack.env.actions import Action
from blackjack.env.game import BlackJack
from blackjack.env.rules import BlackJackRules
from blackjack.player.player import Hand

try:
    from .card_utils import bet_action_idx, stack_draw_order
except ImportError:  # pragma: no cover — running ``python tests/...py`` directly
    from tests.card_utils import bet_action_idx, stack_draw_order


class TestNaturalBlackjackAfterDeal:
    """Naturals end the betting step; correct payoffs."""

    def test_player_natural_pays_three_to_two_when_dealer_not_blackjack(self):
        rules = BlackJackRules(min_bet=50, max_bet=500, dealer_hits_soft=True)
        env = BlackJack(rules=rules)
        env.reset(seed=42)
        # Deal order: player 11, dealer 10, player 10, dealer 10
        stack_draw_order(env.deck, (11, 10, 10, 10))
        _, _, terminated, _, info = env.step(bet_action_idx(rules, 50))
        assert terminated
        h = info["hands"][0]
        assert h["win"]
        assert h["winnings"] == pytest.approx(75.0)

    def test_both_naturals_push(self):
        rules = BlackJackRules(min_bet=10, max_bet=200)
        env = BlackJack(rules=rules)
        env.reset()
        stack_draw_order(env.deck, (11, 11, 10, 10))
        _, _, terminated, _, info = env.step(bet_action_idx(rules, 10))
        assert terminated
        h = info["hands"][0]
        assert h["push"]
        assert h["winnings"] == 0

    def test_dealer_natural_player_loses_non_blackjack_even_if_total_21(self):
        rules = BlackJackRules(min_bet=20, max_bet=400)
        env = BlackJack(rules=rules)
        env.reset()
        stack_draw_order(env.deck, (10, 11, 10, 10))
        _, _, terminated, _, info = env.step(bet_action_idx(rules, 20))
        assert terminated
        assert info["hands"][0]["loss"]
        assert info["hands"][0]["winnings"] == pytest.approx(-20.0)

    def test_natural_requires_only_stand_among_play_actions_when_in_play_phase(self):
        rules = BlackJackRules(min_bet=2, max_bet=100)
        env = BlackJack(rules=rules)
        env.reset()

        env.is_betting_phase = False
        env.player.reset()
        env.dealer.reset()
        env.player.add_hand(Hand(bet=10))
        for c in (11, 10):
            env.player.current_hand.add_card(c)

        assert env.player.current_hand.has_blackjack
        moves = env.get_legal_moves()
        assert moves == [Action.STAND.value]

        env.model_type = "DQN"
        mask = env.get_action_mask()
        assert not mask[Action.HIT.value]
        assert mask[Action.STAND.value]


class TestDealerSoft17AndAutoplay:
    def test_dealer_hits_soft_17_when_configured(self):
        rules = BlackJackRules(
            min_bet=2,
            max_bet=500,
            dealer_stay_value=17,
            dealer_hits_soft=True,
        )
        env = BlackJack(rules=rules)
        env.reset()
        # Player stands on 19. Dealer Ace + 6 = soft 17; must hit; draw 10 -> hard 17.
        stack_draw_order(env.deck, (10, 11, 9, 6, 10))
        env.step(bet_action_idx(rules, 2))
        _, _, terminated, _, _ = env.step(Action.STAND.value)
        assert terminated
        assert len(env.dealer.current_hand.cards) >= 3
        assert env.dealer.hand_total == 17
        assert not env.dealer.current_hand.has_soft_ace

    def test_dealer_stands_on_soft_17_when_dealer_hits_soft_false(self):
        rules = BlackJackRules(
            min_bet=2,
            max_bet=500,
            dealer_stay_value=17,
            dealer_hits_soft=False,
        )
        env = BlackJack(rules=rules)
        env.reset()
        stack_draw_order(env.deck, (10, 11, 9, 6))
        env.step(bet_action_idx(rules, 2))
        _, _, terminated, _, _ = env.step(Action.STAND.value)
        assert terminated
        assert len(env.dealer.current_hand.cards) == 2
        assert env.dealer.hand_total == 17
        assert env.dealer.current_hand.has_soft_ace


class TestShowdownOutcomes:
    def test_dealer_natural_beats_player_non_natural_twenty_one(self):
        """Three-card player 21 must lose when dealer holds a natural (two-card) blackjack."""
        rules = BlackJackRules(min_bet=10, max_bet=400)
        env = BlackJack(rules=rules)
        env.reset()
        env.is_betting_phase = False
        env.player.add_hand(Hand(bet=10))
        env.dealer.reset()
        for c in (7, 7, 7):
            env.player.current_hand.add_card(c)
        env.dealer.current_hand.cards.clear()
        env.dealer.current_hand.add_card(11)
        env.dealer.current_hand.add_card(10)

        assert not env.player.current_hand.has_blackjack
        assert env.dealer.has_blackjack

        diag: dict = {}
        env._get_reward_for_hand(env.player.current_hand, env.dealer.hand_total, diag)
        assert diag["loss"]
        assert diag["push"] is False
        assert diag["winnings"] == pytest.approx(-10.0)


class TestPlayerBustVersusDealer:
    def test_bust_loses_without_dealer_needing_additional_cards_when_dealer_already_strong(self):
        rules = BlackJackRules(min_bet=25, max_bet=250)
        env = BlackJack(rules=rules)
        env.reset()
        env.is_betting_phase = False
        env.player.add_hand(Hand(bet=25))
        env.dealer.reset()
        for c in (10, 6, 8):
            env.player.current_hand.add_card(c)
        env.dealer.current_hand.cards.clear()
        env.dealer.current_hand.add_card(11)
        env.dealer.current_hand.add_card(9)

        assert env.player.current_hand.is_bust
        diag: dict = {}
        env._get_reward_for_hand(env.player.current_hand, env.dealer.hand_total, diag)
        assert diag["loss"]


@pytest.mark.parametrize("allow_double", [True, False])
def test_double_only_with_two_cards(allow_double: bool):
    rules = BlackJackRules(min_bet=2, max_bet=100, allow_double=allow_double)
    env = BlackJack(rules=rules)
    env.reset()
    stack_draw_order(env.deck, (8, 7, 4, 2, 11))
    env.step(bet_action_idx(rules, 2))
    hits = Action.HIT.value
    dd = Action.DOUBLE.value
    moves = sorted(env.get_legal_moves())
    if allow_double:
        assert dd in moves
    else:
        assert dd not in moves
        assert hits in moves
    env.step(hits)
    moves_after_hit = sorted(env.get_legal_moves())
    assert dd not in moves_after_hit
