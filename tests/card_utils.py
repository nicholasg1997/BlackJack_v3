"""Small helpers reused by blackjack rule tests."""

from __future__ import annotations

from blackjack.deck.deck import Deck
from blackjack.env.actions import Action
from blackjack.env.rules import BlackJackRules


def stack_draw_order(deck: Deck, draws_first_to_last: tuple[int, ...]) -> None:
    """Rebuild the shoe so pops match `draws_first_to_last` (deck pops from end)."""
    deck.cards.clear()
    for c in reversed(draws_first_to_last):
        deck.cards.append(c)


def bet_action_idx(rules: BlackJackRules, bet_amount: int) -> int:
    return len(Action) + (bet_amount - rules.min_bet)
