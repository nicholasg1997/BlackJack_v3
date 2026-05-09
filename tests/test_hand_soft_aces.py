"""Hand-level rules for soft/hard totals and ace downgrades."""

from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from blackjack.player.player import Hand


class TestSoftAndHardTotals:
    def test_soft_hand_has_11_in_cards(self):
        hand = Hand()
        hand.add_card(11)
        hand.add_card(6)
        assert hand.total == 17
        assert hand.has_soft_ace

    def test_ace_hardens_when_otherwise_bust_three_cards(self):
        """11 + 10 + 2 would be bust as all 11-valued; Ace downgrades once."""
        hand = Hand()
        hand.add_card(11)
        hand.add_card(10)
        hand.add_card(2)
        assert 11 not in hand.cards or hand.total <= 21
        assert hand.total == 13  # 1 + 10 + 2 after downgrade
        assert not hand.has_soft_ace

    def test_two_aces_then_nine_totals_twenty_one(self):
        hand = Hand()
        hand.add_card(11)
        hand.add_card(11)
        hand.add_card(9)
        assert hand.total == 21, f"Hand total is {hand.total}, should be 21"
        assert not hand.is_bust, f"Hand is bust: {hand.is_bust}, should be False"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
