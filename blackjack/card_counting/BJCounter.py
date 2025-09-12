import numpy as np

class CardCounter:
    def __init__(self):
        self.running_count = 0.0
        self.seen_card_counts = np.zeros(10, dtype=np.float32)

    def update_count(self, card: int) -> None:
        if 2 <= card <= 6:
            self.running_count += 1
        elif card == 10 or card == 1:
            self.running_count -= 1

        if card == 1:  # Ace
            self.seen_card_counts[9] += 1
        else:
            self.seen_card_counts[card - 2] += 1

    def true_count(self, deck_remaining_fraction: float, num_decks: int) -> float:
        """Calculates the true count."""
        if deck_remaining_fraction == 0:
            return 0.0
        decks_left = deck_remaining_fraction * num_decks
        return self.running_count / decks_left

    def reset(self) -> None:
        self.running_count = 0.0
        self.seen_card_counts.fill(0)