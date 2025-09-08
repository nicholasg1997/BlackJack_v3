import numpy as np

class CardCounter:
    def __init__(self):
        # A simple array to count cards [2, 3, 4, 5, 6, 7, 8, 9, 10, A]
        self.seen_card_counts = np.zeros(10, dtype=np.float32)

    def update_count(self, card: int) -> None:
        # Aces (11) are the 10th element, tens are the 9th, etc.
        idx = card - 2
        self.seen_card_counts[idx] += 1

    def reset(self) -> None:
        self.seen_card_counts.fill(0)