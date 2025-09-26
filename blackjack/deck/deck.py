import random
from collections import Counter
import numpy as np
import torch as th

class Deck:
    def __init__(self, num_decks: int = 1, rounding_precision: int = 3):
        self.rounding_precision = rounding_precision
        self.num_decks = num_decks
        self.cards: list[int] = []
        self.create_deck()

    @property
    def deck_remaining(self) -> float:
        return round(len(self.cards) / (52 * self.num_decks), self.rounding_precision)

    def create_deck(self) -> None:
        starting_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        starting_deck *= self.num_decks
        random.shuffle(starting_deck)
        self.cards = starting_deck

    def reset(self) -> None:
        self.cards.clear()
        self.create_deck()

    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def draw_card(self) -> int:
        assert len(self.cards) > 0, "No cards left in the deck."
        return self.cards.pop()

    def draw_cards(self, num_cards: int) -> list[int]:
        assert len(self.cards) >= num_cards, "Not enough cards left in the deck."
        assert num_cards > 0, "Number of cards to draw must be greater than 0."
        return [self.draw_card() for _ in range(num_cards)]

    def get_percentage(self) -> np.ndarray:
        percentages = np.zeros(10,)
        all_tens = 4 * 4 * self.num_decks
        counts = Counter(self.cards)
        for i in range(2, 12):
            if i == 10:
                percentages[i-2] = (counts[i] / all_tens)
            else:
                percentages[i-2] = (counts[i] / (4 * self.num_decks))
        return np.round(percentages, self.rounding_precision)

    def probability_of_drawing(self) -> np.ndarray:
        total = len(self.cards)
        percentages = np.zeros(10,)
        if total == 0:
            return percentages
        for i in range(2, 12):
            percentages[i - 2] = (self.cards.count(i) / total)
        return np.round(percentages, self.rounding_precision)

    def __repr__(self):
        return f"Deck(num_decks={self.num_decks}, cards={self.cards})"

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

if __name__ == "__main__":
    deck = Deck(num_decks=1)
    print(Counter(deck.cards))
    print(deck.get_percentage())
    drawn_cards = deck.draw_cards(5)
    print(f"Drawn cards: {drawn_cards}")
    print(f"Remaining cards: {len(deck)}")
    print(deck.get_percentage())
    print(deck.probability_of_drawing())
    print(deck.deck_remaining)
    for c in deck:
        print(c, end=' ')