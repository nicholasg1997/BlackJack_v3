import random
from collections import Counter

import numpy as np

class Deck:
    """
    Represents a deck of cards with various functionalities such as shuffling,
    drawing, resetting, and probability calculations.

    This class is designed for managing the BlackJack deck with support
    for multiple decks, probabilities of drawing specific cards, and tracking the
    remaining cards. It provides methods to shuffle, reset, and draw cards while
    offering utilities to retrieve statistical insights like percentage composition
    and probabilities.

    :ivar rounding_precision: Number of decimal places for rounding probabilities.
    :type rounding_precision: int
    :ivar num_decks: Number of standard 52-card decks to include.
    :type num_decks: int
    :ivar cards: The list of current cards in the deck.
    :type cards: list[int]
    """
    def __init__(self, num_decks:int=1, rounding_precision:int=3):
        self.rounding_precision = rounding_precision
        self.num_decks = num_decks
        self.cards: list[int] = []
        self.create_deck()

    @property
    def deck_remaining(self) -> float:
        """
        Returns the fraction of the deck that is still remaining.
        This is calculated as the number of cards left divided by the total
        number of cards in the deck (52 * num_decks).
        :return: Fraction of the deck remaining.
        :rtype: float
        """
        return round(len(self.cards) / (52 * self.num_decks), self.rounding_precision)

    def create_deck(self) -> None:
        """
        Creates a new deck of cards based on the number of decks specified.
        Each deck consists of 52 cards, with values from 2 to 11 (where
        10, J, Q, K are all represented as 10, and A is represented as 11).
        The deck is shuffled upon creation.
        :return: None
        """
        starting_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        starting_deck *= self.num_decks
        random.shuffle(starting_deck)
        self.cards = starting_deck

    def reset(self) -> None:
        """
        Resets the deck to its initial state by clearing the current cards
        :return: None
        """
        self.cards.clear()
        self.create_deck()

    def shuffle(self) -> None:
        """
        Shuffles the current deck of cards in place.
        :return:
        """
        random.shuffle(self.cards)

    def draw_card(self) -> int:
        """
        Draws a single card from the deck.
        :return:
        """
        assert len(self.cards) > 0, "No cards left in the deck."
        return self.cards.pop()

    def draw_cards(self, num_cards: int) -> list[int]:
        """
        Draws a specified number of cards from the deck.
        :param num_cards:
        :return:
        """
        assert len(self.cards) >= num_cards, "Not enough cards left in the deck."
        assert num_cards > 0, "Number of cards to draw must be greater than 0."
        return [self.cards.pop() for _ in range(num_cards)]

    def get_percentage(self) -> np.ndarray:
        """
        Returns the percentage composition of each card value (2-11) in the deck.
        :return:
        """
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
        """
        Calculates the probability of drawing each card value (2-11) from the deck.
        :return:
        """
        total = len(self.cards)
        percentages = np.zeros(10,)
        if total == 0:
            return percentages
        for i in range(2, 12):
            idx = i - 2
            percentages[idx] = (self.cards.count(i) / total)
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