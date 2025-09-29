class Hand:
    def __init__(self, bet: float = 0):
        self.cards = []
        self.aces = 0
        self.bet = bet
        self.done = False

    @property
    def total(self) -> int:
        return sum(self.cards)

    @property
    def hard_total(self) -> int:
        return self.total - (10 * self.aces)

    @property
    def is_bust(self) -> bool:
        return self.total > 21

    @property
    def has_blackjack(self) -> bool:
        return self.total == 21 and len(self.cards) == 2

    @property
    def has_soft_ace(self) -> bool:
        return 11 in self.cards

    @property
    def one_hot_total(self) -> list[int]:
        one_hot = [0] * 23
        total = min(22, self.total)
        one_hot[total] = 1
        return one_hot

    def add_card(self, card) -> None:
        self.cards.append(card)
        if card == 11:
            self.aces += 1
        if self.total > 21 and self.aces > 0:
            self.cards[self.cards.index(11)] = 1
            self.aces -= 1

    def __repr__(self):
        return f"Hand(cards={self.cards}, total={self.total}, is_bust={self.is_bust})"

class Player:
    _player_id = 1
    def __init__(self, bet=0, dealer: bool = False, starting_balance: int = 1000):
        self.is_dealer = dealer
        self.hands = []
        self.current_hand_index = 0
        self.balance = float('inf') if dealer else starting_balance
        self.name = "Dealer" if dealer else f"Player_{Player._player_id}"
        if dealer:
            self.hands.append(Hand(bet=0))
        if not dealer:
            Player._player_id += 1
            self.add_hand(Hand(bet=bet))

    @property
    def current_hand(self) -> Hand:
        if not self.hands:
            raise ValueError("No hands available")
        return self.hands[self.current_hand_index]

    @property
    def hand_total(self) -> int:
        return self.current_hand.total

    @property
    def hard_hand_total(self) -> int:
        return self.current_hand.hard_total

    @property
    def is_bust(self) -> bool:
        return self.current_hand.is_bust

    @property
    def has_blackjack(self) -> bool:
        return self.current_hand.has_blackjack

    @property
    def has_soft_ace(self) -> bool:
        return self.current_hand.has_soft_ace

    @property
    def one_hot_total(self) -> list[int]:
        return self.current_hand.one_hot_total

    @property
    def bet(self) -> float:
        if self.hands:
            return self.current_hand.bet
        else:
            return self._initial_bet

    @bet.setter
    def bet(self, value: float):
        if self.hands:
            self.current_hand.bet = value
        else:
            self._initial_bet = value

    def reset(self) -> None:
        self.hands.clear()
        self.current_hand_index = 0
        if self.is_dealer:
            self.hands.append(Hand(bet=0))

    def add_card(self, card) -> None:
        self.current_hand.add_card(card)

    def add_cards(self, cards) -> None:
        for card in cards:
            self.add_card(card)

    def add_hand(self, hand: Hand) -> None:
        self.hands.append(hand)

    def all_hands_done(self) -> bool:
        return all(hand.done for hand in self.hands)

    def __repr__(self):
        return f"{self.name} (hands={self.hands}, balance={self.balance})"

    def __len__(self):
        return len(self.hands)

    def __getitem__(self, index):
        return self.hands[index]





if __name__ == "__main__":
    pass