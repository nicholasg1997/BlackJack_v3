class Player:
    _player_id = 1
    def __init__(self, bet = 0, dealer: bool = False, starting_balance: int = 1000):
        self.is_dealer = dealer
        self.hand = []
        self.aces = 0

        self.bet = bet
        self.balance = starting_balance

        if self.is_dealer:
            self.balance = float('inf')
            self.name = "Dealer"
        else:
            self.name = f"Player_{Player._player_id}"
            Player._player_id += 1

    @property
    def hand_total(self) -> int:
        total = sum(self.hand)
        return total

    @property
    def soft_hand_total(self) -> int:
        if self.aces > 0 and self.hand_total + 10 <= 21:
            return self.hand_total + 10
        return self.hand_total

    @property
    def is_bust(self) -> bool:
        return self.hand_total > 21

    @property
    def has_blackjack(self) -> bool:
        return self.hand_total == 21 and len(self.hand) == 2

    @property
    def has_soft_ace(self) -> bool:
        return 11 in self.hand

    @property
    def one_hot_total(self) -> list[int]:
        one_hot_total = [0] * 23
        total = min(23, self.hand_total)
        one_hot_total[total] = 1
        return one_hot_total

    def reset(self) -> None:
        self.hand.clear()
        self.aces = 0

    def add_card(self, card) -> None:
        self.hand.append(card)
        if card == 11:
            self.aces += 1
        if self.hand_total > 21 and self.aces > 0:
            self.hand[self.hand.index(11)] = 1
            self.aces -= 1

    def add_cards(self, cards) -> None:
        for card in cards:
            self.add_card(card)

    def __repr__(self):
        return f"{self.name} (hand={self.hand}, hand_total={self.hand_total}, is_bust={self.is_bust})"

    def __len__(self):
        return len(self.hand)

    def __getitem__(self, index):
        return self.hand[index]

if __name__ == "__main__":
    player_1 = Player()
    player_1.add_cards([10,11])
    print(player_1)
    player_1.add_card(9)
    print(player_1)
    player_1.add_card(11)
    print(player_1)
    print(player_1.name)
    player_2 = Player()
    print(player_2.name)