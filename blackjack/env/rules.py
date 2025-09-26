from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class BlackJackRules:
    # deck rules and parameters
    num_decks: int = 4
    reshuffle_threshold: float = 0.25

    #dealer rules
    dealer_stay_value: int = 17
    dealer_hits_soft: bool = True

    # player allowed moves
    allow_double: bool = True
    allow_surrender: bool = False # Not yet implemented
    allow_split: bool = False     # Not yet implemented
    max_splits: int = 3           # Not yet implemented
    double_after_split: bool = False # Not yet implemented
    allow_insurance: bool = False # Not yet implemented
    resplit_aces: bool = False   # Not yet implemented

    # betting rules
    min_bet: int = 2
    max_bet: int = 100
    fixed_bet: Optional[int] = None

    # other
    blackjack_reward: float = 1.5
    starting_balance: int = 1000