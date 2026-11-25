"""Card labels, a Chen-style preflop score, and a coarse made-hand tier.

Baselines need a fast read of the hole cards without a solver. Preflop
uses the Chen formula. Postflop asks eval7 for the hand category and then
separates top pair and overpairs from weak pairs.

::

    "As","Kd" ---> preflop_score ---> open / fold thresholds
    hole + board ---> eval7 category ---> postflop_tier 0..6
"""

import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(f"{rank}{suit}" for rank in RANKS for suit in SUITS)

_CHEN = {
    "A": 10.0, "K": 8.0, "Q": 7.0, "J": 6.0, "T": 5.0,
    "9": 4.5, "8": 4.0, "7": 3.5, "6": 3.0, "5": 2.5,
    "4": 2.0, "3": 1.5, "2": 1.0,
}
_CATEGORY = {
    "High Card": 0,
    "Pair": 1,
    "Two Pair": 3,
    "Trips": 4,
    "Straight": 5,
    "Flush": 5,
    "Full House": 6,
    "Quads": 6,
    "Straight Flush": 6,
}
_EVAL_CARD = {label: eval7.Card(label) for label in DECK}


def card_key(label: str) -> eval7.Card:
    """Return the cached eval7 card for a label such as ``As``."""
    return _EVAL_CARD[label]


def preflop_score(left: str, right: str) -> float:
    """Chen-style score. Pairs and suited broadways rank above junk."""
    if _CHEN[left[0]] < _CHEN[right[0]]:
        left, right = right, left
    high, low = left[0], right[0]
    score = _CHEN[high]
    if high == low:
        return max(score * 2, 5)
    suited = left[1] == right[1]
    if suited:
        score += 2
    gap = RANKS.index(high) - RANKS.index(low) - 1
    score -= _gap_penalty(gap)
    if gap <= 1 and suited and _CHEN[high] < 7:
        score += 1
    return score


def _gap_penalty(gap: int) -> float:
    """Chen penalty for the number of missing ranks between the two cards."""
    if gap <= 0:
        return 0
    if gap == 1:
        return 1
    if gap == 2:
        return 2
    if gap == 3:
        return 4
    return 5


def postflop_tier(hole: list[str], board: list[str]) -> int:
    """Rank a made hand from 0 (air) through 6 (full house or better).

    Tier 1 is a weak pair, tier 2 is top pair or an overpair.
    """
    if len(board) < 3:
        return 0
    cards = [card_key(label) for label in hole + board]
    category = eval7.handtype(eval7.evaluate(cards))
    if category == "Pair":
        return _pair_tier(hole, board)
    return _CATEGORY[category]


def _pair_tier(hole: list[str], board: list[str]) -> int:
    """Split pairs into top pair / overpair (2) and everything else (1)."""
    top = max(RANKS.index(card[0]) for card in board)
    pocket = hole[0][0] == hole[1][0]
    if pocket and RANKS.index(hole[0][0]) > top:
        return 2
    if top in (RANKS.index(hole[0][0]), RANKS.index(hole[1][0])):
        return 2
    return 1
