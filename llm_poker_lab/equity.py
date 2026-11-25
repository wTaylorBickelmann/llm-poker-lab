"""Monte Carlo equity of one hand against a single random opponent.

eval7 ranks the showdowns. Preflop lookups are cached by the 169-hand
class so the equity bot does not repeat the same shuffle all night.
Multiway pots are approximated by raising heads-up equity to the number
of opponents, which is fast and good enough for a baseline.

::

    hole, board, iterations ---> sample runouts ---> eval7 ranks ---> equity
"""

import random

from llm_poker_lab.cards import DECK, card_key

_CACHE: dict[tuple, float] = {}


def equity(hole: list[str], board: list[str] | None = None, iterations: int = 200, seed: int = 0) -> float:
    """Share of the pot the hole cards win against one random hand.

    Ties count as half a win. ``iterations`` must be at least 1.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    board = list(board or [])
    if not board:
        key = (_hand_class(hole), iterations, seed)
        if key not in _CACHE:
            _CACHE[key] = _simulate(hole, board, iterations, seed)
        return _CACHE[key]
    return _simulate(hole, board, iterations, seed)


def multiway(heads_up: float, opponents: int) -> float:
    """Approximate equity against ``opponents`` similar random hands."""
    count = max(1, opponents)
    return heads_up ** count


def mix_seed(parts: list[str], salt: int = 0) -> int:
    """Stable non-cryptographic seed so equity draws repeat across processes."""
    acc = 17 + salt
    for part in parts:
        for char in part:
            acc = (acc * 131 + ord(char)) % 1_000_003
    return acc


def _hand_class(hole: list[str]) -> str:
    """Canonical label such as ``AKs``, ``AKo``, or ``TT``."""
    left, right = hole
    if rank_index(left[0]) < rank_index(right[0]):
        left, right = right, left
    if left[0] == right[0]:
        return left[0] + right[0]
    suffix = "s" if left[1] == right[1] else "o"
    return left[0] + right[0] + suffix


def rank_index(rank: str) -> int:
    """Index of a rank character, deuce low and ace high."""
    return "23456789TJQKA".index(rank)


def _simulate(hole: list[str], board: list[str], iterations: int, seed: int) -> float:
    """Average win/tie rate over ``iterations`` random villain hands."""
    import eval7

    rng = random.Random(seed)
    dead = set(hole + board)
    remaining = [card_key(label) for label in DECK if label not in dead]
    hero = [card_key(label) for label in hole]
    known = [card_key(label) for label in board]
    missing = 5 - len(known)
    won = 0.0
    for _ in range(iterations):
        draw = rng.sample(remaining, missing + 2)
        full = known + draw[:missing]
        villain = draw[missing:missing + 2]
        hero_score = eval7.evaluate(hero + full)
        villain_score = eval7.evaluate(villain + full)
        if hero_score > villain_score:
            won += 1
        elif hero_score == villain_score:
            won += 0.5
    return won / iterations
