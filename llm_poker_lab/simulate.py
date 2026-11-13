"""Play N hands with seat rotation and duplicate dealing.

One shuffle is replayed six times, once with every bot moved one seat
to the right, which cuts the luck of who was on the button. If the
field has more than six bots, a sliding window chooses who sits out
and that window advances every deal. A bot that sets ``disqualified``
ends the match; its error string is kept on the stat book.

::

    shuffle ----+--> lineup of 6 ---> play_hand ---> StatBook
                |         ^
                |         +-- rotate seats, same cards (x6)
                +-- next deal, next sit-out window
"""

import math
import random
import time
from dataclasses import dataclass, field

from llm_poker_lab.cards import DECK
from llm_poker_lab.engine import play_hand
from llm_poker_lab.interface import BIG_BLIND, SEATS, SMALL_BLIND, STACK_BB
from llm_poker_lab.stats import StatBook


@dataclass
class Match:
    """Aggregate result of one simulated match."""

    books: dict[str, StatBook]
    hands_played: int
    hands_requested: int
    elapsed_sec: float
    seed: int
    sb: int
    bb: int
    stack_bb: int
    locked: tuple[str, ...] = field(default_factory=tuple)

    def public(self) -> dict:
        """JSON-ready summary, including a hands/sec measurement."""
        rows = [book.summary() for book in self.books.values()]
        speed = self.hands_played / max(self.elapsed_sec, 1e-9)
        return {
            "hands_requested": self.hands_requested,
            "hands_played": self.hands_played,
            "elapsed_sec": self.elapsed_sec,
            "hands_per_sec": speed,
            "seed": self.seed,
            "small_blind": self.sb,
            "big_blind": self.bb,
            "stack_bb": self.stack_bb,
            "locked": list(self.locked),
            "bots": rows,
        }


def deal_cards(rng: random.Random) -> tuple[list[list[str]], list[str], list[str]]:
    """Shuffle a deck into six hole-card pairs, three burns, and a board."""
    deck = list(DECK)
    rng.shuffle(deck)
    holes = [deck[index * 2:index * 2 + 2] for index in range(SEATS)]
    burns = deck[12:15]
    board = deck[15:20]
    return holes, burns, board


def choose_lineup(bots: list, deal_index: int, locked: tuple[str, ...] = (), seats: int = SEATS) -> list:
    """Pick ``seats`` bots. Names in ``locked`` always play; the rest rotate."""
    locked_bots = [bot for bot in bots if bot.name in locked]
    pool = [bot for bot in bots if bot.name not in locked]
    need = seats - len(locked_bots)
    if need < 0 or len(pool) < need:
        raise ValueError(f"need {need} bots from a pool of {len(pool)}")
    if len(pool) == need:
        chosen = list(pool)
    else:
        start = deal_index % len(pool)
        chosen = [pool[(start + offset) % len(pool)] for offset in range(need)]
    return locked_bots + chosen


def sitout_cycle(bots: list, locked: tuple[str, ...] = (), seats: int = SEATS) -> int:
    """Deals needed before the sit-out window repeats. One when everyone sits."""
    pool = [bot for bot in bots if bot.name not in locked]
    need = seats - (len(bots) - len(pool))
    if len(pool) <= need:
        return 1
    return len(pool)


def plan_deals(hands: int, seats: int = SEATS, cycle: int = 1) -> int:
    """Round a requested hand count up to a whole rotation and sit-out cycle."""
    if hands < 1:
        raise ValueError("hands must be positive")
    deals = math.ceil(hands / seats)
    if cycle > 1:
        deals = math.ceil(deals / cycle) * cycle
    return deals


def simulate(
    bots: list,
    hands: int = 6000,
    seed: int = 1,
    sb: int = SMALL_BLIND,
    bb: int = BIG_BLIND,
    stack_bb: int = STACK_BB,
    locked: tuple[str, ...] = (),
) -> Match:
    """Play ``hands`` (rounded up) and return per-bot books plus the wall clock."""
    names = [bot.name for bot in bots]
    if len(names) != len(set(names)):
        raise ValueError("bot names must be unique")
    cycle = sitout_cycle(bots, locked)
    deals = plan_deals(hands, SEATS, cycle)
    books = {bot.name: StatBook(bot.name) for bot in bots}
    rng = random.Random(seed)
    played = 0
    stack = stack_bb * bb
    started = time.perf_counter()
    for deal_index in range(deals):
        lineup = choose_lineup(bots, deal_index, locked)
        holes, burns, board = deal_cards(rng)
        stopped = False
        for rotation in range(SEATS):
            order = [lineup[(seat - rotation) % SEATS] for seat in range(SEATS)]
            result = play_hand(
                order,
                holes,
                board,
                burns,
                hand_id=played,
                sb=sb,
                bb=bb,
                stack=stack,
            )
            for seat, bot in enumerate(order):
                payoff = result.payoffs[seat]
                books[bot.name].add(
                    payoff,
                    bb,
                    payoff > 0,
                    result.vpip[seat],
                    result.pfr[seat],
                    result.aggressive[seat],
                    result.calls[seat],
                )
            played += 1
            failure = _disqualified(order)
            if failure:
                _mark(books, failure)
                stopped = True
                break
        if stopped:
            break
    elapsed = time.perf_counter() - started
    return Match(
        books=books,
        hands_played=played,
        hands_requested=hands,
        elapsed_sec=elapsed,
        seed=seed,
        sb=sb,
        bb=bb,
        stack_bb=stack_bb,
        locked=tuple(locked),
    )


def _disqualified(order: list) -> dict[str, str]:
    """Map disqualified bot names in this lineup to their error strings."""
    found = {}
    for bot in order:
        if getattr(bot, "disqualified", False):
            found[bot.name] = getattr(bot, "last_error", None) or "disqualified"
    return found


def _mark(books: dict[str, StatBook], failure: dict[str, str]) -> None:
    """Copy sandbox errors onto the stat books."""
    for name, message in failure.items():
        books[name].disqualified = True
        books[name].error = message
