"""Seven fixed opponents. Each one has a few knobs and implements ``act``.

random, station, rock, tag, lag, maniac, and a pot-odds player that
calls eval7. They are intentionally simple: the evolution loop is the
part that is supposed to get clever.

::

    hole cards --+--> preflop_score ----+--> fold / check / raise
    board -------+--> postflop_tier ----+
    pot odds ----+--> equity() ---------+
"""

import random

from llm_poker_lab.cards import postflop_tier, preflop_score
from llm_poker_lab.equity import equity, mix_seed, multiway
from llm_poker_lab.interface import Bot

_POSITION_BONUS = {"UTG": 0.0, "HJ": 1.0, "CO": 1.5, "BTN": 2.0, "SB": 0.5, "BB": 1.0}


def position_bonus(position: str) -> float:
    """Loosen a hand score slightly in later position."""
    return _POSITION_BONUS.get(position, 0.0)


def pot_sized(state: dict, fraction: float) -> int:
    """Raise-to that adds ``fraction`` of the pot after calling."""
    extra = int(round((state["pot"] + state["to_call"]) * fraction))
    return int(state["hero_bet"] + state["to_call"] + max(extra, state["big_blind"]))


def sized_raise(state: dict, amount: int) -> dict:
    """A legal raise-to, or a check/call when a raise is not allowed."""
    if not state.get("can_raise") or state.get("min_raise_to") is None:
        return {"kind": "check_call"}
    low = int(state["min_raise_to"])
    high = int(state["max_raise_to"])
    return {"kind": "raise", "amount": max(low, min(high, int(amount)))}


def check_or_fold(state: dict) -> dict:
    """Check when it is free and fold when a call costs chips."""
    if state["to_call"] == 0:
        return {"kind": "check_call"}
    return {"kind": "fold"}


class RandomBot(Bot):
    """Mix of fold, call, and a fixed-size raise. Probabilities are normalized."""

    name = "random"

    def __init__(self, fold_prob: float = 0.45, call_prob: float = 0.30, raise_prob: float = 0.25, raise_bb: float = 8.0, seed: int = 0) -> None:
        """Store the mix and a private RNG."""
        total = fold_prob + call_prob + raise_prob
        if total <= 0:
            raise ValueError("probabilities must sum to a positive number")
        self.fold_p = fold_prob / total
        self.call_p = call_prob / total
        self.raise_bb = raise_bb
        self.rng = random.Random(seed)

    def act(self, state: dict) -> dict:
        """Draw one action from the configured mix."""
        roll = self.rng.random()
        if roll < self.fold_p:
            return check_or_fold(state)
        if roll < self.fold_p + self.call_p:
            return {"kind": "check_call"}
        return sized_raise(state, int(state["big_blind"] * self.raise_bb))


class StationBot(Bot):
    """Call almost everything. ``max_call_bb`` is the only fold threshold."""

    name = "station"

    def __init__(self, max_call_bb: float = 10_000) -> None:
        """Remember how large a bet this station will still call."""
        self.max_call_bb = max_call_bb

    def act(self, state: dict) -> dict:
        """Check or call unless the price exceeds ``max_call_bb``."""
        if state["to_call"] > self.max_call_bb * state["big_blind"]:
            return {"kind": "fold"}
        return {"kind": "check_call"}


class RockBot(Bot):
    """Tight and passive: enter with strong hands, raise only the top of that range."""

    name = "rock"

    def __init__(self, open_score: float = 9.0, value_score: float = 14.0, open_bb: float = 3.0) -> None:
        """Set the Chen thresholds for calling and for the rare raise."""
        self.open_score = open_score
        self.value_score = value_score
        self.open_bb = open_bb

    def act(self, state: dict) -> dict:
        """Call premiums, raise the nuts of the range, fold the rest."""
        if state["street"] == "preflop":
            return self._preflop(state)
        return self._postflop(state)

    def _preflop(self, state: dict) -> dict:
        """Apply the Chen thresholds, loosened a little in position."""
        score = preflop_score(*state["hole_cards"]) + position_bonus(state["hero_position"])
        facing = state["to_call"] > state["big_blind"]
        if score >= self.value_score and not facing:
            return sized_raise(state, int(state["big_blind"] * self.open_bb))
        if score >= self.open_score:
            return {"kind": "check_call"}
        return check_or_fold(state)

    def _postflop(self, state: dict) -> dict:
        """Call top pair, raise trips or better, give up air."""
        tier = postflop_tier(state["hole_cards"], state["board"])
        if tier >= 4 and state["can_raise"]:
            return sized_raise(state, pot_sized(state, 0.75))
        if tier >= 2:
            return {"kind": "check_call"}
        return check_or_fold(state)


class TagBot(Bot):
    """Tight-aggressive: raise a position-dependent range and continuation-bet value."""

    name = "tag"

    def __init__(self, open_scores: dict | None = None, three_bet_score: float = 12.0, continue_score: float = 9.0, open_bb: float = 2.5) -> None:
        """Set open thresholds per position plus 3-bet and continue scores."""
        self.open_scores = open_scores or {"UTG": 9.0, "HJ": 8.0, "CO": 7.0, "BTN": 6.0, "SB": 7.5, "BB": 5.0}
        self.three_bet_score = three_bet_score
        self.continue_score = continue_score
        self.open_bb = open_bb

    def act(self, state: dict) -> dict:
        """Open-raise, 3-bet strong hands, and bet made hands after the flop."""
        if state["street"] == "preflop":
            return self._preflop(state)
        tier = postflop_tier(state["hole_cards"], state["board"])
        if state["to_call"] == 0:
            if tier >= 2:
                return sized_raise(state, pot_sized(state, 0.66))
            return {"kind": "check_call"}
        if tier >= 4:
            return sized_raise(state, pot_sized(state, 1.0))
        if tier >= 2 or (tier >= 1 and state["to_call"] <= state["pot"] * 0.5):
            return {"kind": "check_call"}
        return {"kind": "fold"}

    def _preflop(self, state: dict) -> dict:
        """Raise unopened pots and continue only with a real hand versus a raise."""
        score = preflop_score(*state["hole_cards"])
        facing = state["to_call"] > state["big_blind"]
        if facing and score >= self.three_bet_score:
            return sized_raise(state, pot_sized(state, 1.0))
        if facing and score >= self.continue_score:
            return {"kind": "check_call"}
        if facing:
            return {"kind": "fold"}
        needed = self.open_scores[state["hero_position"]]
        if score >= needed:
            target = max(int(round(self.open_bb * state["big_blind"])), int(state["min_raise_to"] or 0))
            return sized_raise(state, target)
        return check_or_fold(state)


class LagBot(Bot):
    """Loose-aggressive: wider opens than TAG plus a bluff frequency."""

    name = "lag"

    def __init__(self, open_scores: dict | None = None, three_bet_score: float = 10.0, continue_score: float = 6.0, bluff_freq: float = 0.35, open_bb: float = 3.0, seed: int = 0) -> None:
        """Set a wide range and how often weak hands turn into a raise."""
        self.open_scores = open_scores or {"UTG": 6.0, "HJ": 5.0, "CO": 4.0, "BTN": 3.0, "SB": 4.0, "BB": 2.0}
        self.three_bet_score = three_bet_score
        self.continue_score = continue_score
        self.bluff_freq = bluff_freq
        self.open_bb = open_bb
        self.rng = random.Random(seed)

    def act(self, state: dict) -> dict:
        """Play a wide range and fire extra raises when the bluff roll hits."""
        bluff = self.rng.random() < self.bluff_freq
        if state["street"] == "preflop":
            return self._preflop(state, bluff)
        tier = postflop_tier(state["hole_cards"], state["board"])
        if bluff and state["can_raise"]:
            return sized_raise(state, pot_sized(state, 0.8))
        if state["to_call"] == 0:
            if tier >= 1:
                return sized_raise(state, pot_sized(state, 0.7))
            return {"kind": "check_call"}
        if tier >= 3:
            return sized_raise(state, pot_sized(state, 1.0))
        if tier >= 1:
            return {"kind": "check_call"}
        return {"kind": "fold"}

    def _preflop(self, state: dict, bluff: bool) -> dict:
        """Open wide, and 3-bet light when a bluff is due."""
        score = preflop_score(*state["hole_cards"])
        facing = state["to_call"] > state["big_blind"]
        if facing and (score >= self.three_bet_score or bluff):
            return sized_raise(state, pot_sized(state, 1.0))
        if facing and score >= self.continue_score:
            return {"kind": "check_call"}
        if facing:
            return {"kind": "fold"}
        needed = self.open_scores[state["hero_position"]]
        if score >= needed or bluff:
            target = max(int(round(self.open_bb * state["big_blind"])), int(state["min_raise_to"] or 0))
            return sized_raise(state, target)
        return check_or_fold(state)


class ManiacBot(Bot):
    """Raise almost every hand to a large size."""

    name = "maniac"

    def __init__(self, raise_prob: float = 0.97, open_bb: float = 12.0, seed: int = 0) -> None:
        """Set how often the maniac raises and the minimum open size in big blinds."""
        self.raise_prob = raise_prob
        self.open_bb = open_bb
        self.rng = random.Random(seed)

    def act(self, state: dict) -> dict:
        """Raise pot or ``open_bb``, whichever is larger, unless the roll says fold."""
        if self.rng.random() >= self.raise_prob:
            return check_or_fold(state)
        target = max(pot_sized(state, 1.2), int(state["big_blind"] * self.open_bb))
        return sized_raise(state, target)


class EquityBot(Bot):
    """Call when Monte Carlo equity clears the pot odds, and raise the strong hands."""

    name = "equity"

    def __init__(self, iterations: int = 40, margin: float = 1.05, raise_equity: float = 0.70, open_equity: dict | None = None, bet_fraction: float = 0.75, seed: int = 1) -> None:
        """Set the Monte Carlo budget and the equity thresholds."""
        self.iterations = iterations
        self.margin = margin
        self.raise_equity = raise_equity
        self.open_equity = open_equity or {"UTG": 0.58, "HJ": 0.55, "CO": 0.52, "BTN": 0.48, "SB": 0.52, "BB": 0.40}
        self.bet_fraction = bet_fraction
        self.seed = seed

    def act(self, state: dict) -> dict:
        """Compare a heads-up Monte Carlo share with the price and the open chart."""
        hole = state["hole_cards"]
        board = state["board"]
        if board:
            draw_seed = mix_seed(list(hole) + list(board) + [str(state["hand_id"])], self.seed)
            share = multiway(equity(hole, board, self.iterations, draw_seed), state["num_active"] - 1)
        else:
            draw_seed = mix_seed(sorted(hole), 0)
            share = equity(hole, [], 160, draw_seed)
        if state["street"] == "preflop" and state["to_call"] <= state["big_blind"]:
            if share >= self.open_equity[state["hero_position"]]:
                target = max(int(state["big_blind"] * 2.5), pot_sized(state, 0.5))
                return sized_raise(state, target)
            return check_or_fold(state)
        price = state["pot_odds"] * self.margin + 0.04 * max(0, state["num_active"] - 2)
        if share >= self.raise_equity and state["can_raise"]:
            return sized_raise(state, pot_sized(state, self.bet_fraction))
        if share + 1e-12 >= price or state["to_call"] == 0:
            return {"kind": "check_call"}
        return {"kind": "fold"}


def make_baselines(seed: int = 0) -> list[Bot]:
    """The default six-max field, in a stable order, each with its own seed."""
    return [
        RandomBot(seed=seed + 1),
        StationBot(),
        RockBot(),
        TagBot(),
        LagBot(seed=seed + 2),
        ManiacBot(seed=seed + 3),
        EquityBot(seed=seed + 4),
    ]
