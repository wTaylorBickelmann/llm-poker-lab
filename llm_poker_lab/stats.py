"""Chip results turned into bb/100, a 95% interval, and the usual HUD stats.

Each hand contributes one big-blind result. The interval is the normal
approximation ``1.96 * s / sqrt(n)`` on those per-hand results, scaled to
bb/100. Duplicate deals make the hands correlated, so the interval is a
little optimistic, but it is the number poker players expect to see.

::

    payoffs ---> sum and sum of squares ---> bb/100 ± 1.96 * se * 100
    vpip, pfr, bets, calls ---> percentages and aggression factor
"""

import math


class StatBook:
    """Running HUD for one bot. ``add`` is called once per hand that bot played."""

    def __init__(self, name: str) -> None:
        """Start every counter at zero for ``name``."""
        self.name = name
        self.hands = 0
        self.profit = 0
        self.won = 0
        self.vpip = 0
        self.pfr = 0
        self.aggressive = 0
        self.calls = 0
        self.sum_bb = 0.0
        self.sum_bb_sq = 0.0
        self.disqualified = False
        self.error: str | None = None

    def add(self, payoff: int, bb: int, won: bool, vpip: bool, pfr: bool, aggressive: int, calls: int) -> None:
        """Fold one hand's chip result and action counts into the book."""
        if bb <= 0:
            raise ValueError("bb must be positive")
        result = payoff / bb
        self.hands += 1
        self.profit += payoff
        self.won += 1 if won else 0
        self.vpip += 1 if vpip else 0
        self.pfr += 1 if pfr else 0
        self.aggressive += aggressive
        self.calls += calls
        self.sum_bb += result
        self.sum_bb_sq += result * result

    def summary(self) -> dict:
        """bb/100, confidence bounds, and HUD percentages. AF is null with no calls."""
        hands = self.hands
        mean = (self.sum_bb / hands) if hands else 0.0
        bb100 = mean * 100.0
        if hands > 1:
            variance = (self.sum_bb_sq - hands * mean * mean) / (hands - 1)
            half = 1.96 * math.sqrt(max(0.0, variance) / hands) * 100.0
        else:
            half = 0.0
        factor = (self.aggressive / self.calls) if self.calls else None
        return {
            "name": self.name,
            "hands": hands,
            "bb_per_100": bb100,
            "ci_low": bb100 - half,
            "ci_high": bb100 + half,
            "hands_won": self.won,
            "vpip": (100.0 * self.vpip / hands) if hands else 0.0,
            "pfr": (100.0 * self.pfr / hands) if hands else 0.0,
            "aggression_factor": factor,
            "aggressive_actions": self.aggressive,
            "calls": self.calls,
            "profit_chips": self.profit,
            "disqualified": self.disqualified,
            "error": self.error,
        }
