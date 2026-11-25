"""The only contract a poker bot has to satisfy.

Every baseline and every LLM-written bot exposes ``act(state) -> action``.
``state`` is a plain dict the engine can turn into JSON. ``action`` is one
of fold, check/call, or raise. The engine coerces anything illegal so a
buggy bot loses the pot instead of crashing the table.

::

    bot.act(state) ---> coerce ---> PokerKit operation
         ^                |
         |                v
         +-------- next public state

Seat map used by the engine (PokerKit posts the blinds at the front):

    seat:  0   1   2    3   4    5
    pos:  SB  BB  UTG  HJ  CO  BTN
"""

SEATS = 6
SMALL_BLIND = 1
BIG_BLIND = 2
STACK_BB = 200
POSITIONS = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
STREETS = ("preflop", "flop", "turn", "river")

# Keys present on every decision. The prompt and the README mirror this list.
STATE_KEYS = (
    "hole_cards",
    "board",
    "street",
    "pot",
    "to_call",
    "min_raise_to",
    "max_raise_to",
    "can_raise",
    "can_check",
    "hero_index",
    "hero_position",
    "hero_stack",
    "hero_bet",
    "stacks",
    "positions",
    "action_history",
    "big_blind",
    "small_blind",
    "num_active",
    "hand_id",
    "pot_odds",
)

STATE_DOC = """
state keys (all JSON-serializable):
- hole_cards: two strings, e.g. ["As", "Kd"]. Only the acting player's cards.
- board: community cards dealt so far, e.g. ["Qh", "7c", "2d"].
- street: "preflop", "flop", "turn", or "river".
- pot: chips in the pot, including bets on the current street.
- to_call: extra chips needed to call. 0 means a check is free.
- min_raise_to, max_raise_to: legal raise-to totals for this street, or null.
- can_raise: true when a raise is legal. can_check: true when to_call is 0.
- hero_index: seat 0-5. hero_position: SB, BB, UTG, HJ, CO, or BTN.
- hero_stack: chips behind. hero_bet: chips already invested this street.
- stacks: chips behind for every seat. positions: label per seat.
- action_history: list of {seat, position, street, kind, amount}.
  amount is chips put in for a call, the raise-to total for a raise, else 0.
- big_blind, small_blind: chip sizes (default 2 and 1). Stacks reset to 400.
- num_active: players who have not folded. hand_id: int. pot_odds: float.
""".strip()

ACTION_DOC = """
Return one dict:
- {"kind": "fold"}
- {"kind": "check_call"}
- {"kind": "raise", "amount": <int>}
  amount is the total chips you will have committed this street (a raise-to),
  between min_raise_to and max_raise_to, not the extra chips on top of a call.
Illegal output is coerced: an unknown kind folds if calling costs chips and
checks otherwise; folding when a check is free becomes a check; a raise
outside the legal interval is clamped; a raise when can_raise is false
becomes a check/call.
""".strip()


class Bot:
    """Documented base. Subclasses override ``act`` and set ``name``."""

    name = "bot"

    def act(self, state: dict) -> dict:
        """Return a fold, check/call, or raise for this decision."""
        raise NotImplementedError


def coerce(raw, state: dict) -> dict:
    """Turn any bot output into a legal fold, check/call, or clamped raise.

    A raise that cannot be made becomes a check/call, because the bot meant
    to stay in the hand. Anything unrecognizable folds if a call costs chips.
    """
    kind = raw.get("kind") if isinstance(raw, dict) else None
    to_call = int(state["to_call"])
    can_raise = bool(state.get("can_raise")) and state.get("min_raise_to") is not None
    if kind == "raise" and can_raise:
        low = int(state["min_raise_to"])
        high = int(state["max_raise_to"])
        amount = _as_int(raw.get("amount"), low)
        return {"kind": "raise", "amount": max(low, min(high, amount))}
    if kind in ("check_call", "raise"):
        return {"kind": "check_call"}
    if to_call == 0:
        return {"kind": "check_call"}
    return {"kind": "fold"}


def _as_int(value, default: int) -> int:
    """Parse a chip count. Non-numeric values fall back to ``default``."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
