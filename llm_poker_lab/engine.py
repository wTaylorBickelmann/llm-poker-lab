"""Play one six-max no-limit cash hand with PokerKit.

Stacks reset every hand. The caller supplies the hole cards, three burn
cards, and the five board cards so the same deal can be replayed with
the bots rotated. Each decision is a public dict; the returned action
is coerced before PokerKit sees it.

::

    seat: 0   1   2    3   4    5
    pos: SB  BB  UTG  HJ  CO  BTN

    holes/board ---> PokerKit ---> public state ---> bot.act ---> coerce
                         ^                                      |
                         +---------- fold / call / raise -------+
    payoffs <--- showdown or last player standing
"""

import warnings
from dataclasses import dataclass, field

from pokerkit import Automation, Mode, NoLimitTexasHoldem

from llm_poker_lab.interface import BIG_BLIND, POSITIONS, SEATS, SMALL_BLIND, STACK_BB, STREETS, coerce

warnings.filterwarnings("ignore", module="pokerkit")

_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
    Automation.RUNOUT_COUNT_SELECTION,
)
_MAX_STEPS = 500


@dataclass
class HandResult:
    """Per-seat outcome of a single hand. Payoffs are chips, not big blinds."""

    payoffs: list[int]
    vpip: list[bool]
    pfr: list[bool]
    aggressive: list[int]
    calls: list[int]
    stacks: list[int] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)


def play_hand(
    bots: list,
    holes: list[list[str]],
    board: list[str],
    burns: list[str],
    hand_id: int = 0,
    sb: int = SMALL_BLIND,
    bb: int = BIG_BLIND,
    stack: int | None = None,
) -> HandResult:
    """Deal one hand and return chip payoffs plus VPIP / PFR / aggression counts."""
    if len(bots) != SEATS or len(holes) != SEATS:
        raise ValueError("play_hand deals exactly 6 seats")
    if len(board) < 5 or len(burns) < 3:
        raise ValueError("need 5 board cards and 3 burn cards")
    if stack is None:
        stack = STACK_BB * bb
    state = NoLimitTexasHoldem.create_state(
        _AUTOMATIONS,
        True,
        0,
        (sb, bb),
        bb,
        (stack,) * SEATS,
        SEATS,
        mode=Mode.CASH_GAME,
    )
    tracker = _Tracker(hand_id=hand_id, sb=sb, bb=bb)
    dealt_holes = 0
    dealt_board = 0
    dealt_burns = 0
    for _ in range(_MAX_STEPS):
        if not state.status:
            break
        if state.can_deal_hole():
            state.deal_hole("".join(holes[dealt_holes]))
            dealt_holes += 1
            continue
        if state.can_burn_card():
            state.burn_card(burns[dealt_burns])
            dealt_burns += 1
            continue
        if state.can_deal_board():
            count = int(state.board_dealing_count)
            state.deal_board("".join(board[dealt_board:dealt_board + count]))
            dealt_board += count
            continue
        actor = state.actor_index
        if actor is None:  # pragma: no cover
            raise RuntimeError("poker state stalled before the hand ended")
        view = _view(state, holes[actor], board[:dealt_board], tracker, actor)
        raw = _ask(bots[actor], view)
        action = coerce(raw, view)
        _apply(state, action)
        tracker.note(actor, view, action)
    else:
        raise RuntimeError("hand did not finish")  # pragma: no cover
    payoffs = [int(value) for value in state.payoffs]
    stacks = [int(value) for value in state.stacks]
    return HandResult(
        payoffs=payoffs,
        vpip=list(tracker.vpip),
        pfr=list(tracker.pfr),
        aggressive=list(tracker.aggressive),
        calls=list(tracker.calls),
        stacks=stacks,
        actions=list(tracker.actions),
    )


def _ask(bot, view: dict):
    """Call the bot. A crash is a fold, matching the sandbox rule."""
    try:
        return bot.act(view)
    except Exception:
        return {"kind": "fold"}


def _apply(state, action: dict) -> None:
    """Send a coerced action to PokerKit."""
    if action["kind"] == "raise":
        state.complete_bet_or_raise_to(int(action["amount"]))
        return
    if action["kind"] == "fold":
        state.fold()
        return
    state.check_or_call()


def _view(state, hole: list[str], board: list[str], tracker, actor: int) -> dict:
    """Build the public, serializable decision dict for ``actor``."""
    street = STREETS[int(state.street_index)]
    to_call = int(state.checking_or_calling_amount)
    pot = int(state.total_pot_amount)
    min_to = state.min_completion_betting_or_raising_to_amount
    max_to = state.max_completion_betting_or_raising_to_amount
    can_raise = bool(state.can_complete_bet_or_raise_to()) and min_to is not None
    denominator = pot + to_call
    return {
        "hole_cards": list(hole),
        "board": list(board),
        "street": street,
        "pot": pot,
        "to_call": to_call,
        "min_raise_to": None if min_to is None else int(min_to),
        "max_raise_to": None if max_to is None else int(max_to),
        "can_raise": can_raise,
        "can_check": to_call == 0,
        "hero_index": actor,
        "hero_position": POSITIONS[actor],
        "hero_stack": int(state.stacks[actor]),
        "hero_bet": int(state.bets[actor]),
        "stacks": [int(value) for value in state.stacks],
        "positions": list(POSITIONS),
        "action_history": [dict(item) for item in tracker.actions],
        "big_blind": tracker.bb,
        "small_blind": tracker.sb,
        "num_active": sum(1 for flag in state.statuses if flag),
        "hand_id": tracker.hand_id,
        "pot_odds": (to_call / denominator) if denominator else 0.0,
    }


class _Tracker:
    """VPIP, PFR, aggression, and the action log for one hand."""

    def __init__(self, hand_id: int, sb: int, bb: int) -> None:
        """Start empty counters for six seats."""
        self.hand_id = hand_id
        self.sb = sb
        self.bb = bb
        self.vpip = [False] * SEATS
        self.pfr = [False] * SEATS
        self.aggressive = [0] * SEATS
        self.calls = [0] * SEATS
        self.actions: list[dict] = []

    def note(self, actor: int, view: dict, action: dict) -> None:
        """Record the coerced action and update the running tallies."""
        kind = action["kind"]
        to_call = int(view["to_call"])
        amount = int(action.get("amount", to_call if kind == "check_call" else 0))
        if view["street"] == "preflop":
            if kind == "raise":
                self.vpip[actor] = True
                self.pfr[actor] = True
            elif kind == "check_call" and to_call > 0:
                self.vpip[actor] = True
        if kind == "raise":
            self.aggressive[actor] += 1
        elif kind == "check_call" and to_call > 0:
            self.calls[actor] += 1
        self.actions.append(
            {
                "seat": actor,
                "position": POSITIONS[actor],
                "street": view["street"],
                "kind": kind,
                "amount": amount,
            }
        )
