"""Shared decision dicts and scripted bots for the unit tests."""


def decision(**overrides) -> dict:
    """A legal public state. Override any key for the branch under test."""
    state = {
        "hole_cards": ["7c", "2d"],
        "board": [],
        "street": "preflop",
        "pot": 3,
        "to_call": 2,
        "min_raise_to": 4,
        "max_raise_to": 400,
        "can_raise": True,
        "can_check": False,
        "hero_index": 2,
        "hero_position": "UTG",
        "hero_stack": 400,
        "hero_bet": 0,
        "stacks": [399, 398, 400, 400, 400, 400],
        "positions": ["SB", "BB", "UTG", "HJ", "CO", "BTN"],
        "action_history": [],
        "big_blind": 2,
        "small_blind": 1,
        "num_active": 6,
        "hand_id": 1,
        "pot_odds": 0.4,
    }
    state.update(overrides)
    return state


class Scripted:
    """Returns a fixed action, or calls a function of the state."""

    def __init__(self, name, action):
        self.name = name
        self.action = action
        self.seen = []

    def act(self, state):
        """Record the state and return the scripted action."""
        self.seen.append(state)
        if callable(self.action):
            return self.action(state)
        return self.action


def fold_bot(name: str) -> Scripted:
    """A bot that always folds."""
    return Scripted(name, {"kind": "fold"})


SAMPLE_HOLES = [
    ["9h", "9d"],
    ["8c", "8d"],
    ["As", "Ad"],
    ["Kc", "Qd"],
    ["Jc", "Th"],
    ["7s", "6s"],
]
SAMPLE_BURNS = ["2c", "3c", "4d"]
SAMPLE_BOARD = ["2h", "3d", "4c", "5s", "9c"]

VALID_BOT = '''
class EvolvedBot:
    def act(self, state):
        if state["to_call"] == 0:
            return {"kind": "check_call"}
        return {"kind": "fold"}
'''

LOOP_BOT = '''
class EvolvedBot:
    def act(self, state):
        while True:
            pass
'''

CRASH_BOT = '''
class EvolvedBot:
    def act(self, state):
        raise RuntimeError("boom")
'''

BAD_IMPORT_BOT = '''
import os
class EvolvedBot:
    def act(self, state):
        return {"kind": "fold"}
'''
