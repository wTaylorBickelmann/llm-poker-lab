"""Prompt assembly for the evolution loop."""

import pytest

from llm_poker_lab.prompt import SYSTEM_PROMPT, ParseError, build_user_prompt, parse_response


def _row(generation, bb=1.0, **extra):
    """One history record for the prompt builder."""
    row = {
        "generation": generation,
        "bb_per_100": bb,
        "ci_low": None if bb is None else bb - 1,
        "ci_high": None if bb is None else bb + 1,
        "vpip": 20,
        "pfr": 15,
        "aggression_factor": 2,
        "hands": 100,
        "strategy": f"strategy {generation}",
        "code": f"class EvolvedBot:\n    def act(self, state):\n        return {{'kind': 'fold'}}  # {generation}\n",
        "disqualified": False,
        "error": None,
    }
    row.update(extra)
    return row


def test_system_prompt_and_parser():
    """The system prompt states the contract, and replies split into two parts."""
    assert "EvolvedBot" in SYSTEM_PROMPT
    assert "check_call" in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT) < 4500
    text = (
        "STRATEGY:\nPlay tight from early position.\nCODE:\n"
        "```python\nclass EvolvedBot:\n    def act(self, state):\n        return {'kind': 'fold'}\n```\n"
    )
    strategy, code = parse_response(text)
    assert "tight" in strategy
    assert code.startswith("class EvolvedBot")
    assert code.endswith("\n")
    strategy, code = parse_response(
        "```python\nprint(1)\n```\n```python\nclass EvolvedBot:\n    def act(self, state):\n        return {'kind': 'fold'}\n```"
    )
    assert "def act" in code
    with pytest.raises(ParseError):
        parse_response("no code here")
    with pytest.raises(ParseError):
        parse_response("```python\nx = 1\n```")
    strategy, code = parse_response(
        "```python\nclass EvolvedBot:\n    def act(self, state):\n        return {'kind': 'fold'}\n```"
    )
    assert strategy.startswith("No strategy")


def test_prompt_bounds_and_selection():
    """Top results and recent failures are kept, and the text stays bounded."""
    assert "No previous generations" in build_user_prompt([])
    best = _row(1, bb=5)
    text = build_user_prompt([best], top_k=3, last_m=2)
    assert "top, recent" in text
    assert "AF 2.00" in text
    bad = _row(2, bb=None, disqualified=True, error="import os is not allowed", strategy="bad", code="import os\n")
    text = build_user_prompt([best, bad], top_k=1, last_m=1)
    assert "DISQUALIFIED: import os is not allowed" in text
    assert "Generation 1" in text
    odd = _row(3, bb=None, aggression_factor=None, strategy="s", code="c")
    text = build_user_prompt([odd], top_k=0, last_m=1)
    assert "bb/100 n/a" in text
    assert "AF n/a" in text
    assert "Previous generations" in build_user_prompt([best], top_k=0, last_m=0)
    long = _row(1, strategy="word " * 1000, code="c" * 3000)
    clipped = build_user_prompt([long], top_k=1, last_m=0, max_chars=12000)
    assert "...truncated..." in clipped
    history = [_row(i, bb=float(i), strategy="s" * 2000, code="c" * 4000) for i in range(1, 6)]
    bounded = build_user_prompt(history, top_k=4, last_m=3, max_chars=500)
    assert len(bounded) <= 500
