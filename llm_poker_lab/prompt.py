"""The fixed system prompt and a bounded history of past generations.

A local model has a small context window, so the user message keeps only
the top-K strategies by bb/100 and the last-M generations. Code and the
plain-English strategy are clipped, then whole generations are dropped,
and the text is hard-cut if it is still too long.

::

    history ---> top K + last M ---> clipped blocks ---> user message
"""

import re

from llm_poker_lab.interface import ACTION_DOC, STATE_DOC

SYSTEM_PROMPT = f"""You write one Texas Hold'em bot per generation for a 6-max no-limit cash game.
Blinds are 1/2 chips. Every stack resets to 400 chips (200 big blinds) at the start of each hand.
Seats are SB, BB, UTG, HJ, CO, BTN.

{STATE_DOC}

{ACTION_DOC}

Constraints:
- Define class EvolvedBot with def act(self, state).
- You may import only math, random, statistics, collections, itertools, functools, json, re.
- Do not use os, sys, subprocess, sockets, open, eval, exec, or files.
- act must return quickly. Do not write an unbounded loop.
- Prefer a check over a fold when to_call is 0.

Reply in exactly this shape and nothing else:

STRATEGY:
<at most 120 words: what the bot does and what changed from last time>

CODE:
```python
class EvolvedBot:
    def act(self, state: dict) -> dict:
        ...
```
"""


class ParseError(ValueError):
    """The model reply did not contain a bot with an act method."""


def parse_response(text: str) -> tuple[str, str]:
    """Pull the strategy paragraph and the python block out of a reply."""
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.S | re.I)
    code = ""
    for block in blocks:
        if "def act" in block:
            code = block.strip()
            break
    if not code and blocks:
        code = blocks[0].strip()
    if "def act" not in code:
        raise ParseError("response has no act() method")
    match = re.search(r"STRATEGY:\s*(.*?)(?:\n\s*CODE:|\n```)", text, flags=re.S | re.I)
    strategy = match.group(1).strip() if match else ""
    if not strategy:
        strategy = "No strategy description was provided."
    return strategy, code.strip() + "\n"


def build_user_prompt(history: list[dict], top_k: int = 3, last_m: int = 2, max_chars: int = 12000) -> str:
    """Describe past generations without exceeding ``max_chars``."""
    if not history:
        return "No previous generations. Write a solid first bot that beats a loose, aggressive field."
    chosen = _select(history, top_k, last_m)
    code_limit = 2500
    strategy_limit = 700
    text = _render(chosen, code_limit, strategy_limit)
    while len(text) > max_chars and code_limit > 400:
        code_limit = max(400, code_limit // 2)
        strategy_limit = max(120, strategy_limit // 2)
        text = _render(chosen, code_limit, strategy_limit)
    while len(text) > max_chars and len(chosen) > 1:
        chosen = _shrink(chosen)
        text = _render(chosen, code_limit, strategy_limit)
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _select(history: list[dict], top_k: int, last_m: int) -> list[tuple[dict, str]]:
    """Pick the best finished bots and the most recent attempts, deduped."""
    viable = [row for row in history if not row.get("disqualified") and row.get("bb_per_100") is not None]
    best = sorted(viable, key=lambda row: row["bb_per_100"], reverse=True)[: max(0, top_k)]
    recent = list(history[-max(0, last_m):]) if last_m > 0 else []
    chosen = []
    seen = set()
    for row in best + recent:
        generation = row["generation"]
        if generation in seen:
            continue
        seen.add(generation)
        tags = []
        if row in best:
            tags.append("top")
        if row in recent:
            tags.append("recent")
        chosen.append((row, ", ".join(tags)))
    return chosen


def _shrink(chosen: list[tuple[dict, str]]) -> list[tuple[dict, str]]:
    """Drop a middle generation, keeping the first and the last blocks."""
    if len(chosen) < 3:
        return chosen[:1]
    return [chosen[0], *chosen[2:]]


def _render(chosen: list[tuple[dict, str]], code_limit: int, strategy_limit: int) -> str:
    """Join the selected generations into the user message."""
    blocks = [
        "Previous generations follow. Beat the best bb/100 and do not repeat a disqualified mistake."
    ]
    for row, tag in chosen:
        blocks.append(_one(row, tag, code_limit, strategy_limit))
    return "\n\n".join(blocks)


def _one(row: dict, tag: str, code_limit: int, strategy_limit: int) -> str:
    """Format a single generation's stats, strategy, and source."""
    header = f"### Generation {row['generation']} ({tag})"
    return "\n".join(
        [
            header,
            _stat_line(row),
            "Strategy:",
            _clip(str(row.get("strategy") or ""), strategy_limit),
            "```python",
            _clip(str(row.get("code") or ""), code_limit),
            "```",
        ]
    )


def _stat_line(row: dict) -> str:
    """One line of results, or the disqualification error."""
    if row.get("disqualified"):
        return f"DISQUALIFIED: {row.get('error')}"
    factor = row.get("aggression_factor")
    factor_text = "n/a" if factor is None else f"{factor:.2f}"
    bb = row.get("bb_per_100")
    bb_text = "n/a" if bb is None else f"{bb:+.1f}"
    low = row.get("ci_low")
    high = row.get("ci_high")
    return (
        f"bb/100 {bb_text} CI [{low}, {high}] "
        f"VPIP {row.get('vpip')} PFR {row.get('pfr')} AF {factor_text} hands {row.get('hands')}"
    )


def _clip(text: str, limit: int) -> str:
    """Shorten text that would crowd out the rest of the prompt."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...truncated..."
