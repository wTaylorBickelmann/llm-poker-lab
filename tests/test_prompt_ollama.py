"""Prompt assembly and the Ollama HTTP client."""

import io
import json
import urllib.error

import pytest

from llm_poker_lab.ollama import OllamaError, chat, normalize_host
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


def test_ollama_chat(monkeypatch):
    """The client reads message content and turns transport errors into OllamaError."""
    seen = {}

    class Response:
        """A context manager around a fixed body."""

        def __init__(self, payload):
            self.payload = payload

        def read(self):
            """Return the canned body."""
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(request, timeout=0):
        """Capture the outgoing chat request."""
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data.decode())
        return Response(b'{"message": {"content": " STRATEGY:\\nfold\\n "}}')

    monkeypatch.setattr("llm_poker_lab.ollama.urllib.request.urlopen", fake_open)
    assert "fold" in chat("localhost:11434", "qwen", "sys", "user", timeout=3, temperature=0.2)
    assert seen["timeout"] == 3
    assert seen["body"]["model"] == "qwen"
    assert seen["body"]["messages"][0]["content"] == "sys"
    assert normalize_host("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"

    def raise_http(request, timeout=0):
        """Simulate an HTTP 500 from Ollama."""
        raise urllib.error.HTTPError("http://x", 500, "bad", hdrs=None, fp=io.BytesIO(b"nope"))

    monkeypatch.setattr("llm_poker_lab.ollama.urllib.request.urlopen", raise_http)
    with pytest.raises(OllamaError, match="HTTP 500"):
        chat("http://127.0.0.1:9", "m", "s", "u", timeout=1)

    def raise_url(request, timeout=0):
        """Simulate a refused connection."""
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("llm_poker_lab.ollama.urllib.request.urlopen", raise_url)
    with pytest.raises(OllamaError, match="Could not reach"):
        chat("http://127.0.0.1:9", "m", "s", "u", timeout=1)

    def raise_timeout(request, timeout=0):
        """Simulate a stalled server."""
        raise TimeoutError("slow")

    monkeypatch.setattr("llm_poker_lab.ollama.urllib.request.urlopen", raise_timeout)
    with pytest.raises(OllamaError, match="timed out"):
        chat("http://127.0.0.1:9", "m", "s", "u", timeout=1)

    def bad_body(payload):
        """Return a response with the given bytes."""

        def opener(request, timeout=0):
            return Response(payload)

        return opener

    monkeypatch.setattr("llm_poker_lab.ollama.urllib.request.urlopen", bad_body(b"not-json"))
    with pytest.raises(OllamaError, match="Unexpected"):
        chat("http://h", "m", "s", "u")
    monkeypatch.setattr("llm_poker_lab.ollama.urllib.request.urlopen", bad_body(b'{"message": {}}'))
    with pytest.raises(OllamaError, match="Unexpected"):
        chat("http://h", "m", "s", "u")
    monkeypatch.setattr("llm_poker_lab.ollama.urllib.request.urlopen", bad_body(b'{"message": {"content": "  "}}'))
    with pytest.raises(OllamaError, match="empty"):
        chat("http://h", "m", "s", "u")
    monkeypatch.setattr("llm_poker_lab.ollama.urllib.request.urlopen", bad_body(b'{"message": {"content": 3}}'))
    with pytest.raises(OllamaError, match="empty"):
        chat("http://h", "m", "s", "u")
