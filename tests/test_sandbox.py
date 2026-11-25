"""Static checks, the in-process worker, and the child-process leash."""

import json
import resource
import subprocess
from io import StringIO

import pytest

from llm_poker_lab.sandbox import (
    SandboxBot,
    SandboxError,
    apply_limits,
    guarded_import,
    limit_resource,
    load_bot,
    parse_reply,
    silent_print,
    validate_source,
    wait_for,
)
from llm_poker_lab.sandbox_worker import serve
from tests.support import CRASH_BOT, LOOP_BOT, VALID_BOT


def test_validate_source_accepts_and_rejects():
    """The auditor allows a normal bot and blocks the obvious escapes."""
    assert validate_source(VALID_BOT) == "EvolvedBot"
    other = VALID_BOT.replace("EvolvedBot", "OtherBot")
    second = other + "\nclass SecondBot:\n    def act(self, state, extra=0):\n        return {'kind': 'fold'}\n"
    assert validate_source(second) == "OtherBot"
    named = "class OtherBot:\n    def act(self, state):\n        return {'kind': 'fold'}\n\n" + VALID_BOT
    assert validate_source(named) == "EvolvedBot"
    star = "class EvolvedBot:\n    def act(self, *args):\n        return {'kind': 'fold'}\n"
    assert validate_source(star) == "EvolvedBot"
    allowed = (
        "from math import sqrt\n"
        "import collections.abc\n"
        "class EvolvedBot:\n"
        "    def __init__(self):\n"
        "        self.n = 1\n"
        "    def act(self, state):\n"
        "        x = __init__\n"
        "        return {'kind': 'fold'}\n"
    )
    assert validate_source(allowed) == "EvolvedBot"
    with pytest.raises(SandboxError):
        validate_source("import os\n" + VALID_BOT)
    with pytest.raises(SandboxError):
        validate_source("from os.path import join\n" + VALID_BOT)
    with pytest.raises(SandboxError):
        validate_source("from .math import sqrt\n" + VALID_BOT)
    with pytest.raises(SandboxError):
        validate_source("class EvolvedBot:\n    def act(self, state):\n        open('x')\n")
    with pytest.raises(SandboxError):
        validate_source("class EvolvedBot:\n    def act(self, state):\n        return self.__class__\n")
    with pytest.raises(SandboxError):
        validate_source("class EvolvedBot:\n    def act(self, state):\n        return __loader__\n")
    with pytest.raises(SandboxError):
        validate_source("class EvolvedBot:\n    def act(self):\n        return {'kind': 'fold'}\n")
    with pytest.raises(SandboxError):
        validate_source("x = 1\n")
    with pytest.raises(SandboxError):
        validate_source("def act(:\n")
    with pytest.raises(SandboxError):
        validate_source("x" * 20_001)
    with pytest.raises(SandboxError):
        validate_source("```python\nx = 1\n```\n")


def test_import_hook_and_loader():
    """Runtime imports stay on the allow list, and exec builds the class."""
    import math

    assert guarded_import("math").sqrt is math.sqrt
    assert guarded_import("collections.abc", fromlist=("Iterable",)).Iterable
    with pytest.raises(ImportError):
        guarded_import("os")
    with pytest.raises(ImportError):
        guarded_import("math", level=1)
    silent_print("hidden")
    bot = load_bot(VALID_BOT, seed=3)
    assert bot.act({"to_call": 0})["kind"] == "check_call"
    assert bot.act({"to_call": 2})["kind"] == "fold"
    renamed = load_bot(VALID_BOT.replace("EvolvedBot", "OtherBot"))
    assert renamed.act({"to_call": 2})["kind"] == "fold"
    math_bot = load_bot(
        "import math\nclass EvolvedBot:\n"
        "    def act(self, state):\n"
        "        return {'kind': 'check_call'} if math.floor(state['to_call']) == 0 else {'kind': 'fold'}\n"
    )
    assert math_bot.act({"to_call": 0})["kind"] == "check_call"
    overwritten = VALID_BOT + "\nEvolvedBot = 1\n"
    with pytest.raises(SandboxError):
        load_bot(overwritten)


def test_worker_protocol():
    """The line protocol loads a bot, reports crashes, and stops cleanly."""

    def run(lines):
        """Feed JSON lines to serve() and parse the replies."""
        incoming = StringIO("\n".join(lines) + "\n")
        outgoing = StringIO()
        serve(incoming, outgoing)
        return [json.loads(line) for line in outgoing.getvalue().splitlines() if line]

    replies = run(
        [
            "",
            "not-json",
            "1",
            json.dumps({"op": "act", "state": {}}),
            json.dumps({"op": "nope"}),
            json.dumps({"op": "load", "seed": "bad"}),
            json.dumps({"op": "load", "code": "import os\n"}),
            json.dumps({"op": "load", "code": VALID_BOT, "seed": 1}),
            json.dumps({"op": "act", "state": {"to_call": 0}}),
            json.dumps({"op": "act", "state": {"to_call": 2}}),
            json.dumps({"op": "stop"}),
            json.dumps({"op": "act", "state": {}}),
        ]
    )
    assert replies[0]["error"] == "invalid json"
    assert replies[1]["error"] == "message must be an object"
    assert replies[2]["error"] == "bot not loaded"
    assert replies[3]["error"] == "unknown op"
    assert replies[4]["ok"] is False
    assert replies[5]["ok"] is False
    assert replies[6]["ok"] is True
    assert replies[7]["action"]["kind"] == "check_call"
    assert replies[8]["action"]["kind"] == "fold"
    crash = run(
        [
            json.dumps({"op": "load", "code": CRASH_BOT}),
            json.dumps({"op": "act", "state": {}}),
            json.dumps(
                {
                    "op": "load",
                    "code": "class EvolvedBot:\n    def act(self, state):\n        return ['fold']\n",
                }
            ),
            json.dumps({"op": "act", "state": {}}),
            json.dumps(
                {
                    "op": "load",
                    "code": "class EvolvedBot:\n    def act(self, state):\n        return {'kind': 'dance'}\n",
                }
            ),
            json.dumps({"op": "act", "state": {}}),
            json.dumps(
                {
                    "op": "load",
                    "code": "class EvolvedBot:\n    def act(self, state):\n        return {'kind': 'raise', 'amount': 8}\n",
                }
            ),
            json.dumps({"op": "act", "state": {}}),
        ]
    )
    assert "boom" in crash[1]["error"]
    assert crash[3]["error"] == "action must be a dict"
    assert "dance" in crash[5]["error"]
    assert crash[7]["action"] == {"kind": "raise", "amount": 8}


def test_parse_reply_and_limits(monkeypatch):
    """Bad child output is rejected, and rlimits survive a platform refusal."""
    with pytest.raises(SandboxError):
        parse_reply(b"not-json")
    with pytest.raises(SandboxError):
        parse_reply(b"[1]")
    assert parse_reply(b'{"ok": true}')["ok"] is True

    class Hung:
        """A process handle whose wait never returns in time."""

        def wait(self, timeout):
            """Pretend the process ignored the kill."""
            raise subprocess.TimeoutExpired(cmd="bot", timeout=timeout)

    wait_for(Hung())
    calls = {"n": 0}

    def fake_set(kind, limits):
        """Count rlimit updates and fail on the CPU cap."""
        calls["n"] += 1
        if kind == resource.RLIMIT_CPU:
            raise OSError("nope")
        if kind == resource.RLIMIT_FSIZE:
            raise ValueError("nope")

    monkeypatch.setattr("llm_poker_lab.sandbox.resource.setrlimit", fake_set)
    limit_resource(resource.RLIMIT_AS, 1024)
    limit_resource(resource.RLIMIT_CPU, 1)
    limit_resource(resource.RLIMIT_FSIZE, 0)
    apply_limits()
    assert calls["n"] >= 4
    monkeypatch.delattr("llm_poker_lab.sandbox.resource.RLIMIT_NPROC")
    before = calls["n"]
    apply_limits()
    assert calls["n"] - before == 3


def test_sandbox_process_paths():
    """The child folds on timeout, crash, bad output, and a dead pipe."""
    broken = SandboxBot("import os\n")
    assert broken.disqualified
    broken.start()
    assert broken.act({})["kind"] == "fold"
    broken.close()

    limited = SandboxBot(VALID_BOT, max_seconds=-1)
    assert limited.act({})["kind"] == "fold"
    assert "time limit" in limited.last_error
    assert limited.act({})["kind"] == "fold"

    with pytest.raises(SandboxError):
        SandboxBot(VALID_BOT)._send({"op": "stop"})
    with pytest.raises(SandboxError):
        SandboxBot(VALID_BOT)._readline(0.1)

    live = SandboxBot(VALID_BOT, timeout=1, max_failures=3)
    live.start()
    pid = live.proc.pid
    live.start()
    assert live.proc.pid == pid
    assert live.act({"to_call": 2}) == {"kind": "fold"}
    live.proc.kill()
    live.proc.wait(timeout=2)
    assert live.act({"to_call": 0})["kind"] == "check_call"
    live.close()

    rushed = SandboxBot(VALID_BOT, timeout=0, max_failures=1)
    rushed.start()
    assert rushed.act({"to_call": 0})["kind"] == "fold"
    assert rushed.disqualified
    rushed.close()

    looping = SandboxBot(LOOP_BOT, timeout=0.25, max_failures=2, load_timeout=2)
    looping.start()
    assert looping.act({"to_call": 1})["kind"] == "fold"
    assert looping.disqualified is False
    assert looping.act({"to_call": 1})["kind"] == "fold"
    assert looping.disqualified is True
    assert "timed out" in looping.last_error
    looping.close()

    crashing = SandboxBot(CRASH_BOT, timeout=1, max_failures=1)
    crashing.start()
    assert crashing.act({})["kind"] == "fold"
    assert "boom" in crashing.last_error
    crashing.close()

    dancer = SandboxBot(
        "class EvolvedBot:\n    def act(self, state):\n        return {'kind': 'dance'}\n",
        timeout=1,
        max_failures=1,
    )
    dancer.start()
    assert dancer.act({})["kind"] == "fold"
    assert dancer.disqualified
    dancer.close()

    exiting = SandboxBot(
        "class EvolvedBot:\n    def act(self, state):\n        raise SystemExit(3)\n",
        timeout=1,
        max_failures=2,
    )
    exiting.start()
    assert exiting.act({})["kind"] == "fold"
    assert exiting.failures == 1
    assert "exited" in (exiting.last_error or "")
    exiting.close()

    ctor = SandboxBot(
        "class EvolvedBot:\n    def __init__(self):\n        raise RuntimeError('ctor')\n"
        "    def act(self, state):\n        return {'kind': 'fold'}\n",
        timeout=1,
        load_timeout=2,
    )
    assert ctor.act({})["kind"] == "fold"
    assert ctor.disqualified
    ctor.close()

    dancer_reply = SandboxBot(VALID_BOT, timeout=1, max_failures=3)
    dancer_reply.start()
    dancer_reply._recv = lambda timeout: {"ok": True, "action": {"kind": "dance"}}
    assert dancer_reply.act({"to_call": 0})["kind"] == "fold"
    dancer_reply.close()

    blank = SandboxBot(VALID_BOT, timeout=1, max_failures=3)
    blank.start()
    blank._recv = lambda timeout: {"ok": False}
    assert blank.act({"to_call": 0})["kind"] == "fold"
    assert blank.last_error == "invalid sandbox reply"
    blank.close()

    hung = SandboxBot(
        "class EvolvedBot:\n    x = 0\n    while x >= 0:\n        x += 1\n    def act(self, state):\n        return {'kind': 'fold'}\n",
        load_timeout=0.4,
    )
    hung.start()
    assert hung.disqualified
    assert hung.last_error
    hung.close()

    dead = SandboxBot(VALID_BOT, load_timeout=2)
    dead._spawn()
    dead.proc.stdin.close()
    dead.proc.wait(timeout=2)
    with pytest.raises(SandboxError):
        dead._readline(1)
    dead.close()
