"""Child-process entry for one sandboxed bot.

The parent speaks newline-delimited JSON on stdin and stdout. ``load``
execs the bot under the restricted builtins. ``act`` calls it and turns
exceptions into an error reply so the parent can count a fold.

::

    stdin JSON ---> load / act / stop ---> stdout JSON
"""

import json
import sys

from llm_poker_lab.sandbox import apply_limits, load_bot


def serve(stdin, stdout) -> None:
    """Read requests until EOF or a stop message."""
    bot = None
    for line in stdin:
        text = line.strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            _write(stdout, {"ok": False, "error": "invalid json"})
            continue
        if not isinstance(message, dict):
            _write(stdout, {"ok": False, "error": "message must be an object"})
            continue
        op = message.get("op")
        if op == "load":
            bot, reply = _load(message)
            _write(stdout, reply)
        elif op == "act":
            _write(stdout, _act(bot, message.get("state")))
        elif op == "stop":
            break
        else:
            _write(stdout, {"ok": False, "error": "unknown op"})


def _load(message: dict) -> tuple:
    """Construct the bot. Any failure becomes an ``ok: false`` reply."""
    try:
        instance = load_bot(str(message.get("code", "")), int(message.get("seed", 0)))
    except Exception as exc:
        return None, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return instance, {"ok": True}


def _act(bot, state) -> dict:
    """Call ``act`` and normalize the action dict."""
    if bot is None:
        return {"ok": False, "error": "bot not loaded"}
    try:
        action = bot.act(state)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(action, dict):
        return {"ok": False, "error": "action must be a dict"}
    kind = action.get("kind")
    if kind not in {"fold", "check_call", "raise"}:
        return {"ok": False, "error": f"bad kind {kind}"}
    payload = {"kind": kind}
    if kind == "raise":
        payload["amount"] = action.get("amount")
    return {"ok": True, "action": payload}


def _write(stdout, payload: dict) -> None:
    """Send one JSON line and flush it so the parent can select on it."""
    stdout.write(json.dumps(payload) + "\n")
    stdout.flush()


def main() -> None:  # pragma: no cover
    """Process entry: apply rlimits, then serve stdin/stdout."""
    apply_limits()
    serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
