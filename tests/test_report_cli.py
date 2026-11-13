"""Charts, leaderboards, the CLI, and one fake-Ollama evolution."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from llm_poker_lab.cli import main
from llm_poker_lab.evolve import _default_dir, _progress, evolve, run_generation
from llm_poker_lab.ollama import OllamaError
from llm_poker_lab.report import format_af, format_bb, format_ci, write_report
from tests.support import BAD_IMPORT_BOT, CRASH_BOT, LOOP_BOT, VALID_BOT


def _bot_row(name, bb, **extra):
    """A summary row shaped like StatBook.summary()."""
    row = {
        "name": name,
        "hands": 20,
        "bb_per_100": bb,
        "ci_low": bb - 2,
        "ci_high": bb + 2,
        "hands_won": 3,
        "vpip": 22.0,
        "pfr": 18.0,
        "aggression_factor": 1.5,
        "aggressive_actions": 3,
        "calls": 2,
        "profit_chips": 10,
        "disqualified": False,
        "error": None,
    }
    row.update(extra)
    return row


def test_formats_and_report_files(tmp_path):
    """Markdown treats missing aggression as n/a or inf, and charts are written."""
    assert format_af({"aggression_factor": None, "aggressive_actions": 0}) == "n/a"
    assert format_af({"aggression_factor": None, "aggressive_actions": 4}) == "inf"
    assert format_af({"aggression_factor": 2.5, "aggressive_actions": 5}) == "2.50"
    assert format_bb({"disqualified": True, "bb_per_100": 1}) == "DQ"
    assert format_bb({"disqualified": False, "bb_per_100": 1.26}) == "+1.3"
    assert format_ci({"disqualified": True, "hands": 4}) == "—"
    assert format_ci({"disqualified": False, "hands": 0}) == "—"
    assert format_ci({"disqualified": False, "hands": 4, "ci_low": -1.2, "ci_high": 3.4}) == "[-1.2, +3.4]"
    data = {
        "hands_played": 12,
        "hands_requested": 10,
        "hands_per_sec": 100.0,
        "elapsed_sec": 0.12,
        "seed": 1,
        "small_blind": 1,
        "big_blind": 2,
        "stack_bb": 200,
        "bots": [
            _bot_row("tag", 4.0),
            _bot_row("maniac", -3.0, aggression_factor=None, aggressive_actions=8, calls=0),
            _bot_row("rock", 0.0, disqualified=True, error="boom"),
        ],
    }
    (tmp_path / "results.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "gen-skip").write_text("not a directory", encoding="utf-8")
    path = write_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "tag" in text and "DQ" in text and "inf" in text
    assert (tmp_path / "baseline_comparison.png").stat().st_size > 1000
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "results.json").write_text(json.dumps({**data, "bots": []}), encoding="utf-8")
    assert "—" in write_report(empty).read_text(encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        write_report(tmp_path / "missing")


def test_generation_report(tmp_path):
    """An evolve directory with one good generation and one failure still charts."""
    good = tmp_path / "gen-002"
    bad = tmp_path / "gen-001"
    good.mkdir()
    bad.mkdir()
    (bad / "results.json").write_text(
        json.dumps({"generation": 1, "disqualified": True, "error": "syntax error\nline", "bots": []}),
        encoding="utf-8",
    )
    (good / "results.json").write_text(
        json.dumps(
            {
                "generation": 2,
                "disqualified": False,
                "error": None,
                "bots": [_bot_row("evolved", 3.5), _bot_row("tag", 1.0)],
            }
        ),
        encoding="utf-8",
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "Generation 2" in text
    assert "syntax error line" in text
    assert (tmp_path / "bb100_by_generation.png").stat().st_size > 1000
    only = tmp_path / "failed"
    only.mkdir()
    (only / "gen-001").mkdir()
    (only / "gen-001" / "results.json").write_text(
        json.dumps({"generation": 1, "disqualified": True, "error": "nope", "bots": []}),
        encoding="utf-8",
    )
    failed = write_report(only).read_text(encoding="utf-8")
    assert "No generation finished" in failed
    assert _progress(1, {"disqualified": True, "error": "nope"}).startswith("gen 001")
    assert "n/a" in _progress(2, {"disqualified": False, "bb_per_100": None})
    assert "+1.5" in _progress(3, {"disqualified": False, "bb_per_100": 1.5})
    assert _default_dir().name.startswith("evolve-")


def test_cli_simulate_and_errors(tmp_path, monkeypatch, capsys):
    """The simulate command writes a run, and bad arguments return exit code 1."""
    assert main(["simulate", "--hands", "6", "--seed", "1", "--out", str(tmp_path / "sim")]) == 0
    assert (tmp_path / "sim" / "leaderboard.md").is_file()
    assert "hands/sec" in capsys.readouterr().out
    assert main(["report", str(tmp_path / "sim")]) == 0
    assert "wrote" in capsys.readouterr().out
    assert main(["simulate", "--hands", "0"]) == 1
    assert main(["report", str(tmp_path / "missing")]) == 1
    from llm_poker_lab.cli import _stamp

    assert _stamp("sim").name.startswith("sim-")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_evolve(**kwargs):
        """Record CLI kwargs without calling Ollama."""
        captured.update(kwargs)
        target = Path(kwargs["out"]) if kwargs.get("out") else tmp_path / "evo"
        target.mkdir(exist_ok=True)
        (target / "results.json").write_text(
            json.dumps(
                {
                    "hands_played": 1,
                    "hands_requested": 1,
                    "hands_per_sec": 1,
                    "elapsed_sec": 1,
                    "seed": 1,
                    "small_blind": 1,
                    "big_blind": 2,
                    "stack_bb": 200,
                    "bots": [],
                }
            ),
            encoding="utf-8",
        )
        return target

    monkeypatch.setattr("llm_poker_lab.cli.evolve", fake_evolve)
    assert main(["evolve", "--generations", "2", "--hands", "6", "--model", "qwen2.5-coder"]) == 0
    assert captured["out"] is None
    assert captured["model"] == "qwen2.5-coder"
    assert main(["evolve", "--generations", "1", "--hands", "6", "--out", str(tmp_path / "explicit")]) == 0
    assert captured["out"] == tmp_path / "explicit"
    assert main(["evolve", "--generations", "0", "--hands", "6"]) == 1
    assert main(["evolve", "--generations", "1", "--hands", "0"]) == 1

    def boom(**kwargs):
        """Stand in for a down Ollama server."""
        raise OllamaError("down")

    monkeypatch.setattr("llm_poker_lab.cli.evolve", boom)
    assert main(["evolve", "--generations", "1", "--hands", "6", "--out", str(tmp_path / "down")]) == 1


def _reply(strategy: str, code: str) -> str:
    """Wrap code the way the system prompt asks the model to answer."""
    return f"STRATEGY:\n{strategy}\nCODE:\n```python\n{code}\n```\n"


CTOR_BOT = """
class EvolvedBot:
    def __init__(self):
        raise RuntimeError("ctor")
    def act(self, state):
        return {"kind": "fold"}
"""


def test_run_generation_paths(tmp_path):
    """Parse failures, import failures, and a constructor crash are recorded."""
    from llm_poker_lab.bots import make_baselines

    baselines = make_baselines(1)
    kwargs = dict(
        baselines=baselines,
        hands=6,
        seed=1,
        decision_timeout=0.4,
        max_failures=2,
        max_seconds=30,
        sb=1,
        bb=2,
        stack_bb=200,
        load_timeout=2,
    )
    parsed = run_generation(tmp_path, 1, "hello", "prompt", **kwargs)
    assert parsed["disqualified"] is True
    assert "act()" in parsed["error"]
    blocked = run_generation(tmp_path, 2, _reply("imports os", BAD_IMPORT_BOT), "prompt", **kwargs)
    assert blocked["disqualified"] is True
    assert "os" in blocked["error"]
    crashed = run_generation(tmp_path, 3, _reply("constructor fails", CTOR_BOT), "prompt", **kwargs)
    assert crashed["disqualified"] is True
    assert (tmp_path / "gen-003" / "bot.py").is_file()
    assert (tmp_path / "gen-003" / "strategy.md").is_file()
    with pytest.raises(ValueError):
        evolve(generations=0, out=tmp_path / "nope")


class _Handler(BaseHTTPRequestHandler):
    """Serves scripted chat completions for the end-to-end evolve test."""

    replies: list[str] = []
    seen: list[dict] = []

    def do_POST(self):
        """Record the prompt and return the next canned assistant message."""
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode())
        _Handler.seen.append(body)
        content = _Handler.replies.pop(0)
        payload = json.dumps({"message": {"content": content}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        """Keep the test log quiet."""
        return


def test_evolve_end_to_end_fake_llm(tmp_path):
    """A local fake Ollama drives a valid bot, a bad import, a crash, and a loop."""
    _Handler.replies = [
        _reply("fold junk and check when it is free", VALID_BOT),
        _reply("this one cheats", BAD_IMPORT_BOT),
        _reply("this one crashes", CRASH_BOT),
        _reply("this one loops", LOOP_BOT),
    ]
    _Handler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        code = main(
            [
                "evolve",
                "--generations",
                "4",
                "--hands",
                "6",
                "--host",
                host,
                "--model",
                "fake-qwen",
                "--seed",
                "1",
                "--top-k",
                "2",
                "--last-m",
                "2",
                "--decision-timeout",
                "0.3",
                "--max-failures",
                "2",
                "--ollama-timeout",
                "5",
                "--out",
                str(tmp_path),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
    assert code == 0
    assert len(_Handler.seen) == 4
    assert _Handler.seen[0]["model"] == "fake-qwen"
    assert "No previous generations" in _Handler.seen[0]["messages"][1]["content"]
    assert "Generation 1" in _Handler.seen[1]["messages"][1]["content"]
    results = [
        json.loads((tmp_path / f"gen-{index:03d}" / "results.json").read_text(encoding="utf-8"))
        for index in range(1, 5)
    ]
    assert results[0]["disqualified"] is False
    assert results[0]["hands_played"] > 0
    assert any(row["name"] == "evolved" for row in results[0]["bots"])
    assert results[1]["disqualified"] is True
    assert results[2]["disqualified"] is True
    assert results[3]["disqualified"] is True
    assert "timed out" in (results[3]["error"] or "")
    board = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert "Evolved bot by generation" in board
    assert (tmp_path / "bb100_by_generation.png").stat().st_size > 1000
    assert (tmp_path / "baseline_comparison.png").stat().st_size > 1000
