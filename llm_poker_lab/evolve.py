"""One generation: ask the model, sandbox the bot, measure it, save the files.

Ollama is called once per generation. The bot then plays the configured
number of hands against the baseline field. Whatever comes back, including
a disqualification, is written under ``runs/<id>/gen-NNN/`` and becomes
context for the next prompt.

::

    prompt + history ---> Ollama ---> bot.py + strategy.md
                              |
                              v
                         sandbox match ---> results.json ---> next prompt
"""

import json
from pathlib import Path

from llm_poker_lab.bots import make_baselines
from llm_poker_lab.ollama import chat
from llm_poker_lab.prompt import SYSTEM_PROMPT, ParseError, build_user_prompt, parse_response
from llm_poker_lab.report import write_report
from llm_poker_lab.sandbox import SandboxBot, SandboxError, validate_source
from llm_poker_lab.simulate import simulate


def evolve(
    generations: int = 20,
    hands: int = 3000,
    model: str = "qwen2.5-coder",
    host: str = "http://127.0.0.1:11434",
    seed: int = 1,
    out: Path | None = None,
    top_k: int = 3,
    last_m: int = 2,
    decision_timeout: float = 0.5,
    max_failures: int = 5,
    max_seconds: float = 120.0,
    temperature: float = 0.7,
    ollama_timeout: float = 180.0,
    sb: int = 1,
    bb: int = 2,
    stack_bb: int = 200,
    max_chars: int = 12000,
    chatter=None,
) -> Path:
    """Run ``generations`` of write-and-measure and return the run directory."""
    if generations < 1:
        raise ValueError("generations must be positive")
    run_dir = Path(out) if out is not None else _default_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    talk = chatter or chat
    config = {
        "generations": generations,
        "hands": hands,
        "model": model,
        "host": host,
        "seed": seed,
        "top_k": top_k,
        "last_m": last_m,
        "decision_timeout": decision_timeout,
        "max_failures": max_failures,
        "temperature": temperature,
        "sb": sb,
        "bb": bb,
        "stack_bb": stack_bb,
    }
    _dump(run_dir / "config.json", config)
    history: list[dict] = []
    baselines = make_baselines(seed)
    for generation in range(1, generations + 1):
        user_prompt = build_user_prompt(history, top_k=top_k, last_m=last_m, max_chars=max_chars)
        raw = talk(host, model, SYSTEM_PROMPT, user_prompt, timeout=ollama_timeout, temperature=temperature)
        record = run_generation(
            run_dir,
            generation,
            raw,
            user_prompt,
            baselines,
            hands=hands,
            seed=seed,
            decision_timeout=decision_timeout,
            max_failures=max_failures,
            max_seconds=max_seconds,
            sb=sb,
            bb=bb,
            stack_bb=stack_bb,
        )
        history.append(record)
        print(_progress(generation, record))
    _dump(run_dir / "generations.json", history)
    write_report(run_dir)
    return run_dir


def run_generation(
    run_dir: Path,
    generation: int,
    raw: str,
    user_prompt: str,
    baselines: list,
    hands: int,
    seed: int,
    decision_timeout: float,
    max_failures: int,
    max_seconds: float,
    sb: int,
    bb: int,
    stack_bb: int,
    load_timeout: float = 5.0,
) -> dict:
    """Save one generation's files, play the match, and return prompt context."""
    gen_dir = run_dir / f"gen-{generation:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "prompt.txt").write_text(user_prompt, encoding="utf-8")
    (gen_dir / "response.txt").write_text(raw, encoding="utf-8")
    strategy, code, error = _extract(raw)
    (gen_dir / "strategy.md").write_text((strategy or "No strategy.") + "\n", encoding="utf-8")
    (gen_dir / "bot.py").write_text(code or "# no code extracted\n", encoding="utf-8")
    results = {
        "generation": generation,
        "strategy": strategy,
        "disqualified": error is not None,
        "error": error,
        "hands_played": 0,
        "elapsed_sec": 0.0,
        "hands_per_sec": 0.0,
        "bots": [],
    }
    if error is None:
        _play(
            results, code, baselines, hands, seed, generation,
            decision_timeout, max_failures, max_seconds, sb, bb, stack_bb, load_timeout,
        )
    _dump(gen_dir / "results.json", results)
    evolved = _evolved(results)
    return {
        "generation": generation,
        "strategy": strategy,
        "code": code,
        "bb_per_100": None if results["disqualified"] or evolved is None else evolved["bb_per_100"],
        "ci_low": None if evolved is None else evolved.get("ci_low"),
        "ci_high": None if evolved is None else evolved.get("ci_high"),
        "vpip": None if evolved is None else evolved.get("vpip"),
        "pfr": None if evolved is None else evolved.get("pfr"),
        "aggression_factor": None if evolved is None else evolved.get("aggression_factor"),
        "hands": 0 if evolved is None else evolved.get("hands", 0),
        "disqualified": results["disqualified"],
        "error": results["error"],
    }


def _play(results, code, baselines, hands, seed, generation, decision_timeout, max_failures, max_seconds, sb, bb, stack_bb, load_timeout) -> None:
    """Run the sandboxed bot against the baselines and copy the match stats."""
    bot = SandboxBot(
        code,
        timeout=decision_timeout,
        max_failures=max_failures,
        max_seconds=max_seconds,
        load_timeout=load_timeout,
        seed=seed + generation,
    )
    try:
        bot.start()
        if bot.disqualified:
            results["disqualified"] = True
            results["error"] = bot.last_error
            return
        match = simulate(
            [bot, *baselines],
            hands=hands,
            seed=seed + generation,
            sb=sb,
            bb=bb,
            stack_bb=stack_bb,
            locked=(bot.name,),
        )
        public = match.public()
        results["hands_played"] = public["hands_played"]
        results["elapsed_sec"] = public["elapsed_sec"]
        results["hands_per_sec"] = public["hands_per_sec"]
        results["bots"] = public["bots"]
        evolved = _find(public["bots"], bot.name)
        if evolved is not None and evolved.get("disqualified"):
            results["disqualified"] = True
            results["error"] = evolved.get("error")
    finally:
        bot.close()


def _extract(raw: str) -> tuple[str, str, str | None]:
    """Return strategy, code, and an error string when the reply is unusable."""
    try:
        strategy, code = parse_response(raw)
    except ParseError as exc:
        return "", "", str(exc)
    try:
        validate_source(code)
    except SandboxError as exc:
        return strategy, code, str(exc)
    return strategy, code, None


def _evolved(results: dict) -> dict | None:
    """The evolved bot's row, if the match recorded one."""
    return _find(results.get("bots") or [], "evolved")


def _find(rows: list[dict], name: str) -> dict | None:
    """Return the summary dict with this bot name."""
    for row in rows:
        if row.get("name") == name:
            return row
    return None


def _progress(generation: int, record: dict) -> str:
    """One console line so a long run shows which generation just finished."""
    if record["disqualified"]:
        return f"gen {generation:03d}  disqualified  {record.get('error')}"
    bb = record.get("bb_per_100")
    shown = "n/a" if bb is None else f"{bb:+.1f}"
    return f"gen {generation:03d}  bb/100 {shown}"


def _default_dir() -> Path:
    """Timestamped folder under ``runs/`` for a new evolution."""
    import time

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path("runs") / f"evolve-{stamp}"


def _dump(path: Path, payload) -> None:
    """Write pretty JSON."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
