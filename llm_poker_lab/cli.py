"""Command line: simulate the field, evolve with a local or remote LLM, or redraw a report.

::

    poker-lab simulate ---> Match ---> runs/sim-*/leaderboard.md
    poker-lab evolve ----> LLMClient + sandbox ---> runs/evolve-*/gen-NNN/
    poker-lab report ----> charts and markdown for an existing run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from llm_poker_lab.bots import make_baselines
from llm_poker_lab.evolve import evolve
from llm_poker_lab.llm import LLMError, build_client, resolve_settings
from llm_poker_lab.report import format_match, write_report
from llm_poker_lab.simulate import simulate


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run one subcommand. Returns a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "simulate":
            return _cmd_simulate(args)
        if args.cmd == "evolve":
            return _cmd_evolve(args)
        return _cmd_report(args)
    except (LLMError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    """Build the simulate / evolve / report parsers."""
    parser = argparse.ArgumentParser(
        prog="poker-lab",
        description="Measure hold'em bots and let a local LLM write the next one.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sim = sub.add_parser("simulate", help="play the baseline field")
    _add_table_args(sim)
    sim.add_argument("--hands", type=int, default=6000)
    sim.add_argument("--seed", type=int, default=1)
    sim.add_argument("--out", default=None)
    evo = sub.add_parser("evolve", help="ask a local or remote LLM for a new bot each generation")
    _add_table_args(evo)
    evo.add_argument("--generations", type=int, default=20)
    evo.add_argument("--hands", type=int, default=3000)
    evo.add_argument("--provider", choices=["ollama", "api"], default=None)
    evo.add_argument("--model", default=None)
    evo.add_argument("--base-url", default=None)
    evo.add_argument("--host", default=None, help="alias for --base-url, usually the Ollama server")
    evo.add_argument("--api-key-env", default=None, help="env var that holds the API key (default POKER_LAB_API_KEY)")
    evo.add_argument("--config", default=None, help="toml settings file (default ./poker-lab.toml if present)")
    evo.add_argument("--seed", type=int, default=1)
    evo.add_argument("--out", default=None)
    evo.add_argument("--top-k", type=int, default=3)
    evo.add_argument("--last-m", type=int, default=2)
    evo.add_argument("--decision-timeout", type=float, default=0.5)
    evo.add_argument("--max-failures", type=int, default=5)
    evo.add_argument("--temperature", type=float, default=0.7)
    evo.add_argument("--timeout", type=float, default=180.0)
    evo.add_argument("--ollama-timeout", type=float, default=None, help=argparse.SUPPRESS)
    rep = sub.add_parser("report", help="write charts and a leaderboard for a saved run")
    rep.add_argument("run")
    return parser


def _add_table_args(parser: argparse.ArgumentParser) -> None:
    """Blinds and stack depth shared by simulate and evolve."""
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--stack-bb", type=int, default=200)


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Play the seven baselines and write a leaderboard."""
    if args.hands < 1:
        raise ValueError("hands must be positive")
    match = simulate(
        make_baselines(args.seed),
        hands=args.hands,
        seed=args.seed,
        sb=args.sb,
        bb=args.bb,
        stack_bb=args.stack_bb,
    )
    out = Path(args.out) if args.out else _stamp("sim")
    out.mkdir(parents=True, exist_ok=True)
    data = match.public()
    (out / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_report(out)
    print(format_match(data))
    print(f"wrote {out}")
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    """Run the Ollama loop and print where the run was saved."""
    if args.generations < 1:
        raise ValueError("generations must be positive")
    if args.hands < 1:
        raise ValueError("hands must be positive")
    out = Path(args.out) if args.out else None
    settings = resolve_settings(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        host=args.host,
        api_key_env=args.api_key_env,
        config_path=args.config,
        environ=os.environ,
    )
    timeout = args.ollama_timeout if args.ollama_timeout is not None else args.timeout
    client = build_client(settings, timeout=timeout, temperature=args.temperature)
    run = evolve(
        generations=args.generations,
        hands=args.hands,
        seed=args.seed,
        out=out,
        top_k=args.top_k,
        last_m=args.last_m,
        decision_timeout=args.decision_timeout,
        max_failures=args.max_failures,
        sb=args.sb,
        bb=args.bb,
        stack_bb=args.stack_bb,
        client=client,
    )
    print(f"wrote {run}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Regenerate markdown and charts for an existing run directory."""
    path = write_report(args.run)
    print(path.read_text(encoding="utf-8"))
    print(f"wrote {path}")
    return 0


def _stamp(prefix: str) -> Path:
    """A new ``runs/<prefix>-<utc>`` directory."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path("runs") / f"{prefix}-{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
