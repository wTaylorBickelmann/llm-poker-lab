"""Markdown leaderboards and PNG charts for a simulate or evolve run.

``poker-lab report`` rebuilds both from the JSON already on disk. The
generation chart shows the evolved bot's bb/100 with a 95% band. The bar
chart compares whoever sat in the latest useful match.

::

    results.json / gen-*/results.json ---> leaderboard.md
                                        ---> baseline_comparison.png
                                        ---> bb100_by_generation.png
"""

import json
from pathlib import Path

_READY = False


def write_report(run_dir: Path | str) -> Path:
    """Write ``leaderboard.md`` and charts into ``run_dir`` and return the markdown path."""
    directory = Path(run_dir)
    generations = _load_generations(directory)
    if generations:
        text = format_evolve(directory, generations)
        plot_generations(generations, directory / "bb100_by_generation.png")
        plot_bars(_field(generations), directory / "baseline_comparison.png", "Best generation versus the field")
    elif (directory / "results.json").is_file():
        data = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        text = format_match(data)
        plot_bars(data.get("bots") or [], directory / "baseline_comparison.png", "Baseline field")
    else:
        raise FileNotFoundError(f"no poker-lab results in {directory}")
    path = directory / "leaderboard.md"
    path.write_text(text, encoding="utf-8")
    return path


def format_match(data: dict) -> str:
    """Leaderboard for a single baseline (or mixed) match."""
    lines = [
        "# Baseline leaderboard",
        "",
        f"Hands played: {data['hands_played']} (requested {data['hands_requested']}).",
        f"Speed: {data['hands_per_sec']:.1f} hands/sec in {data['elapsed_sec']:.2f}s.",
        (
            f"Seed {data['seed']}, blinds {data['small_blind']}/{data['big_blind']}, "
            f"stacks {data['stack_bb']}bb."
        ),
        "",
        "bb/100 uses a normal 95% interval on per-hand results. "
        "Duplicate dealing reduces variance, so the band is slightly narrow.",
        "",
        _table(data.get("bots") or []),
        "",
    ]
    return "\n".join(lines)


def format_evolve(directory: Path, generations: list[dict]) -> str:
    """Generation table plus the field from the best finished generation."""
    lines = [
        "# Evolution leaderboard",
        "",
        f"Run: `{directory}`",
        "",
        "## Evolved bot by generation",
        "",
        _generation_table(generations),
        "",
        "## Best generation versus the field",
        "",
    ]
    best = _best(generations)
    if best is None:
        lines.append("No generation finished a match.")
    else:
        lines.append(f"Generation {best['generation']} is the best finished evolved bot.")
        lines.append("")
        lines.append(_table(best.get("bots") or []))
    lines.append("")
    return "\n".join(lines)


def format_af(row: dict) -> str:
    """Aggression factor, ``inf`` when the bot never called, ``n/a`` when it never bet."""
    factor = row.get("aggression_factor")
    if factor is None:
        if row.get("aggressive_actions"):
            return "inf"
        return "n/a"
    return f"{factor:.2f}"


def format_bb(row: dict) -> str:
    """Signed bb/100, or DQ when the bot was disqualified."""
    if row.get("disqualified"):
        return "DQ"
    return f"{row['bb_per_100']:+.1f}"


def format_ci(row: dict) -> str:
    """The 95% band, or a dash when there is nothing to estimate."""
    if row.get("disqualified") or not row.get("hands"):
        return "—"
    return f"[{row['ci_low']:+.1f}, {row['ci_high']:+.1f}]"


def plot_generations(generations: list[dict], path: Path) -> None:
    """Line chart of evolved bb/100 with the confidence band filled in."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs, ys, lows, highs = [], [], [], []
    for row in generations:
        evolved = _evolved_row(row)
        if row.get("disqualified") or evolved is None or evolved.get("bb_per_100") is None:
            continue
        xs.append(row["generation"])
        ys.append(evolved["bb_per_100"])
        lows.append(evolved["ci_low"])
        highs.append(evolved["ci_high"])
    if xs:
        ax.plot(xs, ys, marker="o", color="#1f4e79", label="evolved bb/100")
        ax.fill_between(xs, lows, highs, color="#1f4e79", alpha=0.2, label="95% CI")
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.legend()
        ax.set_xlabel("generation")
        ax.set_ylabel("bb/100")
    else:
        ax.text(0.5, 0.5, "No completed generations", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Evolved bot by generation")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bars(rows: list[dict], path: Path, title: str) -> None:
    """Horizontal bb/100 bars with the confidence interval as the error bar."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    usable = [row for row in rows if not row.get("disqualified") and row.get("bb_per_100") is not None]
    usable.sort(key=lambda row: row["bb_per_100"])
    if usable:
        labels = [row["name"] for row in usable]
        values = [row["bb_per_100"] for row in usable]
        errors = [row["bb_per_100"] - row["ci_low"] for row in usable]
        ax.barh(labels, values, xerr=errors, color="#3d7ea6")
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set_xlabel("bb/100")
    else:
        ax.text(0.5, 0.5, "No bots to compare", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _table(rows: list[dict]) -> str:
    """Markdown table sorted by bb/100, disqualified bots last."""
    ordered = sorted(rows, key=_sort_key)
    header = "| Bot | bb/100 | 95% CI | Hands | Won | VPIP | PFR | AF |"
    sep = "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"
    body = [
        "| {name} | {bb} | {ci} | {hands} | {won} | {vpip:.1f} | {pfr:.1f} | {af} |".format(
            name=row["name"],
            bb=format_bb(row),
            ci=format_ci(row),
            hands=row.get("hands", 0),
            won=row.get("hands_won", 0),
            vpip=row.get("vpip") or 0.0,
            pfr=row.get("pfr") or 0.0,
            af=format_af(row),
        )
        for row in ordered
    ]
    if not body:
        body = ["| — | — | — | — | — | — | — | — |"]
    return "\n".join([header, sep, *body])


def _generation_table(generations: list[dict]) -> str:
    """One row per generation for the evolved bot."""
    header = "| Gen | bb/100 | 95% CI | Hands | VPIP | PFR | AF | Note |"
    sep = "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for row in generations:
        evolved = _evolved_row(row) or {
            "name": "evolved",
            "bb_per_100": 0.0,
            "disqualified": row.get("disqualified"),
            "hands": 0,
            "vpip": 0.0,
            "pfr": 0.0,
            "aggression_factor": None,
            "aggressive_actions": 0,
        }
        note = (row.get("error") or "").replace("|", "/").replace("\n", " ")
        lines.append(
            "| {gen} | {bb} | {ci} | {hands} | {vpip:.1f} | {pfr:.1f} | {af} | {note} |".format(
                gen=row["generation"],
                bb=format_bb(evolved) if not row.get("disqualified") else "DQ",
                ci=format_ci(evolved) if not row.get("disqualified") else "—",
                hands=evolved.get("hands", 0),
                vpip=evolved.get("vpip") or 0.0,
                pfr=evolved.get("pfr") or 0.0,
                af=format_af(evolved),
                note=note.replace("|", "/"),
            )
        )
    return "\n".join(lines)


def _sort_key(row: dict) -> tuple:
    """Disqualified rows sink. Everyone else sorts by bb/100 descending."""
    if row.get("disqualified"):
        return (1, 0.0)
    return (0, -float(row.get("bb_per_100") or 0.0))


def _load_generations(directory: Path) -> list[dict]:
    """Read ``gen-NNN/results.json`` files in order."""
    rows = []
    for path in sorted(directory.glob("gen-*")):
        results = path / "results.json"
        if results.is_file():
            rows.append(json.loads(results.read_text(encoding="utf-8")))
    return rows


def _best(generations: list[dict]) -> dict | None:
    """The finished generation with the highest evolved bb/100."""
    finished = []
    for row in generations:
        evolved = _evolved_row(row)
        if row.get("disqualified") or evolved is None or evolved.get("bb_per_100") is None:
            continue
        finished.append(row)
    if not finished:
        return None
    return max(finished, key=lambda row: _evolved_row(row)["bb_per_100"])


def _field(generations: list[dict]) -> list[dict]:
    """Bots from the best generation, or an empty list when all failed."""
    best = _best(generations)
    if best is None:
        return []
    return list(best.get("bots") or [])


def _evolved_row(results: dict) -> dict | None:
    """The summary named ``evolved`` inside a generation result."""
    for row in results.get("bots") or []:
        if row.get("name") == "evolved":
            return row
    return None


def _pyplot():
    """Import pyplot after selecting the headless backend, once."""
    global _READY
    import matplotlib

    if not _READY:
        matplotlib.use("Agg")
        _READY = True
    import matplotlib.pyplot as plt

    return plt
