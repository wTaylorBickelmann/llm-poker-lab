# llm-poker-lab

A local or remote LLM writes a 6-max no-limit Texas Hold'em bot, plays it for thousands of
hands against a fixed field, and tries again with the measured results in the
prompt. Rules and showdowns come from [PokerKit](https://github.com/uoftcprg/pokerkit)
(MIT). Equity estimates come from [eval7](https://github.com/julianandrews/pyeval7)
(MIT). No GPL dependencies.

```
                         poker-lab
            simulate        |        evolve          report
               |            |           |               |
               v            |           v               v
          +---------+       |      +----------+    +-----------+
          | matcher |<------+------|   loop   |    | charts +  |
          | N hands |              | 1 call   |    | markdown  |
          +----+----+              +----+-----+    +-----------+
               |                        |
               |                  +-----+------+
               |                  | LLMClient  |
               |                  | ollama|api |
               v                  +-----+------+
          +---------+                   |
          | engine  |                   v
          | PokerKit|              +---------+
          +----+----+              | sandbox |
               |                   +---------+
       +-------+--------+
       v                v
  baseline bots    evolved bot
  random .. equity   act(state)
```

Seat map inside one hand. PokerKit posts the blinds on the first two seats,
and the matcher rotates bots through those seats so the button is not luck:

```
seat:  0   1   2    3   4    5
pos:  SB  BB  UTG  HJ  CO  BTN
```

Stacks reset to 200bb every hand (blinds 1/2, so 400 chips). A shuffle is
replayed six times with the bots moved one seat each time. If more than six
bots are in the field, a sliding window chooses who sits out, and that window
advances every deal.

## Setup

Python 3.11 or newer.

```bash
pip install -e ".[dev]"
ollama pull qwen2.5-coder
```

## Commands

```bash
# Seven baselines, duplicate deals. 10000 is rounded up to a full rotation.
poker-lab simulate --hands 10000 --seed 1

# Local Ollama. One model call per generation, then the hand count.
ollama pull qwen2.5-coder
poker-lab evolve --generations 20 --hands 5000 --model qwen2.5-coder

# Any OpenAI-compatible server: OpenAI, OpenRouter, Together, Groq, vLLM, LM Studio.
export POKER_LAB_API_KEY=sk-...
# OPENAI_API_KEY is used when POKER_LAB_API_KEY is unset.
poker-lab evolve --provider api --base-url https://api.openai.com/v1 --model gpt-4o-mini

# Rebuild the markdown leaderboard and the PNGs for a saved run.
poker-lab report runs/evolve-20261113T230000Z
```

## LLM providers

`--provider ollama` (the default) posts to `{base-url}/api/chat`. The default
base URL is `http://127.0.0.1:11434` and the default model is `qwen2.5-coder`.
`--host` is still accepted as an alias for `--base-url`.

`--provider api` posts to `{base-url}/chat/completions` with
`Authorization: Bearer ...`. The default base URL is `https://api.openai.com/v1`
and the default model is `gpt-4o-mini`. The key is read from the environment
variable named by `--api-key-env` (default `POKER_LAB_API_KEY`). If that
variable is empty, `OPENAI_API_KEY` is the fallback. The key is never read
from a config file and never written to `results.json`, `config.json`, or the
console.

The same three settings can live in the environment or in `poker-lab.toml`
in the working directory. Precedence is command line, then environment, then
the file, then the defaults above.

```bash
export POKER_LAB_PROVIDER=api
export POKER_LAB_MODEL=gpt-4o-mini
export POKER_LAB_BASE_URL=https://openrouter.ai/api/v1
export POKER_LAB_API_KEY=sk-...
```

```toml
provider = "api"
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
```

Do not put the API key in that file. A field named `api_key` is ignored.

Each evolution generation is saved as:

```
runs/<run-id>/gen-NNN/bot.py
runs/<run-id>/gen-NNN/strategy.md
runs/<run-id>/gen-NNN/results.json
```

The next prompt sees the top 3 finished strategies and the 2 most recent
attempts, including disqualifications. Code and prose are clipped so the
prompt stays small enough for a local model.

## Bot interface

Every bot implements one method. `state` is a plain dict (hole cards, board,
street, pot, to-call, stacks, positions, action history, and the legal
raise-to interval). The return value is one of:

```python
{"kind": "fold"}
{"kind": "check_call"}
{"kind": "raise", "amount": 12}  # amount is the raise-to, in chips
```

Illegal output is coerced. An unknown kind folds when a call costs chips and
checks otherwise. Folding when a check is free becomes a check. A raise
outside `[min_raise_to, max_raise_to]` is clamped. A raise that is not legal
becomes a check/call. The full key list lives in `llm_poker_lab/interface.py`
and is copied into the system prompt.

## Baselines

| Bot | Idea | Knobs |
| --- | --- | --- |
| random | fold / call / raise mix | probabilities, raise size, seed |
| station | calls, almost never folds | `max_call_bb` |
| rock | tight and passive | open score, value score, open size |
| tag | position opens, 3-bets, bets value | open scores, continue score |
| lag | wider than TAG, plus bluffs | bluff frequency, open scores, seed |
| maniac | raises almost every hand | raise probability, open size, seed |
| equity | Monte Carlo share vs pot odds | iterations, margin, raise threshold |

`equity` uses eval7 to sample one random opponent. Preflop classes are cached.
Multiway pots raise that heads-up equity to the number of opponents still in.

## Sandbox

LLM code is checked before it runs (syntax, a class with `act(self, state)`,
no `os` / `sys` / `open` / `eval`, no dunder access). It then runs in a child
process. Imports are limited to a short standard-library list. The child gets
a zero file-size rlimit, an address-space cap, and a CPU cap. Each decision
has a timeout (default 0.5s). A crash, timeout, or invalid reply counts as a
fold. After a few of those (default 5) the bot is disqualified for that
generation, the error is saved, and the next prompt is told what went wrong.

## Reports

`poker-lab report <run>` writes `leaderboard.md`, `baseline_comparison.png`,
and, for an evolve run, `bb100_by_generation.png`. bb/100 is the mean big-blind
result per hand, times 100. The band is a normal 95% interval on those per-hand
results. Duplicate deals make the hands correlated, so the band is a little
narrower than a fully independent sample.

## Measured speed

On the Linux environment used to develop this (Python 3.12, no GPU):

```text
poker-lab simulate --hands 10000 --seed 1
Hands played: 10038 (requested 10000).
Speed: 380.4 hands/sec in 26.39s.
```

Ten thousand hands of the seven baselines finished in under half a minute,
which is well under a few minutes. Showdowns are the slow part: PokerKit's
pure-Python hand comparison is what a calling station forces the table into.
Fold-heavy lineups are several times faster. The sum of every bot's chip
delta on this run is 0. Each hand starts from a fresh 200bb stack; payoffs
are the change from that stack, not a running balance.

That same run, seed 1, blinds 1/2, 200bb:

| Bot | bb/100 | 95% CI | Hands | Won | VPIP | PFR | AF |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| equity | +1331.3 | [+1069.1, +1593.5] | 8604 | 767 | 29.6 | 8.7 | 0.41 |
| tag | +705.7 | [+533.4, +878.0] | 8604 | 296 | 10.5 | 8.8 | 1.49 |
| rock | +615.2 | [+443.1, +787.3] | 8604 | 275 | 9.7 | 0.6 | 0.08 |
| lag | +313.5 | [-5.3, +632.2] | 8604 | 1087 | 58.4 | 55.8 | 4.41 |
| maniac | -649.7 | [-1130.7, -168.6] | 8604 | 3713 | 96.8 | 96.8 | 18.90 |
| station | -834.8 | [-1309.0, -360.6] | 8604 | 3543 | 99.7 | 0.0 | 0.00 |
| random | -1481.2 | [-1735.1, -1227.3] | 8604 | 636 | 54.6 | 26.9 | 0.75 |

The gaps are large because the maniac and the station pay off strong hands for
200bb. That is why the equity bot's +1331 bb/100 is not a scoring bug: it won
229095 chips over 8604 hands, and the other six bots lost exactly that many.
A normal 95% interval on those per-hand results still sits well above zero.

## Tests

```bash
pytest
```

Coverage of `llm_poker_lab` is required to be 100%. CI runs that on Python
3.11 and 3.12. The suite includes an end-to-end `poker-lab evolve` against a
fake Ollama HTTP server that returns, in order, a valid bot, a bot that
imports `os`, a bot that raises, and a bot that loops.

A live Ollama smoke test is not part of CI. Pulling even `qwen2.5-coder:0.5b`
is a large, slow download; the model's code is non-deterministic, so the job
would fail whenever it emitted a bot the sandbox rejects; and GitHub-hosted
runners have no GPU, so a generation is both slow and flaky. The fake server
covers the same control flow: a good bot is measured, and invalid, crashing,
and looping bots are disqualified and fed back into the next prompt.

## License

MIT. See `LICENSE`.
