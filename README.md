# bench — reasoning-bench

A small, **contamination-proof** harness for measuring how well an LLM *reasons* —
not just whether it scores high on a static, memorizable test.

Every problem is generated on the fly from parameterized templates, so it cannot
exist in any model's training data. Gold answers are computed *while the problem is
built* and then **independently re-derived** from the prompt text before the item is
ever stored. Instead of collapsing everything into one accuracy number, the harness
reports the signals that actually separate reasoning from pattern-matching:
degradation under depth, variance across equivalent items, distractibility,
surface invariance, calibration, self-consistency, and confabulation on ill-posed
problems.

> The benchmark code lives in [`files/`](files/). This top-level README is the
> orientation map; [`files/README.md`](files/README.md) is the in-depth operator
> manual for the same tool.

> **Just want to run it?** → **[GETTING_STARTED.md](GETTING_STARTED.md)** is a
> copy-paste walkthrough: install, launch, point it at a model, and read the results.
> The 30-second version: `cd files && uv sync && uv run python cli.py start`.

---

## Table of contents

- [What it measures](#what-it-measures)
- [Problem families](#problem-families)
- [Why these design choices](#why-these-design-choices)
- [Install](#install)
- [Easiest start (one command)](#easiest-start-one-command)
- [Quickstart](#quickstart-no-model-needed)
- [Running real models](#running-real-models)
- [Command reference](#command-reference)
- [Reading the output](#reading-the-output)
- [How it works (architecture)](#how-it-works-architecture)
- [Repository layout](#repository-layout)
- [Extending it](#extending-it)
- [Testing](#testing)
- [Limits — read these](#limits--read-these)
- [Project workflow (beads)](#project-workflow-beads)

---

## What it measures

| Signal | What it asks | Why it matters |
|---|---|---|
| **Degradation curve** | Does accuracy hold as the difficulty axis rises? | Real reasoning degrades *gracefully*; pattern-matching falls off a cliff. The axis is **per-family and not commensurable**: for most families it is the number of reasoning steps, but `sequences` scales a *rule tier* (AP→cubic) and the CSP families scale *problem size n*. Read each curve against itself across difficulty, not across families. |
| **Variance** | How much does accuracy wobble across structurally-identical items? | High variance on equivalent items = brittle. |
| **Distractibility** | Does an irrelevant clause break it? | The GSM-NoOp probe. A big paired drop means it isn't tracking relevance. |
| **Surface invariance** | Same computation, different names/items/phrasing — does the answer flip? | Answer flips mean it's keying on surface, not structure. **Scope:** the probe varies *cosmetic* surface (names, items, and lexical phrasing such as verb choice) while holding the gold fixed. It does **not** vary numeric magnitude — that would change the gold, so magnitude robustness can't be measured by a same-gold flip metric (read it from the degradation/variance signals instead). |
| **Calibration (ECE)** | Does stated confidence track correctness? | A smart system is uncertain when it should be. |
| **pass@1 / maj@k / pass@k** | Single-shot vs majority-vote vs best-of-k | `maj@k` (self-consistency) is a deployable strategy; `pass@k` is an oracle upper bound. All three are reported. |
| **Chance-corrected accuracy** | Accuracy normalized against the random-guess baseline per family | A family with many possible answers looks harder than it is; this exposes true headroom above guessing. |
| **Confabulation rate** | On ill-posed items (`UNDETERMINED`, `NO_SOLUTION`), does it answer with a concrete value anyway? | Inventing answers for contradictory/under-constrained problems is worse than saying "unknown". |
| **Grading fragility** | How often does the `ANSWER:` marker parse disagree with the fallback parse? | High fragility means the score depends on format compliance, not reasoning. |
| **Behavioral uncertainty** | Disagreement entropy + self-consistency gap across samples | Low stated confidence but high sample disagreement flags uncertainty the model doesn't report. |

There is **no single intelligence score** by design — the output is a *profile*. Two
models with the same overall accuracy can have very different degradation and
robustness fingerprints.

## Problem families

Eleven procedurally-generated families, grouped by the kind of reasoning load they
apply:

| Family | What it probes | Difficulty axis |
|---|---|---|
| `arithmetic` | multi-hop quantitative word problems (`+ − × ÷`, exact) | number of operations |
| `state_tracking` | item counts across containers through updates | number of updates |
| `ordering` | transitive comparison | number of entities |
| `sequences` | next-term rule induction (AP, GP, quadratic, fibonacci, interleaved, cubic) | rule-complexity tier 1..6 |
| `retroactive_edit` | single-turn dynamic constraint: a late clause rewrites an earlier value | updates / edit distance |
| `multi_turn_inject` | multi-turn state tracking: turn 1 sets state, turn 2 injects a new rule | number of state facts |
| `knights_knaves` | truth-teller / liar **deduction** under self-reference | islanders (diff+2, ≤8) |
| `logic_grid` | **constraint satisfaction** — place N people on N floors | floors (diff+2, ≤7) |
| `unsat_csp` | premise-flaw detection: controlled ill-posed knights-and-knaves puzzles | n (diff+2); brute-forced solution count |
| `unsat_localize` | **justification localization**: an unsatisfiable knights-and-knaves puzzle where you must name the single statement whose removal restores a unique solution; graded by an entailment check (`justified_choice`), so the prior-guessable sentinel of `unsat_csp` no longer wins | n (diff+2) |
| `composed` | 5 execution-dependency hops (knights → arithmetic → ordering → two parity-gated op-selections) where an early slip flips a parity and selects a **different operator** downstream, not just a shifted magnitude | chain length / per-hop difficulty |
| `redefined_ops` | arithmetic with counterfactually redefined operators; at difficulty ≥3 the same symbol changes meaning by **position** in the chain, or one operator is defined **compositionally** in terms of another | number of operations |
| `dynamic_pivot` | **genuine backtracking**: turn 1 commits a count, turn 2 reveals the moves never happened, forcing a revision | number of updates before the pivot |
| `false_lemma` | **premise-flaw trap**: a plausible-but-false "the total is conserved" note must be rejected, not trusted | number of updates |
| `noise_haystack` | **high-similarity needle**: a real arithmetic chain buried under structurally-identical decoy chains about other people | number of operations / decoys |

- **Numeric robustness probes** (`arithmetic`, `state_tracking`, `ordering`,
  `sequences`) measure compositional and perturbation-resilient reasoning — not just
  final accuracy.
- **CSP / deductive families** (`knights_knaves`, `logic_grid`, `unsat_csp`) add load
  that scales steeply (8 islanders = 2⁸ assignments; 7 floors = 7! arrangements) and
  is where strong reasoners separate from pattern-matchers.
- **Dynamic / counterfactual families** (`retroactive_edit`, `multi_turn_inject`,
  `composed`, `redefined_ops`) test late edits, multi-turn injection, multi-hop
  execution, and reasoning under redefined operators.

**Perturbation support** (from `generators.py`):

| Perturbation | Families that support it |
|---|---|
| Distractor (matched NoOp clause) | `arithmetic`, `state_tracking`, `ordering`, `retroactive_edit`, `redefined_ops` |
| Surface rename (gold held fixed) | `arithmetic`, `state_tracking`, `ordering`, `retroactive_edit`, `knights_knaves`, `logic_grid`, `redefined_ops`, `unsat_csp` |

Run `python files/cli.py families` to print the live support matrix.

## Why these design choices

- **Generated, not scraped.** Because items are synthesized per run, any gap between
  two models *cannot* be a contamination/memorization artifact — it's reasoning.
- **Gold by construction + independent re-verification.** Each family ships a
  `gen_*` builder that computes the answer in code, and a matching `_verify_*` that
  re-derives the answer from the *prompt text alone*. `build_dataset` runs that check
  on every item, so a prompt/gold mismatch fails at generation time rather than
  silently mis-scoring a run.
- **Three independent random streams per item** — *structure* (the logic + gold),
  *surface* (names/items/clause order), and *distractor* (the irrelevant clause).
  Splitting them lets the harness hold the computation fixed while varying surface,
  which is exactly the invariance probe.
- **Raw responses stored verbatim.** Everything is recomputed from the responses
  table, so you can re-grade or add metrics by editing `metrics.py` and re-running
  `report` — without spending tokens again.

---

## Install

This project uses [uv](https://docs.astral.sh/uv/). One command creates the environment
and installs the dependencies (`numpy`, `matplotlib`, `pytest`):

```bash
cd files
uv sync
```

Run the commands below with the `uv run` prefix (e.g. `uv run python cli.py start`), or
activate the environment once with `source .venv/bin/activate` and drop the prefix — the
examples that follow use the bare `python cli.py …` form. (No uv yet?
`curl -LsSf https://astral.sh/uv/install.sh | sh`.)

The runner itself uses only the Python standard library and talks to any
**OpenAI-compatible** `/chat/completions` endpoint — covering Ollama, vLLM, LM
Studio, llama.cpp, TGI, and hosted APIs. An optional native Anthropic Messages
streaming path activates when a provider is tagged with the `native_anthropic`
capability (captures per-phase wall-clock timings).

## Easiest start (one command)

New here? Run **one** command and follow the prompts. It asks whether to use a model
you've already configured or add a new one, builds a problem set, runs the benchmark
with a live progress bar, and writes the report — start to finish:

```bash
cd files
uv sync
uv run python cli.py start          # or just: uv run python cli.py
```

`start` walks you through everything:

1. **Pick or add a model.** With nothing configured yet it drops straight into the
   setup wizard, which asks four questions — **provider endpoint, API key, model ID,
   context window** — and saves them to `providers.json` (the API key defaults to an
   environment-variable reference, so no secret lands in the file; leave it blank for
   keyless local servers like Ollama). Already have models? It lists them and you pick
   one by number, or choose to add another.
2. **Choose a dataset.** Reuse the one already in your DB, or generate a fresh
   `quick` / `standard` / `thorough` set — problems are generated on the fly, nothing
   to download.
3. **Run + report.** It runs the model with a **live progress bar** — percent
   complete, items done, OK / error counts, throughput, and ETA — then builds the
   report:

```text
  [████████████░░░░░░░░░░░░░░░░]  43.5%  217/499  ok=214 err=3  6.1 it/s  ETA 0:46
```

Want to add a model without running anything? `python cli.py setup` is the wizard on
its own. Prefer to drive each step yourself? They're all normal commands
(`generate`, `run`, `report`) — see the [Command reference](#command-reference). Or
sanity-check the whole pipeline with no model at all via the
[Quickstart](#quickstart-no-model-needed) below.

## Quickstart (no model needed)

Sanity-check the whole pipeline with a deterministic mock that synthesizes answers:

```bash
cd files
python cli.py generate --db bench.db --max-diff 6 --reps 12 --distractor --surface-variants 3
python cli.py run      --db bench.db --mock noisy --run-id smoke --confidence --samples 3
python cli.py report   --db bench.db --runs smoke --out report_smoke
```

If that prints a metrics table and writes `report_smoke/`, the pipeline is healthy
and ready for real models.

## Running real models

### Register a provider + model once (recommended)

Endpoints and models live in [`files/providers.json`](files/providers.json) (point
`$BENCH_PROVIDERS` elsewhere to override). Add an entry once and refer to it by a
short **alias** instead of repeating `--base-url` / `--model` / `--api-key`:

```json
{
  "providers": {
    "openai": { "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY" }
  },
  "models": {
    "gpt4omini": { "provider": "openai", "model": "gpt-4o-mini", "context_window": 128000, "max_tokens": 4096 }
  }
}
```

```bash
python cli.py providers   # list configured endpoints
python cli.py models      # list aliases (provider, model id, context window)
python cli.py run --db bench.db --model gpt4omini --run-id gpt4omini --confidence
```

- `api_key_env` reads the key from an environment variable; `api_key` is a literal
  fallback (handy for local servers that want a dummy key, e.g. vLLM's `EMPTY`).
- `context_window` is recorded with the run; `max_tokens`, if set, becomes that
  model's default completion budget (the runner warns if `--max-tokens` ≥ context).
- `--provider` / `--base-url` / `--api-key` / `--max-tokens` always override the
  registry; a raw model id with no provider falls back to a local Ollama endpoint —
  so the explicit forms below work with no config file at all.

### Local (Ollama)

```bash
ollama serve            # in another terminal
ollama pull llama3.2:3b

python cli.py run --db bench.db \
    --base-url http://localhost:11434/v1 \
    --model llama3.2:3b --run-id llama32 \
    --confidence --workers 4 --temperature 0
```

### Local (vLLM)

```bash
python cli.py run --db bench.db \
    --base-url http://localhost:8000/v1 \
    --model google/gemma-3-1b-it --run-id gemma31 \
    --api-key EMPTY --confidence
```

### Hosted API

```bash
export OPENAI_API_KEY=sk-...
python cli.py run --db bench.db \
    --base-url https://api.openai.com/v1 \
    --model gpt-4o-mini --run-id gpt4omini --confidence
```

### Compare models in one report

```bash
python cli.py report --db bench.db --runs llama32 gemma31 --out report
# -> report/report.md, degradation.png, distractibility.png, metrics.csv
```

To isolate **scale vs alignment-recipe** effects: generate one dataset, run both
models on the *identical* items, and compare degradation curves.

---

## Command reference

`python files/cli.py <command>` — run `<command> -h` for full options.

| Command | Purpose |
|---|---|
| `start` | **One-command launcher** (also the default for bare `cli.py`) — pick or add a model, choose/reuse a dataset, then run + report, all interactively. |
| `setup` | **Interactive wizard** — register a provider + model (endpoint, API key, model id, context window) into `providers.json` and print the next steps. |
| `generate` | Build a procedurally-generated problem set into a SQLite DB. |
| `run` | Run a model (or mock) over the dataset and store graded responses. |
| `report` | Compute metrics + accessible charts for one or more runs. |
| `dashboard` | Render a rich **in-terminal** stats dashboard for a run (or compare several side by side) — no image viewer needed. |
| `list` | List runs in a DB. |
| `families` | List families and their distractor/surface support. |
| `providers` | List configured providers from `providers.json`. |
| `models` | List configured model aliases. |

### `generate` knobs

| Flag | Meaning |
|---|---|
| `--families a b c` | Subset of the eleven families, or `all` (default). |
| `--min-diff` / `--max-diff` | Difficulty range = number of reasoning steps (default `1..6`). Families that select a discrete structure are capped via `FAMILY_MAX_DIFF` so a higher `--max-diff` can't emit "difficulties" that aren't actually harder (or feasible to gold-verify). |
| `--reps N` | Distinct structures per (family, difficulty). Higher N = tighter variance estimates; 12–25 is a good range. |
| `--distractor` | Also emit a matched NoOp-distractor copy of every eligible item. |
| `--surface-variants N` | Cosmetic (rename) variants per item for the invariance probe. |

A run with `--reps 20 --distractor --surface-variants 3` over difficulties 1–6 and
all eleven families is ~5,000 items (≈5,000 model calls at `--samples 1`).

### `run` knobs

| Flag | Meaning |
|---|---|
| `--model` / `--provider` / `--base-url` / `--api-key` | A model alias, or a raw model id plus a provider alias / explicit endpoint. |
| `--samples N` | Completions per item, drawn one request at a time (so `pass@k` works even on servers that ignore the OpenAI `n` param). `>1` enables `pass@k`, `maj@k`, and self-consistency. |
| `--temperature` | Use `0` for the cleanest single-shot signal; raise it only when measuring `pass@k` / `maj@k`. |
| `--confidence` | Ask the model for `CONFIDENCE: 0-100` → enables calibration/ECE. |
| `--workers` | Concurrent requests. Keep modest for local servers. |
| `--resume` | Skip items already scored OK; items that only errored are retried. |
| `--mock {perfect,random,noisy}` | Synthesize answers without a server (deterministic per item+sample). |
| `--dataset-tag` | Tag the run's dataset version; `report` warns if you compare runs with mismatched tags that share families. |

Re-running the same `--run-id` overwrites its rows (keyed on item+sample), so it
never double-counts — `--resume` only *skips* finished work.

## Reading the output

- **Flat degradation curve** across difficulties is the strongest positive signal. A
  cliff at d3–d4 means the model handles short chains by recall but can't actually
  compose steps. Error bars are **Wilson 95% intervals** — shrink them by collecting
  more structures (`--reps`), not by reading the raw per-cell std.
- **Distractor drop near zero** is good. The comparison is **paired** (same structure
  with and without the clause); the summary shows how many items the clause *hurt* vs
  *helped*. A large drop is the original GSM-NoOp finding, though newer models
  sometimes treat the clause as a hint — read it alongside the curve.
- **Answer-flip rate near zero** means the model is keying on structure, not names.
- **Low ECE** means confidence is trustworthy. Confidently-wrong on hard items is the
  pattern-matching signature.
- **Confabulation rate** on `unsat_csp` is the meaningful metric there: concrete
  answers to `UNDETERMINED` / `NO_SOLUTION` items.

### Terminal dashboard

Don't want to open PNGs? `dashboard` renders the whole profile **in the terminal** —
colored value meters, an inline degradation sparkline per family (so a cliff jumps
out at a glance), the calibration diagram, distractibility, and runtime/token cost:

```bash
python cli.py dashboard --db bench.db --runs gpt4omini          # one run, full dashboard
python cli.py dashboard --db bench.db --runs llama32 gemma31    # many runs, leaderboard
```

```text
╭──────────────────────────────────────────────────────────────────────────╮
│ reasoning-bench · gpt4omini                                    gpt-4o-mini │
│ 1,512 items · 3 samples/item · 2026-06-25 14:53                            │
╰──────────────────────────────────────────────────────────────────────────╯

 HEADLINE ──────────────────────────────────────────────────────────────────
   accuracy          ███████████▌░░░░░░ 0.637  ~  single-shot, base items
   coverage          ██████████████████ 1.000  ✓  1512/1512 answered
   calibration ECE   ██▎░░░░░░░░░░░░░░░ 0.125  ✓  lower is better
   answer-flip rate  █████████████▌░░░░ 0.753  ✗  lower is better
   self-consistency  pass@1 0.637 → maj@3 0.704 → oracle 0.931  (+0.294 headroom)

 ACCURACY BY FAMILY ────────────────────────────────────────────────────────
   arithmetic        ████████████▏░░░░░ 0.68 ✓  █▇▇▇▃  reasoning steps
   knights_knaves    ████████████▋░░░░░ 0.70 ✓  ██▇▅▅  islanders (n) · 0.67 above chance
   state_tracking    ██████████████▍░░░ 0.80 ✓  ███▆▇  reasoning steps
   …
```

It's **dependency-free** (standard library only — no matplotlib) and capability-aware:
color and box-drawing on a real terminal, clean plain ASCII when piped to a file, and
it honors `NO_COLOR`. `start` shows it automatically as the final summary. Use
`--no-color` / `--width` to override detection. Every bar still carries its number and
a ✓ / ~ / ✗ mark, so meaning never rests on color alone.

### Accessibility

Charts use the Okabe-Ito colorblind-safe palette with **redundant encoding** — every
model gets a distinct color *and* marker shape *and* line style, so plots stay
readable in greyscale and under color-vision deficiency. `metrics.csv` is provided
for screen-reader / spreadsheet use instead of relying on the images. The terminal
`dashboard` follows the same rule (number + glyph + color on every bar).

---

## How it works (architecture)

```
generate ─▶ SQLite (dataset) ─▶ run ─▶ SQLite (responses, verbatim) ─▶ report
   │                              │                                       │
   ▼                              ▼                                       ▼
gold computed in code      OpenAI-compatible client            metrics recomputed
+ independently            (+ deterministic mock,              from raw responses
re-verified per item       + optional native Anthropic)        → md / png / csv
```

1. **Generate** — `generators.build_dataset` instantiates each `gen_*` family across
   the requested difficulties/reps, computes gold by construction, re-derives it via
   the matching `_verify_*` (`verify_gold`), and stores items in SQLite.
2. **Run** — `runner.run` fans out requests to an OpenAI-compatible endpoint with a
   `ThreadPoolExecutor`, grades each completion with `grading.grade`
   (`ANSWER:`-marker parse with conservative fallbacks), and stores the **raw
   response verbatim** alongside the parse + per-call telemetry.
3. **Report** — `metrics.compute` rebuilds every signal from the responses table;
   `report.build_report` writes Markdown, accessible PNG charts, and `metrics.csv`.

Because step 3 reads only stored rows, you can iterate on metrics and charts without
re-spending tokens. The terminal `dashboard` is a fourth consumer of the same
responses table — it renders the metrics profile in-place (no matplotlib), so you can
eyeball a run the moment it finishes.

## Repository layout

```
bench/
├── README.md                 ← you are here (repo orientation)
├── CLAUDE.md / AGENTS.md      project + agent workflow instructions
├── .beads/                    beads (bd) issue tracker data, synced via Dolt
└── files/                     the reasoning-bench tool
    ├── cli.py                 start / setup / generate / run / report / list / families / providers / models
    ├── generators.py          11 problem families + dataset builder + independent gold verifiers
    ├── grading.py             ANSWER:/CONFIDENCE: extraction + exact-match scoring + fragility
    ├── storage.py             SQLite layer (dataset, runs, responses, telemetry)
    ├── runner.py              OpenAI-compatible client + orchestration (+ deterministic mock)
    ├── metrics.py             all metrics computed from raw responses (+ runtime/token stats)
    ├── report.py              accessible charts + CSV + Markdown
    ├── dashboard.py           dependency-free terminal dashboard + run comparison
    ├── providers.py           provider/model registry loader + resolver
    ├── providers.json         the endpoints + model aliases you edit
    ├── pyproject.toml         deps (numpy, matplotlib) + dev group (pytest), uv-managed
    ├── uv.lock                pinned, reproducible dependency lockfile
    ├── README.md              in-depth operator manual
    ├── test_*.py              pytest suite (no model or network needed)
    └── fixtures/ patches/     test fixtures and patch artifacts
```

## Extending it

Add a family in `files/generators.py`:

1. Write `gen_yourfamily(difficulty, structure_seed, surface_seed, distractor)`
   returning `(prompt, gold, answer_type, choices)` — or with a trailing `turns`
   element for multi-turn families. **Compute the gold in code.**
2. Add a matching `_verify_yourfamily(prompt, gold)` in `_VERIFIERS` so `verify_gold`
   can re-derive the answer from the prompt text independently.
3. Register it in `GENERATORS`, add it to `SUPPORTS_DISTRACTOR` / `SUPPORTS_SURFACE`
   if it can hold its gold fixed under those perturbations, and set `FAMILY_MAX_DIFF`
   if it uses a discrete tier or brute-forced structure.

Everything else (storage, running, metrics, charts) picks it up automatically.

## Testing

```bash
cd files
uv run pytest -q   # generators, CSP uniqueness, grading, error/dup handling, metrics, registry
```

`pytest` is in the `dev` dependency group, so `uv sync` already installed it.

The suite drives the full pipeline through the deterministic mock — no model,
network, or matplotlib required. Every generated gold is independently re-derived
from the prompt text, and `build_dataset` runs that check on every item, so a
prompt/gold mismatch fails fast at generation time.

## Limits — read these

- **Sequences are the noisiest family.** "Next term" can admit more than one rule;
  the generators use enough terms to make the intended rule the simplest fit, but
  treat this family as a soft signal and compare it *within-family* (its axis is a
  rule-complexity tier, not a step count).
- **Grading is exact-match** on a parsed `ANSWER:` line with conservative fallbacks.
  A model that refuses the format will look wrong — spot-check raw responses
  (`SELECT raw FROM responses LIMIT 5`) the first time you run a new model.
  `grading_fragility` quantifies how much the score depends on the fallback path.
- **CSP difficulty is bounded by gold-verification cost, not the model.** Uniqueness
  (or controlled non-uniqueness for `unsat_csp`) is proven by brute force at build
  time, so the `FAMILY_MAX_DIFF` caps keep generation honest. Harder puzzles require
  raising the cap *and* the verifier's (exponential/factorial) search alongside it.

---

## Project workflow (beads)

This repository tracks work with **bd (beads)**, backed by Dolt, not markdown TODOs.

```bash
bd ready                # find available work
bd show <id>            # view issue details
bd update <id> --claim  # claim work
bd close <id>           # complete work
bd dolt push            # push tracker data to the Dolt remote
```

Run `bd prime` for the full command reference and session-close protocol. See
[`CLAUDE.md`](CLAUDE.md) and [`AGENTS.md`](AGENTS.md) for the agent workflow rules.
