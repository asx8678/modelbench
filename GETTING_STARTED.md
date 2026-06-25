# Getting started — running the reasoning-bench

A practical, copy-paste walkthrough: install it, launch it, point it at a model, and
read the results. For *what* it measures and *why*, see the [main README](README.md);
for every flag, see the [operator manual](files/README.md).

> **TL;DR**
> ```bash
> cd files
> pip install -r requirements.txt
> python cli.py start          # interactive: pick a model, build a set, run, report
> ```

All commands run from the **`files/`** directory.

> **`python` vs `python3`:** the examples use `python`. If your shell says
> `command not found: python`, just use `python3` instead — everything else is identical.

---

## 1. Install

Needs Python 3.9+ and two packages (`numpy`, plus `matplotlib` only for the PNG charts):

```bash
cd files
pip install -r requirements.txt
```

The runner itself is pure standard library and talks to any **OpenAI-compatible**
`/chat/completions` endpoint — Ollama, vLLM, LM Studio, llama.cpp, TGI, and hosted APIs.

---

## 2. Launch — one command

```bash
python cli.py start          # or just: python cli.py
```

`start` walks you through the whole thing interactively:

1. **Pick or add a model.** With nothing configured it drops into a 4-question setup
   wizard (endpoint, API key, model id, context window) and saves it to `providers.json`.
2. **Choose a dataset** — reuse one, or generate a fresh `quick` / `standard` / `thorough`
   set. Problems are generated on the fly, so there's nothing to download.
3. **Run + report + dashboard.** It runs the model with a live progress bar, builds the
   report, and prints the terminal dashboard as the finale.

That's the easy path. The rest of this guide shows the steps run by hand, so you can
script them or tweak the knobs.

---

## 3. Try it with no model (30-second smoke test)

Sanity-check the whole pipeline using a built-in deterministic mock that synthesizes
answers — no server, no API key:

```bash
python cli.py generate --db bench.db --max-diff 4 --reps 4 --distractor
python cli.py run      --db bench.db --mock noisy --run-id smoke --confidence --samples 3
python cli.py dashboard --db bench.db --runs smoke          # pretty terminal view
python cli.py report   --db bench.db --runs smoke --out report_smoke   # PNG + CSV + md
```

If the dashboard paints and `report_smoke/` gets written, the install is healthy and
you're ready for a real model. `--mock` accepts `perfect`, `random`, or `noisy`.

---

## 4. Run a real model

The three pieces of a run are **endpoint** + **model id** + (optional) **API key**. Pass
them directly, or register them once in `providers.json` and use a short alias.

### Local — Ollama

```bash
ollama serve            # in another terminal
ollama pull llama3.2:3b

python cli.py generate --db bench.db --reps 12 --distractor --surface-variants 3
python cli.py run --db bench.db \
    --base-url http://localhost:11434/v1 \
    --model llama3.2:3b --run-id llama32 \
    --confidence --workers 4 --temperature 0
```

### Local — vLLM

```bash
python cli.py run --db bench.db \
    --base-url http://localhost:8000/v1 \
    --model google/gemma-3-1b-it --run-id gemma31 --api-key EMPTY --confidence
```

### Hosted API

```bash
export OPENAI_API_KEY=sk-...
python cli.py run --db bench.db \
    --base-url https://api.openai.com/v1 \
    --model gpt-4o-mini --run-id gpt4omini --confidence
```

### Register once, then use an alias (recommended)

Add an entry to `files/providers.json`:

```json
{
  "providers": {
    "openai": { "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY" }
  },
  "models": {
    "gpt4omini": { "provider": "openai", "model": "gpt-4o-mini", "context_window": 128000 }
  }
}
```

```bash
python cli.py models                              # list configured aliases
python cli.py run --db bench.db --model gpt4omini --run-id gpt4omini --confidence
```

`python cli.py setup` is the same registration wizard `start` uses, on its own.

Useful `run` knobs: `--samples N` (enables `pass@k` / `maj@k` / self-consistency),
`--temperature 0` (cleanest single-shot signal), `--confidence` (enables calibration),
`--workers N` (concurrency — keep modest for local servers), `--resume` (skip finished
items). Re-running the same `--run-id` overwrites its rows, so it never double-counts.

---

## 5. See your results

### Terminal dashboard (no image viewer needed)

```bash
python cli.py dashboard --db bench.db --runs llama32
```

```text
╭──────────────────────────────────────────────────────────────────────────╮
│ reasoning-bench · smoke                                             mock   │
│ 320 items · 3 samples/item                                                 │
╰──────────────────────────────────────────────────────────────────────────╯

 HEADLINE ──────────────────────────────────────────────────────────────────
   accuracy          ███████████▉░░░░░░ 0.662  ✓  single-shot, base items
   coverage          ██████████████████ 1.000  ✓  320/320 answered
   calibration ECE   ██▍░░░░░░░░░░░░░░░ 0.134  ~  lower is better
   self-consistency  pass@1 0.662 → maj@3 0.767 → oracle 0.954  (+0.292 headroom)

 ACCURACY BY FAMILY ─────────────────────────────────────────────────────────
   arithmetic        ██████████████▋░░░ 0.81 ✓  █▇▇▇   reasoning steps
   knights_knaves    █████████████▌░░░░ 0.75 ✓  ▇▇█▅   islanders (n) · 0.71 above chance
   …
```

Each bar shows the number **and** a ✓ / ~ / ✗ mark; the per-family sparkline is accuracy
from easy to hard difficulty — **flat-and-high is the good signal; a cliff is bad**. It
auto-detects color/width; `--no-color` and `--width N` override, and it prints clean
ASCII when piped to a file.

### Full report (charts + spreadsheet)

```bash
python cli.py report --db bench.db --runs llama32 --out report
# -> report/report.md, degradation.png, distractibility.png, …, metrics.csv
```

`report` re-derives every metric from the stored raw responses, so you can re-run it (or
edit `metrics.py`) without spending tokens again.

---

## 6. Compare models

Run two models over the **same** dataset, then pass both run-ids:

```bash
python cli.py dashboard --db bench.db --runs llama32 gemma31   # side-by-side leaderboard
python cli.py report    --db bench.db --runs llama32 gemma31 --out report
```

The dashboard's compare view scales each row's meters so the **best run has the longest
bar** — even on "lower is better" rows like ECE. Comparing on identical generated items
means any gap is reasoning, not a contamination/memorization artifact.

---

## 7. Command cheat-sheet

| Command | What it does |
|---|---|
| `python cli.py start` | Interactive launcher: pick/add a model → dataset → run → report → dashboard. |
| `python cli.py setup` | Register a model + endpoint (wizard), writes `providers.json`. |
| `python cli.py generate` | Build a procedurally-generated problem set into a SQLite DB. |
| `python cli.py run` | Run a model (or `--mock`) over the dataset; stores graded responses. |
| `python cli.py dashboard` | Rich in-terminal stats / multi-run comparison. |
| `python cli.py report` | Metrics + accessible PNG charts + `metrics.csv` + `report.md`. |
| `python cli.py list` | List runs in a DB. |
| `python cli.py models` / `providers` | List configured aliases / endpoints. |
| `python cli.py families` | List problem families and their probe support. |

Add `-h` to any command for its full options.

---

## 8. Notes & troubleshooting

- **`command not found: python`** → use `python3` (and `pip3`).
- **A real model scores 0 everywhere** → it's probably ignoring the `ANSWER:` format.
  Spot-check raw responses: `sqlite3 bench.db "SELECT raw FROM responses LIMIT 5"`. The
  `grading_fragility` metric quantifies how much the score leans on fallback parsing.
- **No charts** → `report` needs matplotlib (`pip install matplotlib`); the **dashboard
  needs nothing extra** and still works.
- **Run interrupted** → re-run with `--resume` to skip finished items (errored items are
  retried). Same `--run-id` overwrites rather than duplicating.
- **Sizing** → `--reps 20 --distractor --surface-variants 3` over difficulties 1–6 and
  all families is ~5,000 items (≈5,000 calls at `--samples 1`). Start small.

---

## Where to go deeper

- **[README.md](README.md)** — what each signal measures, the problem families, and the
  design rationale (generated-not-scraped, gold-by-construction, etc.).
- **[files/README.md](files/README.md)** — the in-depth operator manual: every flag,
  metric definitions, accessibility notes, and how to add your own problem family.
- **Run the tests** (no model/network needed): `cd files && pip install pytest && pytest -q`.
