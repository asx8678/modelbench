# reasoning-bench

A small, **contamination-proof** harness for measuring how well an LLM *reasons* —
not just whether it scores high on a static test.

Every problem is generated on the fly from parameterized templates, so it cannot be
in any model's training data. Gold answers are computed while the problem is built
(and independently re-verified). Instead of one accuracy number, you get the things
that actually distinguish reasoning from pattern-matching:

| Signal | What it asks | Why it matters |
|---|---|---|
| **Degradation curve** | Does accuracy hold as the difficulty axis rises? | Real reasoning degrades *gracefully*; pattern-matching falls off a cliff. The axis is **per-family, not commensurable**: most families scale reasoning steps, but `sequences` scales a *rule tier* (AP→cubic) and the CSP families scale *problem size n*. Read each curve against itself, not across families. |
| **Variance** | How much does accuracy wobble across equivalent problems? | High variance on structurally-identical items = brittle. |
| **Distractibility** | Does an irrelevant clause break it? | The GSM-NoOp probe. A big drop means it isn't tracking relevance. |
| **Surface invariance** | Same computation, different names/items/phrasing — does the answer flip? | Answer flips mean it's keying on surface, not structure. **Scope:** varies cosmetic surface (names, items, lexical phrasing) with the gold held fixed; it does **not** vary numeric magnitude (that changes the gold, so it can't be a same-gold flip pair — see degradation/variance for magnitude effects). |
| **Calibration (ECE)** | Does stated confidence track correctness? | A smart system is uncertain when it should be. |
| **pass@1 / maj@k / pass@k** | Single-shot vs majority-vote vs best-of-k | maj@k (self-consistency) is a deployable strategy; pass@k is an oracle upper bound. Report all three. |
| **Chance-corrected accuracy (`acc_above_chance`)** | Accuracy normalized against the random-guess baseline for each family | A family with many possible answers looks harder than it is; this exposes true headroom above guessing. |
| **Confabulation rate** | On ill-posed items (`UNDETERMINED`, `NO_SOLUTION`), does the model answer with a concrete value anyway? | Inventing answers for contradictory or under-constrained problems is worse than saying "unknown". |
| **Grading fragility** | How often does the `ANSWER:` marker parse disagree with the fallback parse? | High fragility means the score depends on format compliance, not just reasoning. |
| **Behavioral uncertainty** | Disagreement entropy and self-consistency gap across samples | Low stated confidence but high sample disagreement flags uncertainty the model doesn't report. |

Eleven problem families:

| Family | What it probes | Difficulty axis |
|---|---|---|
| `arithmetic` | multi-hop quantitative word problems (add/sub/×/÷ — exact) | number of operations |
| `state_tracking` | item counts across containers through updates | number of updates |
| `ordering` | transitive comparison | number of entities |
| `sequences` | next-term rule induction (AP, GP, quadratic, fibonacci, interleaved, cubic) | rule-complexity tier 1..6 |
| `retroactive_edit` | single-turn dynamic constraint: a late clause rewrites an earlier value | number of updates / edit distance |
| `multi_turn_inject` | multi-turn state tracking: turn 1 sets state, turn 2 injects a new rule | number of state facts |
| `knights_knaves` | truth-teller / liar **deduction** under self-reference | islanders (diff+2, up to 8) |
| `logic_grid` | **constraint satisfaction** — place N people on N floors | floors (diff+2, up to 7) |
|`unsat_csp` | premise-flaw detection: controlled ill-posed knights-and-knaves puzzles | n (diff+2); brute-force solution count |
| `composed` | ≥3 execution-dependency hops (e.g. knights → arithmetic → ordering) | chain length / difficulty of each hop |
| `redefined_ops` | arithmetic with counterfactually redefined operators | number of operations |
| `dynamic_pivot` | genuine backtracking: turn 1 commits a count, turn 2 reveals the moves never happened → revise | updates before the pivot |

The four numeric families (`arithmetic`, `state_tracking`, `ordering`, `sequences`) are **robustness probes**: they measure compositional and perturbation-resilient reasoning, not just final accuracy. The CSP families (`knights_knaves`, `logic_grid`, `unsat_csp`) add **deductive / constraint-satisfaction** load that scales steeply (8 islanders = 2⁸ assignments; 7 floors = 7! arrangements) and is where strong reasoners separate from pattern-matchers. `retroactive_edit`, `multi_turn_inject`, `composed`, and `redefined_ops` test dynamic, multi-hop, and counterfactual execution.

`arithmetic`, `state_tracking`, `ordering`, `retroactive_edit`, `knights_knaves`, `logic_grid`, and `redefined_ops` support the surface probe; the first three of those also support distractor probes. `composed`, `multi_turn_inject`, `unsat_csp`, and `sequences` do not currently support surface renaming.
---

## Install

```bash
pip install -r requirements.txt        # numpy + matplotlib (only needed for `report`)
```

The runner itself uses only the standard library and talks to any **OpenAI-compatible**
`/chat/completions` endpoint — which covers Ollama, vLLM, LM Studio, llama.cpp, TGI,
and hosted APIs.

---

## Easiest start: `python cli.py start`

One command runs the whole thing interactively — no flags to remember:

```bash
python cli.py start          # or just: python cli.py
```

It walks you through three steps:

1. **Pick or add a model.** If models are already configured it lists them and you pick
   one by number (or choose to add another). If none are configured it drops straight
   into the setup wizard, which asks step by step for the **provider endpoint, API key,
   model ID, and context window**, writes them to `providers.json` (reusing an existing
   provider when the endpoint matches), and offers to test the connection.
   - The API key is read with hidden input and, by default, stored as an
     **environment-variable reference** (`api_key_env`) so no secret lands in the file;
     choose "file literal" only for throwaway/local keys (`providers.json` is tracked by git).
   - Leave the key blank for keyless local servers (e.g. Ollama).
2. **Choose a dataset** — reuse the one already in the DB, or generate a fresh
   `quick` / `standard` / `thorough` set.
3. **Run + report** — runs the model with the live progress bar (below) and writes the report.

Just want to register a model without running? `python cli.py setup` is the wizard on
its own (it prints the `generate` / `run` / `report` commands to run next). Re-run it
any time to add another model; `python cli.py models` lists what's configured. Prefer
to edit config by hand? The [registry section](#register-a-provider--model-once-recommended)
below documents the `providers.json` format both write.

---

## Quickstart (3 steps)

```bash
# 1) sanity-check the whole pipeline with NO model (synthesizes answers)
python cli.py generate --db bench.db --max-diff 6 --reps 12 --distractor --surface-variants 3
python cli.py run --db bench.db --mock noisy --run-id smoke --confidence --samples 3
python cli.py report --db bench.db --runs smoke --out report_smoke
```

If that prints a metrics table and writes `report_smoke/`, you're ready for real models.

---

## Running real models

### Register a provider + model once (recommended)

Endpoints and models live in **`providers.json`** (next to the code; set `$BENCH_PROVIDERS`
to point elsewhere). Add an entry once and refer to it by a short **alias** instead of
repeating `--base-url` / `--model` / `--api-key` on every run:

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
python cli.py providers      # list configured endpoints
python cli.py models         # list aliases (provider, model id, context window)
python cli.py run --db bench.db --model gpt4omini --run-id gpt4omini --confidence
```

- `api_key_env` reads the key from an environment variable; `api_key` is a literal fallback
  (handy for local servers that want a dummy key, e.g. vLLM's `EMPTY`).
- `context_window` is recorded with the run; `max_tokens`, if set, becomes that model's
  default completion budget (the runner warns if `--max-tokens` ≥ the context window).
- `--provider` / `--base-url` / `--api-key` / `--max-tokens` always override the registry,
  and a raw model id with no provider falls back to a local Ollama endpoint — so the
  explicit forms below still work with no config file at all.

### Local model via Ollama
```bash
ollama serve            # in another terminal
ollama pull llama3.2:3b

python cli.py run --db bench.db \
    --base-url http://localhost:11434/v1 \
    --model llama3.2:3b --run-id llama32 \
    --confidence --workers 4 --temperature 0
```

### Local model via vLLM
```bash
# vLLM serves an OpenAI-compatible API on :8000
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

---

## The dataset knobs (`generate`)

```
--families a b c     subset of: arithmetic state_tracking retroactive_edit multi_turn_inject
                     ordering sequences knights_knaves logic_grid unsat_csp composed
                     redefined_ops  (default: all)
--min-diff / --max-diff   difficulty range = number of reasoning steps (default 1..6).
                     Some families select a discrete structure instead of a step count
                     and are capped so a higher --max-diff can't emit "difficulties"
                     that aren't actually harder (or aren't feasible to gold-verify):
                     sequences = rule-complexity tier 1..6;
                     knights_knaves / unsat_csp = diff+2 islanders (≤8);
                     logic_grid = diff+2 floors (≤7);
                     composed / redefined_ops / retroactive_edit / multi_turn_inject
                     inherit the numeric caps of their constituent stages.
--reps N             distinct structures per (family, difficulty). Higher N = tighter
                     variance estimates. 12–25 is a good range.
--distractor         also emit a matched NoOp-distractor copy of every base item
A run with `--reps 20 --distractor --surface-variants 3` over difficulties 1–6 and all
eleven families is ~5,000 items. At `--samples 1` that's ~5,000 model calls.


## The run knobs (`run`)

```
--model / --provider / --base-url   a model alias (from `cli.py models`), or a raw
                     model id plus a provider alias / explicit endpoint.
--samples N          completions per item, drawn one request at a time (so pass@k works
                     even on servers that ignore the OpenAI `n` param). >1 enables
                     pass@k, majority-vote (maj@k), and self-consistency.
--temperature        use 0 for the cleanest single-shot reasoning signal;
                     raise it only when measuring pass@k / maj@k.
--confidence         ask the model for CONFIDENCE: 0-100 -> enables calibration/ECE.
--workers            concurrent requests. Keep modest for local servers.
--resume             skip items already scored OK; items that only errored are retried.
--mock {perfect,random,noisy}   synthesize answers without a server (deterministic).
```

While a run is in progress the runner shows a **live progress bar** — percent complete,
items done, OK / error counts, throughput, and ETA — repainted in place on a terminal:

```text
  [████████████░░░░░░░░░░░░░░░░]  43.5%  217/499  ok=214 err=3  6.1 it/s  ETA 0:46
```

When stdout is redirected to a file it falls back to a plain progress line every ~5%
so logs stay readable, and a final `done in M:SS (… items, … ok, … errored)` summary
closes every run.

Re-running the same `--run-id` overwrites its rows (keyed on item+sample), so it never
double-counts — `--resume` is only needed to *skip* finished work, not to avoid duplicates.

Raw responses are stored verbatim in SQLite, so you can re-grade or re-analyze (edit
`metrics.py` and re-run `report`) without spending tokens again.

---

## Reading the output

- **Flat degradation curve** across difficulties is the strongest positive signal. A
  cliff at d3–d4 means the model handles short chains by recall but can't actually compose
  steps. Error bars are **Wilson 95% intervals** (they shrink with `--reps`); widen them
  by collecting more structures, not by reading the raw per-cell std.
- **Distractor drop near zero** is good. The comparison is **paired** (same structure with
  and without the clause), and the summary shows how many items the clause *hurt* vs
  *helped* — a large drop (the original GSM-NoOp finding) means irrelevant info derails it,
  though newer models sometimes treat it as a hint, so read it alongside the curve.
- **Answer-flip rate near zero** means the model is keying on structure, not names.
- **Low ECE** means confidence is trustworthy. Confidently-wrong on hard items is the
  pattern-matching signature.
- There is **no single intelligence score** here by design — it's a profile. Two models
  with the same overall accuracy can have very different degradation and robustness.

To isolate **scale vs alignment-recipe** effects: generate one dataset, run both models
on the *identical* items, and compare degradation curves. Because the items are generated
(not scraped), any gap can't be a contamination artifact.

---

## Accessibility

Charts use the Okabe-Ito colorblind-safe palette with **redundant encoding** — every
model has a distinct color *and* marker shape *and* line style, so the plots stay
readable in greyscale and under color-vision deficiency. Fonts are DejaVu Sans at a
large size. `metrics.csv` is provided for screen-reader / spreadsheet use instead of
relying on the images.

---

## Limits (read these)

- **Sequences are the noisiest family.** "Next term" can admit more than one rule; the
  generators use enough terms to make the intended rule the simplest fit, but treat this
  family as a soft signal. Its difficulty axis is a rule-complexity *tier*, not a step
  count, so compare it within-family rather than against numeric families.
- **Grading is exact-match** on a parsed `ANSWER:` line with fallbacks. If a model
  refuses the format, it will look wrong; spot-check a few raw responses in the DB
  (`SELECT raw FROM responses LIMIT 5`) the first time you run a new model. `grading_fragility`
  tells you how much the reported score depends on the fallback path.
- The four numeric families (`arithmetic`, `state_tracking`, `ordering`, `sequences`) are
  **robustness probes**: they test compositional and perturbation-resilient reasoning, not
  just accuracy. `retroactive_edit`, `multi_turn_inject`, `composed`, and `redefined_ops`
  add dynamic, multi-turn, and counterfactual execution load.
- The CSP families (`knights_knaves`, `logic_grid`, `unsat_csp`) add **deductive /
  constraint-satisfaction** load that scales steeply (8 islanders = 2⁸ assignments;
  7 floors = 7! arrangements) and is where strong reasoners separate from pattern-matchers.
  `unsat_csp` deliberately includes controlled ill-posed instances; the meaningful metric
  there is the **confabulation rate** (concrete answers to `UNDETERMINED`/`NO_SOLUTION`).
- **CSP difficulty is bounded by gold-verification cost, not by the model.** Uniqueness
  (or controlled non-uniqueness for `unsat_csp`) is proven by brute force at build time,
  so the caps above keep generation honest. If you want harder puzzles than the caps allow,
  raise `FAMILY_MAX_DIFF` *and* the verifier's search will need to scale with it
  (it's exponential/factorial).

## Extending it

Add a family in `generators.py`:

1. write `gen_yourfamily(difficulty, structure_seed, surface_seed, distractor)` returning
   `(prompt, gold, answer_type, choices)` — or `(prompt, gold, answer_type, choices, turns)`
   for multi-turn families. Compute the gold *in code*.
2. add a matching `_verify_yourfamily(prompt, gold)` in `_VERIFIERS` so `verify_gold` can
   re-derive the answer from the prompt text independently.
3. register it in `GENERATORS`, and add it to `SUPPORTS_DISTRACTOR` / `SUPPORTS_SURFACE`
   if it can hold its gold fixed under those perturbations. Set `FAMILY_MAX_DIFF` if the
   family uses a discrete tier or brute-forced structure.

Everything else (storage, running, metrics, charts) picks it up automatically.

## Files
```
generators.py   problem families + dataset builder + independent gold verifiers
grading.py      answer extraction + scoring
storage.py      SQLite layer
runner.py       OpenAI-compatible client + orchestration (+ deterministic mock)
metrics.py      all metrics from the raw responses
report.py       accessible charts + CSV + Markdown
providers.py    provider/model registry loader + resolver
providers.json  the endpoints + model aliases you edit (or let `setup` write)
cli.py          start / setup / generate / run / report / list / families / providers / models
test_bench.py   pytest suite (no model or network needed)
```

## Testing

```bash
pip install pytest
pytest -q          # generators, CSP uniqueness, grading, error/dup handling, metrics, registry
```

The suite drives the full pipeline through the deterministic mock, so it needs no model,
network, or matplotlib. Every generated gold is independently re-derived from the prompt
text (`generators.verify_gold`) — `build_dataset` runs that check on every item, so a
prompt/gold mismatch fails fast at generation time rather than silently mis-scoring a run.
