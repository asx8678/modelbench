# reasoning-bench

A small, **contamination-proof** harness for measuring how well an LLM *reasons* —
not just whether it scores high on a static test.

Every problem is generated on the fly from parameterized templates, so it cannot be
in any model's training data. Gold answers are computed while the problem is built
(and independently re-verified). Instead of one accuracy number, you get the things
that actually distinguish reasoning from pattern-matching:

| Signal | What it asks | Why it matters |
|---|---|---|
| **Degradation curve** | Does accuracy hold as you add reasoning steps? | Real reasoning degrades *gracefully*; pattern-matching falls off a cliff. |
| **Variance** | How much does accuracy wobble across equivalent problems? | High variance on structurally-identical items = brittle. |
| **Distractibility** | Does an irrelevant clause break it? | The GSM-NoOp probe. A big drop means it isn't tracking relevance. |
| **Surface invariance** | Same computation, different names — does the answer flip? | Answer flips mean it's keying on surface, not structure. |
| **Calibration (ECE)** | Does stated confidence track correctness? | A smart system is uncertain when it should be. |
| **pass@1 / maj@k / pass@k** | Single-shot vs majority-vote vs best-of-k | maj@k (self-consistency) is a deployable strategy; pass@k is an oracle upper bound. Report all three. |

Six problem families:

| Family | What it probes | Difficulty axis |
|---|---|---|
| `arithmetic` | multi-hop quantitative word problems (add/sub/×/÷-exact) | number of operations |
| `state_tracking` | item counts across containers through updates | number of updates |
| `ordering` | transitive comparison | number of entities |
| `sequences` | next-term rule induction (AP, GP, quadratic, fibonacci, interleaved, cubic) | rule-complexity tier 1..6 |
| `knights_knaves` | truth-teller / liar **deduction** under self-reference | islanders (diff+2, up to 8) |
| `logic_grid` | **constraint satisfaction** — place N people on N floors | floors (diff+2, up to 7) |

`arithmetic` / `state_tracking` / `ordering` support distractor + surface probes.
`knights_knaves` and `logic_grid` are the hardest discriminators: each is a
contamination-proof CSP with a **unique, minimal, brute-force-verified** solution
(drop any one clue and uniqueness breaks), and they support the surface probe
(structure is chosen in slot space and only labelled afterwards, so renaming holds
the answer fixed). `sequences` is difficulty/variance only.

---

## Install

```bash
pip install -r requirements.txt        # numpy + matplotlib (only needed for `report`)
```

The runner itself uses only the standard library and talks to any **OpenAI-compatible**
`/chat/completions` endpoint — which covers Ollama, vLLM, LM Studio, llama.cpp, TGI,
and hosted APIs.

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
--families a b c     subset of: arithmetic state_tracking ordering sequences
                     knights_knaves logic_grid  (default: all)
--min-diff / --max-diff   difficulty range = number of reasoning steps (default 1..6).
                     Some families select a discrete structure instead of a step count
                     and are capped so a higher --max-diff can't emit "difficulties"
                     that aren't actually harder (or aren't feasible to gold-verify):
                     sequences = rule-complexity tier 1..6; knights_knaves = diff+2
                     islanders (≤8); logic_grid = diff+2 floors (≤7).
--reps N             distinct structures per (family, difficulty). Higher N = tighter
                     variance estimates. 12–25 is a good range.
--distractor         also emit a matched NoOp-distractor copy of every base item
--surface-variants K also emit K cosmetic variants (same gold) per base item
```

A run with `--reps 20 --distractor --surface-variants 3` over difficulties 1–6 and all
four families is ~2,000 items. At `--samples 1` that's ~2,000 model calls.

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
  count, so compare it within-family rather than against the other three.
- **Grading is exact-match** on a parsed `ANSWER:` line with fallbacks. If a model
  refuses the format, it will look wrong; spot-check a few raw responses in the DB
  (`SELECT raw FROM responses LIMIT 5`) the first time you run a new model.
- The four numeric families measure **compositional / robustness reasoning**;
  `knights_knaves` and `logic_grid` add **deductive / constraint-satisfaction** load
  that scales steeply (8 islanders = 2⁸ assignments; 7 floors = 7! arrangements) and
  is where strong reasoners separate from pattern-matchers. They remain complementary
  to, not a replacement for, frontier benchmarks (ARC-AGI-2, GPQA-Diamond, etc.).
- **CSP difficulty is bounded by gold-verification cost, not by the model.** Uniqueness
  is proven by brute force at build time, so the caps above keep generation honest. If
  you want harder puzzles than the caps allow, raise `FAMILY_MAX_DIFF` *and* the
  verifier's search will need to scale with it (it's exponential/factorial).

## Extending it

Add a family in `generators.py`:

1. write `gen_yourfamily(difficulty, structure_seed, surface_seed, distractor)` returning
   `(prompt, gold, answer_type, choices)`. Compute the gold *in code*.
2. register it in `GENERATORS`, and add it to `SUPPORTS_DISTRACTOR` / `SUPPORTS_SURFACE`
   if it can hold its gold fixed under those perturbations.

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
providers.json  the endpoints + model aliases you edit
cli.py          generate / run / report / list / families / providers / models
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
