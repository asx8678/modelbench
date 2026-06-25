# reasoning-bench — Hardening Plan

Plan to fix the Phase 1 audit findings and land the Phase 2 hardening. Grounded in the
actual code; line refs are current as of this writing. Baseline: `pytest -q` → **40 passed**.

**How to read:** workstreams are independent unless a dependency is listed. Each has an
effort tag (S ≈ <1h, M ≈ a few hours, L ≈ day+), the files it touches, acceptance criteria,
and the tests that prove it. Recommended order is at the bottom.

---

## Guiding invariants (must not regress)

1. **Gold is computed in code and independently re-verified.** Every family has an entry in
   `generators._VERIFIERS` that re-derives the answer from prompt *text* (not generator
   state). `build_dataset(verify=True)` (`generators.py:641`) runs it on every item.
   `test_all_generated_golds_verify` and the parametrized `test_verifier_catches_wrong_gold`
   enforce this for **every** family in `GENERATORS` — so a new family with no verifier, or a
   verifier that trusts the gold, fails CI.
2. **Determinism.** `test_generation_is_deterministic` asserts `(item_id, gold, prompt)` are
   byte-stable across builds. No `Date.now()`/unseeded RNG in generators.
3. **Analysis is re-runnable without re-querying models.** Raw responses are stored
   (`storage.py` `responses.raw`); all metrics derive from them. Metric changes must not need
   new model calls.
4. **Additive changes are safe; changes to existing families are migrations** (see below).

## Compatibility & migration note (read before touching existing generators)

`_mk` (`generators.py:444`) hashes the `item_id` from
`family|difficulty|structure_seed|surface_seed|distractor|probe` — **not** from the prompt or
gold. Consequences:

- **New families are fully additive** — new `item_id`s, no collision, old data untouched.
- **Editing an existing generator/verifier (WS5) changes the prompt+gold under the *same*
  `item_id`.** Any previously generated `bench.db` dataset and its stored runs become
  stale/incomparable for that family. Required step when shipping WS5: bump a dataset version
  (e.g. a `--dataset-tag` recorded in `runs.params`, or simply regenerate into a fresh db) and
  **do not compare pre/post numbers for changed families.**
- **Telemetry (WS9) needs a schema migration** — new columns/table; `storage.SCHEMA` uses
  `CREATE TABLE IF NOT EXISTS`, so add columns via `ALTER TABLE` guarded by a `PRAGMA
  table_info` check, or a `schema_version` row.

---

## WS0 — Safety net  ·  S  ·  no deps

Lock in current behavior so later changes are measurable.

- Confirm `pytest -q` green (done: 40 passed).
- Generate a reference dataset + `--mock noisy` run and snapshot `metrics.compute` output to a
  fixture (`tests/fixtures/baseline_metrics.json`). Add a test asserting unchanged families’
  metrics stay byte-stable across the whole effort.

**Done when:** a regression in any *unchanged* family’s metrics fails a test.

---

## WS1 — Metrics gaps  ·  M  ·  no deps (analysis-only, safe)

Pure `metrics.py` additions; no generator changes; re-runs on stored data.

| Metric | Definition | Implementation |
|---|---|---|
| `chance_baseline` / `acc_above_chance` | per-family guess rate; `(acc − chance)/(1 − chance)` | chance: `arithmetic`/`state_tracking`/`sequences` ≈ 0 (unbounded int); `ordering` = 1/m; `knights_knaves` = 1/(n+1); `logic_grid` = 1/n. Derive m/n from `difficulty`. |
| `frontier_headroom` | fraction of base items *any* sampled completion got right (`oracle` already computed at `metrics.py:198`) vs `pass@1` | reuse the `oracle` list; expose the gap as a saturation indicator |
| `grading_fragility` | rate at which the `ANSWER:` marker parse disagrees with the last-token fallback | needs WS2’s `parse_source`, or recompute both parses from `raw` here |

**Acceptance / tests:** extend the metrics test block; assert `acc_above_chance ≤ accuracy`,
chance values correct for known m/n, and headroom ∈ [pass@1, 1].

**Why first:** quantifies C1/C2 (answer-space collapse, saturation) with zero risk, and gives
the number that justifies the rest of the plan.

---

## WS2 — Grading robustness (C4)  ·  S–M  ·  no deps

`grading.py`. Today `parse_answer` falls back to "last integer anywhere" (`:45`) / "last
choice mentioned" (`:57`), which both over- and under-credits.

- Return a `parse_source ∈ {marker, fallback, none}` alongside the parsed value (thread through
  `grade`). Optionally add an `int` column `parse_source` to `responses` (WS-migration) or
  derive in metrics from `raw`.
- Add a strict-mode flag (config) that scores only the marker line, to measure format
  compliance separately from reasoning.

**Acceptance / tests:** extend `test_grading*`; assert marker present → `parse_source=marker`;
confidence line never mistaken for the answer (already guarded at `_strip_confidence`,
`grading.py:29` — add a regression case); fragility computed on a crafted disagreement.

---

## WS3 — Verdict answer type (foundation for premise-flaw)  ·  S  ·  blocks WS4

Support non-numeric sentinel answers so a model can say a problem is ill-posed.

- Add handling for gold ∈ {`NO_SOLUTION`, `UNDETERMINED`} as a `choice`-style answer whose
  `choices` include the sentinels (keeps `test_verifier_catches_wrong_gold` happy — it needs a
  populated `choices` to pick a wrong gold for non-int types, see `test_bench.py:63`).
- `grading.parse_answer`: match the sentinel tokens case-insensitively on the `ANSWER:` line.
- `runner.build_messages` (`runner.py:36`): when sentinels are in play, the format hint must
  list them as allowed answers.

**Acceptance / tests:** round-trip grading test for each sentinel; wrong-gold test passes.

---

## WS4 — `unsat_csp` family (premise-flaw vector — highest ROI)  ·  M  ·  deps: WS3

Nearly free: `_kk_all_solutions` (`generators.py:247`) and `_lg_solutions` (`:325`) already
**count** solutions.

- `gen_unsat_csp(difficulty, structure_seed, surface_seed, distractor)`: build a KK or
  logic-grid clue set, then with controlled probability either drop a load-bearing clue
  (→ multiple solutions → `UNDETERMINED`) or inject a contradiction (→ zero → `NO_SOLUTION`),
  else keep it unique (→ the value). `gold` per the solution count.
- `_verify_unsat_csp(prompt, gold)`: re-parse via existing `_kk_parse`/`_lg_parse`, re-count,
  assert the count→sentinel mapping matches gold. Register in `_VERIFIERS`.
- Register in `GENERATORS`, `SUPPORTS_SURFACE`, and `FAMILY_MAX_DIFF`.
- Generate satisfiable vs ill-posed at a known ratio (e.g. 70/30) so we can measure…
- **New metric `confabulation_rate`** (`metrics.py`): fraction of ill-posed items the model
  answered with a concrete value instead of the correct sentinel.

**Acceptance / tests:** new tests asserting we emit all three gold classes; brute-force
solution count matches the sentinel; `build_dataset(verify=True)` passes; the parametrized
`test_verifier_catches_wrong_gold` passes for `unsat_csp`.

---

## WS5 — Fix existing-family shortcuts  ·  M–L  ·  **migration (see note above)**

Each sub-item changes a family’s prompt/gold → regenerate datasets; don’t compare pre/post.

| Sub | Fix | Files | Notes |
|---|---|---|---|
| 5a | **`ordering` — force transitivity.** Ask a *non-extreme* rank ("3rd tallest?") and/or withhold some adjacent pairs so a chain must be assembled. | `gen_order` (`:157`), `_verify_order` (`:500`) | The verifier currently *encodes the shortcut* (`cand = [n for n in names if n not in lows]`, `:514`). It must be rewritten to do real transitive resolution (topo-order the chain, index the asked rank). Highest-value fix. |
| 5b | **`knights_knaves` — de-collapse the answer.** Ask "list all knaves" (or a specific islander’s type) → answer space 2ⁿ instead of n+1. | `gen_knights` (`:258`), `_verify_knights` (`:577`), `grading` (list/choice answer) | Kills the guess/`maj@k` inflation (C1). |
| 5c | **`logic_grid` — second attribute.** Add e.g. floor + color with cross-attribute clues (true zebra-lite). | `gen_logic_grid` (`:339`), `_lg_*` helpers, `_verify_logic_grid` (`:610`) | Bigger change; brute-force search grows — keep `FAMILY_MAX_DIFF` honest. Can defer as stretch. |
| 5d | **`sequences` — independent verifier.** Replace `_detect_next` (`:520`) reuse with a genuinely independent method (finite differences + explicit ambiguity check) so the verifier doesn’t share the generator’s rule ladder. | `_verify_sequence` (`:549`) | Closes the circular-verifier finding. |

**Acceptance / tests:** for each, an updated uniqueness/correctness test; `ordering` test must
prove a *middle*-rank query can’t be answered by degree-counting; `sequences` verifier must
flag a genuinely ambiguous sequence.

---

## WS6 — Counterfactual families  ·  M  ·  deps: WS2 (grading), pattern from WS4

- `redefined_ops`: arithmetic where the prompt redefines operators (e.g. "⊕ means a+b+3").
  Gold computed in code by applying the redefined table; verifier replays the redefined ops
  from prompt text. Supports distractor + surface. Unbounded int (no C1 issue).
- `kk_counterfactual`: knights/knaves with a twisted truth rule ("knaves tell the truth about
  even-numbered speakers"). `_kk_truth` (`:232`) already parameterizes truth — pass the rule
  in; brute-force uniqueness unchanged.

**Acceptance / tests:** golds verify; a model that applies *standard* operator semantics scores
at chance (sanity check via a crafted mock).

---

## WS7 — Dynamic-constraint families  ·  M (retroactive) / L (multi-turn)

- `retroactive_edit` (single-turn): a late clause mutates an earlier value
  ("…actually the metal tin held double"). Gold recomputed in code after the edit; verifier
  replays edits from text. No runner change.
- `multi_turn_inject` (**needs runner change**): turn 1 establishes state, turn 2 injects a new
  rule then asks. `build_messages` (`runner.py:36`) currently emits one user turn — extend the
  `Problem`/message path to carry a turn list. Gold computed across turns.

**Acceptance / tests:** retroactive golds verify; multi-turn path exercised through the mock.

---

## WS8 — `composed` family (≥3 execution-dependency hops)  ·  M  ·  deps: existing gens

Chain generators so a slip in hop A breaks hop C: e.g. a KK knight-count seeds an arithmetic
start value, whose result indexes an ordering query. One final int gold computed by running the
chain in code; verify each hop independently.

**Acceptance / tests:** gold verifies end-to-end; a unit test perturbs hop A and asserts the
final gold changes (proving the dependency is real, not decorative).

---

## WS9 — Telemetry pipeline  ·  L  ·  separate track, schema migration

Reality check (verified against the current Claude API): the runner is **non-streaming** and
captures only `usage.{prompt,completion}_tokens` + latency (`runner.py:49`). The
OpenAI-compatible path **cannot surface Claude reasoning**; the native Anthropic Messages API
streams `thinking_delta`, but on Opus 4.8 `display` defaults to `omitted` (empty thinking text),
the **raw chain of thought is never returned** (summary only), and **there are no logprobs on
any Anthropic endpoint**. So build for what’s real and flag what isn’t.

- **Add an optional streaming + native-Anthropic client path** alongside the OpenAI-compat one,
  behind a provider capability flag.
- **New `telemetry` table** (or columns on `responses`): provenance/`capabilities`, token
  accounting with a `reasoning_token_source ∈ {native_usage, counted_from_think_tags,
  summary_only, unavailable}`, phase timing (ttft / first-reasoning / reasoning-wall /
  answer-wall) **labeled ops-only, not cognition**, surface backtracking-marker counts (only
  when `display:summarized`), and an uncertainty block.
- **Lead with the validated, behavioral uncertainty signal** we already have: `maj@k − pass@1`
  spread, sample-disagreement entropy, stated confidence/ECE. Token-entropy fields stay `null`
  unless an open-weight server exposes logprobs.
- **Explicitly do NOT** ship a "thinking-velocity/TPS struggle" gauge or a "Tree-of-Thought
  branch map" — not observable through current APIs for the target model.

**Acceptance / tests:** schema migration idempotent; telemetry rows join to `responses` by
`(run_id,item_id,sample_idx)`; capability flags gate which fields populate.

---

## WS10 — Docs, report, charts  ·  S–M  ·  after the families land

- Update `README.md` family table + metrics section (new families, chance-corrected accuracy,
  confabulation rate, the repositioning of the four numeric families as robustness probes).
- `report.py`: add **confabulation-vs-confidence** and **accuracy-above-chance** charts; keep
  the colorblind-safe redundant encoding.

---

## Recommended execution order

```
WS0 ─┬─ WS1 ─────────────┐
     ├─ WS2 ─┐           ├─ WS10 (docs/report)
     │       └─ WS3 ─ WS4 ┤
     │                    ├─ WS6
     ├─ WS5 (migration) ──┤
     │                    ├─ WS7
     │                    └─ WS8
     └─ WS9 (telemetry, parallel track) ─┘
```

1. **WS0 → WS1 → WS2** (safe, analysis-layer; quantify the problem).
2. **WS3 → WS4** (premise-flaw family + confabulation rate — biggest reasoning signal for least
   code; counters already exist).
3. **WS5** (fix shortcuts; gated behind the dataset-version bump).
4. **WS6 / WS7 / WS8** (counterfactual, dynamic, composed — independent, parallelizable).
5. **WS9** (telemetry; own track, don’t block the dataset work on it).
6. **WS10** (docs + charts once families are stable).

## Definition of done (whole effort)

- `pytest -q` green, with new tests for every new family and metric.
- Every family in `GENERATORS` has a `_VERIFIERS` entry that rejects corrupted gold.
- `metrics.compute` reports chance-corrected accuracy, confabulation rate, frontier headroom,
  and grading fragility.
- README + report reflect the new families and the numeric-family repositioning.
- Telemetry (if landed) records only what’s actually observable, with capability provenance.

## First slice I’d implement (safe, high-ROI)

WS0 + WS1 + WS3 + WS4: baseline snapshot, chance-corrected/headroom metrics, verdict answers,
and the `unsat_csp` family with `confabulation_rate` — all additive (no migration), all
verifiable, drives straight at the saturation and premise-flaw findings.
