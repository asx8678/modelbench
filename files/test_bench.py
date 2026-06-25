"""
Test suite for reasoning-bench. Run with:  pytest -q

Covers the things that silently corrupt results if they regress: independent gold
verification, grading robustness, error/duplicate handling, and the metrics math.
No model or network needed — the deterministic mock drives the end-to-end paths.
(report.py is not imported here so the suite needs no matplotlib.)
"""

import dataclasses
import itertools
import json

import re
from pathlib import Path
import pytest

import generators
import grading
import storage
import runner
import metrics
import providers


def _cfg(**kw):
    base = dict(mock=None, base_url="", api_key="", model="m", temperature=0,
                max_tokens=64, n=1, workers=2, timeout=5, retries=0,
                ask_confidence=True, resume=False, context_window=None)
    base.update(kw)
    return base


def _db_with(items):
    con = storage.connect(":memory:")
    storage.save_dataset(con, items)
    return con, storage.load_dataset(con)


# --------------------------------------------------------- generators / gold (#12)
def test_all_generated_golds_verify():
    # build_dataset(verify=True) raises if any gold fails independent re-derivation
    ds = generators.build_dataset(list(generators.GENERATORS), 1, 6, 6,
                                  with_distractor=True, surface_variants=2, verify=True)
    assert ds and all(generators.verify_gold(p) for p in ds)


def test_sequences_capped_at_tier_6():
    # tier 6 (cubic) is the hardest rule; asking for more must not silently reuse it
    ds = generators.build_dataset(["sequences"], 1, 9, 3, verify=True)
    assert max(p.difficulty for p in ds) == 6
    cubics = [p for p in ds if p.difficulty == 6]
    assert cubics and all(generators.verify_gold(p) for p in cubics)


@pytest.mark.parametrize("family", list(generators.GENERATORS))
def test_verifier_catches_wrong_gold(family):
    # The verifier must *independently re-derive* the answer, not trust the gold:
    # the true gold passes, a corrupted one fails — for every family.
    p = generators._mk(family, 3, 7, 0, False, "base", "g")
    assert generators.verify_gold(p) is True
    if p.answer_type == "int":
        bad = dataclasses.replace(p, gold=str(int(p.gold) + 1))
    else:
        bad = dataclasses.replace(p, gold=next(c for c in p.choices if c != p.gold))
    assert generators.verify_gold(bad) is False


def test_generation_is_deterministic():
    # item_id/gold/prompt must be byte-stable across builds, or resume, the mock,
    # and cross-run comparisons all silently break.
    a = generators.build_dataset(list(generators.GENERATORS), 1, 5, 3, surface_variants=2, verify=False)
    b = generators.build_dataset(list(generators.GENERATORS), 1, 5, 3, surface_variants=2, verify=False)
    assert [(p.item_id, p.gold, p.prompt) for p in a] == [(p.item_id, p.gold, p.prompt) for p in b]


# --------------------------------------------------- hard CSP families (#new)
# These are the genuine reasoning discriminators: every puzzle must have exactly
# ONE solution (else "accuracy" is meaningless), the gold must be that solution,
# and difficulty must really add entities. The verifier re-solves from the prompt
# text by brute force, so a passing verify already proves uniqueness + correctness.

def test_knights_knaves_is_uniquely_solvable_and_scales():
    small = generators._mk("knights_knaves", 1, 3, 0, False, "base", "g")
    big = generators._mk("knights_knaves", 6, 3, 0, False, "base", "g")
    n_small, _ = generators._kk_parse(small.prompt)
    n_big, _ = generators._kk_parse(big.prompt)
    assert len(n_small) == 3 and len(n_big) == 8           # difficulty + 2 islanders
    for p in (small, big):
        names, stmts = generators._kk_parse(p.prompt)
        sols = generators._kk_all_solutions(names, stmts)
        assert len(sols) == 1                              # unique => well-posed
        # gold is the sorted, comma-separated list of knave names (2^n space)
        knave_names = {n for n, is_knight in sols[0].items() if not is_knight}
        assert set(t.strip() for t in p.gold.split(",") if t.strip()) == knave_names


def test_knights_knaves_breaks_the_global_flip_symmetry():
    # The classic trap: with only "X says Y is a knight/knave" links, flipping every
    # type yields a second solution. A well-formed puzzle must rule that out.
    for r in range(8):
        p = generators._mk("knights_knaves", 4, r, 0, False, "base", "g")
        names, stmts = generators._kk_parse(p.prompt)
        sols = generators._kk_all_solutions(names, stmts)
        assert len(sols) == 1
        flipped = {n: (not v) for n, v in sols[0].items()}
        assert flipped not in generators._kk_all_solutions(names, stmts)


def test_logic_grid_is_uniquely_solvable_and_scales():
    small = generators._mk("logic_grid", 1, 5, 0, False, "base", "g")
    big = generators._mk("logic_grid", 5, 5, 0, False, "base", "g")
    ns, n_s, _, _ = generators._lg_parse(small.prompt)
    nb, n_b, _, _ = generators._lg_parse(big.prompt)
    assert len(ns) == 3 and len(nb) == 7                   # difficulty + 2 floors
    for p in (small, big):
        names, n, clues, q = generators._lg_parse(p.prompt)
        sols = generators._lg_solutions(names, n, clues)
        assert len(sols) == 1
        assert str(sols[0][q]) == p.gold
def test_unsat_csp_dropped_clue_with_invariant_queried_slot_is_determinate():
    # bench-le7.1 regression: when a clue is dropped and the puzzle becomes
    # under-constrained, the queried slot may still be invariant across all
    # remaining solutions. The gold must be the determinate knight/knave, not
    # the generic UNDETERMINED sentinel.
    prompt, gold, _atype, choices = generators.gen_unsat_csp(2, 4, 0, False)
    names, stmts = generators._kk_parse(prompt)
    sols = generators._kk_all_solutions(names, stmts)
    assert len(sols) > 1, "expected an under-constrained (dropped-clue) item"
    query_name = __import__('re').search(r"Is (\w+) a knight or a knave\?", prompt).group(1)
    values = {s.get(query_name) for s in sols}
    assert len(values) == 1, "this seed's queried slot should be invariant"
    assert gold in ("knight", "knave"), gold
    assert gold == ("knight" if next(iter(values)) else "knave")


def test_unsat_csp_verifier_rejects_undetermined_on_invariant_slot():
    # bench-le7.2 regression: the verifier must independently check the set of
    # values for the queried slot, not blindly accept UNDETERMINED whenever there
    # are multiple solutions.
    prompt, gold, _atype, choices = generators.gen_unsat_csp(2, 4, 0, False)
    assert generators._verify_unsat_csp(prompt, gold) is True
    assert generators._verify_unsat_csp(prompt, "UNDETERMINED") is False


def test_csp_puzzles_are_minimally_constrained():
    # Dropping any one clue should destroy uniqueness — proof the puzzle has no
    # redundant giveaways (a redundant clue would make it easier than its tier).
    p = generators._mk("logic_grid", 4, 6, 0, False, "base", "g")
    names, n, clues, _ = generators._lg_parse(p.prompt)
    assert len(generators._lg_solutions(names, n, clues)) == 1
    assert all(len(generators._lg_solutions(names, n, clues[:i] + clues[i + 1:])) != 1
               for i in range(len(clues)))


def test_csp_surface_variants_hold_gold_and_structure_fixed():
    # A surface variant must be the SAME puzzle with different names: the
    # underlying slot structure and the knave slot set must be invariant,
    # while the rendered name list and gold string change with renaming.
    for fam in ("knights_knaves", "logic_grid"):
        if fam == "knights_knaves":
            def shape(pr):
                _, stmts = generators._kk_parse(pr)
                return sorted(s[0] for s in stmts)
            def knave_count(pr):
                names, stmts = generators._kk_parse(pr)
                sols = generators._kk_all_solutions(names, stmts)
                assert len(sols) == 1
                return sum(1 for v in sols[0].values() if not v)
        else:
            def shape(pr):
                names, n, clues, _ = generators._lg_parse(pr)
                return (n, sorted(c[0] for c in clues))
            def knave_count(pr):
                # logic_grid gold is a floor number; surface variants
                # change the names but the queried floor must match.
                return None
        base = generators._mk(fam, 4, 2, 0, False, "base", "g")
        variants = [generators._mk(fam, 4, 2, s, False, "surface", "g") for s in (1, 2, 3)]
        assert all(shape(v.prompt) == shape(base.prompt) for v in variants)
        assert all(v.prompt != base.prompt for v in variants)
        if knave_count(base.prompt) is not None:
            base_kc = knave_count(base.prompt)
            assert all(knave_count(v.prompt) == base_kc for v in variants)
        else:
            # logic_grid: gold floor must be identical across variants
            assert {base.gold} | {v.gold for v in variants} == {base.gold}


def test_arithmetic_division_op_is_generated_and_verified():
    # The hardened arithmetic family must actually emit division, and the
    # independent verifier must handle it (integer-exact by construction).
    div = next((generators._mk("arithmetic", 4, r, 0, False, "base", "g")
                for r in range(200) if "divides the" in
                generators._mk("arithmetic", 4, r, 0, False, "base", "g").prompt), None)
    assert div is not None, "division operation was never generated"
    assert generators.verify_gold(div) is True
    assert generators.verify_gold(dataclasses.replace(div, gold=str(int(div.gold) + 1))) is False


@pytest.mark.parametrize("family,diff", [("knights_knaves", 5), ("logic_grid", 4),
                                         ("arithmetic", 5), ("sequences", 6)])
def test_no_degenerate_constant_gold(family, diff):
    # A generator that ignored its structure seed would emit one repeated answer;
    # require real answer diversity across structures at a fixed difficulty.
    golds = {generators._mk(family, diff, r, 0, False, "base", "g").gold for r in range(20)}
    min_div = 2 if family == "knights_knaves" else 4
    assert len(golds) >= min_div, (family, golds)



def test_composed_gold_verifies_end_to_end():
    # The composed chain (knights -> arithmetic -> ordering) must produce a gold that
    # the independent verifier re-derives from the prompt text.
    p = generators._mk("composed", 3, 7, 0, False, "base", "g")
    assert generators.verify_gold(p) is True
    # Sanity: prompt has all three stages and >=3 dependency hops.
    assert "Stage 1:" in p.prompt and "Stage 2:" in p.prompt and "Stage 3:" in p.prompt
    # The choices come from the ordering names.
    assert p.gold in p.choices


def test_composed_perturb_hop_a_changes_final_gold():
    # Perturbing the first hop (knights count) must propagate and change the final gold.
    p = generators._mk("composed", 3, 7, 0, False, "base", "g")
    orig_kk = p.prompt
    # Build a synthetic variant with the same ordering/arithmetic but a different
    # knight count by re-seeding hop A while holding hops B/C fixed.
    parsed = generators._composed_parse_hops(p.prompt)
    names, stmts = generators._kk_parse(parsed["knights_prompt"])
    # Flip the type of the first speaker in every statement: changes the knight count.
    flipped = []
    for st in stmts:
        if st[0] == "ABS":
            flipped.append((st[0], st[1], st[2], not st[3]))
        else:
            flipped.append((st[0], st[1], st[2], st[3], not st[4]))
    sols = generators._kk_all_solutions(names, flipped)
    if len(sols) != 1:
        # If flipping breaks uniqueness, just verify that a different hop-A seed
        # usually changes the final gold by sampling.
        seen = {p.gold}
        for s in range(1, 30):
            q = generators._mk("composed", 3, s, 0, False, "base", "g")
            if q.gold not in seen:
                break
            seen.add(q.gold)
        else:
            assert False, "composed gold did not vary across 30 hop-A seeds"
        return
    new_knight_count = sum(1 for v in sols[0].values() if v)
    if new_knight_count == sum(1 for v in generators._kk_all_solutions(names, stmts)[0].values() if v):
        # If equal by bad luck, try the next speaker flip.
        for i in range(1, len(stmts)):
            flipped_i = stmts[:i] + [flipped[i]] + stmts[i + 1:]
            sols_i = generators._kk_all_solutions(names, flipped_i)
            if len(sols_i) == 1:
                new_knight_count = sum(1 for v in sols_i[0].values() if v)
                if new_knight_count != sum(1 for v in generators._kk_all_solutions(names, stmts)[0].values() if v):
                    break
    arith_total = generators._verify_arithmetic_raw(parsed["arith_prompt"], new_knight_count)
    names_order, _ = generators._verify_order_raw(parsed["order_prompt"], parsed["order_query"])
    new_gold = names_order[(arith_total - 1) % len(names_order)]
    assert new_gold != p.gold, (p.gold, new_gold)

# ------------------------------------------------------------------ grading (#1)
def test_confidence_value_not_used_as_answer():
    # The dangerous case: confidence equals gold and the model gave no numeric
    # answer. The old fallback grabbed the CONFIDENCE digits -> spurious "correct".
    parsed, correct, conf, _ = grading.grade("I am not sure of the count.\nCONFIDENCE: 13",
                                          "int", "13")
    assert conf == 13
    assert parsed is None
    assert correct is False


def test_int_fallback_still_finds_a_stated_number():
    # stripping the confidence line must not stop us finding a real trailing number
    parsed, correct, _, _ = grading.grade("so the result is 13.\nCONFIDENCE: 80", "int", "13")
    assert parsed == "13" and correct


def test_grading_well_formed():
    parsed, correct, conf, _ = grading.grade("work...\nANSWER: 13\nCONFIDENCE: 80", "int", "13")
    assert parsed == "13" and correct and conf == 80


def test_grading_choice_fallback():
    parsed, correct, _, _ = grading.grade("so the tallest is Diego.", "choice", "Diego",
                                       ["Diego", "Lena"])
    assert correct


def test_grading_choice_fallback_restricted_on_csp_sentinels():
    # Prompt echoes words like knight/knave; without an ANSWER line they must
    # NOT rescue a non-compliant response via whole-text fallback.
    choices = ["knight", "knave", "UNDETERMINED", "NO_SOLUTION"]
    parsed, src = grading.parse_answer(
        "The prompt mentions a knight and a knave.", "choice", choices)
    assert parsed is None and src == "none"

    # ANSWER line still wins.
    parsed, src = grading.parse_answer(
        "I think the speaker is a knight, not a knave.\nANSWER: knight",
        "choice", choices)
    assert parsed == "knight" and src == "marker"


def test_grading_choice_fallback_unrestricted_for_other_families():
    # Non-CSP choice families keep the whole-text fallback.
    parsed, src = grading.parse_answer(
        "Diego is clearly the tallest here.", "choice", ["Diego", "Lena"])
    assert parsed == "Diego" and src == "fallback"

def test_grading_int_strips_thousands_separator():
    # CSP/arithmetic answers can be large; "1,234" must read as 1234, not 1.
    parsed, correct, _, _ = grading.grade("After working it out,\nANSWER: 1,234", "int", "1234")
    assert parsed == "1234" and correct


def test_grading_int_handles_negative():
    parsed, correct, _, _ = grading.grade("the net change is -5.\nANSWER: -5", "int", "-5")
    assert parsed == "-5" and correct


def test_grading_marker_overrides_earlier_wrong_mention():
    # The model second-guesses itself in prose; the ANSWER line is authoritative,
    # so an earlier wrong name/number in the reasoning must not be scored.
    p_choice, ok_choice, _, _ = grading.grade("At first Diego looks tallest, but no.\nANSWER: Lena", "choice", "Lena", ["Diego", "Lena"])
    assert p_choice == "Lena" and ok_choice
    p_int, ok_int, _, _ = grading.grade("I initially get 7, but rechecking it is 4.\nANSWER: 4", "int", "4")
    assert p_int == "4" and ok_int


def test_grading_choice_marker_with_extra_words():
    # marker line carries justification around the choice token
    parsed, correct, _, _ = grading.grade("ANSWER: The knight is Lena.", "choice", "Lena", ["Diego", "Lena", "Omar"])
    assert parsed == "Lena" and correct


# ------------------------------------------------- mock end-to-end, determinism (#13)
def test_mock_perfect_is_full_accuracy():
    items = generators.build_dataset(["arithmetic", "ordering"], 1, 3, 4,
                                     with_distractor=True, surface_variants=2)
    con, ds = _db_with(items)
    runner.run(con, "perfect", ds, _cfg(mock="perfect"))
    res = metrics.compute(con, "perfect")
    assert res["overall_accuracy"] == 1.0
    assert res["coverage"]["errored"] == 0
    assert 0.0 <= res["invariance"]["answer_flip_rate"] <= 1.0


def test_mock_perfect_on_csp_families_flows_through_pipeline():
    # the hard CSP families (int answers, no distractor/surface) must run end-to-end
    items = generators.build_dataset(["knights_knaves", "logic_grid"], 1, 4, 3,
                                     surface_variants=2)
    con, ds = _db_with(items)
    runner.run(con, "csp", ds, _cfg(mock="perfect"))
    res = metrics.compute(con, "csp")
    assert res["overall_accuracy"] == 1.0
    assert res["coverage"]["errored"] == 0
    # surface variants were emitted, so invariance must be measurable on these families
    assert {"knights_knaves", "logic_grid"} <= set(res["invariance"]["by_family"])


def test_mock_is_deterministic():
    items = generators.build_dataset(["arithmetic"], 1, 4, 5)
    con1, ds1 = _db_with(items)
    runner.run(con1, "a", ds1, _cfg(mock="noisy"))
    con2, ds2 = _db_with(items)
    runner.run(con2, "b", ds2, _cfg(mock="noisy"))
    assert metrics.compute(con1, "a")["overall_accuracy"] == \
           metrics.compute(con2, "b")["overall_accuracy"]


def test_rerun_does_not_duplicate():
    items = generators.build_dataset(["arithmetic"], 1, 2, 4)
    con, ds = _db_with(items)
    runner.run(con, "r", ds, _cfg(mock="noisy"))
    n1 = con.execute("SELECT COUNT(*) FROM responses WHERE run_id='r'").fetchone()[0]
    runner.run(con, "r", ds, _cfg(mock="noisy"))
    n2 = con.execute("SELECT COUNT(*) FROM responses WHERE run_id='r'").fetchone()[0]
    assert n1 == n2 == len(ds)



# ------------------------------------------------- telemetry persistence (#mf4.1/2)
def test_mock_run_writes_telemetry_row_with_caps_and_source():
    items = generators.build_dataset(["arithmetic"], 1, 1, 2)
    con, ds = _db_with(items)
    cfg = _cfg(mock="perfect", capabilities=["stream", "mock"], n=2)
    runner.run(con, "r1", ds, cfg)
    row = storage.load_telemetry(con, "r1", ds[0]["item_id"], 0)
    assert row is not None
    assert row["capabilities"] == ["stream", "mock"]
    assert row["reasoning_token_source"] is not None
    # mock path has no honest producer for reasoning tokens
    assert row["reasoning_token_source"] == "unavailable"
    assert row["prompt_tokens"] is None


def test_mock_run_telemetry_marks_unobservable_fields():
    items = generators.build_dataset(["arithmetic"], 1, 1, 2)
    con, ds = _db_with(items)
    cfg = _cfg(mock="perfect", capabilities=["mock"], n=1)
    runner.run(con, "r2", ds, cfg)
    row = storage.load_telemetry(con, "r2", ds[0]["item_id"], 0)
    assert row is not None
    unobs = row["unobservable_fields"]
    assert unobs["reasoning_wall_ms"] == "no_direct_producer"
    assert "token_entropy" in unobs
    assert "thinking_tps" in unobs
    assert "tot_branch_map" in unobs


def test_build_telemetry_reasonable_when_think_tokens_present():
    cfg = _cfg(capabilities=["native_anthropic", "stream"])
    telemetry = runner._build_telemetry(
        cfg, prompt_tokens=10, completion_tokens=100,
        timings={"ttft": 0.1, "first_reasoning": 0.2,
                 "answer_wall": 0.5, "total": 0.6},
        think_tokens=30, text="ANSWER: 42")
    assert telemetry["reasoning_token_source"] == "native_usage"
    assert telemetry["reasoning_tokens"] == 30
    # density proxy = (ct - (ct - rt)) / (ct - rt) = rt / (ct - rt)
    assert telemetry["reasoning_density_proxy"] == pytest.approx(30 / 70)
    assert telemetry["ttft_ms"] == 100
    assert telemetry["first_reasoning_ms"] == 200
    assert telemetry["answer_wall_ms"] == 500
    assert telemetry["reasoning_wall_ms"] is None
    assert telemetry["unobservable_fields"]["reasoning_wall_ms"] == "no_direct_producer"


def test_build_telemetry_honest_null_when_think_tokens_absent():
    cfg = _cfg(capabilities=["native_anthropic"])
    telemetry = runner._build_telemetry(
        cfg, prompt_tokens=10, completion_tokens=100,
        timings={"ttft": 0.1, "first_reasoning": 0.2,
                 "answer_wall": 0.5, "total": 0.6},
        think_tokens=0, text="ANSWER: 42")
    assert telemetry["reasoning_token_source"] == "unavailable"
    assert telemetry["reasoning_tokens"] == 0
    assert telemetry["reasoning_density_proxy"] is None
    assert telemetry["unobservable_fields"]["reasoning_tokens"] == "not_exposed_by_provider"

def test_baseline_metrics_unchanged():
    # Full reference dataset snapshot: every family, difficulties 1..6, reps 12,
    # distractor + 3 surface variants, mock noisy with 3 samples. The deterministic
    # pipeline must reproduce the per-family metrics byte-for-byte.
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "baseline_metrics.json").read_text()
    )
    items = generators.build_dataset(list(generators.GENERATORS), 1, 6, 12,
                                     with_distractor=True, surface_variants=3,
                                     verify=True)
    con, ds = _db_with(items)
    runner.run(con, "baseline-regression", ds,
               _cfg(mock="noisy", model="mock", n=3, workers=1,
                    max_tokens=1024, timeout=30, ask_confidence=False))
    res = metrics.compute(con, "baseline-regression")

    def per_family(d):
        out = {
            "accuracy_by_family": d["accuracy_by_family"],
            "degradation": d["degradation"],
            "distractibility": d["distractibility"],
            "invariance_by_family": d["invariance"]["by_family"],
        }
        # bench-9eb.1 is a MIGRATION: knights_knaves moved from a binary
        # answer to a 2^n set answer, which changes the mock-noisy
        # accuracy and the surface-flip rate. Exclude it from the
        # byte-stable comparison; re-baseline after the migration tag
        # bump.
        for top in out.values():
            if isinstance(top, dict):
                top.pop("knights_knaves", None)
        return out

    assert json.dumps(per_family(res), sort_keys=True, indent=2) == \
           json.dumps(per_family(fixture), sort_keys=True, indent=2)


# ----------------------------------------- errors are coverage gaps, not wrong (#2, #3)
def test_errors_excluded_and_retried_on_resume(monkeypatch):
    items = generators.build_dataset(["arithmetic"], 1, 1, 3)
    con, ds = _db_with(items)
    monkeypatch.setattr(runner, "call_api", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    runner.run(con, "e", ds, _cfg(mock=None, retries=0))
    res = metrics.compute(con, "e")
    assert res["coverage"]["errored"] == len(ds)
    assert res["overall_accuracy"] == 0.0
    assert storage.done_items(con, "e") == set()   # nothing counts as done -> all retried


def test_malformed_200_response_does_not_crash(monkeypatch):
    items = generators.build_dataset(["arithmetic"], 1, 1, 2)
    con, ds = _db_with(items)

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps({"error": "model loading"}).encode()

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **k: _Resp())
    runner.run(con, "m", ds, _cfg(mock=None, retries=1))   # must not raise
    assert metrics.compute(con, "m")["coverage"]["errored"] == len(ds)


# -------------------------------------------------- majority vote / self-consistency (#8)
def test_majority_vote_overrides_unlucky_first_sample():
    con = storage.connect(":memory:")
    p = generators._mk("arithmetic", 1, 0, 0, False, "base", "g")
    storage.save_dataset(con, [p])
    storage.new_run(con, "v", "m", "", {})
    g = p.gold
    storage.save_response(con, "v", p.item_id, 0, "x", str(int(g) + 1), 0, 50, 1, None, None)
    storage.save_response(con, "v", p.item_id, 1, "x", g, 1, 60, 1, None, None)
    storage.save_response(con, "v", p.item_id, 2, "x", g, 1, 60, 1, None, None)
    con.commit()
    pk = metrics.compute(con, "v")["passk"]
    assert pk["pass@1"] == 0.0
    assert pk["maj@k"] == 1.0
    assert pk["pass@k_oracle"] == 1.0


# -------------------------------------------------------- degradation uses a CI (#5)
def test_degradation_has_wilson_interval():
    items = generators.build_dataset(["arithmetic"], 1, 2, 6)
    con, ds = _db_with(items)
    runner.run(con, "c", ds, _cfg(mock="noisy"))
    cell = next(iter(next(iter(metrics.compute(con, "c")["degradation"].values())).values()))
    assert cell["lo"] <= cell["mean"] <= cell["hi"]
    assert {"mean", "lo", "hi", "n", "std"} <= set(cell)


# -------------------------------------------------------------- multi-turn inject (#WS7b)
def test_multi_turn_build_messages_emits_user_per_turn():
    p = generators._mk("multi_turn_inject", 2, 5, 0, False, "base", "g")
    item = p.row()
    msgs = runner.build_messages(item, ask_confidence=False)
    assert msgs[0]["role"] == "system"
    # Each turn becomes a user message; the last includes the format instruction.
    turns = p.turns
    assert len(msgs) == len(turns) + 1
    assert all(m["role"] == "user" for m in msgs[1:])
    assert turns[0] in msgs[1]["content"]
    assert "ANSWER:" in msgs[-1]["content"]


def test_multi_turn_gold_verifies_from_full_prompt():
    p = generators._mk("multi_turn_inject", 3, 7, 0, False, "base", "g")
    assert generators.verify_gold(p)
    bad = dataclasses.replace(p, gold=str(int(p.gold) + 1))
    assert generators.verify_gold(bad) is False


def test_multi_turn_runs_through_mock():
    items = generators.build_dataset(["multi_turn_inject"], 1, 3, 3)
    con, ds = _db_with(items)
    runner.run(con, "mt", ds, _cfg(mock="perfect"))
    res = metrics.compute(con, "mt")
    assert res["coverage"]["errored"] == 0
    assert res["overall_accuracy"] == 1.0


def test_single_turn_families_unaffected_by_turns_field():
    p = generators._mk("arithmetic", 2, 3, 0, False, "base", "g")
    assert p.turns is None
    msgs = runner.build_messages(p.row(), ask_confidence=False)
    assert len(msgs) == 2
    assert "ANSWER:" in msgs[-1]["content"]


# ----------------------------------------------------------- provider registry (feature)
_REG = {
    "providers": {
        "ollama": {"base_url": "http://localhost:11434/v1"},
        "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    },
    "models": {
        "gpt4omini": {"provider": "openai", "model": "gpt-4o-mini",
                      "context_window": 128000, "max_tokens": 4096},
    },
}


def test_resolve_model_alias():
    ep = providers.resolve(_REG, "gpt4omini", None, None, "KEY")
    assert ep["base_url"].endswith("openai.com/v1")
    assert ep["model"] == "gpt-4o-mini"
    assert ep["context_window"] == 128000 and ep["max_tokens"] == 4096
    assert ep["api_key"] == "KEY"


def test_resolve_raw_model_falls_back_to_default_provider():
    ep = providers.resolve(_REG, "llama3.2:3b", None, None, None, default_provider="ollama")
    assert ep["base_url"].endswith("11434/v1")
    assert ep["model"] == "llama3.2:3b"


def test_resolve_base_url_override_wins():
    ep = providers.resolve(_REG, "gpt4omini", None, "http://local/v1", None)
    assert ep["base_url"] == "http://local/v1" and ep["model"] == "gpt-4o-mini"


def test_resolve_unknown_provider_raises():
    with pytest.raises(ValueError):
        providers.resolve(_REG, "x", "nope", None, None)


def test_shipped_providers_json_is_valid():
    reg = providers.load()                 # the real providers.json next to the code
    assert reg["providers"] and reg["models"]
    for name, m in reg["models"].items():
        assert m["provider"] in reg["providers"], f"{name} -> unknown provider"

def test_retroactive_edit_gold_verifies_end_to_end():
    p = generators._mk("retroactive_edit", 3, 7, 0, False, "base", "g")
    assert generators.verify_gold(p)


def test_retroactive_edit_changes_gold_vs_unedited():
    p = generators._mk("retroactive_edit", 3, 7, 0, False, "base", "g")
    # The prompt contains an "Actually..." edit clause.
    assert "Actually," in p.prompt
    # A corrupted gold must fail verification.
    assert not generators.verify_gold(dataclasses.replace(p, gold=str(int(p.gold) + 1)))
