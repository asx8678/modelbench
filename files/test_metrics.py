"""Tests for chance-corrected accuracy and grading fragility metrics."""

import pytest
import generators
import metrics
import storage


def _dataset(families, n_per_family, max_difficulty, reps, **kw):
    items = generators.build_dataset(
        families, n_per_family, max_difficulty, reps, verify=False, **kw)
    con = storage.connect(":memory:")
    storage.save_dataset(con, items)
    return con, items


def _store(con, run_id, item_id, sample_idx, raw, parsed, correct):
    storage.save_response(con, run_id, item_id, sample_idx, raw, parsed,
                          correct, None, 1, None, None)


def test_acc_above_chance_is_at_most_accuracy():
    con, items = _dataset(
        ["ordering", "knights_knaves", "logic_grid", "arithmetic"],
        2, 4, 1)
    for p in items:
        _store(con, "r", p.item_id, 0, f"ANSWER: {p.gold}", p.gold, 1)
    res = metrics.compute(con, "r")
    for fam, acc in res["accuracy_by_family"].items():
        assert res["acc_above_chance"][fam] <= acc + 1e-9


def test_chance_baseline_is_empirical_best_constant_guess():
    # Unbounded-integer families have no winning constant guess -> chance 0,
    # even when golds repeat (composed is unbounded; logic_grid is NOT).
    assert metrics._family_chance("arithmetic", ["3", "3", "3"]) == 0.0
    assert metrics._family_chance("composed", ["6", "51", "12"]) == 0.0
    # Bounded families use the empirical modal-answer frequency.
    assert metrics._modal_frequency(["a", "a", "b"]) == pytest.approx(2 / 3)
    assert metrics._modal_frequency([]) == 0.0
    assert metrics._family_chance(
        "unsat_csp", ["knight", "knight", "knave", "UNDETERMINED"]) == pytest.approx(0.5)
    # logic_grid's gold is a bounded integer (a floor), so it is scored
    # empirically too -- the bug bench-r6u fixes is treating it as unbounded.
    assert metrics._family_chance("logic_grid", ["1", "2", "1", "3"]) == pytest.approx(0.5)


def test_chance_correction_recompute_no_model_calls():
    # yvr.2 / bench-r6u: build a dataset (no model calls), fill all responses
    # with the true gold so acc == 1, and assert acc_above_chance is
    # well-defined and bounded. Chance is the empirical best-constant-guess
    # (modal-answer frequency) for bounded families, and 0 for unbounded-int
    # families (composed is now an unbounded int -> 0, not 1/(d+2)).
    families = ["ordering", "knights_knaves", "logic_grid", "unsat_csp",
                "composed", "arithmetic", "state_tracking"]
    con, items = _dataset(families, 2, 4, 2)
    for p in items:
        _store(con, "r", p.item_id, 0, f"ANSWER: {p.gold}", p.gold, 1)
    res = metrics.compute(con, "r")
    # acc_above_chance <= accuracy for every family.
    for fam, acc in res["accuracy_by_family"].items():
        if acc is None:
            continue
        assert res["acc_above_chance"][fam] <= acc + 1e-9
    int_families = {"composed", "arithmetic", "state_tracking"}
    for fam, baseline in res["chance_baseline"].items():
        if fam in int_families:
            # Unbounded integer answer -> no winning constant guess.
            assert baseline == 0.0
        else:
            # Bounded (choice/set) family -> a real modal-answer frequency.
            assert 0.0 < baseline <= 1.0



def test_zero_chance_families_have_zero_baseline():
    for fam in ["arithmetic", "state_tracking", "sequences",
                "retroactive_edit", "multi_turn_inject"]:
        con, items = _dataset([fam], 2, 3, 1)
        for p in items:
            _store(con, "r", p.item_id, 0, f"ANSWER: {p.gold}", p.gold, 1)
        res = metrics.compute(con, "r")
        assert fam in res["chance_baseline"]
        assert res["chance_baseline"][fam] == 0.0
        assert res["acc_above_chance"][fam] == res["accuracy_by_family"][fam]


def test_frontier_headroom_bounds_and_matches_oracle_gap():
    con, items = _dataset(["arithmetic"], 3, 3, 1)
    for p in items:
        # first sample is wrong, second is right -> pass@1 < oracle
        _store(con, "r", p.item_id, 0, "ANSWER: 999999", "999999", 0)
        _store(con, "r", p.item_id, 1, f"ANSWER: {p.gold}", p.gold, 1)
    res = metrics.compute(con, "r")
    headroom = res["frontier_headroom"]
    assert 0.0 <= headroom <= 1.0
def test_grading_fragility_bounds_and_detects_disagreement():
    con, items = _dataset(["arithmetic"], 1, 2, 1)
    p = items[0]
    # marker line says 5, but a later number in the prose makes the fallback differ
    raw = "ANSWER: 5\nActually the answer is 7."
    _store(con, "r", p.item_id, 0, raw, "5", 0)
    res = metrics.compute(con, "r")
    assert 0.0 <= res["grading_fragility"] <= 1.0
    # one sample, one disagreement -> fragility == 1.0
    assert res["grading_fragility"] == pytest.approx(1.0)


def test_grading_fragility_zero_when_marker_and_fallback_agree():
    con, items = _dataset(["arithmetic"], 2, 2, 1)
    for p in items:
        _store(con, "r", p.item_id, 0, f"ANSWER: {p.gold}", p.gold, 1)
    res = metrics.compute(con, "r")
    assert res["grading_fragility"] == pytest.approx(0.0)

def test_false_undetermined_rate_bounds_and_counts_unique_items():
    # false_undetermined_rate should only consider items whose gold is NOT
    # UNDETERMINED / NO_SOLUTION, and should count sample-0 parsed answers that
    # ARE UNDETERMINED / NO_SOLUTION.
    con, items = _dataset(["unsat_csp"], 3, 3, 2)
    run_id = "fudr"
    for p in items:
        _store(con, run_id, p.item_id, 0, f"ANSWER: {p.gold}", p.gold, 1)
        _store(con, run_id, p.item_id, 1, f"ANSWER: {p.gold}", p.gold, 1)
    res = metrics.compute(con, run_id)
    assert "false_undetermined_rate" in res
    assert res["false_undetermined_rate"] == pytest.approx(0.0)
    # Now inject a false UNDETERMINED on one unique item.
    unique_iid = next(i for i, m in _meta_items(con, run_id)
                      if m["gold"] not in ("UNDETERMINED", "NO_SOLUTION"))
    _store(con, run_id, unique_iid, 0, "ANSWER: UNDETERMINED", "UNDETERMINED", 0)
    res2 = metrics.compute(con, run_id)
    unique_count = sum(1 for _, m in _meta_items(con, run_id)
                       if m["gold"] not in ("UNDETERMINED", "NO_SOLUTION"))
    assert res2["false_undetermined_rate"] == pytest.approx(1.0 / unique_count)


def _meta_items(con, run_id):
    rows = metrics._fetch(con, run_id)
    meta = {}
    for r in rows:
        meta.setdefault(r["item_id"], {k: r[k] for k in ("family", "difficulty", "probe", "grp", "gold", "answer_type", "choices")})
    return list(meta.items())
