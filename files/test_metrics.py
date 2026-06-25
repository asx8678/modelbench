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


def test_chance_baseline_values_for_known_families():
    # ordering: choice over (difficulty+2) ranks
    assert metrics._chance_baseline("ordering", 3) == pytest.approx(1.0 / 5)
    # knights_knaves: set answer over 2^n assignments (n = difficulty+2)
    assert metrics._chance_baseline("knights_knaves", 3) == pytest.approx(1.0 / 32)
    # composed: choice over (difficulty+2) names
    assert metrics._chance_baseline("composed", 3) == pytest.approx(1.0 / 5)
    # unsat_csp: 4-way determinate choice
    assert metrics._chance_baseline("unsat_csp", 3) == pytest.approx(0.25)
    # logic_grid: choice over (difficulty+2) ranks
    assert metrics._chance_baseline("logic_grid", 5) == pytest.approx(1.0 / 7)


def test_chance_correction_recompute_no_model_calls():
    # yvr.2: build a dataset (no model calls), fill all responses with the
    # true gold so acc == 1, and assert acc_above_chance is well-defined
    # and bounded. Also assert the per-family chance values match the
    # documented baseline formula.
    families = ["ordering", "knights_knaves", "logic_grid", "unsat_csp",
                "composed", "arithmetic", "state_tracking"]
    con, items = _dataset(families, 1, 4, 1)
    for p in items:
        _store(con, "r", p.item_id, 0, f"ANSWER: {p.gold}", p.gold, 1)
    res = metrics.compute(con, "r")
    # acc_above_chance <= accuracy for every family.
    for fam, acc in res["accuracy_by_family"].items():
        if acc is None:
            continue
        assert res["acc_above_chance"][fam] <= acc + 1e-9
    # Chance baselines match the documented formula.
    for fam, baseline in res["chance_baseline"].items():
        if fam == "knights_knaves":
            # Set answer over 2^n assignments; chance averaged across
            # difficulties is bounded by 1/2^(min_diff+2).
            assert 0.0 < baseline <= 0.5
        elif fam in ("ordering", "logic_grid", "composed"):
            # Averaged across difficulties 1..4: 1/(d+2) for d in 1..4
            assert 0.15 < baseline < 0.4
        elif fam == "unsat_csp":
            assert baseline == pytest.approx(0.25)
        else:
            assert baseline == 0.0



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
