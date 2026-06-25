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
    for fam, offset, diff in [
        ("ordering", 2, 3),
        ("knights_knaves", 3, 4),
        ("logic_grid", 2, 5),
    ]:
        items = [generators._mk(fam, diff, 0, 0, False, "base", "g")]
        con = storage.connect(":memory:")
        storage.save_dataset(con, items)
        for p in items:
            _store(con, "r", p.item_id, 0, f"ANSWER: {p.gold}", p.gold, 1)
        res = metrics.compute(con, "r")
        assert fam in res["chance_baseline"]
        assert res["chance_baseline"][fam] == pytest.approx(1.0 / (diff + offset))


def test_zero_chance_families_have_zero_baseline():
    for fam in ["arithmetic", "state_tracking", "sequences",
                "composed", "retroactive_edit", "multi_turn_inject"]:
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
