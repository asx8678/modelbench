"""Tests for behavioral-uncertainty metrics."""

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


def _store(con, run_id, item_id, sample_idx, raw, parsed, correct,
           confidence=None):
    storage.save_response(con, run_id, item_id, sample_idx, raw, parsed,
                          correct, confidence, 1, None, None)


def test_disagreement_entropy_zero_when_all_samples_agree():
    con, items = _dataset(["arithmetic"], 2, 2, 3)
    for p in items:
        for s in range(3):
            _store(con, "r", p.item_id, s, f"ANSWER: {p.gold}", p.gold, 1)
    res = metrics.compute(con, "r")
    bu = res["behavioral_uncertainty"]
    assert bu is not None
    assert bu["disagreement_entropy"] == pytest.approx(0.0)


def test_disagreement_entropy_positive_when_samples_disagree():
    con, items = _dataset(["arithmetic"], 2, 2, 3)
    for p in items:
        answers = [str(int(p.gold) + i) for i in range(3)]
        for s, ans in enumerate(answers):
            _store(con, "r", p.item_id, s, f"ANSWER: {ans}", ans, 0)
    res = metrics.compute(con, "r")
    bu = res["behavioral_uncertainty"]
    assert bu is not None
    assert bu["disagreement_entropy"] > 0.0
def test_selfconsistency_gap_between_minus_one_and_one():
    con, items = _dataset(["arithmetic"], 2, 2, 3)
    for p in items:
        # first sample wrong, others correct and consistent -> maj@k > pass@1
        _store(con, "r", p.item_id, 0, "ANSWER: 999999", "999999", 0)
        for s in range(1, 3):
            _store(con, "r", p.item_id, s, f"ANSWER: {p.gold}", p.gold, 1)
    res = metrics.compute(con, "r")
    gap = res["behavioral_uncertainty"]["selfconsistency_gap"]
    assert gap is not None
    assert -1.0 <= gap <= 1.0


def test_stated_confidence_ece_between_zero_and_one():
    con, items = _dataset(["arithmetic"], 4, 6, 1)
    for i, p in enumerate(items):
        # alternate confidence bands that are deliberately miscalibrated
        conf = 90 if i % 2 == 0 else 50
        correct = 1 if i % 2 == 0 else 0
        _store(con, "r", p.item_id, 0, f"ANSWER: {p.gold}", p.gold,
               correct, confidence=conf)
    res = metrics.compute(con, "r")
    ece = res["behavioral_uncertainty"]["stated_confidence_ece"]
    assert ece is not None
    assert 0.0 <= ece <= 1.0
