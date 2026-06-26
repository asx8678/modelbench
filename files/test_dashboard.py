"""Tests for the terminal dashboard (dashboard.py) and runtime_stats.

All rendering is exercised as pure string output with capabilities forced, so the
suite needs no TTY, no color, and no matplotlib — same philosophy as the rest of
the bench tests (driven through the deterministic mock)."""

import io

import pytest

import dashboard
import generators
import metrics
import runner
import storage


# --------------------------------------------------------------- fixtures
def _run(con, run_id, *, families=("arithmetic", "knights_knaves", "ordering"),
         mock="noisy", samples=1, confidence=True, **gen):
    items = generators.build_dataset(list(families), 1, 3, 4, verify=False, **gen)
    storage.save_dataset(con, items)
    cfg = dict(base_url="", api_key="", model="mock", temperature=0.0,
               max_tokens=1024, context_window=None, n=samples, workers=1,
               timeout=120, retries=2, ask_confidence=confidence, resume=False,
               mock=mock, dataset_tag="", capabilities=[])
    runner.run(con, run_id, storage.load_dataset(con), cfg)
    return metrics.compute(con, run_id)


def _caps(color=False, unicode=True, width=88):
    return dashboard.Caps(color=color, unicode=unicode, width=width)


# ------------------------------------------------------- capability detection
def test_no_color_env_disables_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    tty = io.StringIO()
    tty.isatty = lambda: True            # even on a TTY, NO_COLOR wins
    assert dashboard.detect_caps(tty).color is False


def test_non_tty_stream_is_plain_by_default(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    caps = dashboard.detect_caps(io.StringIO())      # StringIO: not a tty, no encoding
    assert caps.color is False and caps.unicode is False


def test_force_overrides_and_width_clamp():
    caps = dashboard.detect_caps(io.StringIO(), force_color=True, force_ascii=False, width=5)
    assert caps.color is True and caps.unicode is True
    assert caps.width == 48                           # clamped up to the minimum


# --------------------------------------------------------------- primitives
def test_meter_unicode_width_is_exact():
    paint = dashboard._Paint(False)
    for frac in (0.0, 0.27, 0.5, 0.999, 1.0):
        assert dashboard._visible_len(dashboard.meter(frac, 16, paint)) == 16


def test_meter_fill_is_proportional():
    paint = dashboard._Paint(False)
    assert dashboard.meter(1.0, 10, paint).count("█") == 10
    assert dashboard.meter(0.0, 10, paint).count("█") == 0
    assert dashboard.meter(0.5, 10, paint).count("█") == 5


def test_meter_ascii_form_is_bracketed():
    paint = dashboard._Paint(False)
    m = dashboard.meter(0.5, 10, paint, unicode=False)
    assert m.startswith("[") and m.endswith("]")
    assert m.count("#") == 5 and m.count("-") == 5


def test_sparkline_is_monotonic_in_value():
    paint = dashboard._Paint(False)
    spark = dashboard.sparkline([0.0, 0.25, 0.5, 0.75, 1.0], paint)
    heights = [dashboard._SPARK_U.index(c) for c in spark]
    assert heights == sorted(heights)
    assert heights[0] == 0 and heights[-1] == len(dashboard._SPARK_U) - 1


def test_sparkline_empty_and_constant():
    paint = dashboard._Paint(False)
    assert dashboard.sparkline([], paint) == ""
    assert dashboard.sparkline([0.5, 0.5, 0.5], paint) == dashboard._SPARK_U[4] * 3


def test_visible_len_ignores_ansi():
    paint = dashboard._Paint(True)
    colored = paint.fg("hello", 41)
    assert "\x1b[" in colored
    assert dashboard._visible_len(colored) == 5


# --------------------------------------------------------------- render_run
def test_render_run_plain_has_no_ansi_and_is_ascii():
    con = storage.connect(":memory:")
    res = _run(con, "r")
    lines = dashboard.render_run(res, {"model": "mock"},
                                 metrics.runtime_stats(con, "r"),
                                 _caps(color=False, unicode=False))
    out = "\n".join(lines)
    assert "\x1b[" not in out                          # no color escapes
    assert all(ord(c) < 128 for c in out)              # encodable on an ASCII terminal


def test_render_run_contains_key_sections():
    con = storage.connect(":memory:")
    res = _run(con, "myrun")
    out = "\n".join(dashboard.render_run(res, {"model": "m"},
                                         metrics.runtime_stats(con, "myrun"), _caps()))
    for token in ("myrun", "HEADLINE", "ACCURACY BY FAMILY", "arithmetic",
                  "CALIBRATION", "RUNTIME"):
        assert token in out


def test_render_run_color_escapes_are_balanced():
    con = storage.connect(":memory:")
    res = _run(con, "r")
    out = "\n".join(dashboard.render_run(res, {"model": "m"},
                                         metrics.runtime_stats(con, "r"),
                                         _caps(color=True)))
    opens = out.count("\x1b[38;5;") + out.count("\x1b[1m") + out.count("\x1b[2m")
    assert opens > 0 and opens == out.count("\x1b[0m")  # every open is reset


def test_render_run_handles_error_dict():
    lines = dashboard.render_run({"error": "no responses for run"}, caps=_caps())
    assert len(lines) == 1 and "no responses" in lines[0]


def test_render_run_minimal_run_omits_optional_sections():
    # A bare run (no calibration / pass@k / distractor, rstats=None) must render the
    # core sections and silently drop the optional ones rather than crash.
    res = {
        "run_id": "min", "n_items": 10, "samples_per_item": 1,
        "coverage": {"n_items": 10, "answered": 10, "errored": 0, "coverage": 1.0},
        "overall_accuracy": 0.5, "overall_accuracy_strict": 0.5, "fallback_reliance": 0.0,
        "accuracy_by_family": {"arithmetic": 0.5}, "acc_above_chance": {"arithmetic": 0.5},
        "chance_baseline": {"arithmetic": 0.0},
        "degradation": {"arithmetic": {1: {"mean": 0.5, "lo": 0.2, "hi": 0.8, "n": 4, "std": 0.1}}},
        "distractibility": {}, "invariance": {"groups": 0},
        "calibration": None, "passk": None, "confabulation_rate": None,
        "frontier_headroom": None,
    }
    out = "\n".join(dashboard.render_run(res, {"model": "m"}, None, _caps()))
    assert "HEADLINE" in out and "ACCURACY BY FAMILY" in out
    for omitted in ("CALIBRATION", "DISTRACTIBILITY", "RUNTIME"):
        assert omitted not in out


# ------------------------------------------------------------- render_compare
def test_render_compare_includes_all_run_labels():
    con = storage.connect(":memory:")
    r1 = _run(con, "alpha", families=("arithmetic",))
    r2 = _run(con, "beta", families=("arithmetic",))
    out = "\n".join(dashboard.render_compare([r1, r2], ["alpha", "beta"], caps=_caps()))
    assert "alpha" in out and "beta" in out
    assert "overall acc" in out


def test_render_compare_best_value_gets_full_meter_even_when_lower_is_better():
    # Construct two results differing only in ECE; the lower (better) ECE must get
    # the longer meter despite being the smaller number.
    base = dict(overall_accuracy=0.5, overall_accuracy_strict=0.5,
                coverage={"coverage": 1.0}, invariance={"groups": 0},
                confabulation_rate=None, passk=None)
    good = {**base, "calibration": {"ece": 0.05}}
    bad = {**base, "calibration": {"ece": 0.40}}
    lines = dashboard.render_compare([good, bad], ["good", "bad"], caps=_caps())
    ece_line = next(l for l in lines if "calibration ECE" in l)
    # the better (0.05) cell should have a fuller bar than the worse (0.40) cell
    full = ece_line.split("0.05")[0].count("█")        # blocks in the good (left) meter
    worse = ece_line.split("0.05")[1].count("█")       # blocks in the bad (right) meter
    assert full > worse


# --------------------------------------------------------------- runtime_stats
def test_runtime_stats_counts_and_percentiles():
    con = storage.connect(":memory:")
    # three OK calls with known latencies/tokens + one error
    storage.save_response(con, "r", "i1", 0, "ANSWER: 1", "1", 1, None, 100, 10, 20)
    storage.save_response(con, "r", "i2", 0, "ANSWER: 2", "2", 1, None, 200, 30, 40)
    storage.save_response(con, "r", "i3", 0, "ANSWER: 3", "3", 1, None, 300, 50, 60)
    storage.save_response(con, "r", "i4", 0, storage.ERROR_MARKER, None, None, None, 0,
                          None, None)
    con.commit()
    rs = metrics.runtime_stats(con, "r")
    assert rs["n_calls"] == 4 and rs["errored"] == 1
    assert rs["latency_p50_ms"] == 150          # median of [0,100,200,300] (np interpolates)
    assert rs["tokens_available"] is True
    assert rs["completion_tokens_total"] == 120 and rs["prompt_tokens_total"] == 90
    assert rs["completion_tokens_mean"] == 40


def test_runtime_stats_tokens_unavailable_when_unreported():
    con = storage.connect(":memory:")
    storage.save_response(con, "r", "i1", 0, "ANSWER: 1", "1", 1, None, 5, None, None)
    con.commit()
    rs = metrics.runtime_stats(con, "r")
    assert rs["tokens_available"] is False
    assert rs["completion_tokens_total"] == 0


def test_runtime_stats_none_for_unknown_run():
    con = storage.connect(":memory:")
    assert metrics.runtime_stats(con, "nope") is None


def test_runtime_stats_reasoning_tokens_aggregated():
    con = storage.connect(":memory:")
    # two correct calls, 100 completion tokens each (200 total)
    storage.save_response(con, "r", "i1", 0, "ANSWER: 1", "1", 1, None, 10, 5, 100)
    storage.save_response(con, "r", "i2", 0, "ANSWER: 2", "1", 1, None, 10, 5, 100)
    # provider exposed 40 reasoning tokens on each (80 total)
    for iid in ("i1", "i2"):
        storage.save_telemetry(con, "r", iid, 0, reasoning_token_source="native_usage",
                               completion_tokens=100, reasoning_tokens=40)
    con.commit()
    rs = metrics.runtime_stats(con, "r")
    assert rs["reasoning_available"] is True
    assert rs["reasoning_tokens_total"] == 80 and rs["reasoning_tokens_mean"] == 40
    assert abs(rs["reasoning_fraction"] - 0.4) < 1e-9          # 80 / 200 completion
    assert abs(rs["reasoning_tokens_per_correct"] - 40.0) < 1e-9   # 80 / 2 correct


def test_runtime_stats_reasoning_absent_when_provider_silent():
    con = storage.connect(":memory:")
    storage.save_response(con, "r", "i1", 0, "ANSWER: 1", "1", 1, None, 10, 5, 100)
    storage.save_telemetry(con, "r", "i1", 0, reasoning_token_source="unavailable",
                           completion_tokens=100, reasoning_tokens=0)
    con.commit()
    rs = metrics.runtime_stats(con, "r")
    assert rs["reasoning_available"] is False
    assert rs["reasoning_fraction"] is None
    assert rs["reasoning_tokens_per_correct"] is None


def test_runtime_stats_intelligence_metrics():
    """Test reasoning_by_difficulty, efficiency (per_1k), and effort scaling."""
    con = storage.connect(":memory:")
    # Build dataset with 2 difficulties: difficulty 1 (easy) and difficulty 3 (hard).
    # Easy items (difficulty 1): less reasoning, higher accuracy (less overthinking).
    # Hard items (difficulty 3): more reasoning, lower accuracy (but still trying).
    items = generators.build_dataset(["arithmetic"], 1, 3, 2, verify=False)
    storage.save_dataset(con, items)

    # Extract the generated item IDs from the dataset and partition by difficulty
    dataset = storage.load_dataset(con)
    easy_items = [d["item_id"] for d in dataset if d["difficulty"] == 1][:2]
    hard_items = [d["item_id"] for d in dataset if d["difficulty"] == 3][:2]

    # Manually create responses and telemetry for controlled testing.
    # Easy items: both correct, 30 reasoning tokens each
    storage.save_response(con, "r", easy_items[0], 0, "ANSWER: 1", "1", 1, None, 10, 5, 100)
    storage.save_response(con, "r", easy_items[1], 0, "ANSWER: 2", "2", 1, None, 10, 5, 100)
    storage.save_telemetry(con, "r", easy_items[0], 0, reasoning_token_source="native_usage",
                           completion_tokens=100, reasoning_tokens=30)
    storage.save_telemetry(con, "r", easy_items[1], 0, reasoning_token_source="native_usage",
                           completion_tokens=100, reasoning_tokens=30)

    # Hard items: 1 correct, 50 reasoning tokens each
    storage.save_response(con, "r", hard_items[0], 0, "ANSWER: 3", "3", 1, None, 10, 5, 100)
    storage.save_response(con, "r", hard_items[1], 0, "ANSWER: 4", "5", 0, None, 10, 5, 100)
    storage.save_telemetry(con, "r", hard_items[0], 0, reasoning_token_source="native_usage",
                           completion_tokens=100, reasoning_tokens=50)
    storage.save_telemetry(con, "r", hard_items[1], 0, reasoning_token_source="native_usage",
                           completion_tokens=100, reasoning_tokens=50)

    con.commit()

    rs = metrics.runtime_stats(con, "r")

    # Check that reasoning is available
    assert rs["reasoning_available"] is True
    assert rs["reasoning_tokens_total"] == 160  # 30+30+50+50

    # Check reasoning_by_difficulty
    by_diff = rs["reasoning_by_difficulty"]
    assert 1 in by_diff and 3 in by_diff
    # Easy (difficulty 1): mean 30, accuracy 1.0, n=2
    assert by_diff[1]["mean_reasoning_tokens"] == 30
    assert by_diff[1]["accuracy"] == 1.0
    assert by_diff[1]["n"] == 2
    # Hard (difficulty 3): mean 50, accuracy 0.5, n=2
    assert by_diff[3]["mean_reasoning_tokens"] == 50
    assert by_diff[3]["accuracy"] == 0.5
    assert by_diff[3]["n"] == 2

    # Check efficiency: 3 correct out of 160 reasoning tokens
    # 3 * 1000 / 160 = 18.75
    assert abs(rs["reasoning_correct_per_1k"] - 18.75) < 1e-6

    # Check effort scaling: easy/hard = 30/50 = 0.6
    # This indicates the model uses less reasoning on easy items (good!)
    assert abs(rs["reasoning_effort_scaling"] - 0.6) < 1e-6


def test_runtime_stats_intelligence_metrics_absent_without_reasoning():
    """When no reasoning data, intelligence metrics should be empty/None."""
    con = storage.connect(":memory:")
    storage.save_response(con, "r", "i1", 0, "ANSWER: 1", "1", 1, None, 10, 5, 100)
    con.commit()
    rs = metrics.runtime_stats(con, "r")
    assert rs["reasoning_available"] is False
    assert rs["reasoning_by_difficulty"] == {}
    assert rs["reasoning_correct_per_1k"] is None
    assert rs["reasoning_effort_scaling"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
