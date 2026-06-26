"""Tests for runreport: per-run solved/unsolved classification and artifact output."""

import storage
import runreport


def _s(correct, parsed, raw="x", parse_source="marker"):
    return {"correct": correct, "parsed": parsed, "raw": raw, "parse_source": parse_source}


def test_classify_errored_when_any_sample_failed():
    out, reason = runreport._classify([_s(None, None, raw=storage.ERROR_MARKER)], "7")
    assert out == runreport.ERRORED
    assert "no answer" in reason


def test_classify_solved_correct():
    out, reason = runreport._classify([_s(True, "7")], "7")
    assert out == runreport.SOLVED and reason == "correct"


def test_classify_solved_via_fallback_is_flagged():
    out, reason = runreport._classify([_s(True, "7", parse_source="fallback")], "7")
    assert out == runreport.SOLVED and "fallback" in reason


def test_classify_wrong_reports_got_vs_expected():
    out, reason = runreport._classify([_s(False, "5")], "7")
    assert out == runreport.WRONG and "'5'" in reason and "'7'" in reason


def test_classify_wrong_no_parse():
    out, reason = runreport._classify([_s(False, None, parse_source="none")], "7")
    assert out == runreport.WRONG and "no parseable answer" in reason


def test_classify_majority():
    samples = [_s(True, "7"), _s(True, "7"), _s(False, "5")]
    out, _ = runreport._classify(samples, "7")
    assert out == runreport.SOLVED


def test_build_writes_all_artifacts(tmp_path):
    import generators
    db = str(tmp_path / "b.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 2, 2))
    con.commit()
    # store a couple of responses by hand (one right, one wrong) for a fake run
    items = storage.load_dataset(con)
    storage.new_run(con, "r", "m", "", {})
    storage.save_response(con, "r", items[0]["item_id"], 0, "ANSWER: 1", "1",
                          items[0]["gold"] == "1", 90, 10, 0, 0,
                          metadata={"parse_source": "marker"})
    storage.save_response(con, "r", items[1]["item_id"], 0, storage.ERROR_MARKER,
                          None, None, None, 5, None, None, metadata={})
    con.commit()

    folder = runreport.build(con, "r", "my-model", str(tmp_path / "out"))
    assert folder is not None
    names = {p.name for p in __import__("pathlib").Path(folder).iterdir()}
    assert "report.md" in names and "run.log" in names and "items.csv" in names
    assert any(n.startswith("my-model_") and n.endswith(".html") for n in names)
