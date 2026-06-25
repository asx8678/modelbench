import re
import io
import sys
import json
import contextlib
import argparse
import sqlite3

import cli
import storage
import generators
import runner
import metrics


def _parse_run_args(argv):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--db", default="bench.db")
    r.add_argument("--model", default=None)
    r.add_argument("--provider", default=None)
    r.add_argument("--base-url", default=None)
    r.add_argument("--api-key", default=None)
    r.add_argument("--run-id", default=None)
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=None)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--timeout", type=int, default=120)
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--confidence", action="store_true")
    r.add_argument("--resume", action="store_true")
def test_dataset_tag_appears_in_argparse_help():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            cli._parse_args(["run", "--help"])
        except SystemExit:
            pass
    help_text = buf.getvalue()
    assert "--dataset-tag" in help_text

def _make_run(con, run_id, dataset_tag=""):
    ds = generators.build_dataset(["arithmetic"], 1, 2, 3)
    storage.save_dataset(con, ds)
    items = storage.load_dataset(con)
    cfg = dict(
        base_url="", api_key="", model="mock",
        temperature=0.0, max_tokens=1024, context_window=None,
        n=1, workers=1, timeout=120, retries=2,
        ask_confidence=False, resume=False, mock="perfect",
        dataset_tag=dataset_tag,
        capabilities=[])
    runner.run(con, run_id, items, cfg)
def test_dataset_tag_recorded_in_run_params():
    con = storage.connect(":memory:")
    _make_run(con, "run-tagged", dataset_tag="v1")
    row = con.execute("SELECT params FROM runs WHERE run_id=?", ("run-tagged",)).fetchone()
    params = json.loads(row["params"])
    assert params["dataset_tag"] == "v1"


def test_report_warns_on_dataset_tag_mismatch(capsys):
    con = storage.connect(":memory:")
    _make_run(con, "run-a", dataset_tag="v1")
    _make_run(con, "run-b", dataset_tag="v2")
    cli._warn_if_dataset_tags_mismatch(con, ["run-a", "run-b"])
    captured = capsys.readouterr()
    assert "warning" in captured.out.lower()
    assert "different dataset tags" in captured.out
    assert "run-a" in captured.out
    assert "run-b" in captured.out


def test_report_silent_when_dataset_tags_match(capsys):
    con = storage.connect(":memory:")
    _make_run(con, "run-a", dataset_tag="v1")
    _make_run(con, "run-b", dataset_tag="v1")
    cli._warn_if_dataset_tags_mismatch(con, ["run-a", "run-b"])
    captured = capsys.readouterr()
    assert "different dataset tags" not in captured.out
