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
import providers


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


# ---- setup wizard / registry (bench-9gn) + mock-run regression (bench-7cs) ----

def test_provider_name_from_url():
    assert providers.provider_name_from_url("https://api.openai.com/v1") == "openai"
    assert providers.provider_name_from_url("https://openrouter.ai/api/v1") == "openrouter"
    assert providers.provider_name_from_url("http://localhost:11434/v1") == "local"
    assert providers.provider_name_from_url("http://127.0.0.1:8000/v1") == "local"


def test_register_model_new_provider_and_model():
    reg = {"providers": {}, "models": {}}
    reg, prov = providers.register_model(
        reg, alias="m1", base_url="https://api.openai.com/v1",
        model_id="gpt-4o-mini", api_key_env="OPENAI_API_KEY",
        context_window=128000, max_tokens=4096)
    assert prov == "openai"
    assert reg["providers"]["openai"]["base_url"] == "https://api.openai.com/v1"
    assert reg["providers"]["openai"]["api_key_env"] == "OPENAI_API_KEY"
    assert reg["models"]["m1"] == {
        "provider": "openai", "model": "gpt-4o-mini",
        "context_window": 128000, "max_tokens": 4096}


def test_register_model_reuses_provider_with_same_url():
    reg = {"providers": {}, "models": {}}
    reg, p1 = providers.register_model(reg, alias="a", base_url="https://x.ai/v1", model_id="m-a")
    reg, p2 = providers.register_model(reg, alias="b", base_url="https://x.ai/v1", model_id="m-b")
    assert p1 == p2
    assert len(reg["providers"]) == 1
    assert set(reg["models"]) == {"a", "b"}


def test_register_model_literal_and_env_keys_are_mutually_exclusive():
    reg = {"providers": {}, "models": {}}
    reg, _ = providers.register_model(reg, alias="a", base_url="https://x.ai/v1",
                                      model_id="m", api_key="sk-literal")
    prov = reg["providers"][reg["models"]["a"]["provider"]]
    assert prov["api_key"] == "sk-literal" and "api_key_env" not in prov
    # re-registering the same endpoint with an env reference drops the literal
    providers.register_model(reg, alias="b", base_url="https://x.ai/v1",
                             model_id="m2", api_key_env="X_KEY")
    assert prov["api_key_env"] == "X_KEY" and "api_key" not in prov


def test_register_model_distinct_urls_get_unique_provider_names():
    reg = {"providers": {"openai": {"base_url": "https://api.openai.com/v1"}}, "models": {}}
    reg, prov = providers.register_model(reg, alias="a",
                                         base_url="https://api.openai.com/v2", model_id="m")
    assert prov != "openai"
    assert reg["providers"][prov]["base_url"] == "https://api.openai.com/v2"


def test_register_model_overwrites_model_alias():
    reg = {"providers": {}, "models": {}}
    providers.register_model(reg, alias="a", base_url="https://x.ai/v1", model_id="old")
    providers.register_model(reg, alias="a", base_url="https://x.ai/v1", model_id="new")
    assert reg["models"]["a"]["model"] == "new"


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "providers.json")
    reg = {"providers": {}, "models": {}}
    providers.register_model(reg, alias="a", base_url="https://x.ai/v1", model_id="m",
                             context_window=1000)
    providers.save(reg, path)
    loaded = providers.load(path)
    assert loaded["models"]["a"]["model"] == "m"
    assert loaded["models"]["a"]["context_window"] == 1000


def test_cmd_run_mock_does_not_crash(tmp_path, capsys):
    """Regression for bench-7cs: cmd_run referenced an undefined `ep` in --mock mode."""
    db = str(tmp_path / "bench.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 2, 2))
    con.commit()
    args = cli._parse_args(["run", "--db", db, "--mock", "perfect", "--run-id", "t"])
    args.func(args)                       # must not raise UnboundLocalError
    out = capsys.readouterr().out
    assert "running" in out and "done in" in out
    n = storage.connect(db).execute(
        "SELECT COUNT(*) c FROM responses WHERE run_id='t'").fetchone()["c"]
    assert n > 0


def test_fmt_dur():
    assert runner._fmt_dur(0) == "0:00"
    assert runner._fmt_dur(65) == "1:05"
    assert runner._fmt_dur(3661) == "1:01:01"


def test_progress_redirected_emits_counts_and_summary():
    buf = io.StringIO()                   # StringIO.isatty() -> False -> plain-line mode
    p = runner._Progress(10, stream=buf)
    p.update(0, 0, 0)
    for i in range(1, 11):
        p.update(i, i, 0)
    p.finish(10, 0)
    out = buf.getvalue()
    assert "10/10" in out and "it/s" in out
    assert "done in" in out and "\r" not in out


def test_progress_tty_uses_bar_and_carriage_return():
    class _TTY(io.StringIO):
        def isatty(self):
            return True
    buf = _TTY()
    p = runner._Progress(4, stream=buf)
    p.update(0, 0, 0)
    p.update(4, 4, 0)                     # final update always paints despite throttle
    p.finish(4, 0)
    out = buf.getvalue()
    assert "\r" in out and "█" in out and "100.0%" in out and "done in" in out
