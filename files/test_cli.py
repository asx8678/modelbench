import re
import io
import sys
import json
import types
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


# ---- edit a model: providers.edit_model + `cli.py edit` ----

def test_edit_model_changes_id_keeps_other_fields():
    reg = {"providers": {}, "models": {}}
    providers.register_model(reg, alias="a", base_url="https://x.ai/v1", model_id="old",
                             api_key_env="X_KEY", context_window=1000, max_tokens=512)
    reg, prov = providers.edit_model(reg, "a", model_id="new")
    assert reg["models"]["a"]["model"] == "new"
    assert reg["models"]["a"]["context_window"] == 1000     # untouched
    assert reg["models"]["a"]["max_tokens"] == 512
    assert reg["providers"][prov]["base_url"] == "https://x.ai/v1"
    assert reg["providers"][prov]["api_key_env"] == "X_KEY"  # key reference preserved


def test_edit_model_moves_endpoint_and_prunes_orphaned_provider():
    reg = {"providers": {}, "models": {}}
    reg, old = providers.register_model(reg, alias="a", base_url="https://old.ai/v1",
                                        model_id="m")
    reg, new = providers.edit_model(reg, "a", base_url="https://new.ai/v1")
    assert new != old
    assert reg["providers"][new]["base_url"] == "https://new.ai/v1"
    assert old not in reg["providers"]                      # last user moved -> pruned


def test_edit_model_moving_endpoint_keeps_provider_shared_by_another():
    reg = {"providers": {}, "models": {}}
    providers.register_model(reg, alias="a", base_url="https://x.ai/v1", model_id="m-a")
    providers.register_model(reg, alias="b", base_url="https://x.ai/v1", model_id="m-b")
    shared = reg["models"]["a"]["provider"]
    providers.edit_model(reg, "a", base_url="https://y.ai/v1")
    assert shared in reg["providers"]                       # 'b' still uses it -> kept
    assert reg["models"]["b"]["provider"] == shared


def test_edit_model_unknown_alias_raises():
    import pytest
    reg = {"providers": {}, "models": {}}
    with pytest.raises(KeyError):
        providers.edit_model(reg, "ghost", model_id="m")


def test_cmd_edit_unknown_model_exits_nonzero(tmp_path, monkeypatch):
    import pytest
    _save_one_model(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_edit(cli._parse_args(["edit", "--model", "ghost"]))


def test_edit_wizard_changes_model_id_via_prompts(tmp_path, monkeypatch):
    prov = _save_one_model(tmp_path, monkeypatch)
    reg = providers.load(prov)
    # keep endpoint (blank), new model id, keep ctx + max_tokens (blank),
    # don't change key (n), don't test connection (n)
    _scripted_input(monkeypatch, ["", "gpt-5", "", "", "n", "n"])
    alias = cli._edit_wizard(reg, prov, alias="m")
    assert alias == "m"
    assert providers.load(prov)["models"]["m"]["model"] == "gpt-5"


def test_edit_subcommand_registered():
    assert cli._parse_args(["edit", "--model", "m"]).func is cli.cmd_edit


# ---- delete a model: providers.delete_model + `cli.py remove` ----

def test_delete_model_removes_and_prunes_orphaned_provider():
    reg = {"providers": {}, "models": {}}
    providers.register_model(reg, alias="a", base_url="https://x.ai/v1", model_id="m")
    removed, pruned = providers.delete_model(reg, "a")
    assert removed is True
    assert "a" not in reg["models"]
    assert pruned and pruned not in reg["providers"]   # last user gone -> provider pruned


def test_delete_model_keeps_provider_shared_by_another_model():
    reg = {"providers": {}, "models": {}}
    providers.register_model(reg, alias="a", base_url="https://x.ai/v1", model_id="m-a")
    providers.register_model(reg, alias="b", base_url="https://x.ai/v1", model_id="m-b")
    removed, pruned = providers.delete_model(reg, "a")
    assert removed is True and pruned is None
    assert "a" not in reg["models"] and "b" in reg["models"]
    assert reg["models"]["b"]["provider"] in reg["providers"]   # still referenced -> kept


def test_delete_model_unknown_alias_is_noop():
    reg = {"providers": {"p": {"base_url": "u"}}, "models": {}}
    removed, pruned = providers.delete_model(reg, "nope")
    assert removed is False and pruned is None
    assert reg["providers"] == {"p": {"base_url": "u"}}


def _save_one_model(tmp_path, monkeypatch, alias="m"):
    prov = str(tmp_path / "providers.json")
    reg = {"providers": {}, "models": {}}
    providers.register_model(reg, alias=alias, base_url="https://x.ai/v1", model_id="mm")
    providers.save(reg, prov)
    monkeypatch.setenv("BENCH_PROVIDERS", prov)
    return prov


def test_cmd_remove_with_yes_flag_deletes_from_file(tmp_path, monkeypatch, capsys):
    prov = _save_one_model(tmp_path, monkeypatch)
    cli.cmd_remove(cli._parse_args(["remove", "--model", "m", "--yes"]))
    assert "m" not in providers.load(prov)["models"]
    assert "removed model 'm'" in capsys.readouterr().out


def test_cmd_remove_unknown_model_exits_nonzero(tmp_path, monkeypatch):
    import pytest
    _save_one_model(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_remove(cli._parse_args(["remove", "--model", "ghost", "--yes"]))


def test_cmd_remove_confirm_no_keeps_model(tmp_path, monkeypatch):
    prov = _save_one_model(tmp_path, monkeypatch)
    _scripted_input(monkeypatch, ["n"])                # confirm prompt -> no
    cli.cmd_remove(cli._parse_args(["remove", "--model", "m"]))
    assert "m" in providers.load(prov)["models"]


def test_ask_secret_falls_back_to_getpass_off_tty(monkeypatch):
    class _FakeStdin:
        def isatty(self):
            return False
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin())          # force the non-TTY path
    monkeypatch.setattr(cli.getpass, "getpass", lambda *a, **k: "  sk-secret  ")
    assert cli._ask_secret("API key") == "sk-secret"             # stripped, from getpass


def test_read_masked_echoes_stars_and_handles_backspace(monkeypatch, capsys):
    # type "sk-abZ", Backspace (drops 'Z'), "9", Enter  -> "sk-ab9", echoed as '*'
    keys = iter("sk-abZ\x7f9\r")

    class _FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return 0

        def read(self, n):
            return next(keys, "")

    fake_termios = types.ModuleType("termios")
    fake_termios.tcgetattr = lambda fd: None
    fake_termios.tcsetattr = lambda fd, when, old: None
    fake_termios.TCSADRAIN = 1
    fake_tty = types.ModuleType("tty")
    fake_tty.setraw = lambda fd: None
    monkeypatch.setitem(sys.modules, "termios", fake_termios)    # _read_masked imports these
    monkeypatch.setitem(sys.modules, "tty", fake_tty)
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin())

    assert cli._read_masked("  API key: ") == "sk-ab9"
    assert "*" in capsys.readouterr().out                        # field visibly reacts


def test_test_connection_unpacks_call_api_tuple(monkeypatch, capsys):
    """Regression for bench-ncu: _test_connection must take only the text and
    ignore however many usage fields call_api returns (now a 5-tuple including
    reasoning_tokens) — it star-unpacks, so adding return values can't break it."""
    calls = []

    def fake_call_api(*a, **k):
        calls.append((a, k))
        return "OK", 5, 1, 0, None               # 5-tuple: +reasoning_tokens

    monkeypatch.setattr(cli.runner, "call_api", fake_call_api)
    ok = cli._test_connection("https://x.ai/v1", "sk-key", "some-model")
    assert ok is True and calls                  # call_api was reached, no unpack error
    # bench-0ms: the test must give reasoning models enough budget to finish
    # thinking and emit content, else they hit finish_reason=length / content=null.
    assert calls[0][1]["max_tokens"] >= 256
    out = capsys.readouterr().out
    assert "OK ✓" in out and "'OK'" in out


def test_call_api_truncation_error_names_token_cap(monkeypatch):
    """bench-0ms: when a reasoning model burns the whole budget on reasoning,
    content is null with finish_reason=length. call_api must blame the token cap
    (and point at --max-tokens), not the generic 'missing message.content'."""
    body = {"choices": [{"finish_reason": "length",
                         "message": {"content": None, "reasoning": "thinking..."}}],
            "usage": {}}

    class FakeResp:
        def read(self): return json.dumps(body).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    try:
        runner.call_api("https://x.ai/v1", "k", "reasoner",
                        [{"role": "user", "content": "hi"}],
                        temperature=0, max_tokens=16, timeout=5)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "truncated" in str(e) and "max_tokens" in str(e)


def test_cmd_run_mock_does_not_crash(tmp_path, capsys):
    """Regression for bench-7cs: cmd_run referenced an undefined `ep` in --mock mode."""
    db = str(tmp_path / "bench.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 2, 2))
    con.commit()
    args = cli._parse_args(["run", "--db", db, "--mock", "perfect", "--run-id", "t",
                            "--results-dir", str(tmp_path / "results")])
    args.func(args)                       # must not raise UnboundLocalError
    out = capsys.readouterr().out
    assert "running" in out and "done in" in out
    n = storage.connect(db).execute(
        "SELECT COUNT(*) c FROM responses WHERE run_id='t'").fetchone()["c"]
    assert n > 0


def test_run_exports_results_file(tmp_path):
    """bench-q8g: a finished run writes a per-run folder under results/ holding the
    detailed report (md + HTML named with model+date), a run log, items.csv, and the
    full results JSON (metrics, runtime stats, per-item responses, REDACTED api key)."""
    db = str(tmp_path / "bench.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 2, 2))
    con.commit()
    rdir = tmp_path / "results"
    args = cli._parse_args(["run", "--db", db, "--mock", "perfect",
                            "--run-id", "my run", "--results-dir", str(rdir)])
    args.func(args)

    folders = [p for p in rdir.iterdir() if p.is_dir()]
    assert len(folders) == 1
    folder = folders[0]
    assert folder.name.startswith("mock_my_run_")                # model_runid_date, slugified
    names = {p.name for p in folder.iterdir()}
    assert "report.md" in names and "run.log" in names and "items.csv" in names
    assert any(n.startswith("mock_") and n.endswith(".html") for n in names)   # model+date HTML

    json_files = list(folder.glob("*.json"))
    assert len(json_files) == 1
    rec = json.loads(json_files[0].read_text())
    assert rec["run_id"] == "my run"
    assert rec["metrics"]["n_items"] == 4          # 2 difficulties x 2 reps
    assert rec["responses"] and "raw" in rec["responses"][0]
    assert rec["params"].get("api_key") == "***redacted***"                  # key never leaks
    # the detailed report names every item as solved/wrong/errored with a reason
    assert "## Not solved" in (folder / "report.md").read_text()


def test_run_skips_export_when_results_dir_blank(tmp_path):
    """bench-pgr: --results-dir '' disables the export (no folder created)."""
    db = str(tmp_path / "bench.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 2, 2))
    con.commit()
    args = cli._parse_args(["run", "--db", db, "--mock", "perfect",
                            "--run-id", "t", "--results-dir", ""])
    args.func(args)
    assert not (tmp_path / "results").exists()


def test_fmt_dur():
    assert runner._fmt_dur(0) == "0:00"
    assert runner._fmt_dur(65) == "1:05"
    assert runner._fmt_dur(3661) == "1:01:01"


def test_progress_redirected_emits_counts_and_summary():
    buf = io.StringIO()                   # StringIO.isatty() -> False -> plain-line mode
    p = runner._Progress(10, stream=buf)
    p.update(0, 0, 0)
    for i in range(1, 11):
        p.update(i, i, 0)                  # done=i, all solved
    p.finish(10, 0)
    out = buf.getvalue()
    assert "10/10" in out and "it/s" in out
    assert "✓10" in out and "acc=100%" in out          # live solved count + accuracy
    assert "10 solved" in out and "acc=100.0%" in out   # summary line
    assert "done in" in out and "\r" not in out


def test_progress_reports_wrong_and_accuracy():
    buf = io.StringIO()
    p = runner._Progress(4, stream=buf)
    p.update(4, 3, 1)                      # 4 done: 3 solved, 1 errored -> 0 wrong
    p.update(4, 2, 0)                      # 4 done: 2 solved, 0 err -> 2 wrong, acc 50%
    p.finish(2, 0)
    out = buf.getvalue()
    assert "✓2 ✗2 err=0  acc=50%" in out                # errors excluded from accuracy
    assert "2 solved, 2 wrong, 0 errored" in out and "acc=50.0%" in out


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


# ---- one-command launcher: `cli.py start` / bare `cli.py` (bench-3tt) ----

def _scripted_input(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))


def test_bare_invocation_routes_to_start():
    assert getattr(cli._parse_args([]), "func", None) is None   # main() falls back to start
    assert cli._parse_args(["start"]).func is cli.cmd_start


def _configure_one_model(tmp_path, monkeypatch, alias="m"):
    prov = str(tmp_path / "providers.json")
    reg = {"providers": {}, "models": {}}
    providers.register_model(reg, alias=alias, base_url="http://x/v1",
                             model_id="mm", context_window=1000)
    providers.save(reg, prov)
    monkeypatch.setenv("BENCH_PROVIDERS", prov)


def test_start_runs_existing_model_and_reuses_dataset(monkeypatch, tmp_path):
    _configure_one_model(tmp_path, monkeypatch)
    db = str(tmp_path / "b.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 2, 2))
    con.commit()
    seen = {}
    monkeypatch.setattr(cli, "cmd_generate", lambda ns: seen.setdefault("gen", ns))
    monkeypatch.setattr(cli, "cmd_run", lambda ns: seen.setdefault("run", ns))
    monkeypatch.setattr(cli, "cmd_report", lambda ns: seen.setdefault("report", ns))
    # existing, pick #1, reuse dataset (yes), launch all (Enter), no cap (Enter),
    # run-id default, confirm yes
    _scripted_input(monkeypatch, ["e", "1", "y", "", "", "", "y"])
    cli.cmd_start(cli._parse_args(["start", "--db", db]))
    assert "gen" not in seen                       # reused, did not regenerate
    assert seen["run"].model == "m" and seen["run"].run_id == "m"
    assert seen["run"].confidence is True and seen["run"].mock is None
    assert seen["report"].runs == ["m"]


def test_start_generates_quick_preset_when_no_dataset(monkeypatch, tmp_path):
    _configure_one_model(tmp_path, monkeypatch)
    db = str(tmp_path / "b.db")                     # empty: no dataset yet
    seen = {}
    monkeypatch.setattr(cli, "cmd_generate", lambda ns: seen.setdefault("gen", ns))
    monkeypatch.setattr(cli, "cmd_run", lambda ns: seen.setdefault("run", ns))
    monkeypatch.setattr(cli, "cmd_report", lambda ns: seen.setdefault("report", ns))
    # existing, pick #1, (no dataset ->) preset 1 = quick, difficulty default (Enter),
    # amount default (Enter), test cap default (Enter), run-id default, confirm yes
    _scripted_input(monkeypatch, ["e", "1", "1", "", "", "", "", "y"])
    cli.cmd_start(cli._parse_args(["start", "--db", db]))
    assert seen["gen"].reps == 3 and seen["gen"].max_diff == 4 and seen["gen"].distractor is False
    assert "run" in seen and "report" in seen


def test_start_hard_band_and_custom_amount_narrows_difficulty(monkeypatch, tmp_path):
    _configure_one_model(tmp_path, monkeypatch)
    db = str(tmp_path / "b.db")                     # empty: no dataset yet
    seen = {}
    monkeypatch.setattr(cli, "cmd_generate", lambda ns: seen.setdefault("gen", ns))
    monkeypatch.setattr(cli, "cmd_run", lambda ns: seen.setdefault("run", ns))
    monkeypatch.setattr(cli, "cmd_report", lambda ns: seen.setdefault("report", ns))
    # existing, pick #1, preset 2 = standard, difficulty 4 = hard (5-6), amount 5,
    # test cap default (Enter), run-id, yes
    _scripted_input(monkeypatch, ["e", "1", "2", "4", "5", "", "", "y"])
    cli.cmd_start(cli._parse_args(["start", "--db", db]))
    assert seen["gen"].min_diff == 5 and seen["gen"].max_diff == 6   # hard only, no easy
    assert seen["gen"].reps == 5                                     # custom amount


def test_start_fresh_generate_caps_test_count(monkeypatch, tmp_path):
    _configure_one_model(tmp_path, monkeypatch)
    db = str(tmp_path / "b.db")                     # empty: no dataset yet
    seen = {}
    monkeypatch.setattr(cli, "cmd_run", lambda ns: seen.setdefault("run", ns))
    monkeypatch.setattr(cli, "cmd_report", lambda ns: seen.setdefault("report", ns))
    monkeypatch.setattr(cli, "cmd_dashboard", lambda ns: None)
    # existing, pick #1, preset 1 = quick, difficulty default, amount default,
    # cap to 7 tests, run-id default, confirm yes
    _scripted_input(monkeypatch, ["e", "1", "1", "", "", "7", "", "y"])
    cli.cmd_start(cli._parse_args(["start", "--db", db]))
    assert seen["run"].limit == 7                   # fresh set capped to 7 tests


def test_start_reuse_launch_filter_passes_run_flags(monkeypatch, tmp_path):
    _configure_one_model(tmp_path, monkeypatch)
    db = str(tmp_path / "b.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 6, 2))
    con.commit()
    seen = {}
    monkeypatch.setattr(cli, "cmd_run", lambda ns: seen.setdefault("run", ns))
    monkeypatch.setattr(cli, "cmd_report", lambda ns: seen.setdefault("report", ns))
    monkeypatch.setattr(cli, "cmd_dashboard", lambda ns: None)
    # reuse, launch hard band (4 = 5-6), cap 3, run-id default, confirm yes
    _scripted_input(monkeypatch, ["e", "1", "y", "4", "3", "", "y"])
    cli.cmd_start(cli._parse_args(["start", "--db", db]))
    assert seen["run"].min_diff == 5 and seen["run"].max_diff == 6   # hard-only launch
    assert seen["run"].limit == 3                                    # capped count


def test_start_cancel_at_confirm_skips_run(monkeypatch, tmp_path):
    _configure_one_model(tmp_path, monkeypatch)
    db = str(tmp_path / "b.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 2, 2))
    con.commit()
    seen = {}
    monkeypatch.setattr(cli, "cmd_run", lambda ns: seen.setdefault("run", ns))
    # reuse, launch all (Enter), no cap (Enter), run-id default, confirm -> no
    _scripted_input(monkeypatch, ["e", "1", "y", "", "", "", "n"])
    cli.cmd_start(cli._parse_args(["start", "--db", db]))
    assert "run" not in seen


def test_filter_dataset_difficulty_and_even_limit():
    items = [{"item_id": str(i), "difficulty": d}
             for d in range(1, 7) for i in range(4)]      # 24 items, 4 per difficulty
    hard = cli._filter_dataset(items, min_diff=5, max_diff=6)
    assert {it["difficulty"] for it in hard} == {5, 6} and len(hard) == 8
    capped = cli._filter_dataset(items, min_diff=1, max_diff=6, limit=6)
    assert len(capped) == 6
    assert {it["difficulty"] for it in capped} == {1, 2, 3, 4, 5, 6}   # even stride, all bands


def test_run_difficulty_limit_filter_only_runs_subset(monkeypatch, tmp_path, capsys):
    db = str(tmp_path / "b.db")
    con = storage.connect(db)
    storage.save_dataset(con, generators.build_dataset(["arithmetic"], 1, 6, 2))
    con.commit()
    cli.cmd_run(cli._parse_args(
        ["run", "--db", db, "--mock", "perfect", "--run-id", "r",
         "--min-diff", "5", "--results-dir", ""]))
    diffs = {r["difficulty"] for r in con.execute(
        "SELECT d.difficulty FROM responses x JOIN dataset d ON d.item_id=x.item_id "
        "WHERE x.run_id='r'")}
    assert diffs == {5, 6}                              # easy items never ran
