#!/usr/bin/env python3
"""
reasoning-bench CLI.

  start      one command — pick/add a model, then generate + run + report (start here)
  setup      interactive wizard: register a model + endpoint
  edit       change a configured model's endpoint, id, key or limits
  remove     delete a configured model alias
  generate   build a procedurally-generated problem set into a SQLite DB
  run        run a model over the dataset and store graded responses
  report     compute metrics + accessible charts for one or more runs
  dashboard  rich in-terminal stats dashboard / multi-run comparison
  list       list runs in a DB
  families   list available problem families

Run `python cli.py` with no command to launch the interactive start menu.
Run `python cli.py <command> -h` for options.
"""

import re
import json
import os
import sys
import time
import getpass
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow running from anywhere

import generators
import providers
import storage
import runner
import metrics
import runreport            # stdlib-only per-run report (no matplotlib)
# report is imported lazily inside cmd_report so generate/run/list don't require matplotlib

ALL_FAMILIES = list(generators.GENERATORS)

# When neither --max-tokens nor a per-model max_tokens is set, the completion budget is
# auto-filled to the model's context window minus this reserve for the prompt — i.e. NO
# artificial reasoning limit: the model may reason up to its hard context ceiling. This
# avoids truncating a reasoning model mid-thought (which scores as wrong/errored — an
# invisible confound). Note: omitting max_tokens entirely is NOT equivalent — each
# provider then applies its own arbitrary default, often small — so we always send an
# explicit context-filling value.
PROMPT_RESERVE_TOKENS = 8192

# Fallback budget, used only when the context window is unknown (mock runs, or a model
# configured without a context_window) so there is no window to fill.
DEFAULT_MAX_TOKENS = 32768


def _resolve_max_tokens(explicit, per_model, context_window):
    """Completion budget for a run, in precedence order:
      1. an explicit --max-tokens,
      2. a per-model max_tokens pinned in providers.json,
      3. else NO limit: fill the context window, less PROMPT_RESERVE_TOKENS for the prompt,
      4. else (context window unknown) DEFAULT_MAX_TOKENS.
    """
    if explicit:
        return explicit
    if per_model:
        return per_model
    if context_window:
        return max(context_window - PROMPT_RESERVE_TOKENS, 1024)
    return DEFAULT_MAX_TOKENS


def cmd_generate(a):
    fams = ALL_FAMILIES if a.families == ["all"] else a.families
    items = generators.build_dataset(
        fams, a.min_diff, a.max_diff, a.reps,
        with_distractor=a.distractor, surface_variants=a.surface_variants)
    con = storage.connect(a.db)
    storage.save_dataset(con, items)
    print(f"generated {len(items)} items -> {a.db}")
    print(f"  families={fams} difficulty={a.min_diff}..{a.max_diff} reps={a.reps} "
          f"distractor={a.distractor} surface_variants={a.surface_variants}")


def _filter_dataset(items, min_diff=None, max_diff=None, limit=None):
    """Subset a loaded dataset for a launch: keep items whose difficulty is within
    [min_diff, max_diff] (each bound optional), then down-sample to `limit` items.
    The limit is taken as an even stride across the filtered list (not the first N)
    so families and difficulties stay represented. Lets one generated pool be run
    in slices (e.g. hard-only, 20 items) without regenerating."""
    if min_diff is not None:
        items = [it for it in items if it["difficulty"] >= min_diff]
    if max_diff is not None:
        items = [it for it in items if it["difficulty"] <= max_diff]
    if limit is not None and 0 <= limit < len(items):
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return items


def cmd_run(a):
    con = storage.connect(a.db)
    items = storage.load_dataset(con)
    if not items:
        sys.exit("no dataset in DB — run `generate` first")
    items = _filter_dataset(items, a.min_diff, a.max_diff, a.limit)
    if not items:
        sys.exit("no items match the --min-diff/--max-diff/--limit filter")

    if a.mock:
        base_url, model_id, api_key = "", a.model or "mock", ""
        context_window, max_tokens, capabilities = None, _resolve_max_tokens(a.max_tokens, None, None), []
    else:
        try:
            ep = providers.resolve(providers.load(), a.model, a.provider,
                                   a.base_url, a.api_key, default_provider="ollama")
        except ValueError as e:
            sys.exit(str(e))
        base_url, model_id, api_key = ep["base_url"], ep["model"], ep["api_key"]
        context_window = ep["context_window"]
        max_tokens = _resolve_max_tokens(a.max_tokens, ep["max_tokens"], context_window)
        capabilities = ep.get("capabilities", [])
        cw = f"  ctx={context_window}" if context_window else ""
        print(f"endpoint: {base_url}  model: {model_id}{cw}")
        if not a.max_tokens and not ep["max_tokens"] and context_window:
            print(f"  no token limit: completion budget = {max_tokens} "
                  f"(context window minus {PROMPT_RESERVE_TOKENS}-token prompt reserve)")
        elif context_window and max_tokens >= context_window:
            print(f"warning: --max-tokens {max_tokens} >= context window {context_window}; "
                  f"lower it to leave room for the prompt.")

    cfg = dict(
        base_url=base_url, api_key=api_key, model=model_id,
        temperature=a.temperature, max_tokens=max_tokens, context_window=context_window,
        n=a.samples, workers=a.workers, timeout=a.timeout, retries=a.retries,
        ask_confidence=a.confidence, resume=a.resume, mock=a.mock,
        dataset_tag=a.dataset_tag,
        capabilities=capabilities)
    run_id = a.run_id or (f"mock-{a.mock}" if a.mock else (a.model or model_id).replace("/", "_"))
    runner.run(con, run_id, items, cfg)
    res = metrics.compute(con, run_id)
    metrics.print_summary(res)
    if a.results_dir:
        try:                                  # a write hiccup must not lose a finished run
            _write_run_artifacts(con, run_id, a.model or model_id, a.results_dir, res)
        except Exception as e:
            print(f"\nwarning: could not write report/results: {type(e).__name__}: {e}")


def _write_run_artifacts(con, run_id, model_label, root, metrics_res):
    """After a run, write a per-run folder under `root` containing the detailed
    report (what solved/failed and why), an HTML file named with model + date,
    a run log, and the full machine-readable results JSON. The folder name and
    HTML name carry the model, run-id and date."""
    runtime = metrics.runtime_stats(con, run_id)
    params = _params_for_run(con, run_id)
    if "api_key" in params:                   # never leak the key into a report
        params["api_key"] = "***redacted***"
    folder = runreport.build(con, run_id, model_label, root, metrics_res=metrics_res,
                             runtime=runtime, params=params,
                             created_iso=_run_meta(con, run_id).get("created"))
    if folder is None:
        print("\n(no stored responses — nothing to report)")
        return
    _export_results(con, run_id, model_label, folder, metrics_res)   # results JSON inside the folder
    print(f"\nreport written -> {folder}/")
    print(f"  report.md · {_slug(model_label)}_{time.strftime('%Y-%m-%d')}.html · run.log · "
          "items.csv · results JSON")



def _slug(s):
    """Filesystem-safe token: keep alnum / dot / dash / underscore, collapse the rest."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def _export_results(con, run_id, model_label, results_dir, metrics_res=None):
    """Write a complete record of a finished run to
    `results_dir/<model>_<run-id>_<date>.json`; return the path (None if the run has
    no stored responses). 'Complete' = run metadata (API key redacted), the full
    metrics + runtime stats, and every graded per-item response (raw output included).
    Same-day re-runs of a run-id overwrite, matching the DB's INSERT-OR-REPLACE run."""
    run = con.execute("SELECT model, base_url, created_at FROM runs WHERE run_id=?",
                      (run_id,)).fetchone()
    if run is None:
        return None
    params = _params_for_run(con, run_id)
    if "api_key" in params:                      # never leak the key into a results file
        params["api_key"] = "***redacted***"

    rows = con.execute(
        """SELECT r.item_id, d.family, d.difficulty, d.gold, r.sample_idx,
                  r.raw, r.parsed, r.correct, r.confidence, r.latency_ms,
                  r.prompt_tokens, r.completion_tokens, r.metadata
             FROM responses r LEFT JOIN dataset d ON d.item_id = r.item_id
            WHERE r.run_id=? ORDER BY r.item_id, r.sample_idx""", (run_id,)).fetchall()
    if not rows:
        return None

    def _resp(row):
        d = dict(row)
        if d.get("metadata"):                    # stored as a JSON string -> inline it
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    created = run["created_at"]
    record = {
        "run_id": run_id,
        "model": model_label,
        "model_id": run["model"],
        "base_url": run["base_url"],
        "created_at": created,
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(created)) if created else None,
        "exported_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": params,
        "metrics": metrics_res if metrics_res is not None else metrics.compute(con, run_id),
        "runtime": metrics.runtime_stats(con, run_id),
        "responses": [_resp(r) for r in rows],
    }

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(
        results_dir,
        f"{_slug(model_label)}_{_slug(run_id)}_{time.strftime('%Y-%m-%d')}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
        f.write("\n")
    return path


def _params_for_run(con, run_id):
    row = con.execute("SELECT params FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not row or not row["params"]:
        return {}
    return json.loads(row["params"])


def _families_for_run(con, run_id):
    rows = con.execute(
        "SELECT DISTINCT d.family FROM dataset d "
        "JOIN responses r ON d.item_id = r.item_id WHERE r.run_id=?",
        (run_id,),
    )
    return {row["family"] for row in rows}


def _warn_if_dataset_tags_mismatch(con, run_ids):
    if len(run_ids) < 2:
        return
    tags, families = {}, {}
    for rid in run_ids:
        tags[rid] = _params_for_run(con, rid).get("dataset_tag", "")
        families[rid] = _families_for_run(con, rid)
    for i, rid1 in enumerate(run_ids):
        for rid2 in run_ids[i + 1 :]:
            if tags[rid1] != tags[rid2] and families[rid1] & families[rid2]:
                print(
                    f"warning: runs {rid1} and {rid2} have different dataset tags "
                    f"('{tags[rid1]}' vs '{tags[rid2]}') and share families; "
                    "comparisons may be invalid."
                )

def cmd_report(a):
    import report  # lazy: only this command needs matplotlib
    con = storage.connect(a.db)
    rids = a.runs or [r["run_id"] for r in storage.list_runs(con)]
    if not rids:
        sys.exit("no runs found")
    _warn_if_dataset_tags_mismatch(con, rids)
    for rid in rids:
        metrics.print_summary(metrics.compute(con, rid))
    report.build_report(con, rids, a.out)



def _run_meta(con, run_id):
    """Model name + human-readable creation time for a run (for the dashboard banner)."""
    row = con.execute("SELECT model, created_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        return {}
    created = ""
    if row["created_at"]:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["created_at"]))
    return {"model": row["model"], "created": created}


def cmd_dashboard(a):
    import dashboard          # stdlib-only; no matplotlib, unlike report
    con = storage.connect(a.db)
    rids = a.runs or [r["run_id"] for r in storage.list_runs(con)]
    if not rids:
        sys.exit("no runs found — run `run` first, or pass --runs")
    caps = dashboard.detect_caps(
        force_color=(False if a.no_color else None), width=a.width)
    results, metas, missing = [], [], []
    for rid in rids:
        res = metrics.compute(con, rid)
        if "error" in res:
            missing.append(rid); continue
        results.append(res)
        metas.append(_run_meta(con, rid))
    for rid in missing:
        print(f"skip {rid}: no responses for run")
    if not results:
        sys.exit("nothing to show")

    if len(results) == 1:
        rstats = metrics.runtime_stats(con, results[0]["run_id"])
        dashboard.show(dashboard.render_run(results[0], metas[0], rstats, caps))
    else:
        _warn_if_dataset_tags_mismatch(con, [r["run_id"] for r in results])
        labels = [r["run_id"] for r in results]
        dashboard.show(dashboard.render_compare(results, labels, metas, caps))


def cmd_list(a):
    con = storage.connect(a.db)
    runs = storage.list_runs(con)
    if not runs:
        print("no runs"); return
    for r in runs:
        print(f"  {r['run_id']:30s} model={r['model']}")


def cmd_families(a):
    for f in ALL_FAMILIES:
        flags = []
        if f in generators.SUPPORTS_DISTRACTOR: flags.append("distractor")
        if f in generators.SUPPORTS_SURFACE: flags.append("surface")
        print(f"  {f:16s} supports: {', '.join(flags) or 'difficulty/variance only'}")


def cmd_providers(a):
    reg = providers.load()
    if not reg["providers"]:
        print(f"no providers configured ({providers.config_path()} missing or empty)"); return
    print(f"providers from {providers.config_path()}:")
    for name, p in reg["providers"].items():
        if "oauth" in (p.get("capabilities") or []):
            auth = "auth=oauth"
        else:
            key = p.get("api_key_env") or ("(literal)" if p.get("api_key") else "none")
            auth = f"api_key={key}"
        print(f"  {name:12s} {p.get('base_url', '?'):34s} {auth}")


def cmd_models(a):
    reg = providers.load()
    if not reg["models"]:
        print(f"no models configured ({providers.config_path()})"); return
    print(f"models from {providers.config_path()}  (use the alias as --model):")
    for name, m in reg["models"].items():
        cw = m.get("context_window")
        mt = f"  max_tokens={m['max_tokens']}" if m.get("max_tokens") else ""
        print(f"  {name:12s} -> {m.get('provider', '?')}:{m.get('model', '?'):30s} "
              f"ctx={cw if cw else '?'}{mt}")
    print("\n  edit one with:    python cli.py edit --model <alias>")
    print("  remove one with:  python cli.py remove --model <alias>")


# ----------------------------------------------------------------------------
# setup wizard — register a model + endpoint step by step, then print next steps
# ----------------------------------------------------------------------------

_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _ask(label, default=None, required=False, validate=None):
    """Prompt until a valid answer is given; returns a stripped string or `default`."""
    hint = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"  {label}{hint}: ").strip()
        except EOFError:
            print()
            if default is not None:
                return default
            if required:
                raise KeyboardInterrupt          # no more input -> abort, don't loop
            return ""
        if not raw:
            if default is not None:
                return default
            if required:
                print("    (required — please enter a value)")
                continue
            return ""
        if validate:
            ok, msg = validate(raw)
            if not ok:
                print(f"    {msg}")
                continue
        return raw


def _ask_int(label, default=None):
    """Prompt for an optional whole number; returns int, or `default` when left
    blank (the current value, shown in the hint, when editing)."""
    hint = f" [{default}]" if default else ""
    while True:
        raw = _ask(label + hint)
        if not raw:
            return default
        try:
            return int(raw.replace(",", "").replace("_", ""))
        except ValueError:
            print("    (please enter a whole number, e.g. 128000)")


def _ask_yes_no(label, default=True):
    suffix = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input(f"  {label} [{suffix}]: ").strip().lower()
        except EOFError:
            print()
            return default
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    (please answer y or n)")


def _read_masked(prompt):
    """Echo a '*' for each typed character on a POSIX terminal and return the
    text. Returns None when raw reads aren't available (e.g. Windows) so the
    caller can fall back to getpass. Honours Backspace, Ctrl-U and Ctrl-C."""
    try:
        import termios, tty
    except ImportError:
        return None                              # no raw-tty support (Windows)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    chars = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n", ""):           # Enter (raw mode) or EOF -> done
                break
            if ch == "\x03":                     # Ctrl-C
                raise KeyboardInterrupt
            if ch in ("\x7f", "\b"):             # Backspace / Delete
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")    # rub out one '*'
                    sys.stdout.flush()
                continue
            if ch == "\x15":                     # Ctrl-U -> clear the line
                while chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                sys.stdout.flush()
                continue
            if ord(ch) < 32:                     # ignore other control chars
                continue
            chars.append(ch)
            sys.stdout.write("*")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return "".join(chars)


def _ask_secret(label):
    """Prompt for a secret value, echoing '*' per character so the field
    visibly reacts as you type — plain getpass shows nothing, which reads as a
    frozen/unresponsive field. Falls back to hidden getpass, then visible
    input(), when stdin isn't a real terminal (pipes, some IDE consoles)."""
    prompt = f"  {label}: "
    if sys.stdin.isatty():
        masked = _read_masked(prompt)
        if masked is not None:
            return masked.strip()
    try:
        return getpass.getpass(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:                            # non-standard stdin: don't crash the wizard
        return input(prompt).strip()


def _valid_alias(s):
    if _ALIAS_RE.match(s):
        return True, ""
    return False, "use letters, digits, '.', '_' or '-' (no spaces), starting alphanumeric"


def _valid_url(s):
    if s.startswith(("http://", "https://")):
        return True, ""
    return False, "should start with http:// or https:// (e.g. https://api.openai.com/v1)"


def _test_connection(base_url, api_key, model_id):
    """Send one tiny completion so the user sees the endpoint actually works."""
    print("  contacting endpoint...", end=" ", flush=True)
    try:
        text, *_ = runner.call_api(          # star-unpack: robust to call_api's arity
            base_url, api_key, model_id,
            [{"role": "user", "content": "Reply with the single word OK."}],
            # Reasoning models (Kimi, DeepSeek-R1/V4, o1-style) burn completion
            # tokens on hidden reasoning before emitting any content; a tiny cap
            # makes them hit finish_reason=length with content=null and the test
            # fails even though the endpoint is fine. Give them room to answer.
            temperature=0, max_tokens=512, timeout=60)
        print("OK ✓")
        print(f"    model replied: {' '.join(text.split())[:60]!r}")
        return True
    except Exception as e:
        print("failed ✗")
        print(f"    {type(e).__name__}: {e}")
        print("    (config was saved — fix the endpoint / key / model id and re-run "
              "`setup`, or just try a real `run`.)")
        return False


def _print_next_steps(alias):
    print("\nNext steps:")
    print("  1) Generate a problem set (procedurally generated — nothing to download):")
    print("       python cli.py generate --db bench.db --reps 12 --distractor --surface-variants 3")
    print("  2) Run the benchmark on your model (live progress is shown):")
    print(f"       python cli.py run --db bench.db --model {alias} --run-id {alias} --confidence")
    print("  3) Build the report (metrics table + accessible charts):")
    print(f"       python cli.py report --db bench.db --runs {alias} --out report")
    print("\n  Re-run `python cli.py setup` to add another model, "
          "or `python cli.py models` to list what's configured.")


def _print_saved_model(path, alias, prov_name, base_url, model_id,
                       context_window, max_tokens, api_key_env, api_key, verb="saved"):
    """Print the confirmation block shared by the setup and edit wizards."""
    print(f"\n✓ {verb} model '{alias}' (provider '{prov_name}') to {path}")
    print(f"    endpoint  : {base_url}")
    print(f"    model id  : {model_id}")
    if context_window:
        print(f"    context   : {context_window} tokens")
    if max_tokens:
        print(f"    max_tokens: {max_tokens}")
    if api_key_env:
        print(f"    api key   : read from ${api_key_env}")
        print(f"\n  Set the key in your shell before running:\n      export {api_key_env}=<your key>")
    elif api_key:
        print("    api key   : stored in file")
    else:
        print("    api key   : none (keyless endpoint)")


def _stored_key_choice(alias, key):
    """Ask how to store an already-typed key. Returns (api_key, api_key_env) with
    at most one set: a file literal, or the name of an env var to read it from."""
    store = _ask("Store key as [e]nv-var reference (recommended) or [f]ile literal?",
                 default="e")
    if store.lower().startswith("f"):
        print(f"    note: the key will be written in plaintext to {providers.config_path()} "
              "(tracked by git — don't commit it).")
        return key, None
    default_env = re.sub(r"[^A-Za-z0-9]", "_", alias).upper() + "_API_KEY"
    return None, _ask("Environment variable name", default=default_env)


def _setup_wizard(reg, path):
    """Interactive prompts -> register + save -> optional connection test.

    Returns the new model alias, or None if the user cancelled."""
    print("\nreasoning-bench setup — register a model to benchmark.")
    print(f"Answers are saved to {path}. Press Ctrl-C to cancel.\n")

    try:
        alias = _ask("Short name (alias) for this model — you'll pass it to --model",
                     required=True, validate=_valid_alias)
        if alias in reg.get("models", {}) and not _ask_yes_no(
                f"'{alias}' already exists — overwrite it?", default=False):
            print("cancelled.")
            return None

        base_url = _ask("Provider endpoint (OpenAI-compatible base URL)",
                        required=True, validate=_valid_url)
        model_id = _ask("Model ID (the provider's own model name, e.g. gpt-4o-mini)",
                        required=True)

        print("  API key (masked with '*' as you type) — leave blank for keyless "
              "local servers like Ollama.")
        key = _ask_secret("API key")
        api_key, api_key_env = (None, None)
        if key:
            api_key, api_key_env = _stored_key_choice(alias, key)

        context_window = _ask_int("Context window in tokens (e.g. 128000)")
        max_tokens = _ask_int("Max completion tokens — optional, blank to skip")
    except (KeyboardInterrupt, EOFError):
        print("\ncancelled.")
        return None

    reg, prov_name = providers.register_model(
        reg, alias=alias, base_url=base_url, model_id=model_id,
        api_key=api_key, api_key_env=api_key_env,
        context_window=context_window, max_tokens=max_tokens)
    providers.save(reg, path)

    _print_saved_model(path, alias, prov_name, base_url, model_id,
                       context_window, max_tokens, api_key_env, api_key)

    print()
    if _ask_yes_no("Test the connection now?", default=True):
        _test_connection(base_url, key, model_id)
    return alias


def cmd_setup(a):
    path = providers.config_path()
    reg = providers.load(path)
    alias = _setup_wizard(reg, path)
    if alias:
        _print_next_steps(alias)


# ----------------------------------------------------------------------------
# remove — delete a configured model alias (and prune its orphaned provider)
# ----------------------------------------------------------------------------

def _remove_model(reg, path, alias=None, assume_yes=False):
    """Delete a model alias, pruning its now-unused provider, and save on
    success. Picks interactively and asks to confirm unless `alias` /
    `assume_yes` are supplied. Returns the removed alias, or None if nothing
    was removed."""
    models = reg.get("models", {})
    if not models:
        print(f"\nNo models are configured ({path}) — nothing to remove.")
        return None
    if alias is None:
        print("\nDelete which model?")
        alias = _choose_model(reg)
    if alias not in models:
        print(f"  no such model '{alias}'.")
        return None
    if not assume_yes and not _ask_yes_no(
            f"Delete model '{alias}'? This can't be undone", default=False):
        print("  cancelled — nothing removed.")
        return None
    _, pruned = providers.delete_model(reg, alias)
    providers.save(reg, path)
    print(f"\n✓ removed model '{alias}' from {path}")
    if pruned:
        print(f"    also removed its now-unused provider '{pruned}'")
    return alias


def cmd_remove(a):
    path = providers.config_path()
    reg = providers.load(path)
    models = reg.get("models", {})
    if not models:
        print(f"no models configured ({path}) — nothing to remove.")
        return
    if a.model and a.model not in models:
        print(f"no such model '{a.model}'. Configured: {', '.join(models)}")
        raise SystemExit(1)
    _remove_model(reg, path, alias=a.model, assume_yes=a.yes)


# ----------------------------------------------------------------------------
# edit — change a configured model's endpoint, id, key or limits (Enter keeps each)
# ----------------------------------------------------------------------------

def _edit_wizard(reg, path, alias=None):
    """Interactively edit an existing model: each prompt defaults to the current
    value, so pressing Enter keeps it. Saves on success and returns the alias,
    or None if there was nothing to edit / the user cancelled."""
    models = reg.get("models", {})
    if not models:
        print(f"\nNo models are configured ({path}) — nothing to edit.")
        return None
    if alias is None:
        print("\nEdit which model?")
        alias = _choose_model(reg)
    if alias not in models:
        print(f"  no such model '{alias}'.")
        return None

    cur = models[alias]
    prov = reg.get("providers", {}).get(cur.get("provider"), {})
    print(f"\nEditing '{alias}' — press Enter to keep the current value.\n")

    try:
        base_url = _ask("Provider endpoint (OpenAI-compatible base URL)",
                        default=prov.get("base_url"), validate=_valid_url)
        model_id = _ask("Model ID (the provider's own model name)",
                        default=cur.get("model"))
        context_window = _ask_int("Context window in tokens", default=cur.get("context_window"))
        max_tokens = _ask_int("Max completion tokens", default=cur.get("max_tokens"))

        api_key = api_key_env = None                     # both None -> keep the current key
        if _ask_yes_no("Change the API key?", default=False):
            key = _ask_secret("API key")
            if key:
                api_key, api_key_env = _stored_key_choice(alias, key)
    except (KeyboardInterrupt, EOFError):
        print("\ncancelled.")
        return None

    reg, prov_name = providers.edit_model(
        reg, alias, base_url=base_url, model_id=model_id,
        api_key=api_key, api_key_env=api_key_env,
        context_window=context_window, max_tokens=max_tokens)
    providers.save(reg, path)

    saved = reg["providers"].get(prov_name, {})
    _print_saved_model(path, alias, prov_name, base_url, model_id,
                       context_window, max_tokens,
                       saved.get("api_key_env"), saved.get("api_key"), verb="updated")

    print()
    if _ask_yes_no("Test the connection now?", default=True):
        _test_connection(base_url, providers._resolve_key(saved, None), model_id)
    return alias


def cmd_edit(a):
    path = providers.config_path()
    reg = providers.load(path)
    models = reg.get("models", {})
    if not models:
        print(f"no models configured ({path}) — nothing to edit.")
        return
    if a.model and a.model not in models:
        print(f"no such model '{a.model}'. Configured: {', '.join(models)}")
        raise SystemExit(1)
    alias = _edit_wizard(reg, path, alias=a.model)
    if alias:
        _print_next_steps(alias)


# ----------------------------------------------------------------------------
# start — one interactive command: pick/add a model, then generate -> run -> report
# ----------------------------------------------------------------------------

# preset key -> (label, generate knobs). Item count grows quick < standard < thorough.
_DATASET_PRESETS = {
    "1": ("quick",    dict(reps=3,  min_diff=1, max_diff=4, distractor=False, surface=0)),
    "2": ("standard", dict(reps=10, min_diff=1, max_diff=6, distractor=True,  surface=0)),
    "3": ("thorough", dict(reps=20, min_diff=1, max_diff=6, distractor=True,  surface=3)),
}

# difficulty-band key -> (label, (min_diff, max_diff)). Lets the start flow target
# only hard problems and skip the easy ones (difficulty runs 1=easiest .. 6=hardest).
_DIFFICULTY_BANDS = {
    "1": ("all (1-6)",    (1, 6)),
    "2": ("easy (1-2)",   (1, 2)),
    "3": ("medium (3-4)", (3, 4)),
    "4": ("hard (5-6)",   (5, 6)),
}


def _choose_model(reg):
    """Print the configured models and return the alias the user picks."""
    models = reg.get("models", {})
    names = list(models)
    print("\nConfigured models:")
    for i, name in enumerate(names, 1):
        m = models[name]
        cw = m.get("context_window")
        print(f"  [{i}] {name}  ->  {m.get('provider', '?')}:{m.get('model', '?')}"
              f"{'  ctx=' + str(cw) if cw else ''}")
    while True:
        raw = _ask(f"Pick a model [1-{len(names)}]", default="1")
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        if raw in models:                                 # also accept the alias itself
            return raw
        print(f"    (enter a number 1-{len(names)}, or a model alias)")


def _ask_launch_filter(db):
    """Ask which difficulty band and how many items to launch from the existing
    dataset (Enter = all / no cap). Returns (min_diff, max_diff, limit), each None
    when left unset. Used when reusing a pool so it can be run in slices."""
    items = storage.load_dataset(storage.connect(db))
    diffs = sorted({it["difficulty"] for it in items}) or [0]
    band = _ask(f"\nLaunch which difficulty — [Enter] all ({diffs[0]}-{diffs[-1]})  "
                "[1] all  [2] easy  [3] medium  [4] hard", default="")
    if band in _DIFFICULTY_BANDS:
        _, (min_diff, max_diff) = _DIFFICULTY_BANDS[band]
    else:
        min_diff = max_diff = None
    limit = _ask_int("How many items to launch (Enter = all)", default=None)
    return min_diff, max_diff, limit


def _ensure_dataset(db):
    """Reuse the dataset already in `db`, else generate one from a size preset.
    Returns True when an existing dataset was reused, False when one was generated."""
    existing = storage.load_dataset(storage.connect(db))
    if existing and _ask_yes_no(
            f"\nReuse the existing dataset in {db} ({len(existing)} items)?", default=True):
        return True
    label, knobs = _DATASET_PRESETS.get(
        _ask("\nDataset size — [1] quick  [2] standard  [3] thorough", default="2"),
        _DATASET_PRESETS["2"])

    # difficulty band: Enter keeps the preset's range; a number narrows it (e.g. hard-only)
    band = _ask("\nDifficulty — [Enter] preset default  [1] all  [2] easy  [3] medium  [4] hard",
                default="")
    if band in _DIFFICULTY_BANDS:
        band_label, (min_diff, max_diff) = _DIFFICULTY_BANDS[band]
    else:
        min_diff, max_diff = knobs["min_diff"], knobs["max_diff"]
        band_label = f"preset ({min_diff}-{max_diff})"

    # amount: problems generated per (family, difficulty); Enter keeps the preset's count
    reps = _ask_int("\nProblems per family+difficulty (amount)", default=knobs["reps"])

    argv = ["generate", "--db", db,
            "--reps", str(reps),
            "--min-diff", str(min_diff), "--max-diff", str(max_diff),
            "--surface-variants", str(knobs["surface"])]
    if knobs["distractor"]:
        argv.append("--distractor")
    print(f"\nbuilding a '{label}' dataset (difficulty {band_label}, {reps} per family/difficulty) "
          "— instant; problems are generated, not downloaded...")
    cmd_generate(_parse_args(argv))
    return False


def cmd_start(a):
    db = a.db
    path = providers.config_path()
    reg = providers.load(path)

    print("\n" + "=" * 64)
    print("  reasoning-bench — interactive launcher")
    print("=" * 64)

    # 1) pick an existing model, add a new one, or delete one — then proceed
    alias = None
    while True:
        if not reg.get("models"):
            print("\nNo models are configured yet — let's add one.")
            alias = _setup_wizard(reg, path)
            break
        choice = _ask("\nRun an [e]xisting model, [a]dd a new one, [m]odify one, "
                      "or [d]elete one?", default="e").lower()
        if choice.startswith("d"):
            _remove_model(reg, path)
            reg = providers.load(path)            # reload after the delete
            continue                              # back to the menu
        if choice.startswith("m"):
            _edit_wizard(reg, path)
            reg = providers.load(path)            # reload after the edit
            continue                              # back to the menu
        alias = _setup_wizard(reg, path) if choice.startswith("a") else _choose_model(reg)
        break
    if not alias:
        print("\nnothing to run. Re-run `python cli.py start` any time.")
        return

    # 2) dataset — reuse or generate
    reused = _ensure_dataset(db)

    # 3) when reusing a pool, let the user launch only a difficulty band / a capped
    #    count of it; a freshly generated set was already scoped at generate time.
    total = storage.load_dataset(storage.connect(db))
    min_diff = max_diff = limit = None
    if reused:
        min_diff, max_diff, limit = _ask_launch_filter(db)
    else:
        # a freshly generated set is already scoped by difficulty/reps, but it can
        # still be large — let the user cap how many tests actually run.
        limit = _ask_int(f"\nHow many tests to run (Enter = all {len(total)})", default=None)
    launch = _filter_dataset(total, min_diff, max_diff, limit)
    if not launch:                                # filter excluded everything — run it all
        print("  (nothing matched that difficulty/amount — launching the full dataset)")
        min_diff = max_diff = limit = None
        launch = total

    # 4) confirm with sensible defaults, then run with the live progress bar
    run_id = _ask("\nRun id (a label for this run)", default=alias)
    n_items = len(launch)
    print(f"\nReady:  model='{alias}'  launching {n_items} of {len(total)} items  "
          f"run-id='{run_id}'  samples=1  confidence=on")
    if not _ask_yes_no(f"Run the benchmark now (~{n_items} model calls)?", default=True):
        print("cancelled — config and dataset are saved; re-run with `python cli.py start`.")
        return

    run_args = ["run", "--db", db, "--model", alias, "--run-id", run_id, "--confidence"]
    for flag, val in (("--min-diff", min_diff), ("--max-diff", max_diff), ("--limit", limit)):
        if val is not None:
            run_args += [flag, str(val)]
    print()
    cmd_run(_parse_args(run_args))

    # 4) report — charts need matplotlib; degrade gracefully without it
    out = "report"
    charts = True
    try:
        cmd_report(_parse_args(["report", "--db", db, "--runs", run_id, "--out", out]))
    except (ImportError, ModuleNotFoundError):
        charts = False
        print(f"\n(PNG charts need matplotlib — `uv sync` installs it; then re-run: "
              f"uv run python cli.py report --db {db} --runs {run_id})")

    # 5) terminal dashboard — the at-a-glance visual finale (stdlib only, no
    # matplotlib). Best-effort: a run with nothing scorable must not abort start.
    try:
        cmd_dashboard(_parse_args(["dashboard", "--db", db, "--runs", run_id]))
    except SystemExit:
        pass

    tail = f"report written to {out}/ (report.md + charts + metrics.csv)" if charts \
        else "metrics shown above"
    print(f"\n✓ all done — {tail}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="reasoning-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=False)

    st = sub.add_parser("start", help="one command: pick/add a model, then run the whole benchmark")
    st.add_argument("--db", default="bench.db")
    st.set_defaults(func=cmd_start)

    se = sub.add_parser("setup", help="interactive wizard: register a model + endpoint")
    se.set_defaults(func=cmd_setup)

    g = sub.add_parser("generate", help="build a problem set")
    g.add_argument("--db", default="bench.db")
    g.add_argument("--families", nargs="+", default=["all"],
                   help=f"subset of {ALL_FAMILIES} or 'all'")
    g.add_argument("--min-diff", type=int, default=1)
    g.add_argument("--max-diff", type=int, default=6)
    g.add_argument("--reps", type=int, default=15, help="structures per (family,difficulty)")
    g.add_argument("--distractor", action="store_true", help="add matched NoOp-distractor items")
    g.add_argument("--surface-variants", type=int, default=0, help="cosmetic variants per item")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run a model over the dataset")
    r.add_argument("--db", default="bench.db")
    r.add_argument("--model", default=None,
                   help="model alias from providers.json (see `cli.py models`), or a raw model id")
    r.add_argument("--provider", default=None,
                   help="provider alias from providers.json (see `cli.py providers`)")
    r.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint; overrides the provider's base_url")
    r.add_argument("--api-key", default=None, help="overrides provider key / OPENAI_API_KEY")
    r.add_argument("--run-id", default=None, help="defaults to model name")
    r.add_argument("--min-diff", type=int, default=None,
                   help="only run items with difficulty >= this (filter the existing dataset)")
    r.add_argument("--max-diff", type=int, default=None,
                   help="only run items with difficulty <= this (e.g. skip easy: --min-diff 5)")
    r.add_argument("--limit", type=int, default=None,
                   help="cap the number of items run, sampled evenly across the filtered set")
    r.add_argument("--samples", type=int, default=1, help="n samples/item (pass@k, self-consistency)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=None,
                   help="completion budget (default: no limit — fills the model's context window)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--timeout", type=int, default=120)
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--confidence", action="store_true", help="ask for confidence (enables calibration)")
    r.add_argument("--resume", action="store_true", help="skip items already done")
    r.add_argument("--mock", choices=["perfect", "random", "noisy"], default=None,
                   help="synthesize answers without a server (pipeline test)")
    r.add_argument("--dataset-tag", default="", help="dataset version tag for run comparisons")
    r.add_argument("--results-dir", default="results",
                   help="write a full per-run results JSON here when the run finishes "
                        "(default: results/). Pass '' to skip the export.")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="metrics + charts")
    rep.add_argument("--db", default="bench.db")
    rep.add_argument("--runs", nargs="*", default=None, help="run_ids (default: all)")
    rep.add_argument("--out", default="report", help="output directory")
    rep.set_defaults(func=cmd_report)

    da = sub.add_parser("dashboard", help="rich terminal dashboard / run comparison")
    da.add_argument("--db", default="bench.db")
    da.add_argument("--runs", nargs="*", default=None,
                    help="run_ids: one renders a full dashboard, many compare (default: all)")
    da.add_argument("--no-color", action="store_true", help="disable ANSI color")
    da.add_argument("--width", type=int, default=None, help="override terminal width")
    da.set_defaults(func=cmd_dashboard)

    li = sub.add_parser("list", help="list runs"); li.add_argument("--db", default="bench.db")
    li.set_defaults(func=cmd_list)

    fa = sub.add_parser("families", help="list families"); fa.set_defaults(func=cmd_families)

    pr = sub.add_parser("providers", help="list configured providers"); pr.set_defaults(func=cmd_providers)
    mo = sub.add_parser("models", help="list configured model aliases"); mo.set_defaults(func=cmd_models)

    ed = sub.add_parser("edit", help="edit a configured model alias")
    ed.add_argument("--model", default=None,
                    help="alias to edit (omit to pick from a list interactively)")
    ed.set_defaults(func=cmd_edit)

    rm = sub.add_parser("remove", help="delete a configured model alias")
    rm.add_argument("--model", default=None,
                    help="alias to remove (omit to pick from a list interactively)")
    rm.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    rm.set_defaults(func=cmd_remove)

    return p.parse_args(argv)


def main():
    a = _parse_args()
    if getattr(a, "func", None) is None:        # bare `python cli.py` -> interactive launcher
        a = _parse_args(["start"])
    try:
        a.func(a)
    except (KeyboardInterrupt, EOFError):
        print("\ncancelled.")


if __name__ == "__main__":
    main()
