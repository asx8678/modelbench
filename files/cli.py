#!/usr/bin/env python3
"""
reasoning-bench CLI.

  setup      interactive wizard: register a model + endpoint (start here)
  generate   build a procedurally-generated problem set into a SQLite DB
  run        run a model over the dataset and store graded responses
  report     compute metrics + accessible charts for one or more runs
  list       list runs in a DB
  families   list available problem families

Run `python cli.py <command> -h` for options.
"""

import re
import json
import os
import sys
import getpass
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow running from anywhere

import generators
import providers
import storage
import runner
import metrics
# report is imported lazily inside cmd_report so generate/run/list don't require matplotlib

ALL_FAMILIES = list(generators.GENERATORS)


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


def cmd_run(a):
    con = storage.connect(a.db)
    items = storage.load_dataset(con)
    if not items:
        sys.exit("no dataset in DB — run `generate` first")

    if a.mock:
        base_url, model_id, api_key = "", a.model or "mock", ""
        context_window, max_tokens, capabilities = None, a.max_tokens or 1024, []
    else:
        try:
            ep = providers.resolve(providers.load(), a.model, a.provider,
                                   a.base_url, a.api_key, default_provider="ollama")
        except ValueError as e:
            sys.exit(str(e))
        base_url, model_id, api_key = ep["base_url"], ep["model"], ep["api_key"]
        context_window = ep["context_window"]
        max_tokens = a.max_tokens or ep["max_tokens"] or 1024
        capabilities = ep.get("capabilities", [])
        cw = f"  ctx={context_window}" if context_window else ""
        print(f"endpoint: {base_url}  model: {model_id}{cw}")
        if context_window and max_tokens >= context_window:
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
    metrics.print_summary(metrics.compute(con, run_id))



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
        key = p.get("api_key_env") or ("(literal)" if p.get("api_key") else "none")
        print(f"  {name:12s} {p.get('base_url', '?'):34s} api_key={key}")


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
            raw = ""
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


def _ask_int(label):
    """Prompt for an optional whole number; returns int or None if left blank."""
    while True:
        raw = _ask(label)
        if not raw:
            return None
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
        text, _, _ = runner.call_api(
            base_url, api_key, model_id,
            [{"role": "user", "content": "Reply with the single word OK."}],
            temperature=0, max_tokens=16, timeout=30)
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


def cmd_setup(a):
    path = providers.config_path()
    reg = providers.load(path)
    print("\nreasoning-bench setup — register a model to benchmark.")
    print(f"Answers are saved to {path}. Press Ctrl-C to cancel.\n")

    try:
        alias = _ask("Short name (alias) for this model — you'll pass it to --model",
                     required=True, validate=_valid_alias)
        if alias in reg.get("models", {}) and not _ask_yes_no(
                f"'{alias}' already exists — overwrite it?", default=False):
            print("cancelled.")
            return

        base_url = _ask("Provider endpoint (OpenAI-compatible base URL)",
                        required=True, validate=_valid_url)
        model_id = _ask("Model ID (the provider's own model name, e.g. gpt-4o-mini)",
                        required=True)

        print("  API key (input hidden) — leave blank for keyless local servers like Ollama.")
        key = getpass.getpass("  API key: ").strip()
        api_key = api_key_env = None
        if key:
            store = _ask("Store key as [e]nv-var reference (recommended) or [f]ile literal?",
                         default="e")
            if store.lower().startswith("f"):
                api_key = key
                print(f"    note: the key will be written in plaintext to {path} "
                      "(tracked by git — don't commit it).")
            else:
                default_env = re.sub(r"[^A-Za-z0-9]", "_", alias).upper() + "_API_KEY"
                api_key_env = _ask("Environment variable name", default=default_env)

        context_window = _ask_int("Context window in tokens (e.g. 128000)")
        max_tokens = _ask_int("Max completion tokens — optional, blank to skip")
    except KeyboardInterrupt:
        print("\ncancelled.")
        return

    reg, prov_name = providers.register_model(
        reg, alias=alias, base_url=base_url, model_id=model_id,
        api_key=api_key, api_key_env=api_key_env,
        context_window=context_window, max_tokens=max_tokens)
    providers.save(reg, path)

    print(f"\n✓ saved model '{alias}' (provider '{prov_name}') to {path}")
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

    print()
    if _ask_yes_no("Test the connection now?", default=True):
        _test_connection(base_url, key, model_id)

    _print_next_steps(alias)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="reasoning-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

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
    r.add_argument("--samples", type=int, default=1, help="n samples/item (pass@k, self-consistency)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=None, help="completion budget (default: model's or 1024)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--timeout", type=int, default=120)
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--confidence", action="store_true", help="ask for confidence (enables calibration)")
    r.add_argument("--resume", action="store_true", help="skip items already done")
    r.add_argument("--mock", choices=["perfect", "random", "noisy"], default=None,
                   help="synthesize answers without a server (pipeline test)")
    r.add_argument("--dataset-tag", default="", help="dataset version tag for run comparisons")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="metrics + charts")
    rep.add_argument("--db", default="bench.db")
    rep.add_argument("--runs", nargs="*", default=None, help="run_ids (default: all)")
    rep.add_argument("--out", default="report", help="output directory")
    rep.set_defaults(func=cmd_report)

    li = sub.add_parser("list", help="list runs"); li.add_argument("--db", default="bench.db")
    li.set_defaults(func=cmd_list)

    fa = sub.add_parser("families", help="list families"); fa.set_defaults(func=cmd_families)

    pr = sub.add_parser("providers", help="list configured providers"); pr.set_defaults(func=cmd_providers)
    mo = sub.add_parser("models", help="list configured model aliases"); mo.set_defaults(func=cmd_models)

    return p.parse_args(argv)


def main():
    a = _parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
