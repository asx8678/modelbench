#!/usr/bin/env python3
"""
reasoning-bench CLI.

  generate   build a procedurally-generated problem set into a SQLite DB
  run        run a model over the dataset and store graded responses
  report     compute metrics + accessible charts for one or more runs
  list       list runs in a DB
  families   list available problem families

Run `python cli.py <command> -h` for options.
"""

import os
import sys
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
        context_window, max_tokens = None, a.max_tokens or 1024
    else:
        try:
            ep = providers.resolve(providers.load(), a.model, a.provider,
                                   a.base_url, a.api_key, default_provider="ollama")
        except ValueError as e:
            sys.exit(str(e))
        base_url, model_id, api_key = ep["base_url"], ep["model"], ep["api_key"]
        context_window = ep["context_window"]
        max_tokens = a.max_tokens or ep["max_tokens"] or 1024
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
        capabilities=ep.get("capabilities", []))
    run_id = a.run_id or (f"mock-{a.mock}" if a.mock else (a.model or model_id).replace("/", "_"))
    runner.run(con, run_id, items, cfg)
    metrics.print_summary(metrics.compute(con, run_id))


def cmd_report(a):
    import report  # lazy: only this command needs matplotlib
    con = storage.connect(a.db)
    rids = a.runs or [r["run_id"] for r in storage.list_runs(con)]
    if not rids:
        sys.exit("no runs found")
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


def main():
    p = argparse.ArgumentParser(prog="reasoning-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

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

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
