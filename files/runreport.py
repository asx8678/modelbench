"""
Per-run reporting. After a benchmark finishes, this writes a self-contained
folder for that single run with a human-readable account of *what was solved,
what was not, and why* — plus a styled HTML file named with the model and date.

Unlike report.py (which builds cross-run comparison charts and needs matplotlib),
this module is stdlib-only so it can always run right after a `run`. It reads the
already-stored responses from the DB; it makes no model calls.

Folder layout (under <root>/<model>_<run-id>_<date>/):
    report.md                 detailed solved/unsolved breakdown with reasons
    <model>_<date>.html       same content, styled, named with model + date
    run.log                   plain-text log: params, summary, one line per item

The full machine-readable results JSON is written alongside by the CLI.
"""

import os
import re
import csv
import json
import html
import time

import storage

# Outcome buckets. An item is "errored" when its call failed (no answer was
# recorded), "solved" when a majority of its samples graded correct, else "wrong".
SOLVED, WRONG, ERRORED = "solved", "wrong", "errored"


def _slug(s):
    """Filesystem-safe token: keep alnum / dot / dash / underscore, collapse the rest."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def _reasoning_summary(runtime):
    """One-line reasoning-token summary (total · mean/item · % of completion ·
    tokens/correct · efficiency · effort scaling), or '' when the provider exposed no reasoning count."""
    if not runtime or not runtime.get("reasoning_available"):
        return ""
    parts = [f"{runtime['reasoning_tokens_total']} total",
             f"mean {runtime['reasoning_tokens_mean']}/item"]
    if runtime.get("reasoning_fraction") is not None:
        parts.append(f"{runtime['reasoning_fraction'] * 100:.0f}% of completion")
    if runtime.get("reasoning_tokens_per_correct") is not None:
        parts.append(f"{runtime['reasoning_tokens_per_correct']:.0f}/correct")
    # Intelligence metrics
    if runtime.get("reasoning_correct_per_1k") is not None:
        parts.append(f"{runtime['reasoning_correct_per_1k']:.1f} correct/1k")
    if runtime.get("reasoning_effort_scaling") is not None:
        parts.append(f"effort {runtime['reasoning_effort_scaling']:.2f}×")
    return " · ".join(parts)


def _load_metadata(md):
    if isinstance(md, str):
        try:
            return json.loads(md)
        except (json.JSONDecodeError, TypeError):
            return {}
    return md or {}


def _classify(samples, gold):
    """Decide an item's outcome and a short, honest reason from its samples.

    `samples` is a list of dicts with keys correct/parsed/raw/parse_source.
    Mirrors runner._process: errored if any sample failed, else solved when a
    majority graded correct.
    """
    n = len(samples)
    errored = [s for s in samples if s["raw"] == storage.ERROR_MARKER]
    if errored:
        return ERRORED, "model call failed — no answer was recorded (transport/HTTP/parse error)"

    n_correct = sum(1 for s in samples if s["correct"])
    s0 = samples[0]
    src = s0.get("parse_source")
    parsed = s0.get("parsed")

    if n_correct * 2 > n:
        if src == "fallback":
            return SOLVED, "correct, but the ANSWER: marker was missing — rescued by fallback parse"
        if n > 1 and n_correct < n:
            return SOLVED, f"correct by majority ({n_correct}/{n} samples agreed)"
        return SOLVED, "correct"

    # wrong: explain the failure mode as precisely as the stored data allows
    if src == "none" or parsed in (None, ""):
        return WRONG, "no parseable answer — the response had no ANSWER: line and no usable fallback"
    if src == "fallback":
        return WRONG, f"wrong answer (got {parsed!r}, expected {gold!r}); format off — taken from fallback parse"
    if n > 1 and n_correct > 0:
        return WRONG, f"wrong by majority (got {parsed!r}, expected {gold!r}; only {n_correct}/{n} correct)"
    return WRONG, f"wrong answer (got {parsed!r}, expected {gold!r})"


def collect(con, run_id):
    """Gather per-item outcomes for a run, ordered by family then item.

    Returns (items, totals) where each item is a dict with family, difficulty,
    probe, gold, outcome and reason, and totals counts the three buckets.
    """
    rows = con.execute(
        """SELECT r.item_id, r.sample_idx, r.raw, r.parsed, r.correct, r.metadata,
                  d.family, d.difficulty, d.probe, d.gold, d.answer_type
             FROM responses r LEFT JOIN dataset d ON d.item_id = r.item_id
            WHERE r.run_id=? ORDER BY r.item_id, r.sample_idx""", (run_id,)).fetchall()

    by_item = {}
    for r in rows:
        iid = r["item_id"]
        it = by_item.setdefault(iid, {
            "item_id": iid, "family": r["family"], "difficulty": r["difficulty"],
            "probe": r["probe"], "gold": r["gold"], "answer_type": r["answer_type"],
            "samples": []})
        it["samples"].append({
            "correct": r["correct"], "parsed": r["parsed"], "raw": r["raw"],
            "parse_source": _load_metadata(r["metadata"]).get("parse_source")})

    items, totals = [], {SOLVED: 0, WRONG: 0, ERRORED: 0}
    for it in by_item.values():
        outcome, reason = _classify(it["samples"], it["gold"])
        totals[outcome] += 1
        items.append({k: it[k] for k in ("item_id", "family", "difficulty", "probe", "gold")}
                     | {"outcome": outcome, "reason": reason})
    items.sort(key=lambda x: (x["family"] or "", x["difficulty"] or 0, x["item_id"]))
    return items, totals


def _by_family(items):
    """family -> {solved, wrong, errored, total} counts."""
    fams = {}
    for it in items:
        f = it["family"] or "(unknown)"
        c = fams.setdefault(f, {SOLVED: 0, WRONG: 0, ERRORED: 0, "total": 0})
        c[it["outcome"]] += 1
        c["total"] += 1
    return dict(sorted(fams.items()))


def _acc(c):
    answered = c[SOLVED] + c[WRONG]
    return c[SOLVED] / answered if answered else 0.0


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _md(model_label, run_id, items, totals, fam_counts, metrics_res, runtime, created_iso):
    total = sum(totals.values())
    L = [f"# Benchmark report — {model_label}", "",
         f"- **Run id:** `{run_id}`",
         f"- **Model:** {model_label}",
         f"- **Run started:** {created_iso or '—'}",
         f"- **Report generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    acc = metrics_res.get("overall_accuracy") if metrics_res else None
    L += ["## Summary", "",
          f"- **Items:** {total}",
          f"- **Solved:** {totals[SOLVED]}  ({totals[SOLVED] / total * 100:.1f}%)" if total else "- Solved: 0",
          f"- **Wrong:** {totals[WRONG]}  ({totals[WRONG] / total * 100:.1f}%)" if total else "- Wrong: 0",
          f"- **Errored:** {totals[ERRORED]}  ({totals[ERRORED] / total * 100:.1f}%)" if total else "- Errored: 0"]
    if acc is not None:
        L.append(f"- **Overall single-shot accuracy (base items):** {acc:.3f}")
    if runtime:
        L.append(f"- **Latency:** p50 {runtime['latency_p50_ms']} ms · "
                 f"p95 {runtime['latency_p95_ms']} ms · mean {runtime['latency_mean_ms']} ms")
        if runtime.get("tokens_available"):
            L.append(f"- **Tokens:** {runtime['prompt_tokens_total']} prompt · "
                     f"{runtime['completion_tokens_total']} completion")
        rs = _reasoning_summary(runtime)
        if rs:
            L.append(f"- **Reasoning tokens:** {rs}")
        # Reasoning by difficulty (intelligence metrics)
        if runtime.get("reasoning_by_difficulty"):
            by_diff = runtime["reasoning_by_difficulty"]
            L.append("")
            L.append("### Reasoning by difficulty")
            L.append("")
            L.append("| difficulty | mean tokens | accuracy | n |")
            L.append("| --- | --- | --- | --- |")
            for diff in sorted(by_diff.keys()):
                d = by_diff[diff]
                L.append(f"| {diff} | {d['mean_reasoning_tokens']} | {d['accuracy']:.3f} | {d['n']} |")
    L.append("")

    L += ["## By family", "",
          "| family | solved | wrong | errored | accuracy |",
          "| --- | --- | --- | --- | --- |"]
    for f, c in fam_counts.items():
        L.append(f"| {f} | {c[SOLVED]} | {c[WRONG]} | {c[ERRORED]} | {_acc(c):.3f} |")
    L.append("")

    not_solved = [it for it in items if it["outcome"] != SOLVED]
    L += [f"## Not solved ({len(not_solved)}) — what failed and why", ""]
    if not not_solved:
        L.append("_Everything was solved._")
    else:
        L += ["| item | family | diff | probe | outcome | reason |",
              "| --- | --- | --- | --- | --- | --- |"]
        for it in not_solved:
            L.append(f"| `{it['item_id']}` | {it['family']} | {it['difficulty']} | "
                     f"{it['probe']} | {it['outcome']} | {it['reason']} |")
    L.append("")

    solved = [it for it in items if it["outcome"] == SOLVED]
    L += [f"## Solved ({len(solved)})", ""]
    if solved:
        L += ["| item | family | diff | probe |",
              "| --- | --- | --- | --- |"]
        for it in solved:
            L.append(f"| `{it['item_id']}` | {it['family']} | {it['difficulty']} | {it['probe']} |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
body{font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     color:#1a1a1a;max-width:1000px;margin:2rem auto;padding:0 1.2rem}
h1{margin-bottom:.2rem}.sub{color:#666;margin-bottom:1.4rem}
.cards{display:flex;gap:.8rem;flex-wrap:wrap;margin:1rem 0 1.6rem}
.card{flex:1;min-width:120px;border:1px solid #e3e3e3;border-radius:10px;padding:.8rem 1rem}
.card .n{font-size:1.7rem;font-weight:700}.card .l{color:#666;font-size:.85rem}
.solved .n{color:#198754}.wrong .n{color:#b8860b}.errored .n{color:#c0392b}
table{border-collapse:collapse;width:100%;margin:.6rem 0 1.6rem;font-size:.92rem}
th,td{border:1px solid #e3e3e3;padding:.4rem .6rem;text-align:left;vertical-align:top}
th{background:#f6f8fa}tr:nth-child(even) td{background:#fafbfc}
code{background:#f0f2f4;padding:.05rem .35rem;border-radius:4px;font-size:.86rem}
.tag{display:inline-block;padding:.05rem .5rem;border-radius:999px;font-size:.8rem;font-weight:600}
.t-solved{background:#d8f3e3;color:#0f5132}.t-wrong{background:#fcefca;color:#7a5b00}
.t-errored{background:#fadbd8;color:#922b21}
"""


def _tag(outcome):
    return f'<span class="tag t-{outcome}">{outcome}</span>'


def _html(model_label, run_id, items, totals, fam_counts, metrics_res, runtime, created_iso):
    total = sum(totals.values())
    esc = html.escape
    acc = metrics_res.get("overall_accuracy") if metrics_res else None
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>{esc(model_label)} — benchmark {time.strftime('%Y-%m-%d')}</title>",
           f"<style>{_CSS}</style></head><body>"]
    out.append(f"<h1>Benchmark report — {esc(model_label)}</h1>")
    sub = f"run <code>{esc(run_id)}</code> · started {esc(created_iso or '—')} · " \
          f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}"
    out.append(f"<div class='sub'>{sub}</div>")

    def card(cls, n, label):
        return f"<div class='card {cls}'><div class='n'>{n}</div><div class='l'>{label}</div></div>"
    pct = (lambda x: f"{x / total * 100:.1f}%") if total else (lambda x: "—")
    out.append("<div class='cards'>")
    out.append(card("", total, "items"))
    out.append(card("solved", f"{totals[SOLVED]}", f"solved · {pct(totals[SOLVED])}"))
    out.append(card("wrong", f"{totals[WRONG]}", f"wrong · {pct(totals[WRONG])}"))
    out.append(card("errored", f"{totals[ERRORED]}", f"errored · {pct(totals[ERRORED])}"))
    if acc is not None:
        out.append(card("", f"{acc:.3f}", "overall accuracy"))
    out.append("</div>")
    if runtime:
        out.append(f"<p class='sub'>Latency p50 {runtime['latency_p50_ms']} ms · "
                   f"p95 {runtime['latency_p95_ms']} ms · mean {runtime['latency_mean_ms']} ms")
        if runtime.get("tokens_available"):
            out.append(f" · tokens {runtime['prompt_tokens_total']} prompt / "
                       f"{runtime['completion_tokens_total']} completion")
        out.append("</p>")
        rs = _reasoning_summary(runtime)
        if rs:
            out.append(f"<p class='sub'>Reasoning tokens · {rs}</p>")

    out.append("<h2>By family</h2><table><tr><th>family</th><th>solved</th>"
               "<th>wrong</th><th>errored</th><th>accuracy</th></tr>")
    for f, c in fam_counts.items():
        out.append(f"<tr><td>{esc(f)}</td><td>{c[SOLVED]}</td><td>{c[WRONG]}</td>"
                   f"<td>{c[ERRORED]}</td><td>{_acc(c):.3f}</td></tr>")
    out.append("</table>")

    not_solved = [it for it in items if it["outcome"] != SOLVED]
    out.append(f"<h2>Not solved ({len(not_solved)}) — what failed and why</h2>")
    if not not_solved:
        out.append("<p>Everything was solved.</p>")
    else:
        out.append("<table><tr><th>item</th><th>family</th><th>diff</th><th>probe</th>"
                   "<th>outcome</th><th>reason</th></tr>")
        for it in not_solved:
            out.append(f"<tr><td><code>{esc(it['item_id'])}</code></td><td>{esc(it['family'] or '')}</td>"
                       f"<td>{it['difficulty']}</td><td>{esc(it['probe'] or '')}</td>"
                       f"<td>{_tag(it['outcome'])}</td><td>{esc(it['reason'])}</td></tr>")
        out.append("</table>")

    solved = [it for it in items if it["outcome"] == SOLVED]
    out.append(f"<h2>Solved ({len(solved)})</h2>")
    if solved:
        out.append("<table><tr><th>item</th><th>family</th><th>diff</th><th>probe</th></tr>")
        for it in solved:
            out.append(f"<tr><td><code>{esc(it['item_id'])}</code></td><td>{esc(it['family'] or '')}</td>"
                       f"<td>{it['difficulty']}</td><td>{esc(it['probe'] or '')}</td></tr>")
        out.append("</table>")
    out.append("</body></html>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Plain-text log + CSV
# ---------------------------------------------------------------------------

def _log(model_label, run_id, params, items, totals, metrics_res, runtime, created_iso):
    total = sum(totals.values())
    L = ["benchmark run log",
         f"model      : {model_label}",
         f"run id     : {run_id}",
         f"started    : {created_iso or '—'}",
         f"generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
         ""]
    if params:
        keys = ("model", "n", "temperature", "max_tokens", "workers", "timeout",
                "retries", "ask_confidence", "mock", "dataset_tag")
        L.append("params:")
        for k in keys:
            if k in params:
                L.append(f"  {k:14s} = {params[k]}")
        L.append("")
    L += ["summary:",
          f"  items   = {total}",
          f"  solved  = {totals[SOLVED]}",
          f"  wrong   = {totals[WRONG]}",
          f"  errored = {totals[ERRORED]}"]
    if metrics_res and metrics_res.get("overall_accuracy") is not None:
        L.append(f"  overall single-shot accuracy = {metrics_res['overall_accuracy']:.4f}")
    if runtime:
        L.append(f"  latency p50/p95/mean ms = {runtime['latency_p50_ms']}/"
                 f"{runtime['latency_p95_ms']}/{runtime['latency_mean_ms']}")
    L += ["", "per-item (item_id  family  diff  probe  outcome  reason):"]
    for it in items:
        L.append(f"  {it['item_id']:24s} {(it['family'] or ''):16s} "
                 f"d{it['difficulty']} {(it['probe'] or ''):10s} "
                 f"{it['outcome']:8s} {it['reason']}")
    return "\n".join(L) + "\n"


def _write_item_csv(items, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "family", "difficulty", "probe", "gold", "outcome", "reason"])
        for it in items:
            w.writerow([it["item_id"], it["family"], it["difficulty"], it["probe"],
                        it["gold"], it["outcome"], it["reason"]])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build(con, run_id, model_label, root, metrics_res=None, runtime=None,
          params=None, created_iso=None):
    """Write the per-run report folder. Returns the folder path, or None if the
    run has no stored responses. Best-effort: never raises on a single artifact.

    The folder is named <model>_<run-id>_<date>/ under `root`. The HTML file is
    named <model>_<date>.html (model + date + results), as requested.
    """
    items, totals = collect(con, run_id)
    if not items:
        return None
    fam_counts = _by_family(items)
    date = time.strftime("%Y-%m-%d")
    folder = os.path.join(root, f"{_slug(model_label)}_{_slug(run_id)}_{date}")
    os.makedirs(folder, exist_ok=True)

    md = _md(model_label, run_id, items, totals, fam_counts, metrics_res, runtime, created_iso)
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write(md)

    html_name = f"{_slug(model_label)}_{date}.html"
    with open(os.path.join(folder, html_name), "w") as f:
        f.write(_html(model_label, run_id, items, totals, fam_counts,
                      metrics_res, runtime, created_iso))

    with open(os.path.join(folder, "run.log"), "w") as f:
        f.write(_log(model_label, run_id, params, items, totals, metrics_res, runtime, created_iso))

    _write_item_csv(items, os.path.join(folder, "items.csv"))
    return folder
