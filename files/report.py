"""
Reporting. Builds an accessibility-first comparison report across one or more runs
(e.g. Llama vs Gemma): degradation curves, distractibility, and a Markdown summary.

Accessibility choices (deliberate):
  * Okabe-Ito colorblind-safe palette.
  * Redundant encoding: each model gets a distinct color AND marker AND line style,
    so the charts are readable in greyscale and for color-vision deficiency.
  * DejaVu Sans, large fonts, high-contrast gridlines.
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import metrics
import generators

# Okabe-Ito (skip pure black for lines on white; keep for text)
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), "-"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130,
})


def _style(i):
    return dict(color=OKABE[i % len(OKABE)],
                marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=2.2, markersize=8, markeredgecolor="black", markeredgewidth=0.6)


def degradation_chart(run_results, labels, outpath):
    """One subplot per family; lines = models; error bars = std across structures."""
    fams = sorted({f for res in run_results for f in res.get("degradation", {})})
    if not fams:
        return None
    cols = min(2, len(fams)); rows = (len(fams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat[len(fams):]:
        ax.axis("off")

    for ax, fam in zip(axes.flat, fams):
        for i, (res, lab) in enumerate(zip(run_results, labels)):
            d = res.get("degradation", {}).get(fam, {})
            if not d:
                continue
            xs = sorted(d)
            ys = [d[x]["mean"] for x in xs]
            lower = [max(0.0, d[x]["mean"] - d[x]["lo"]) for x in xs]
            upper = [max(0.0, d[x]["hi"] - d[x]["mean"]) for x in xs]
            ax.errorbar(xs, ys, yerr=[lower, upper], label=lab, capsize=4, **_style(i))
        ax.set_title(fam, fontweight="bold")
        # E6: the difficulty axis is not the same quantity across families
        # (reasoning steps vs rule tier vs problem size). Label each honestly so
        # the curves are not misread as one shared "more steps" axis.
        ax.set_xlabel(generators.difficulty_axis(fam))
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=11, framealpha=0.9)
    fig.suptitle("Accuracy vs difficulty axis  (per-family; axes are NOT commensurable — "
                 "error bars = Wilson 95% CI)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def distractibility_chart(run_results, labels, outpath):
    """Grouped bars: base vs distractor accuracy per family, per model."""
    fams = sorted({f for res in run_results for f in res.get("distractibility", {})})
    if not fams:
        return None
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(fams) * len(labels)), 4.8))
    n = len(labels); width = 0.8 / max(n, 1)
    x = range(len(fams))
    for i, (res, lab) in enumerate(zip(run_results, labels)):
        dd = res.get("distractibility", {})
        base = [dd.get(f, {}).get("base_acc", 0) for f in fams]
        dist = [dd.get(f, {}).get("distractor_acc", 0) for f in fams]
        off = (i - (n - 1) / 2) * width
        c = OKABE[i % len(OKABE)]
        ax.bar([xi + off for xi in x], base, width * 0.92, color=c,
               edgecolor="black", linewidth=0.6, label=f"{lab} — base")
        ax.bar([xi + off for xi in x], dist, width * 0.92, color=c, alpha=0.45,
               edgecolor="black", linewidth=0.6, hatch="///",
               label=f"{lab} — +distractor")
    ax.set_xticks(list(x)); ax.set_xticklabels(fams)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.03)
    ax.set_title("Distractibility: solid = base, hatched = irrelevant clause added",
                 fontweight="bold")
    ax.legend(fontsize=10, ncol=max(1, n), framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath

def confabulation_vs_confidence_chart(run_results, labels, outpath):
    """Scatter: confabulation rate vs stated confidence (ECE/average) per model."""
    xs, ys, labs = [], [], []
    for i, (res, lab) in enumerate(zip(run_results, labels)):
        cr = res.get("confabulation_rate")
        if cr is None:
            continue
        conf = res.get("calibration")
        y = conf["ece"] if conf else None
        if y is None:
            continue
        xs.append(cr)
        ys.append(y)
        labs.append(lab)
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i, (x, y, lab) in enumerate(zip(xs, ys, labs)):
        style = _style(i)
        ax.plot(x, y, **{k: v for k, v in style.items() if k not in ("linestyle", "markersize")},
                linestyle="none", markersize=10, label=lab)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=10)

    ax.set_xlabel("confabulation rate")
    ax.set_ylabel("expected calibration error (ECE)")
    ax.set_title("Confabulation vs Confidence", fontweight="bold")
    ax.set_xlim(-0.03, max(1.03, max(xs) * 1.05))
    ax.set_ylim(-0.03, max(1.03, max(ys) * 1.05) if ys else 1.03)
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def behavioral_uncertainty_chart(run_results, labels, outpath):
    """Grouped bars: disagreement entropy and self-consistency gap per model.

    These are the validated behavioral-uncertainty signals (mf4.3): the
    gap between maj@k and pass@1 (selfconsistency_gap) and the Shannon
    entropy of the k-sample answer distribution. Both are derived from
    the stored responses, no model calls.
    """
    if not labels:
        return None
    n = len(labels)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n * 2.0), 4.6))
    xs = list(range(n))
    for i, (res, lab) in enumerate(zip(run_results, labels)):
        bu = res.get("behavioral_uncertainty") or {}
        ent = bu.get("disagreement_entropy") or 0.0
        sc = bu.get("selfconsistency_gap")
        sc = sc if sc is not None else 0.0
        off = (i - (n - 1) / 2) * width
        c = OKABE[i % len(OKABE)]
        ax.bar([x + off - width / 2 for x in xs], [ent] * n,
               width, color=c, edgecolor="black", linewidth=0.6)
        ax.bar([x + off + width / 2 for x in xs], [sc] * n,
               width, color=c, alpha=0.55, edgecolor="black", linewidth=0.6,
               hatch="//")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("bits / gap")
    ax.set_title("Behavioral uncertainty: entropy (solid) + self-consistency gap (hatched)",
                 fontweight="bold")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor="grey", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="grey", alpha=0.55, hatch="//",
                       edgecolor="black"),
    ], labels=["disagreement entropy (bits)", "maj@k − pass@1"], fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath



def accuracy_above_chance_chart(run_results, labels, outpath):
    """Grouped bars: chance-corrected accuracy per family, per model."""
    fams = sorted({f for res in run_results for f in res.get("acc_above_chance", {})})
    if not fams:
        return None
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(fams) * len(labels)), 4.8))
    n = len(labels); width = 0.8 / max(n, 1)
    x = range(len(fams))
    for i, (res, lab) in enumerate(zip(run_results, labels)):
        vals = res.get("acc_above_chance", {})
        ys = [vals.get(f) for f in fams]
        off = (i - (n - 1) / 2) * width
        c = OKABE[i % len(OKABE)]
        ax.bar([xi + off for xi in x],
               [y if y is not None else 0 for y in ys],
               width * 0.92, color=c, edgecolor="black", linewidth=0.6,
               label=lab)
    ax.set_xticks(list(x)); ax.set_xticklabels(fams)
    ax.set_ylabel("accuracy above chance")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Accuracy Above Chance by Family", fontweight="bold")
    ax.legend(fontsize=10, ncol=max(1, n), framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath



def write_csv(run_results, labels, outpath):
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "family", "difficulty", "value"])
        for res, lab in zip(run_results, labels):
            w.writerow([lab, "overall_accuracy", "", "", round(res["overall_accuracy"], 4)])
            if res.get("overall_accuracy_strict") is not None:
                w.writerow([lab, "overall_accuracy_strict", "", "", round(res["overall_accuracy_strict"], 4)])
            if res.get("fallback_reliance") is not None:
                w.writerow([lab, "fallback_reliance", "", "", round(res["fallback_reliance"], 4)])
            for fam, a in res["accuracy_by_family"].items():
                w.writerow([lab, "family_accuracy", fam, "", round(a, 4)])
            for fam, d in res["degradation"].items():
                for diff, v in d.items():
                    w.writerow([lab, "acc_mean", fam, diff, round(v["mean"], 4)])
                    w.writerow([lab, "acc_ci_lo", fam, diff, round(v["lo"], 4)])
                    w.writerow([lab, "acc_ci_hi", fam, diff, round(v["hi"], 4)])
            for fam, v in res.get("distractibility", {}).items():
                w.writerow([lab, "distractor_drop", fam, "", round(v["drop"], 4)])
            inv = res["invariance"]
            if inv["groups"]:
                w.writerow([lab, "answer_flip_rate", "", "", round(inv["answer_flip_rate"], 4)])
                for fam, v in inv.get("by_family", {}).items():
                    if v["answer_flip_rate"] is not None:
                        w.writerow([lab, "answer_flip_rate", fam, "", round(v["answer_flip_rate"], 4)])
            if res["coverage"]["errored"]:
                w.writerow([lab, "coverage", "", "", round(res["coverage"]["coverage"], 4)])
            if res["calibration"]:
                w.writerow([lab, "ece", "", "", round(res["calibration"]["ece"], 4)])
            if res.get("passk"):                     # only when n>1 (single-sample runs have no pass@k)
                w.writerow([lab, "pass@k_oracle", "", "", round(res["passk"]["pass@k_oracle"], 4)])
            if res.get("confabulation_rate") is not None:
                w.writerow([lab, "confabulation_rate", "", "", round(res["confabulation_rate"], 4)])
            for fam, v in res.get("acc_above_chance", {}).items():
                if v is not None:
                    w.writerow([lab, "acc_above_chance", fam, "", round(v, 4)])


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _confab_cell(r):
    return f"{r['confabulation_rate']:.3f}" if r.get("confabulation_rate") is not None else "—"


def _above_chance_cell(r):
    vals = [v for v in r.get("acc_above_chance", {}).values() if v is not None]
    if not vals:
        return "—"
    return f"{float(np.mean(vals)):.3f}"


def _acc_above_chance_cell(r, fam):
    v = r.get("acc_above_chance", {}).get(fam)
    return f"{v:.3f}" if v is not None else "—"

def _strict_cell(r):
    s = r.get("overall_accuracy_strict")
    return f"{s:.3f}" if s is not None else "—"


def _fallback_cell(r):
    fr = r.get("fallback_reliance")
    return f"{fr:.3f}" if fr is not None else "—"


def _coverage_cell(r):
    c = r["coverage"]
    return f"{c['coverage']:.3f}" if c["errored"] else "1.000"


def _flip_cell(r):
    return f"{r['invariance']['answer_flip_rate']:.3f}" if r["invariance"]["groups"] else "—"


def _ece_cell(r):
    return f"{r['calibration']['ece']:.3f}" if r["calibration"] else "—"


def _passk_cell(r):
    pk = r["passk"]
    if not pk:
        return "—"
    return f"{pk['pass@1']:.3f} → maj {pk['maj@k']:.3f} → oracle {pk['pass@k_oracle']:.3f}"

def write_markdown(run_results, labels, charts, outpath):
    L = ["# Reasoning benchmark report", ""]
    L.append(_md_table(
        ["model", "overall acc", "strict acc", "fallback-reliance", "coverage",
         "confabulation", "answer-flip rate", "ECE",
         "acc above chance", "pass@1 → maj@k → oracle"],
        [[lab, f"{r['overall_accuracy']:.3f}", _strict_cell(r), _fallback_cell(r),
          _coverage_cell(r), _confab_cell(r),
          _flip_cell(r), _ece_cell(r), _above_chance_cell(r), _passk_cell(r)]
         for r, lab in zip(run_results, labels)]))
    L += ["", "## Accuracy above chance by family", ""]
    afams = sorted({f for r in run_results for f in r.get("acc_above_chance", {})})
    if afams:
        L.append(_md_table(["family"] + labels,
                           [[f] + [_acc_above_chance_cell(r, f) for r in run_results]
                            for f in afams]))
    if any(r.get("distractibility") for r in run_results):
        L += ["", "## Distractibility (accuracy drop from irrelevant clause)", ""]
        dfams = sorted({f for r in run_results for f in r.get("distractibility", {})})
        L.append(_md_table(["family"] + labels,
                           [[f] + [f"{r.get('distractibility', {}).get(f, {}).get('drop', float('nan')):+.3f}"
                                   for r in run_results] for f in dfams]))
    L += ["", "## Charts", ""]
    for c in charts:
        if c:
            L.append(f"![{os.path.basename(c)}]({os.path.basename(c)})")
    L += ["", "---",
          "*Interpreting the numbers:* high accuracy that **also** holds its value as "
          "difficulty rises (flat degradation curve), survives distractors (small drop), "
          "and stays consistent across surface variants (low flip rate) is the signature of "
          "reasoning rather than pattern-matching. A single accuracy number is not."]
    with open(outpath, "w") as f:
        f.write("\n".join(L))
    return outpath


def build_report(con, run_ids, outdir):
    os.makedirs(outdir, exist_ok=True)
    results, labels = [], []
    for rid in run_ids:
        res = metrics.compute(con, rid)
        if "error" in res:
            print(f"skip {rid}: {res['error']}"); continue
        # label by model name from runs table
        row = con.execute("SELECT model FROM runs WHERE run_id=?", (rid,)).fetchone()
        results.append(res)
        labels.append([(row["model"] if row else rid) or rid, rid])
    if not results:
        print("nothing to report"); return None
    # if two runs share a model name, fall back to run_id so the legend stays distinct
    names = [m for m, _ in labels]
    labels = [m if names.count(m) == 1 else rid for m, rid in labels]
    charts = [
        degradation_chart(results, labels, os.path.join(outdir, "degradation.png")),
        distractibility_chart(results, labels, os.path.join(outdir, "distractibility.png")),
        confabulation_vs_confidence_chart(results, labels, os.path.join(outdir, "confabulation_vs_confidence.png")),
        accuracy_above_chance_chart(results, labels, os.path.join(outdir, "accuracy_above_chance.png")),
        behavioral_uncertainty_chart(results, labels, os.path.join(outdir, "behavioral_uncertainty.png")),
    ]
    write_csv(results, labels, os.path.join(outdir, "metrics.csv"))
    md = write_markdown(results, labels, charts, os.path.join(outdir, "report.md"))
    print(f"report written to {outdir}")
    return md
