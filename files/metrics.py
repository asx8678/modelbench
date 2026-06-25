"""
Metrics computed from stored responses. Everything is derived from the raw
responses table, so you can recompute after editing this file without re-running.

Key principles from the methodology:
  * Report DISTRIBUTIONS with uncertainty, not single numbers. Cells use a Wilson
    95% interval (correct for proportions and shrinks with n) rather than the raw
    Bernoulli std (which doesn't).
  * Errors are not wrong answers. Items with no usable response are excluded from
    accuracy and reported as a coverage gap.
  * The distractor/surface probes are MATCHED pairs, so they're compared pairwise.
"""

import math
import re
from collections import defaultdict, Counter
import numpy as np


def _fetch(con, run_id):
    """One row per (item, sample) joined with its dataset metadata."""
    q = """SELECT r.item_id, r.sample_idx, r.correct, r.confidence, r.parsed, r.raw,
                  d.family, d.difficulty, d.probe, d.grp, d.gold, d.answer_type,
                  d.choices
           FROM responses r JOIN dataset d ON r.item_id = d.item_id
           WHERE r.run_id = ?"""
    return [dict(x) for x in con.execute(q, (run_id,))]


def _wilson(k, n, z=1.96):
    """Return (point, lo, hi): point estimate and Wilson score interval."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _is_correct(parsed, gold, answer_type):
    if parsed is None:
        return False
    if answer_type == "int":
        try:
            return int(parsed) == int(gold)
        except (ValueError, TypeError):
            return False
    return str(parsed).lower() == str(gold).lower()


def _modal_frequency(golds):
    """Best constant-guess accuracy: the frequency of the most common answer.

    This is the honest 'chance' floor for a family with a bounded answer
    space -- the score a model gets by always guessing the single most likely
    label without doing any reasoning. It captures skew in the realized answer
    distribution that a flat 1/k baseline misses.
    """
    if not golds:
        return 0.0
    counts = Counter(golds)
    return counts.most_common(1)[0][1] / len(golds)


# Families whose answer is an UNBOUNDED integer: no constant guess can win, so
# their chance baseline is ~0. Everything else has a bounded answer space and is
# scored empirically -- including logic_grid, whose gold is an int (a floor) but
# bounded to 1..n, so it must NOT be lumped in here.
_UNBOUNDED_INT_FAMILIES = frozenset({
    "arithmetic", "state_tracking", "sequences", "composed",
    "retroactive_edit", "multi_turn_inject",
})


def _family_chance(family, golds):
    """Per-family chance baseline from the realized gold distribution.

    Unbounded-integer families have no winning constant guess, so chance ~ 0.
    Bounded families (ordering, knights_knaves, unsat_csp, and the bounded-int
    logic_grid) use the empirical modal-answer frequency, which reflects skew in
    the answer prior (e.g. unsat_csp's label distribution or logic_grid's
    low-floor bias) that a flat 1/k or 1/2^n value understates.
    """
    if family in _UNBOUNDED_INT_FAMILIES:
        return 0.0
    return _modal_frequency(golds)


# ---- helpers for re-parsing raw responses (marker vs fallback)
_ANS = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
_CONF = re.compile(r"(?im)^\s*confidence\s*[:=]\s*(\d{1,3})")
_INT = re.compile(r"-?\d+")


def _last_marker(text):
    m = list(_ANS.finditer(text))
    return m[-1].group(1).strip() if m else None


def _strip_confidence(text):
    return _CONF.sub("", text)


def _parse_int(text):
    nums = _INT.findall(text.replace(",", ""))
    return str(int(nums[-1])) if nums else None


def _parse_choice(text, choices):
    if not choices:
        return None
    last, pos = None, -1
    for c in choices:
        for hit in re.finditer(rf"\b{re.escape(c.lower())}\b", text.lower()):
            if hit.start() > pos:
                pos, last = hit.start(), c
    return last


def _parse_marker_only(text, answer_type, choices):
    marker = _last_marker(text)
    if marker is None:
        return None
    if answer_type == "int":
        return _parse_int(marker)
    if answer_type == "choice":
        mk = marker.lower()
        for c in choices or []:
            if re.search(rf"\b{re.escape(c.lower())}\b", mk):
                return c
        return None
    return marker


def _parse_fallback(text, answer_type, choices):
    text = _strip_confidence(text)
    if answer_type == "int":
        return _parse_int(text)
    if answer_type == "choice":
        return _parse_choice(text, choices)
    return _last_marker(text)


def compute(con, run_id):
    rows = _fetch(con, run_id)
    if not rows:
        return {"error": "no responses for run"}

    META_KEYS = ("family", "difficulty", "probe", "grp", "gold", "answer_type", "choices")
    meta = {}                       # item_id -> static metadata
    s0_correct = {}                 # item_id -> bool (sample 0, only if not an error)
    s0_parsed = {}                  # item_id -> parsed answer (sample 0)
    s0_conf = {}                    # item_id -> confidence (sample 0)
    samples = defaultdict(list)     # item_id -> [(idx, correct, parsed, raw), ...]
    nsamples = 0
    for r in rows:
        iid = r["item_id"]
        meta.setdefault(iid, {k: r[k] for k in META_KEYS})
        samples[iid].append((r["sample_idx"], r["correct"], r["parsed"], r["raw"]))
        nsamples = max(nsamples, r["sample_idx"] + 1)
        if r["sample_idx"] == 0:
            s0_parsed[iid] = r["parsed"]
            s0_conf[iid] = r["confidence"]
            if r["correct"] is not None:           # None == errored, exclude from accuracy
                s0_correct[iid] = bool(r["correct"])

    base = [i for i, m in meta.items() if m["probe"] == "base"]

    # ---- coverage (errors are a gap, not a wrong answer)
    n_items = len(meta)
    answered = sum(1 for i in meta if i in s0_correct)
    coverage = {"n_items": n_items, "answered": answered,
                "errored": n_items - answered,
                "coverage": float(answered / n_items) if n_items else 0.0}

    # ---- accuracy by family (single-shot, base items, errors excluded)
    by_fam = defaultdict(list)
    for i in base:
        if i in s0_correct:
            by_fam[meta[i]["family"]].append(s0_correct[i])
    fam_acc = {f: float(np.mean(v)) for f, v in by_fam.items()}
    base_ok = [s0_correct[i] for i in base if i in s0_correct]
    overall = float(np.mean(base_ok)) if base_ok else 0.0

    # ---- strict (marker-only) accuracy: gives grading-fragility teeth (E4).
    # The lenient grader can rescue a format-noncompliant response via the
    # last-integer / trailing-choice fallback. Strict accuracy counts only
    # responses whose ANSWER: marker parses; fallback_reliance is the share of
    # ANSWERED base items graded correct ONLY because of that rescue -- a high
    # value means the headline reflects format luck, not reasoning.
    def _choices_of(i):
        ch = meta[i].get("choices")
        if isinstance(ch, str) and ch:
            return ch.split("|")
        return ch or None
    s0_raw = {}
    for i in base:
        for idx, _c, _p, rw in samples[i]:
            if idx == 0:
                s0_raw[i] = rw
                break
    strict_correct = {}
    for i in base:
        if i not in s0_correct:               # errored / unanswered: not scored either way
            continue
        raw = s0_raw.get(i)
        pm = _parse_marker_only(raw, meta[i]["answer_type"], _choices_of(i)) if raw else None
        strict_correct[i] = _is_correct(pm, meta[i]["gold"], meta[i]["answer_type"])
    strict_base = [strict_correct[i] for i in base if i in strict_correct]
    overall_strict = float(np.mean(strict_base)) if strict_base else 0.0
    fallback_dependent = sum(1 for i in base
                             if s0_correct.get(i) and not strict_correct.get(i, False))
    fallback_reliance = float(fallback_dependent / answered) if answered else None

    # ---- degradation curve + Wilson CI: accuracy vs difficulty, per family
    curve = defaultdict(dict)
    bucket = defaultdict(lambda: defaultdict(list))
    for i in base:
        if i in s0_correct:
            bucket[meta[i]["family"]][meta[i]["difficulty"]].append(s0_correct[i])
    for fam, diffs in bucket.items():
        for d, vals in sorted(diffs.items()):
            k, n = int(np.sum(vals)), len(vals)
            mean, lo, hi = _wilson(k, n)
            curve[fam][d] = {"mean": mean, "lo": lo, "hi": hi, "n": n,
                             "std": float(np.std(vals))}

    # ---- distractibility: PAIRED base vs matched NoOp-distractor (by grp)
    grp = defaultdict(dict)
    for i, m in meta.items():
        if m["probe"] in ("base", "distractor"):
            grp[m["grp"]][m["probe"]] = i
    fam_pairs = defaultdict(list)
    for g, d in grp.items():
        bi, di = d.get("base"), d.get("distractor")
        if bi in s0_correct and di in s0_correct:
            fam_pairs[meta[bi]["family"]].append((s0_correct[bi], s0_correct[di]))
    distract = {}
    for fam, pairs in fam_pairs.items():
        b = float(np.mean([bo for bo, _ in pairs]))
        dd = float(np.mean([do for _, do in pairs]))
        distract[fam] = {
            "base_acc": b, "distractor_acc": dd, "drop": b - dd, "n": len(pairs),
            "hurt": sum(1 for bo, do in pairs if bo and not do),    # distractor broke it
            "helped": sum(1 for bo, do in pairs if do and not bo),  # distractor "fixed" it
        }

    # ---- surface invariance: PAIRED vs base, per family (sample 0)
    grp_base = {}
    grp_surf = defaultdict(list)
    for i, m in meta.items():
        if m["probe"] == "base":
            grp_base[m["grp"]] = i
        elif m["probe"] == "surface":
            grp_surf[m["grp"]].append(i)
    inv_fam = defaultdict(lambda: {"groups": 0, "flip_groups": 0,
                                   "surf_total": 0, "surf_correct": 0})
    for g, surf_items in grp_surf.items():
        bi = grp_base.get(g)
        base_ans = s0_parsed.get(bi) if bi is not None else None
        if base_ans is None:                       # can't compare if base didn't parse
            continue
        f = inv_fam[meta[bi]["family"]]
        f["groups"] += 1
        flipped = False
        for si in surf_items:
            if s0_parsed.get(si) != base_ans:      # None counts as a flip, not "consistent"
                flipped = True
            if si in s0_correct:
                f["surf_total"] += 1
                f["surf_correct"] += int(s0_correct[si])
        if flipped:
            f["flip_groups"] += 1

    def _inv(d):
        g = d["groups"]
        return {"groups": g,
                "answer_flip_rate": float(d["flip_groups"] / g) if g else None,
                "consistent_rate": float((g - d["flip_groups"]) / g) if g else None,
                "surface_accuracy": float(d["surf_correct"] / d["surf_total"]) if d["surf_total"] else None}
    by_family_inv = {fam: _inv(d) for fam, d in inv_fam.items()}
    agg = {"groups": sum(d["groups"] for d in inv_fam.values()),
           "flip_groups": sum(d["flip_groups"] for d in inv_fam.values()),
           "surf_total": sum(d["surf_total"] for d in inv_fam.values()),
           "surf_correct": sum(d["surf_correct"] for d in inv_fam.values())}
    invariance = _inv(agg)
    invariance["by_family"] = by_family_inv

    # ---- calibration (base items only, sample 0, if confidences were collected)
    conf_rows = [(s0_conf[i], s0_correct[i]) for i in base
                 if i in s0_correct and s0_conf.get(i) is not None]
    calibration = None
    if conf_rows:
        bins = defaultdict(list)
        for c, ok in conf_rows:
            bins[min(9, c // 10)].append((c, ok))
        ece = 0.0
        total = len(conf_rows)
        table = []
        for b in sorted(bins):
            cs = bins[b]
            acc = np.mean([ok for _, ok in cs])
            conf = np.mean([c for c, _ in cs]) / 100
            ece += (len(cs) / total) * abs(acc - conf)
            hi = b * 10 + (10 if b == 9 else 9)
            table.append({"bin": f"{b*10}-{hi}", "n": len(cs),
                          "avg_conf": float(conf), "accuracy": float(acc)})
        calibration = {"ece": float(ece), "bins": table}

    # ---- pass@1, majority-vote (self-consistency), pass@k oracle upper bound
    passk = None
    if nsamples > 1:
        p1, maj, oracle = [], [], []
        for i in base:
            ss = samples[i]
            cs = [c for _, c, _, _ in ss if c is not None]
            if not cs:
                continue                            # all samples errored
            if i in s0_correct:
                p1.append(s0_correct[i])
            oracle.append(any(bool(c) for c in cs))
            votes = [p for _, _, p, _ in ss if p is not None]
            if votes:
                win = Counter(votes).most_common(1)[0][0]
                maj.append(_is_correct(win, meta[i]["gold"], meta[i]["answer_type"]))
        passk = {"k": nsamples,
                 "pass@1": float(np.mean(p1)) if p1 else 0.0,
                 "maj@k": float(np.mean(maj)) if maj else 0.0,
                 "pass@k_oracle": float(np.mean(oracle)) if oracle else 0.0}

    # ---- behavioral uncertainty (uses sampled answer distribution per item)
    entropies = []
    for i in base:
        votes = [p for _, _, p, _ in samples[i] if p is not None]
        if not votes:
            continue
        total = len(votes)
        counts = Counter(votes)
        h = 0.0
        for cnt in counts.values():
            p = cnt / total
            h -= p * math.log2(p)
        entropies.append(h)
    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
    sc_gap = (passk["maj@k"] - passk["pass@1"]) if passk else None
    stated_ece = calibration["ece"] if calibration else None
    behavioral_uncertainty = {
        "disagreement_entropy": avg_entropy,
        "selfconsistency_gap": sc_gap,
        "stated_confidence_ece": stated_ece,
    }


    # ---- chance-corrected accuracy per family (empirical best-constant-guess)
    chance_baseline = {}
    acc_above_chance = {}
    for fam in by_fam:
        golds = [meta[i]["gold"] for i in base
                 if meta[i]["family"] == fam and i in s0_correct]
        chance = _family_chance(fam, golds)
        chance_baseline[fam] = chance
        acc = fam_acc[fam]
        if 0.0 < chance < 1.0:
            acc_above_chance[fam] = float((acc - chance) / (1.0 - chance))
        else:
            acc_above_chance[fam] = acc

    frontier_headroom = None
    if passk is not None:
        frontier_headroom = passk["pass@k_oracle"] - passk["pass@1"]

    # ---- grading fragility: marker-only vs fallback parse disagreement rate
    frag_disagree = 0
    frag_total = 0
    for i in base:
        answer_type = meta[i]["answer_type"]
        choices = meta[i].get("choices")
        if isinstance(choices, str) and choices:
            choices = choices.split("|")
        elif not choices:
            choices = None
        for _, _, _, raw in samples[i]:
            if raw == "__ERROR__" or raw is None:
                continue
            pm = _parse_marker_only(raw, answer_type, choices)
            pf = _parse_fallback(raw, answer_type, choices)
            frag_total += 1
            if pm != pf:
                frag_disagree += 1
    grading_fragility = float(frag_disagree / frag_total) if frag_total else None


    # ---- confabulation rate: ill-posed items answered with concrete value
    ill_posed = [i for i in base if meta[i]["gold"] in ("UNDETERMINED", "NO_SOLUTION")]
    confab_count = 0
    for i in ill_posed:
        if i in s0_correct:
            parsed = s0_parsed.get(i)
            if parsed and parsed not in ("UNDETERMINED", "NO_SOLUTION"):
                confab_count += 1
    confabulation_rate = float(confab_count / len(ill_posed)) if ill_posed else None

    # ---- false undetermined rate: unique items answered as UNDETERMINED or NO_SOLUTION
    unique = [i for i in base if meta[i]["gold"] not in ("UNDETERMINED", "NO_SOLUTION")]
    false_undet_count = 0
    for i in unique:
        parsed = s0_parsed.get(i)
        if parsed in ("UNDETERMINED", "NO_SOLUTION"):
            false_undet_count += 1
    false_undetermined_rate = float(false_undet_count / len(unique)) if unique else None

    return {
        "run_id": run_id, "n_items": n_items, "samples_per_item": nsamples,
        "coverage": coverage,
        "overall_accuracy": overall, "accuracy_by_family": fam_acc,
        "overall_accuracy_strict": overall_strict,
        "fallback_reliance": fallback_reliance,
        "chance_baseline": chance_baseline,
        "acc_above_chance": acc_above_chance,
        "frontier_headroom": frontier_headroom,
        "grading_fragility": grading_fragility,
        "confabulation_rate": confabulation_rate,
        "degradation": {f: dict(d) for f, d in curve.items()},
        "distractibility": distract, "invariance": invariance,
        "calibration": calibration, "passk": passk,
        "false_undetermined_rate": false_undetermined_rate,
        "behavioral_uncertainty": behavioral_uncertainty,
    }


def print_summary(res):
    if "error" in res:
        print(res["error"]); return
    cov = res["coverage"]
    print(f"\n=== run {res['run_id']}  ({res['n_items']} items, {res['samples_per_item']} sample/item) ===")
    if cov["errored"]:
        print(f"coverage: {cov['answered']}/{cov['n_items']} answered "
              f"({cov['errored']} errored, excluded from accuracy)")
    print(f"overall single-shot accuracy: {res['overall_accuracy']:.3f}")
    if res.get("overall_accuracy_strict") is not None:
        fr = res.get("fallback_reliance")
        tail = f"   fallback-reliance {fr:.3f}" if fr is not None else ""
        print(f"  strict (marker-only) accuracy: {res['overall_accuracy_strict']:.3f}{tail}")
    print()
    print("accuracy by family:")
    for f, a in sorted(res["accuracy_by_family"].items()):
        print(f"  {f:16s} {a:.3f}")
    print("\ndegradation (accuracy ± 95% CI half-width by difficulty):")
    for f, d in sorted(res["degradation"].items()):
        cells = "  ".join(f"d{k}:{v['mean']:.2f}±{(v['hi']-v['lo'])/2:.2f}"
                          for k, v in sorted(d.items()))
        print(f"  {f:16s} {cells}")
    if res["distractibility"]:
        print("\ndistractibility (paired base vs +irrelevant clause):")
        for f, v in sorted(res["distractibility"].items()):
            print(f"  {f:16s} base {v['base_acc']:.2f} -> distractor {v['distractor_acc']:.2f}  "
                  f"drop {v['drop']:+.2f}  (n={v['n']}, hurt {v['hurt']}/helped {v['helped']})")
    inv = res["invariance"]
    if inv["groups"]:
        print(f"\nsurface invariance: answer-flip rate {inv['answer_flip_rate']:.3f} "
              f"over {inv['groups']} groups (lower = more robust); "
              f"surface accuracy {inv['surface_accuracy']:.3f}")
        for f, v in sorted(inv["by_family"].items()):
            print(f"  {f:16s} flip {v['answer_flip_rate']:.3f}  over {v['groups']} groups")
    if res["calibration"]:
        print(f"\ncalibration ECE: {res['calibration']['ece']:.3f} (0 = perfectly calibrated)")
    if res["passk"]:
        pk = res["passk"]
        print(f"\npass@1 {pk['pass@1']:.3f}  |  maj@{pk['k']} {pk['maj@k']:.3f}  |  "
              f"pass@{pk['k']} (oracle) {pk['pass@k_oracle']:.3f}")
    print()
