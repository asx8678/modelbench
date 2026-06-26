"""
Terminal dashboard. Renders a run's metrics (from `metrics.compute`) as a rich,
colored, in-terminal display — meters, inline degradation sparklines, a
calibration diagram, distractibility, and runtime/token stats — plus a side-by-side
comparison view for multiple runs.

Design choices mirror runner._Progress (the other piece of live terminal output):

  * Dependency-free. Pure standard library, like the runner. No curses, no rich —
    just ANSI escapes computed from the already-computed metrics dict, so this stays
    importable everywhere `metrics` is (no matplotlib needed, unlike `report`).
  * Capability-aware and graceful. Color is emitted only to a real TTY that hasn't
    opted out (NO_COLOR / TERM=dumb); piping to a file yields clean plain text.
    Box-drawing/blocks degrade to ASCII when the stream can't encode them.
  * Accessible by construction. Every bar carries its numeric value AND a status
    glyph (✓ / ~ / ✗), so meaning never rests on color or length alone — the same
    redundant-encoding rule the PNG charts follow.
"""

import os
import sys
import shutil
from dataclasses import dataclass

import generators


# ---------------------------------------------------------------- capabilities
@dataclass
class Caps:
    """What the output stream can render: color, unicode glyphs, and width."""
    color: bool
    unicode: bool
    width: int


def detect_caps(stream=None, force_color=None, force_ascii=None, width=None) -> Caps:
    """Inspect the stream + environment to decide what we may emit.

    Honors the de-facto standards: NO_COLOR disables color, FORCE_COLOR enables it,
    TERM=dumb and non-TTYs get plain text. `force_*`/`width` are explicit overrides
    (used by tests and the --no-color / --width CLI flags)."""
    stream = stream if stream is not None else sys.stdout
    tty = bool(getattr(stream, "isatty", lambda: False)())

    if force_color is None:
        if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
            color = False
        elif os.environ.get("FORCE_COLOR"):
            color = True
        else:
            color = tty
    else:
        color = force_color

    if force_ascii is None:
        enc = (getattr(stream, "encoding", None) or "").lower()
        unicode = "utf" in enc
    else:
        unicode = not force_ascii

    if width is None:
        width = shutil.get_terminal_size((80, 24)).columns
    width = max(48, min(width, 100))
    return Caps(color=color, unicode=unicode, width=width)


# ----------------------------------------------------------------------- color
# 256-color ramp from poor (red) to good (green), plus a few semantic accents.
# These are mid-brightness codes chosen to stay legible on both light and dark
# terminal themes.
_RAMP = [196, 208, 178, 112, 41]          # red, orange, gold, light-green, green
_ACCENT = 45                              # cyan: headings / frame
_GREY = 244                               # dim secondary text & meter track


def _sgr(text, *codes):
    return "\x1b[" + ";".join(str(c) for c in codes) + "m" + text + "\x1b[0m"


class _Paint:
    """Tiny color helper that becomes a no-op when color is disabled."""

    def __init__(self, on):
        self.on = on

    def fg(self, text, code):
        return _sgr(text, 38, 5, code) if self.on else text

    def bold(self, text):
        return _sgr(text, 1) if self.on else text

    def dim(self, text):
        return _sgr(text, 2) if self.on else text

    def ramp(self, text, goodness):
        """Color `text` by a 0..1 goodness score (1 = best)."""
        if not self.on:
            return text
        g = 0.0 if goodness is None else max(0.0, min(1.0, goodness))
        idx = min(len(_RAMP) - 1, int(g * len(_RAMP)))
        return self.fg(text, _RAMP[idx])

    def accent(self, text):
        return self.fg(text, _ACCENT)


# ----------------------------------------------------------------- primitives
_EIGHTHS = " ▏▎▍▌▋▊▉█"                    # 0..8 eighths of a cell (index by /8)
_SPARK_U = "▁▂▃▄▅▆▇█"
_SPARK_A = ".:-=+*#@"


def meter(frac, width, paint, goodness=None, unicode=True, neutral=False):
    """A horizontal value meter for a single 0..1 proportion.

    Unicode form has 8x sub-cell resolution (a partial final block) over a dim track;
    ASCII form is `[####----]`. `goodness` (default = frac) drives the red→green fill
    color; `neutral` paints it grey instead (for purely informational meters). No end
    caps in the unicode form: the 1/8 block is glyph-identical to a cap, so the dim
    track alone marks the extent — which keeps fixed-width columns aligned."""
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    goodness = frac if goodness is None else goodness
    color = (lambda t: paint.fg(t, _GREY)) if neutral else (lambda t: paint.ramp(t, goodness))
    if not unicode:
        filled = int(round(frac * width))
        return "[" + color("#" * filled) + paint.dim("-" * (width - filled)) + "]"
    eighths = int(round(frac * width * 8))
    full, rem = divmod(eighths, 8)
    full = min(full, width)
    body = "█" * full
    if rem and full < width:
        body += _EIGHTHS[rem]
        full += 1
    return color(body) + paint.dim("░" * (width - full))


def sparkline(values, paint, lo=0.0, hi=1.0, goodness_each=None, unicode=True):
    """Compact inline chart: one glyph per value, mapped onto [lo, hi].

    Used for degradation curves (accuracy across the difficulty axis), where a flat
    high line is the positive signal and a cliff jumps out at a glance."""
    if not values:
        return ""
    ramp = _SPARK_U if unicode else _SPARK_A
    span = (hi - lo) or 1.0
    out = []
    for i, v in enumerate(values):
        t = max(0.0, min(1.0, (v - lo) / span))
        glyph = ramp[min(len(ramp) - 1, int(t * len(ramp)))]
        g = v if goodness_each is None else goodness_each[i]
        out.append(paint.ramp(glyph, g))
    return "".join(out)


# Safety net: any decorative glyph that might slip into output gets a plain-ASCII
# fallback, applied once when the stream can't encode unicode. Structural forms
# (meters, boxes, sparklines) are chosen up front from caps.unicode; this only
# guards inline punctuation/marks so an ASCII terminal never hits an encode error.
_ASCII_MAP = str.maketrans({
    "·": "-", "→": ">", "↓": "v", "—": "-",
    "✓": "+", "✗": "x",
    "█": "#", "░": "-", "▏": "|", "▎": "#", "▍": "#", "▌": "#",
    "▋": "#", "▊": "#", "▉": "#",
    "▁": ".", "▂": ".", "▃": ":", "▄": "-", "▅": "=", "▆": "+", "▇": "*",
    "╭": "+", "╮": "+", "╰": "+", "╯": "+", "─": "-", "│": "|", "▕": "|",
})


def _ascii_safe(lines):
    return [ln.translate(_ASCII_MAP) for ln in lines]


def _visible_len(s):
    """Length of `s` ignoring ANSI escapes, so padding lines up under color."""
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\x1b":
            j = s.find("m", i)
            i = len(s) if j < 0 else j + 1
        else:
            out += 1
            i += 1
    return out


def _pad(s, width):
    return s + " " * max(0, width - _visible_len(s))


def _glyph(goodness, paint, unicode=True):
    """Status mark, redundant with color: good / middling / poor."""
    if goodness is None:
        return " "
    if unicode:
        mark = "✓" if goodness >= 0.66 else ("~" if goodness >= 0.33 else "✗")
    else:
        mark = "+" if goodness >= 0.66 else ("~" if goodness >= 0.33 else "x")
    return paint.ramp(mark, goodness)


def _banner(lines, paint, caps):
    """A framed title box across the dashboard width."""
    w = caps.width
    tl, tr, bl, br, h, v = ("╭", "╮", "╰", "╯", "─", "│") if caps.unicode else \
                           ("+", "+", "+", "+", "-", "|")
    inner = w - 2
    out = [paint.accent(tl + h * inner + tr)]
    for ln in lines:
        out.append(paint.accent(v) + " " + _pad(ln, inner - 2) + " " + paint.accent(v))
    out.append(paint.accent(bl + h * inner + br))
    return out


def _heading(text, paint, caps):
    """A section heading: bold accent label with a rule to the right margin."""
    label = paint.bold(paint.accent(text))
    rule = "─" if caps.unicode else "-"
    fill = max(0, caps.width - _visible_len(label) - 2)
    return ["", label + " " + paint.dim(rule * fill)]


# ------------------------------------------------------------ row composition
_LABEL_W = 17
_METER_W = 18


def _bar_row(label, frac, value_str, paint, caps, goodness=None, note=""):
    """`label   ████░░░ value  glyph  note` — one aligned headline metric."""
    goodness = frac if goodness is None else goodness
    m = meter(frac, _METER_W, paint, goodness=goodness, unicode=caps.unicode)
    g = _glyph(goodness, paint, caps.unicode)
    row = f"  {_pad(label, _LABEL_W)} {m} {_pad(value_str, 6)} {g}"
    if note:
        row += "  " + paint.dim(note)
    return row


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


# ------------------------------------------------------------------- sections
def _section_headline(res, paint, caps):
    out = _heading("HEADLINE", paint, caps)
    out.append(_bar_row("accuracy", res["overall_accuracy"],
                        _fmt(res["overall_accuracy"]), paint, caps,
                        note="single-shot, base items"))
    strict = res.get("overall_accuracy_strict")
    if strict is not None:
        fr = res.get("fallback_reliance")
        note = "marker-only" + (f" · fallback-reliance {_fmt(fr)}" if fr is not None else "")
        out.append(_bar_row("strict accuracy", strict, _fmt(strict), paint, caps, note=note))
    cov = res["coverage"]
    out.append(_bar_row("coverage", cov["coverage"], _fmt(cov["coverage"]), paint, caps,
                        note=f"{cov['answered']}/{cov['n_items']} answered"
                             + (f" · {cov['errored']} errored" if cov["errored"] else "")))
    cal = res.get("calibration")
    if cal:
        ece = cal["ece"]
        # ECE lives on a compressed scale (good < 0.05, poor > 0.2), so a linear
        # 1-ece would flatter a badly-calibrated model; map against a 0.25 ceiling
        # so the ✓/~/✗ mark tracks how calibration is actually read.
        out.append(_bar_row("calibration ECE", ece, _fmt(ece), paint, caps,
                            goodness=1 - min(1.0, ece / 0.25), note="lower is better"))
    inv = res.get("invariance", {})
    if inv.get("groups"):
        fr = inv["answer_flip_rate"]
        out.append(_bar_row("answer-flip rate", fr, _fmt(fr), paint, caps,
                            goodness=1 - fr,
                            note=f"over {inv['groups']} surface groups · lower is better"))
    conf = res.get("confabulation_rate")
    if conf is not None:
        out.append(_bar_row("confabulation", conf, _fmt(conf), paint, caps,
                            goodness=1 - conf, note="ill-posed answered concretely"))
    pk = res.get("passk")
    if pk:
        head = res.get("frontier_headroom")
        tail = f"  (+{_fmt(head)} headroom)" if head is not None else ""
        line = (f"pass@1 {paint.bold(_fmt(pk['pass@1']))} → "
                f"maj@{pk['k']} {paint.bold(_fmt(pk['maj@k']))} → "
                f"oracle {paint.bold(_fmt(pk['pass@k_oracle']))}")
        out.append(f"  {_pad('self-consistency', _LABEL_W)} {line}{paint.dim(tail)}")
    return out


def _section_families(res, paint, caps):
    fam_acc = res.get("accuracy_by_family", {})
    if not fam_acc:
        return []
    out = _heading("ACCURACY BY FAMILY", paint, caps)
    arrow = "→" if caps.unicode else "->"
    out.append("  " + paint.dim(_pad("family", _LABEL_W) + " " + _pad("accuracy", _METER_W + 9)
                                 + f"trend (easy {arrow} hard difficulty)"))
    deg = res.get("degradation", {})
    above = res.get("acc_above_chance", {})
    chance = res.get("chance_baseline", {})
    for fam in sorted(fam_acc):
        acc = fam_acc[fam]
        m = meter(acc, _METER_W, paint, unicode=caps.unicode)
        g = _glyph(acc, paint, caps.unicode)
        cells = deg.get(fam, {})
        diffs = sorted(cells)
        spark = sparkline([cells[d]["mean"] for d in diffs], paint, unicode=caps.unicode)
        axis = generators.difficulty_axis(fam)
        # acc_above_chance only adds information when the family has a non-trivial
        # guessing floor (bounded answer space); for unbounded-int families it equals
        # raw accuracy, so showing it would just be noise.
        ac = above.get(fam)
        note = ""
        if chance.get(fam, 0) > 0.01 and ac is not None:
            note = paint.dim(f"  · {_fmt(ac, 2)} above chance")
        out.append(f"  {_pad(fam, _LABEL_W)} {m} {_fmt(acc, 2)} {g}  "
                   f"{_pad(spark, 6)} {paint.dim(_pad(axis, 24))}{note}")
    return out


def _section_distract(res, paint, caps):
    dd = res.get("distractibility", {})
    if not dd:
        return []
    out = _heading("DISTRACTIBILITY  (base → +irrelevant clause; a big drop is bad)",
                   paint, caps)
    for fam in sorted(dd):
        v = dd[fam]
        drop = v["drop"]
        # drop near 0 is good; scale goodness so a 0.3 drop reads as clearly bad.
        good = max(0.0, 1 - max(drop, 0) / 0.3)
        bar = meter(min(max(drop, 0) / 0.5, 1.0), 10, paint, goodness=good,
                    unicode=caps.unicode)
        arrow = "→" if caps.unicode else "->"
        out.append(f"  {_pad(fam, _LABEL_W)} {_fmt(v['base_acc'], 2)} {arrow} "
                   f"{_fmt(v['distractor_acc'], 2)}  drop {drop:+.2f} {bar}  "
                   + paint.dim(f"hurt {v['hurt']} / helped {v['helped']} · n={v['n']}"))
    return out


def _section_calibration(res, paint, caps):
    cal = res.get("calibration")
    if not cal or not cal.get("bins"):
        return []
    out = _heading(f"CALIBRATION  (ECE {_fmt(cal['ece'])} · stated confidence vs actual accuracy)",
                   paint, caps)
    for b in cal["bins"]:
        cm = meter(b["avg_conf"], 12, paint, neutral=True, unicode=caps.unicode)
        am = meter(b["accuracy"], 12, paint, neutral=True, unicode=caps.unicode)
        # well-calibrated when stated confidence ≈ realized accuracy in the bin
        gap = abs(b["avg_conf"] - b["accuracy"])
        g = _glyph(1 - gap, paint, caps.unicode)
        n = _pad(paint.dim(f"n={b['n']}"), 7)
        out.append(f"  conf {_pad(b['bin'], 6)} {n} {paint.dim('conf')}{cm} "
                   f"{paint.dim('acc')}{am} {g}")
    return out


def _section_runtime(rstats, paint, caps):
    if not rstats:
        return []
    out = _heading("RUNTIME", paint, caps)
    lat = (f"p50 {rstats['latency_p50_ms']}ms · p95 {rstats['latency_p95_ms']}ms · "
           f"mean {rstats['latency_mean_ms']}ms")
    out.append(f"  {_pad('latency', _LABEL_W)} {lat}")
    if rstats.get("tokens_available"):
        tok = (f"{rstats['completion_tokens_total']:,} completion "
               f"(mean {rstats['completion_tokens_mean']}/item) · "
               f"{rstats['prompt_tokens_total']:,} prompt")
    else:
        tok = paint.dim("not reported by provider")
    out.append(f"  {_pad('tokens', _LABEL_W)} {tok}")
    if rstats.get("reasoning_available"):
        parts = [f"{rstats['reasoning_tokens_total']:,} reasoning "
                 f"(mean {rstats['reasoning_tokens_mean']}/item)"]
        if rstats.get("reasoning_fraction") is not None:
            parts.append(f"{rstats['reasoning_fraction'] * 100:.0f}% of completion")
        # Intelligence metrics: efficiency (correct/1k tokens) and effort scaling (easy/hard ratio)
        if rstats.get("reasoning_correct_per_1k") is not None:
            parts.append(f"{rstats['reasoning_correct_per_1k']:.1f} correct/1k")
        if rstats.get("reasoning_effort_scaling") is not None:
            parts.append(f"effort {rstats['reasoning_effort_scaling']:.2f}×")
        out.append(f"  {_pad('reasoning', _LABEL_W)} {' · '.join(parts)}")
    out.append(f"  {_pad('calls', _LABEL_W)} {rstats['n_calls']:,}"
               + (paint.fg(f"  · {rstats['errored']} errored", _RAMP[0])
                  if rstats["errored"] else paint.dim("  · 0 errored")))
    return out


def render_run(res, run_meta=None, rstats=None, caps=None):
    """Full single-run dashboard as a list of lines (no trailing newline)."""
    caps = caps or detect_caps()
    paint = _Paint(caps.color)
    if "error" in res:
        return [paint.ramp(res["error"], 0.0)]
    run_meta = run_meta or {}

    title = f"reasoning-bench · {paint.bold(res['run_id'])}"
    model = run_meta.get("model") or "—"
    sub = (f"{res['n_items']:,} items · {res['samples_per_item']} sample"
           f"{'s' if res['samples_per_item'] != 1 else ''}/item")
    if run_meta.get("created"):
        sub += f" · {run_meta['created']}"
    head_r = paint.accent(model)
    line1 = title + " " * max(1, caps.width - 4 - _visible_len(title) - _visible_len(head_r)) + head_r

    lines = _banner([line1, paint.dim(sub)], paint, caps)
    for section in (_section_headline(res, paint, caps),
                    _section_families(res, paint, caps),
                    _section_distract(res, paint, caps),
                    _section_calibration(res, paint, caps),
                    _section_runtime(rstats, paint, caps)):
        lines += section
    lines.append("")
    lines.append(paint.dim("  legend: ███░░ meter = value · ✓ good  ~ middling  ✗ poor · "
                           "trend sparkline = accuracy from easy to hard")
                 if caps.unicode else
                 paint.dim("  legend: [#-] meter = value · + good  ~ middling  x poor"))
    return lines if caps.unicode else _ascii_safe(lines)


# ----------------------------------------------------------- comparison view
def _cmp_metric(res, key):
    if key == "ece":
        c = res.get("calibration")
        return c["ece"] if c else None
    if key == "flip":
        inv = res.get("invariance", {})
        return inv.get("answer_flip_rate") if inv.get("groups") else None
    if key == "maj":
        pk = res.get("passk")
        return pk["maj@k"] if pk else None
    return res.get(key)


# (label, metric-key, higher_is_better)
_CMP_ROWS = [
    ("overall acc", "overall_accuracy", True),
    ("strict acc", "overall_accuracy_strict", True),
    ("coverage", None, True),                      # special-cased below
    ("calibration ECE", "ece", False),
    ("answer-flip", "flip", False),
    ("confabulation", "confabulation_rate", False),
    ("maj@k", "maj", True),
]


def render_compare(results, labels, metas=None, caps=None):
    """Leaderboard across runs: one metric per row, one meter per model."""
    caps = caps or detect_caps()
    paint = _Paint(caps.color)
    metas = metas or [{} for _ in results]
    lines = _banner([paint.bold(f"reasoning-bench · comparing {len(results)} runs")],
                    paint, caps)

    name_w = max([_LABEL_W] + [len(l) for l in labels])
    metric_w = 18                              # fits the widest label + a "↓" mark
    header = "  " + _pad("", metric_w) + "  " + "  ".join(_pad(l, name_w) for l in labels)
    lines += _heading("METRIC  (meter scaled to best run in row)", paint, caps)
    lines.append(paint.dim(header))

    for label, key, higher in _CMP_ROWS:
        if key is None:
            vals = [r["coverage"]["coverage"] for r in results]
        else:
            vals = [_cmp_metric(r, key) for r in results]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        # Meter LENGTH encodes goodness (1 = best), scaled to the best run in the
        # row — so the longest bar is always the best, even on "lower is better"
        # rows. Color encodes absolute goodness; the printed number is the raw value.
        goods = [(v if higher else 1 - v) for v in present]
        gmax = max(goods) or 1.0
        cells = []
        for v in vals:
            if v is None:
                cells.append(_pad(paint.dim("—"), name_w))
                continue
            good = v if higher else 1 - v
            m = meter(good / gmax, max(8, name_w - 7), paint, goodness=good,
                      unicode=caps.unicode)
            cells.append(_pad(f"{m} {_fmt(v, 2)}", name_w))
        arrow = "" if higher else (" ↓" if caps.unicode else " v")
        lines.append("  " + _pad(label + paint.dim(arrow), metric_w) + "  " + "  ".join(cells))
    lines.append("")
    lines.append(paint.dim("  ↓ = lower is better. Meters are scaled to the best value in each row."
                           if caps.unicode else
                           "  v = lower is better. Meters scaled to the best value in each row."))
    return lines if caps.unicode else _ascii_safe(lines)


# ----------------------------------------------------------------- entrypoint
def show(lines, stream=None):
    stream = stream if stream is not None else sys.stdout
    stream.write("\n".join(lines) + "\n")
    stream.flush()
