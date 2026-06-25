"""
Answer extraction and grading.

Models are asked to end with `ANSWER: <value>` (and optionally `CONFIDENCE: <0-100>`).
We parse that marker first, with conservative fallbacks. Grading is exact-match on a
normalized form, because an unreliable grader silently corrupts every downstream metric.
"""

import re
from typing import Optional, List, Tuple

_ANS = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
_CONF = re.compile(r"(?im)^\s*confidence\s*[:=]\s*(\d{1,3})")
_INT = re.compile(r"-?\d+")
# Sentinel golds for premise-flaw detection
NO_SOLUTION = "NO_SOLUTION"
UNDETERMINED = "UNDETERMINED"


def parse_confidence(text: str) -> Optional[int]:
    m = list(_CONF.finditer(text))
    if not m:
        return None
    return max(0, min(100, int(m[-1].group(1))))




def _strip_confidence(text: str) -> str:
    """Drop CONFIDENCE: lines so the integer fallback can't mistake the confidence
    value for the answer (a real bug when --confidence is on)."""
    return _CONF.sub("", text)


def _last_marker(text: str) -> Optional[Tuple[str, str]]:
    """Return (source, raw_marker_line_value) or (None, None).

    source is 'marker' when an ANSWER: line is found, otherwise None.
    """
    m = list(_ANS.finditer(text))
    if m:
        return "marker", m[-1].group(1).strip()
    return None, None


def parse_answer(text: str, answer_type: str, choices: Optional[List[str]] = None,
                 strict_mode: bool = False) -> Tuple[Optional[str], str]:
    """Return (normalized_answer, parse_source).

    parse_source is one of {'marker', 'fallback', 'none'}.
    If strict_mode is True, only the ANSWER: line is used.

    For answer_type 'set' the gold and parsed values are comma-separated
    tokens; comparison is order-insensitive (sorted tuple). 'set' is used
    by families like knights_knaves whose answer space is 2^n.
    """
    marker_src, marker = _last_marker(text)
    if marker_src == "marker":
        parsed = None
        if answer_type == "int":
            # Deliberate asymmetry with the fallback below: an ANSWER: marker
            # LEADS with the answer ("ANSWER: 12 apples"), so take the first int;
            # free prose CONCLUDES with it, so the fallback takes the last int.
            # grading_fragility() measures when these disagree, and metrics'
            # strict-accuracy / fallback_reliance quantify the consequence.
            nums = _INT.findall(marker.replace(",", ""))
            if nums:
                parsed = str(int(nums[0]))
        elif answer_type == "choice":
            cands = choices or []
            mk = marker.lower()
            for c in cands:
                if re.search(rf"\b{re.escape(c.lower())}\b", mk):
                    parsed = c
                    break
            if parsed is None:
                # an un-recognized marker value should not fall back unless strict off
                pass
        elif answer_type == "set":
            tokens = _split_set(marker)
            if tokens:
                parsed = ",".join(sorted(tokens))
        else:
            parsed = marker
        if parsed is not None:
            return parsed, "marker"
        # marker present but unusable (e.g. word number). In strict mode stop here.
        if strict_mode:
            return None, "none"

    if strict_mode:
        return None, "none"

    # Fallbacks. No ANSWER: marker: scan the whole response. For ints the
    # conclusion trails, so take the LAST int (vs the marker's first); see the
    # asymmetry note above.
    if answer_type == "int":
        nums = _INT.findall(_strip_confidence(text).replace(",", ""))
        if nums:
            return str(int(nums[-1])), "fallback"
        return None, "none"

    if answer_type == "set":
        tokens = _split_set(_strip_confidence(text))
        if tokens:
            return ",".join(sorted(tokens)), "fallback"
        return None, "none"

    if answer_type == "choice":
        cands = choices or []
        # CSP/sentinel families (knights_knaves, unsat_csp) saturate the prompt
        # with words like knight/knave/UNDETERMINED/NO_SOLUTION. For these,
        # only accept the marker line; do not scan the full reasoning text.
        restricted = any(
            c.lower() in {"undetermined", "no_solution", "knight", "knave"}
            for c in cands
        )
        if restricted:
            if marker_src == "marker":
                # already handled above; any valid marker line was accepted.
                return None, "none"
            else:
                # no marker line -> no fallback rescue.
                return None, "none"

        last, pos = None, -1
        for c in cands:
            for hit in re.finditer(rf"\b{re.escape(c.lower())}\b", text.lower()):
                if hit.start() > pos:
                    pos, last = hit.start(), c
        if last is not None:
            return last, "fallback"
        return None, "none"

    # string/free-form answer_type: no fallback semantics defined
    return None, "none"


def _split_set(text: str) -> List[str]:
    """Split a comma- or slash-separated token list into lowercased names.

    Ignores the literal token 'none' and 'and' connectors. Empty after
    strip means the list was empty (e.g. 'ANSWER: none' for an all-knight
    puzzle).
    """
    if not text:
        return []
    raw = re.split(r"[,;]|\band\b|\bor\b", text)
    out = []
    for tok in raw:
        t = tok.strip().strip("'\".").lower()
        if not t or t in {"none", "no", "nothing", "n/a", "knight,knave,knight"}:
            # the last guard prevents the marker being mis-parsed as a list
            # if the model happened to echo the gold format with no list.
            if t in {"none", "no", "nothing", "n/a"}:
                return []                       # explicit empty
            continue
        out.append(t)
    return out


def grade(text: str, answer_type: str, gold: str, choices=None,
          strict_mode: bool = False) -> Tuple[Optional[str], bool, Optional[int], str]:
    """Return (parsed_answer, is_correct, confidence, parse_source)."""
    parsed, parse_source = parse_answer(text, answer_type, choices, strict_mode=strict_mode)
    if answer_type == "int":
        try:
            correct = parsed is not None and int(parsed) == int(gold)
        except ValueError:
            correct = False
    elif answer_type == "set":
        gold_set = set(_split_set(gold))
        parsed_set = set(_split_set(parsed)) if parsed else set()
        correct = parsed_set == gold_set
    else:
        correct = parsed is not None and parsed.lower() == str(gold).lower()
    return parsed, bool(correct), parse_confidence(text), parse_source



def grading_fragility(examples: List[Tuple[str, str, str, Optional[List[str]]]]) -> float:
    """Rate of marker-vs-fallback disagreement on crafted examples.

    For each example, compare grade() (lenient) and grade() with strict_mode=True.
    Returns the fraction where lenient parse_source == 'fallback' and strict
    produces a different parsed answer (or no answer). 0.0 if no examples.
    """
    if not examples:
        return 0.0
    disagreements = 0
    for text, answer_type, gold, choices in examples:
        parsed_lenient, _, _, source_lenient = grade(text, answer_type, gold, choices)
        parsed_strict, _, _, _ = grade(text, answer_type, gold, choices, strict_mode=True)
        if source_lenient == "fallback" and parsed_lenient != parsed_strict:
            disagreements += 1
    return disagreements / len(examples)


if __name__ == "__main__":
    tests = [
        ("Step 1... so 8+5=13.\nANSWER: 13", "int", "13", None, True),
        ("I think the result is 42 apples.\nANSWER: 42 apples", "int", "42", None, True),
        ("blah\nANSWER: forty", "int", "12", None, False),     # word -> falls back, wrong
        ("Reasoning...\nANSWER: Lena\nCONFIDENCE: 80", "choice", "Lena", ["Diego", "Lena", "Omar"], True),
        ("...the tallest must be Diego.", "choice", "Diego", ["Diego", "Lena"], True),
    ]
    for text, at, gold, ch, exp in tests:
        p, c, conf, src = grade(text, at, gold, ch)
        print(f"parsed={p!r:>14}  correct={c}  conf={conf}  src={src}  (expected {exp})  {'OK' if c == exp else 'FAIL'}")


