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


def parse_confidence(text: str) -> Optional[int]:
    m = list(_CONF.finditer(text))
    if not m:
        return None
    return max(0, min(100, int(m[-1].group(1))))


def _last_marker(text: str) -> Optional[str]:
    m = list(_ANS.finditer(text))
    return m[-1].group(1).strip() if m else None


def _strip_confidence(text: str) -> str:
    """Drop CONFIDENCE: lines so the integer fallback can't mistake the confidence
    value for the answer (a real bug when --confidence is on)."""
    return _CONF.sub("", text)


def parse_answer(text: str, answer_type: str, choices: Optional[List[str]] = None) -> Optional[str]:
    """Return the model's final answer in normalized string form, or None."""
    marker = _last_marker(text)

    if answer_type == "int":
        if marker:
            nums = _INT.findall(marker.replace(",", ""))
            if nums:
                return str(int(nums[0]))
        # fallback: last integer anywhere — but never inside a CONFIDENCE line
        nums = _INT.findall(_strip_confidence(text).replace(",", ""))
        return str(int(nums[-1])) if nums else None

    if answer_type == "choice":
        cands = choices or []
        low = {c.lower(): c for c in cands}
        if marker:
            mk = marker.lower()
            for c in cands:                              # exact token in the marker line
                if re.search(rf"\b{re.escape(c.lower())}\b", mk):
                    return c
        # fallback: last choice mentioned anywhere in the text
        last, pos = None, -1
        for c in cands:
            for hit in re.finditer(rf"\b{re.escape(c.lower())}\b", text.lower()):
                if hit.start() > pos:
                    pos, last = hit.start(), c
        return last

    return marker


def grade(text: str, answer_type: str, gold: str, choices=None) -> Tuple[Optional[str], bool, Optional[int]]:
    """Return (parsed_answer, is_correct, confidence)."""
    parsed = parse_answer(text, answer_type, choices)
    if answer_type == "int":
        try:
            correct = parsed is not None and int(parsed) == int(gold)
        except ValueError:
            correct = False
    else:
        correct = parsed is not None and parsed.lower() == str(gold).lower()
    return parsed, bool(correct), parse_confidence(text)


if __name__ == "__main__":
    tests = [
        ("Step 1... so 8+5=13.\nANSWER: 13", "int", "13", None, True),
        ("I think the result is 42 apples.\nANSWER: 42 apples", "int", "42", None, True),
        ("blah\nANSWER: forty", "int", "12", None, False),     # word -> falls back, wrong
        ("Reasoning...\nANSWER: Lena\nCONFIDENCE: 80", "choice", "Lena", ["Diego", "Lena", "Omar"], True),
        ("...the tallest must be Diego.", "choice", "Diego", ["Diego", "Lena"], True),  # fallback
    ]
    for text, at, gold, ch, exp in tests:
        p, c, conf = grade(text, at, gold, ch)
        print(f"parsed={p!r:>14}  correct={c}  conf={conf}  (expected {exp})  {'OK' if c == exp else 'FAIL'}")
