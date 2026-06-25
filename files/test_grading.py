import pytest

from grading import grade, parse_answer, grading_fragility


def test_marker_present_parse_source_marker():
    parsed, correct, conf, src = grade("work...\nANSWER: 13", "int", "13")
    assert parsed == "13"
    assert correct
    assert src == "marker"


def test_choice_marker_present_parse_source_marker():
    parsed, correct, _, src = grade(
        "Reasoning...\nANSWER: Lena\nCONFIDENCE: 80", "choice", "Lena",
        ["Diego", "Lena", "Omar"])
    assert parsed == "Lena"
    assert correct
    assert src == "marker"


def test_int_fallback_used_parse_source_fallback():
    parsed, correct, _, src = grade("so the result is 13.\nCONFIDENCE: 80", "int", "13")
    assert parsed == "13"
    assert correct
    assert src == "fallback"


def test_choice_fallback_used_parse_source_fallback():
    parsed, correct, _, src = grade("...the tallest must be Diego.", "choice", "Diego",
                                       ["Diego", "Lena"])
    assert parsed == "Diego"
    assert correct
    assert src == "fallback"


def test_no_answer_parse_source_none():
    parsed, correct, _, src = grade("I am not sure of the count.", "int", "13")
    assert parsed is None
    assert not correct
    assert src == "none"


def test_confidence_line_not_used_as_answer():
    parsed, correct, conf, src = grade("I am not sure of the count.\nCONFIDENCE: 13",
                                          "int", "13")
    assert conf == 13
    assert parsed is None
    assert not correct
    assert src == "none"


def test_strict_mode_rejects_fallback_int():
    parsed, correct, _, src = grade("so the result is 13.\nCONFIDENCE: 80", "int", "13",
                                       strict_mode=True)
    assert parsed is None
    assert not correct
    assert src == "none"


def test_strict_mode_rejects_fallback_choice():
    parsed, correct, _, src = grade("...the tallest must be Diego.", "choice", "Diego",
                                       ["Diego", "Lena"], strict_mode=True)
    assert parsed is None
    assert not correct
    assert src == "none"


def test_strict_mode_accepts_marker():
    parsed, correct, _, src = grade("work...\nANSWER: 13", "int", "13", strict_mode=True)
    assert parsed == "13"
    assert correct
    assert src == "marker"


def test_grading_fragility_rate():
    examples = [
        # marker and fallback agree -> not a fragility hit
        ("ANSWER: 13", "int", "13", None),
        # fallback takes a number that strict mode misses -> fragility hit
        ("so the result is 7.\nANSWER: forty", "int", "7", None),
        # fallback takes a trailing choice that strict mode misses -> fragility hit
        ("maybe Diego but maybe Lena.", "choice", "Lena", ["Diego", "Lena"]),
    ]
    fragility = grading_fragility(examples)
    assert fragility == pytest.approx(2 / 3)


def test_grading_fragility_empty():
    assert grading_fragility([]) == 0.0

def test_marker_unusable_word_triggers_fallback_lenient():
    parsed, correct, _, src = grade("blah 40\nANSWER: forty", "int", "40")
    assert src == "fallback"
    assert parsed == "40"
    assert correct
def test_marker_unusable_word_strict_none():
    parsed, correct, _, src = grade("blah\nANSWER: forty", "int", "40", strict_mode=True)
    assert parsed is None
    assert src == "none"
    assert not correct


def test_parse_answer_returns_tuple():
    parsed, src = parse_answer("ANSWER: 42", "int")
    assert parsed == "42"
    assert src == "marker"

    parsed, src = parse_answer("the value is 9", "int")
    assert parsed == "9"
    assert src == "fallback"

    parsed, src = parse_answer("no answer here", "int")
    assert parsed is None
    assert src == "none"


def test_marker_takes_first_int_fallback_takes_last_int():
    # bench-lzl / E4: the marker/fallback int asymmetry is deliberate and documented.
    # A marker LEADS with the answer (first int); free prose CONCLUDES with it (last int).
    parsed, src = parse_answer("ANSWER: 12 apples then 99 total", "int")
    assert (parsed, src) == ("12", "marker")
    parsed, src = parse_answer("first I had 12, ending with 99", "int")
    assert (parsed, src) == ("99", "fallback")
