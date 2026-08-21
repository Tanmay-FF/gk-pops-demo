# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""A case report must not state a POPS score the system never produced.

Run with:  python tests/test_vlm_score_fidelity.py

WHAT BROKE
----------
An end-to-end run on 1763941071670_B8A44F2BC734-medium-OUTSIDE scored Cart C2
at 100 and the case report's executive summary said:

    "the highest POPS score recorded for Cart C2 reaching 101"

100 appeared correctly four times elsewhere in the same document. The prose
was the only place it was wrong, which is the worst place for it to be wrong:
the executive summary is the line a manager reads.

It was not sampling. Generation is greedy (do_sample=False). It was
`no_repeat_ngram_size=4`, which forbids any four-token sequence that has
already appeared -- and the prompt hands the model "peak POPS score of 100"
before asking it to write a sentence containing that fact. The block leaves
the model no way to say the true thing, so it says the nearest permitted one.
Measured on Qwen3-VL-2B-Instruct, greedy, same prompt:

    no_repeat_ngram_size=4, repetition_penalty=1.15  ->  "C(sub2) ... 1(sub0)(sub0)"
    no_repeat_ngram_size=4, repetition_penalty=1.0   ->  "peak score of 99"
    no_repeat_ngram_size=0, repetition_penalty=1.15  ->  "peak POPS score of 100"
    no_repeat_ngram_size=0, repetition_penalty=1.0   ->  "peak POPS score of 100"

Two defences, both tested here. The decoding change is the fix: an n-gram
window longer than a fact still blocks the repeated-sentence loops it was
added for. _repair_scores is the backstop, because the next model may find
some other way to paraphrase a number.

These are pure-logic tests: no GPU, no model, no video.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import VLM_NO_REPEAT_NGRAM
from engine.vlm_analyzer import _repair_scores

#: Cart C2 is the suspect. These are the only scores this run produced, so
#: they are the only scores its report may quote.
POPS = {
    "C1": {"max_score": 5,   "peak_event": "INBOUND"},
    "C2": {"max_score": 100, "peak_event": "PUSHOUT ALERT"},
    "C4": {"max_score": 55,  "peak_event": "OUTBOUND"},
    "C5": {"max_score": 35,  "peak_event": "OUTBOUND"},
}

_FAILURES = []


def check(name, got, want):
    if got != want:
        _FAILURES.append(f"{name}\n     got:  {got!r}\n     want: {want!r}")
        print(f"FAIL {name}")
    else:
        print(f"PASS {name}")


# ---------------------------------------------------------------------------
# The decoding setting
# ---------------------------------------------------------------------------

def test_ngram_window_is_longer_than_a_fact():
    """Four tokens is about "POPS score of 100". Anything that short turns a
    repeated fact into a forbidden sequence."""
    assert VLM_NO_REPEAT_NGRAM == 0 or VLM_NO_REPEAT_NGRAM >= 8, (
        f"VLM_NO_REPEAT_NGRAM is {VLM_NO_REPEAT_NGRAM}. At that size the "
        f"n-gram block forbids the model from repeating the score it was "
        f"given in the prompt, and it writes a nearby number instead."
    )
    print("PASS test_ngram_window_is_longer_than_a_fact")


# ---------------------------------------------------------------------------
# The backstop
# ---------------------------------------------------------------------------

def test_the_reported_regression():
    check("test_the_reported_regression",
          _repair_scores(
              "the highest POPS score recorded for Cart C2 reaching 101 "
              "indicating a confirmed push-out alert.", POPS),
          "the highest POPS score recorded for Cart C2 reaching 100 "
          "indicating a confirmed push-out alert.")


def test_low_side_drift():
    check("test_low_side_drift",
          _repair_scores("reached a peak score of 99 in the POPS system", POPS),
          "reached a peak score of 100 in the POPS system")


def test_lookalike_digits():
    """The subscript form the model reached for when both defences were on."""
    check("test_lookalike_digits",
          _repair_scores("peak POPS score of 1₀₀", POPS),
          "peak POPS score of 100")


def test_a_correct_score_is_left_alone():
    text = "Cart C2 reached a peak POPS score of 100 during a PUSHOUT ALERT."
    check("test_a_correct_score_is_left_alone", _repair_scores(text, POPS), text)


def test_every_cart_in_the_table_counts_as_correct():
    text = "Cart C4 scored 55 and Cart C1 scored 5."
    check("test_every_cart_in_the_table_counts_as_correct",
          _repair_scores(text, POPS), text)


def test_an_invented_number_is_not_promoted_to_a_finding():
    """A model that writes 42 is not miscopying 100. Rewriting that to 100
    would manufacture an incident rather than repair a typo."""
    text = "the POPS score was 42"
    check("test_an_invented_number_is_not_promoted_to_a_finding",
          _repair_scores(text, POPS), text)


def test_numbers_outside_a_score_sentence_are_untouched():
    """Ages, bullet counts, times, resolutions. The report is full of numbers
    that are not scores."""
    text = ("man, 30s, average build. Seen at 34.2s. Three bullets follow. "
            "Video is 1280x720 at 20 fps.")
    check("test_numbers_outside_a_score_sentence_are_untouched",
          _repair_scores(text, POPS), text)


def test_a_near_miss_far_from_a_score_sentence_is_untouched():
    """The word "score" must be near the number for it to be a score. This
    sentence has 34 in it and 35 is a real score, but the mention is a
    timestamp two sentences later."""
    text = "No score is quoted here. Some other sentence. He left at 34 minutes."
    check("test_a_near_miss_far_from_a_score_sentence_is_untouched",
          _repair_scores(text, POPS), text)


def test_empty_and_missing_inputs():
    check("test_empty_and_missing_inputs.empty", _repair_scores("", POPS), "")
    check("test_empty_and_missing_inputs.no_table",
          _repair_scores("score of 101", {}), "score of 101")
    check("test_empty_and_missing_inputs.junk_table",
          _repair_scores("score of 101", {"C1": {"max_score": None}}),
          "score of 101")


def test_multiple_mentions_in_one_paragraph():
    check("test_multiple_mentions_in_one_paragraph",
          _repair_scores(
              "Cart C2 hit a POPS score of 101. Later the score of 54 for "
              "Cart C4 was noted.", POPS),
          "Cart C2 hit a POPS score of 100. Later the score of 55 for "
          "Cart C4 was noted.")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if _FAILURES:
        print(f"\n{len(_FAILURES)} failed:")
        for f in _FAILURES:
            print("  " + f)
        sys.exit(1)
    print(f"\n{len(tests)} passed, 0 failed")
