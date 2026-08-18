# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
VLM summary-parser tests — pure string handling, no GPU and no model load.

Run with:  python tests/test_vlm_parse.py

Every failure this pins is SILENT: the parser never raises, it just leaves a
section empty or a level UNKNOWN, and the case report renders as if the model
had said nothing. The anchor case below is the verbatim reply Qwen3-VL-2B
produced on a forced-HIGH run of try.mp4 — a well-formed report that the
original parser reduced to "Risk Assessment: UNKNOWN" with no suspect profile.

Deliberately stdlib only, no pytest — matches tests/test_ui_builders.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.vlm_analyzer import CaseReportData, VLMAnalyzer

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


def parse(text):
    r = CaseReportData()
    VLMAnalyzer._parse_summary(text, r)
    return r


# ---------------------------------------------------------------------------
# The anchor: real Qwen3-VL-2B output, captured verbatim.
#
# Note the three renamed headers ("RISK ASSESSENT", "SUSPECT PROFILES",
# "LAW ENFORCER SUMMARY") and the `**HEADER:**` bold-after-colon form. All
# four are what broke the original parser.
# ---------------------------------------------------------------------------
REAL_QWEN_REPLY = """\
**EXECUTIVE SUMMARy:**
A person linked to cart C1 pushed the cart toward the entrance during the inbound phase, exhibiting signs of attempted departure before completion of checkout. This action triggered a peak Pops score of 6, indicating a high likelihood of push-out behavior associated with unsecured goods and incomplete purchases.

**RISK LEVEL:** HIGH
**CONFIDENCE:** 0.9

**RISK ASSESSENT:**
This individual demonstrated outbound intent through active cart manipulation while still holding unbagged items, suggesting possible attempt to abandon purchase.

**SUSPECT PROFILES:**
- Gender: male
- Age: 30-35 years
- Hair & Headwear: long braided, black headband
- Top: green hoodie, no logos

**LP TEAM RECOMMENDATION:**
- Initiate immediate surveillance targeting the area where the cart was last seen.
- Deploy trained personnel to verify ownership of cart C1.

**LAW ENFORCER SUMMARY:**
- The suspect's actions align with the highest scoring threshold recorded for cart C1.
- No other individuals were present in proximity when the cart reached its peak score.
"""

section("real Qwen3-VL-2B reply (regression anchor)")
r = parse(REAL_QWEN_REPLY)

check("risk_level is HIGH, not UNKNOWN", r.risk_level == "HIGH",
      f"got {r.risk_level!r}")
check("confidence parsed", abs(r.confidence - 0.9) < 1e-6, f"got {r.confidence!r}")
check("executive summary populated", "pushed the cart toward" in r.executive_summary)
check("executive summary drops the dangling bold",
      not r.executive_summary.startswith("*"), f"got {r.executive_summary[:12]!r}")
check("risk assessment from its own header (typo'd ASSESSENT)",
      "outbound intent" in r.risk_assessment, f"got {r.risk_assessment[:60]!r}")
check("suspect profile populated (plural PROFILES)",
      "green hoodie" in r.suspect_profile, f"got {r.suspect_profile[:60]!r}")
check("LP recommendations populated",
      "surveillance" in r.actionable_insights_lp)
check("LEO summary populated (LAW ENFORCER, not ENFORCEMENT)",
      "highest scoring threshold" in r.actionable_insights_leo,
      f"got {r.actionable_insights_leo[:60]!r}")
check("sections do not bleed into each other",
      "RISK LEVEL" not in r.executive_summary
      and "SUSPECT" not in r.risk_assessment)


# ---------------------------------------------------------------------------
# Header-format variants. Each has been seen from some model or other; the
# point is that none of them may silently drop a section.
# ---------------------------------------------------------------------------
section("header format variants")

VARIANTS = {
    "bold after colon":      "**RISK LEVEL:** HIGH",
    "bold before colon":     "**RISK LEVEL**: HIGH",
    "bold value":            "RISK LEVEL: **HIGH**",
    "plain":                 "RISK LEVEL: HIGH",
    "markdown h3":           "### RISK LEVEL: HIGH",
    "bullet":                "- Risk Level: High",
    "underscore bold":       "__RISK LEVEL:__ HIGH",
    "en-dash separator":     "RISK LEVEL – HIGH",
    "trailing prose":        "RISK LEVEL: HIGH (push-out confirmed)",
    "lowercase":             "risk level: high",
}
for label, line in VARIANTS.items():
    got = parse(f"EXECUTIVE SUMMARY: something happened.\n{line}\n").risk_level
    check(f"risk level from {label}", got == "HIGH", f"{line!r} -> {got!r}")

check("MODERATE normalises to MEDIUM",
      parse("RISK LEVEL: MODERATE").risk_level == "MEDIUM")


# ---------------------------------------------------------------------------
# Salvage path: no recognisable headers at all, everything inline.
# ---------------------------------------------------------------------------
section("inline-prose salvage")

r = parse("The system reports **RISK LEVEL:** HIGH with **CONFIDENCE:** 0.85 "
          "after reviewing the clip.")
check("salvages risk level through markdown", r.risk_level == "HIGH",
      f"got {r.risk_level!r}")
check("salvages confidence through markdown", abs(r.confidence - 0.85) < 1e-6,
      f"got {r.confidence!r}")

r = parse("This clip shows a HIGH risk push-out event.")
check("salvages '<LEVEL> risk' word order", r.risk_level == "HIGH")
check("salvages a risk sentence into risk_assessment",
      "push-out" in r.risk_assessment)

r = parse("Confidence is 70% overall and the risk level is medium here.")
check("percent confidence normalises to 0..1", abs(r.confidence - 0.7) < 1e-6,
      f"got {r.confidence!r}")
check("salvage finds medium", r.risk_level == "MEDIUM", f"got {r.risk_level!r}")


# ---------------------------------------------------------------------------
# Degenerate inputs must not raise.
# ---------------------------------------------------------------------------
section("degenerate inputs")
for label, txt in [("empty", ""), ("whitespace", "   \n\n  "),
                   ("headers with no bodies", "EXECUTIVE SUMMARY:\nRISK LEVEL:\n"),
                   ("no structure", "blah blah blah")]:
    try:
        parse(txt)
        check(f"{label} does not raise", True)
    except Exception as e:  # noqa: BLE001 - the whole point is that it can't
        check(f"{label} does not raise", False, f"{type(e).__name__}: {e}")


print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    print("FAILED: " + ", ".join(_FAIL))
sys.exit(1 if _FAIL else 0)
