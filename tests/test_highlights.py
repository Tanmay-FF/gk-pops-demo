# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
engine.highlights tests — the selection logic shared between the case
report's Operations Highlights section and the tracking JSON's
operational_highlights key.

The precedence test below is the one this module exists to pin: zone-
independent rules (abandoned cart, incoming empty) can produce findings while
rules_unavailable_reason is also set (no door/aisle zones drawn, so
blocked-door/static-cart didn't run). Checking emptiness of rule_findings
before checking the reason would let those findings leak into a "findings"
state when the correct answer is "did not run" — that bug would not raise, it
would just quietly disagree with what the HTML report says for the exact same
clip.

Deliberately stdlib only, no pytest — matches tests/test_case_report.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import highlights
from engine.analytics_models import QueueSpike, RuleFinding

_PASS: list = []
_FAIL: list = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


def finding(**kw):
    base = dict(
        rule_id="unattended_cart", label="UNATTENDED CART (OPS)",
        severity="ACTION", cart_display_id=1, zone_id="z1", zone_name="Aisle 3",
        start_t=12.0, end_t=52.0, duration_s=40.0, threshold_s=30.0,
        reasons=["no linked person for 40s"],
    )
    base.update(kw)
    return RuleFinding(**base)


def spike(**kw):
    base = dict(zone_id="z2", zone_name="Checkout 1", avg_dwell_s=44.0,
                max_dwell_s=90.0, n_visits=12, threshold_s=30.0,
                severity="BACKED_UP", peak_occupancy=7,
                reasons=["occupancy above threshold"])
    base.update(kw)
    return QueueSpike(**base)


class _BareFinding:
    """A duck-typed stand-in missing most attributes, like a hand-rolled test
    double elsewhere in this codebase might be. finding_to_dict() must not
    raise AttributeError on it."""
    rule_id = "static_cart"


class _BareSpike:
    severity = "QUEUE_FORMING"


# ---------------------------------------------------------------------------
section("ops_findings_state precedence")

state, shown, remainder = highlights.ops_findings_state(None, [])
check("no reason, no findings -> clean", state == "clean")
check("clean carries no findings", shown == [] and remainder == 0)

state, shown, remainder = highlights.ops_findings_state(None, [finding()])
check("no reason, findings present -> findings", state == "findings")
check("findings state shows the finding", shown == [finding()] or len(shown) == 1)

state, shown, remainder = highlights.ops_findings_state("no zones drawn", [])
check("reason set, no findings -> unavailable", state == "unavailable")

state, shown, remainder = highlights.ops_findings_state("no door/aisle zones drawn",
                                                        [finding()])
check("reason set AND findings present -> STILL unavailable "
      "(zone-independent rules can find something while others didn't run)",
      state == "unavailable")
check("unavailable state discards the findings", shown == [] and remainder == 0)


# ---------------------------------------------------------------------------
section("select_top_findings — severe-first, else top 3, with remainder")

many = ([finding(severity="SAFETY", label="BLOCKED DOOR (OPS)", cart_display_id=2)]
        + [finding(severity="INFO", label=f"INFO FINDING {i}", cart_display_id=10 + i)
           for i in range(4)])
shown, remainder = highlights.select_top_findings(many)
check("only the severe finding is shown", len(shown) == 1 and shown[0].severity == "SAFETY")
check("remainder counts the rest", remainder == 4)

only_info = [finding(severity="INFO", label=f"INFO FINDING {i}") for i in range(5)]
shown, remainder = highlights.select_top_findings(only_info)
check("with nothing severe, falls back to top 3", len(shown) == 3)
check("remainder is the rest", remainder == 2)

sev_order = [finding(severity="ACTION", label="SECOND"),
             finding(severity="SAFETY", label="FIRST")]
shown, _ = highlights.select_top_findings(sev_order)
check("SAFETY sorts above ACTION", shown[0].label == "FIRST")


# ---------------------------------------------------------------------------
section("select_congestion — independent of the rule engine, top-3 dwell")

spikes, dwell = highlights.select_congestion(
    [spike()], [{"zone_name": "Checkout 1", "avg_dwell_s": 44.0, "n_visits": 12}])
check("severe spike kept", len(spikes) == 1)
check("dwell zone kept", len(dwell) == 1)

spikes, _ = highlights.select_congestion([spike(severity="WATCH")], [])
check("non-severe spike filtered out", spikes == [])

dwell_rows = [{"zone_name": f"Zone {i}", "avg_dwell_s": float(i), "n_visits": 3}
              for i in range(6)]
_, dwell = highlights.select_congestion([], dwell_rows)
check("dwell capped at top 3", len(dwell) == 3)
check("dwell ranked by avg descending",
      [d["zone_name"] for d in dwell] == ["Zone 5", "Zone 4", "Zone 3"])

_, dwell = highlights.select_congestion([], [{"zone_name": "Zone 9",
                                               "avg_dwell_s": 99.0, "n_visits": 0}])
check("zero-visit zones excluded", dwell == [])


# ---------------------------------------------------------------------------
section("finding_to_dict / spike_to_dict — JSON round-trip")

f = finding(evidence={"also_carts": [3, 4]}, ongoing_at_eov=True, confidence="degraded")
d = highlights.finding_to_dict(f)
check("finding_to_dict does not raise", True)
check("JSON round-trip does not raise",
      json.loads(json.dumps(d)) == d)
check("wire field name is cart_id, not cart_display_id",
      "cart_id" in d and "cart_display_id" not in d)
check("wire field name is ongoing_at_end_of_video, not ongoing_at_eov",
      d["ongoing_at_end_of_video"] is True)
check("evidence survives intact", d["evidence"] == {"also_carts": [3, 4]})

s = spike(reasons=["occupancy above threshold", "long dwell"])
sd = highlights.spike_to_dict(s)
check("JSON round-trip does not raise", json.loads(json.dumps(sd)) == sd)

bare_d = highlights.finding_to_dict(_BareFinding())
check("duck-typed stand-in missing attrs does not raise AttributeError",
      bare_d["rule_id"] == "static_cart")
check("duck-typed stand-in JSON round-trips", json.loads(json.dumps(bare_d)) == bare_d)

bare_sd = highlights.spike_to_dict(_BareSpike())
check("duck-typed spike stand-in does not raise",
      bare_sd["severity"] == "QUEUE_FORMING")
check("duck-typed spike stand-in JSON round-trips",
      json.loads(json.dumps(bare_sd)) == bare_sd)


print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    print("FAILED: " + ", ".join(_FAIL))
sys.exit(1 if _FAIL else 0)
