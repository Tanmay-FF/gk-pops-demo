# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Placement-free clip review label: does a human need to look at this clip?

POPS answers "how likely is a push-out", and it needs the camera placement to
tell an outbound cart from an inbound one. The review label answers a narrower
question that needs no placement at all: did any cart in the clip hold goods
that nobody can account for? It is the same shape as the review decision in
Dillion's system (g2-ai-purchek-video review_scoring/score_video.py): one
NEEDS_REVIEW / SKIP bit per clip plus the reasons behind it.

The rule looks for the NORMAL cases first: every cart empty, or bagged goods
still with their shopper. Anything that is not positively normal needs review.

It differs from Dillion's rules where the evaluation clips showed them wrong:

  * an empty cart is never a reason to review, abandoned or not, as long as it
    never held goods (Dillion reviews every abandoned cart, which sends most
    empty-cart clips to review);
  * a cart holding only bagged goods with its shopper still beside it is a
    paying customer and is skipped; once the shopper leaves a cart that held
    goods, bagged or unbagged, it needs review;
  * a cart nobody could read is reviewed only when it was on screen long
    enough to matter (REVIEW_UNREAD_MIN_S); a brief unread cart is skipped.

A third answer, NOTHING_HAPPENED, is a confident SKIP: every cart was seen
clearly, read empty throughout and never left by its shopper
(confident_nothing()). It is produced only when REVIEW_CONFIDENT_TIER is on;
the evidence behind it is always reported.

The candidate rules of docs/review_label_empty_cart_plan.md (static fixtures,
brief strongly-loaded carts, occluded reads, child and stacked reads) each sit
behind a config switch that is off until the fresh validation set passes them.

INPUTS ARE PLACEMENT-FREE BY CONSTRUCTION. Every fact below comes from the
raw classification history, the loaded-read sequence and its per-read
evidence, the track span, box movement, the linker's link and abandonment
flags and the detector's per-frame P(non-empty). Nothing here reads the event
log, the POPS peak snapshot or the clip verdict, all of which depend on
direction and so on camera placement (the INBOUND kill switch decides which
events get logged and which frame becomes a cart's peak).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import (
    CART_BLIP_S, REVIEW_UNREAD_MIN_S, REVIEW_LOADED_BASIS, REVIEW_LOADED_MIN_RUN,
    REVIEW_ABANDON_MODE, STACKED_DRAW_THRESH,
    REVIEW_IGNORE_STATIC_UNLINKED, REVIEW_STATIC_MOVE_PX,
    REVIEW_PROMOTE_BRIEF_STRONG, REVIEW_BRIEF_MIN_READS, REVIEW_BRIEF_MIN_P,
    REVIEW_DROP_OCCLUDED_READS, REVIEW_OCCLUDED_COVER, REVIEW_OCCLUDED_KEEP_P,
    REVIEW_DROP_CHILD_READS, REVIEW_CHILD_INSIDE, REVIEW_STACKED_NOT_HELD,
    REVIEW_CHILD_SOURCE, REVIEW_CLS_CHILD_MIN, REVIEW_STACKED_SOURCE, REVIEW_CLS_STACKED_MIN,
    REVIEW_CONFIDENT_TIER, REVIEW_CONFIDENT_MIN_READS, REVIEW_CONFIDENT_MAX_P,
)
from .scoring import (
    vote_classification, vote_bag_for_loaded_cart, merchandise_removed,
)

REVIEW = "NEEDS_REVIEW"
SKIP = "SKIP"
# A SKIP clip whose every cart was seen clearly and read empty, with nothing
# unread and nothing left by its shopper (plan section 6).
NOTHING_HAPPENED = "NOTHING_HAPPENED"

# Per-cart decisions. The rule looks for the NORMAL cases first (an empty cart,
# bagged goods still with their shopper); anything that is not positively
# normal needs review.
CART_BLIP = "blip"                      # seen < CART_BLIP_S, never read
CART_UNREAD_BRIEF = "unread_brief"      # never read, on screen < REVIEW_UNREAD_MIN_S
CART_EMPTY = "empty"                    # never held goods
CART_FIXTURE = "fixture"                # never moved, never linked: a display, not a cart
CART_STACKED = "stacked"                # a nest of empty carts
CART_BAGGED_WITH_SHOPPER = "bagged_with_shopper"  # bagged goods, shopper never left
CART_UNREAD = "unread"                  # never read, on screen long enough to matter
CART_SPLIT_TRACK = "split_track"        # empty, abandoned, may be a loaded cart re-acquired
CART_GOODS_REMOVED = "goods_removed"    # held goods, ends empty, shopper left
CART_LEFT_WITH_GOODS = "left_with_goods"  # held goods (bagged or unbagged), shopper left
CART_LOADED = "loaded"                  # unbagged goods, shopper with it

_NORMAL_DECISIONS = frozenset({CART_BLIP, CART_UNREAD_BRIEF, CART_EMPTY, CART_FIXTURE,
                               CART_STACKED, CART_BAGGED_WITH_SHOPPER})
# Decisions that read as "empty" for the clip-level skip reason.
_EMPTY_DECISIONS = frozenset({CART_EMPTY, CART_FIXTURE, CART_STACKED})

_REASON = {
    CART_UNREAD: "Cart {c} was on screen {s:.1f} s and never read clearly",
    CART_SPLIT_TRACK: "Cart {c} was left empty where a loaded cart had just been lost",
    CART_GOODS_REMOVED: "Goods were taken out of Cart {c} and its shopper left it",
    CART_LEFT_WITH_GOODS: "Cart {c} held {b} goods and its shopper left it",
    CART_LOADED: "Cart {c} held unbagged goods",
    CART_UNREAD_BRIEF: "Cart {c} was never read but on screen only {s:.1f} s",
    CART_BAGGED_WITH_SHOPPER: "Cart {c} held only bagged goods and stayed with its shopper",
    CART_FIXTURE: "Cart {c} never moved and nobody used it (a fixture, not a cart)",
    CART_STACKED: "Cart {c} is a stack of empty carts",
    CART_EMPTY: "Cart {c} read empty",
}


@dataclass
class CartReviewFacts:
    """What the review label knows about one cart. All placement-free.

    The fields after `abandoned_flags` are optional evidence for the candidate
    rules; a rule whose evidence is missing (None, or reads not aligned) does
    not fire, so older saved runs replay exactly as before.
    """
    seen_s: float                        # first to last frame the cart was tracked
    history: list = field(default_factory=list)       # (fill, bag, fill_conf, bag_conf) per read
    loaded_reads: list = field(default_factory=list)  # bool per read, see tracker._record_loaded_read
    never_loaded: bool = True            # tracker._never_loaded(): includes the split-track guard
    abandoned_flags: list = field(default_factory=list)  # per-frame linker abandonment, in order
    moved_px: Optional[float] = None     # largest extent of the box centre over the cart's life
    linked_ever: Optional[bool] = None   # linked to a person on any frame
    p_mean: Optional[float] = None       # mean detector P(non-empty) over every frame it was scored
    p_stacked_mean: Optional[float] = None  # mean stacked_cart share (5-class backend)
    # One entry per loaded_reads entry: (read P(non-empty) or None, largest
    # person cover of the cart box, largest share of a child box inside it,
    # v2 classifier P(child in cart) or None). Older runs have 3-tuples.
    read_evidence: list = field(default_factory=list)
    cls_stacked_mean: Optional[float] = None  # mean v2 classifier P(stacked) over its reads


def _longest_true_run(seq) -> int:
    best = run = 0
    for x in seq:
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def _abandoned(flags, mode: str) -> bool:
    """`ever`: the shopper left the cart at any point. `end`: the cart ENDS the
    clip without its shopper, i.e. it was left behind rather than stepped away
    from for a moment."""
    if not flags:
        return False
    return bool(flags[-1]) if mode == "end" else any(flags)


def _signal(source: str, det: bool, cls: Optional[bool]) -> bool:
    """Combine the detector and v2 classifier views of one yes / no signal.
    `cls` None means the classifier has no opinion (legacy classifier, or an
    older run): then every source except "classifier" falls back to the detector."""
    if cls is None:
        return det if source != "classifier" else False
    return {"detector": det, "classifier": cls, "either": det or cls,
            "both": det and cls}[source]


def _child_in_read(child, cls_child, source: str) -> bool:
    det = child is not None and child >= REVIEW_CHILD_INSIDE
    return _signal(source, det, None if cls_child is None else cls_child >= REVIEW_CLS_CHILD_MIN)


def _filtered_reads(f: CartReviewFacts, drop_occluded: bool, drop_child: bool,
                    child_source: str = REVIEW_CHILD_SOURCE):
    """The loaded-read sequence with doubtful LOADED reads removed (R4, child).

    A removed read neither counts toward nor breaks a run, the same treatment
    an unreadable read gets. Empty reads are never removed. Returns the reads
    and how many were dropped; without aligned per-read evidence nothing is.
    """
    reads = list(f.loaded_reads)
    ev = f.read_evidence
    if not (drop_occluded or drop_child) or len(ev) != len(reads):
        return reads, 0
    kept = []
    for loaded, e in zip(reads, ev):
        p, cover, child, cls_child = (tuple(e) + (None, None, None, None))[:4]
        if loaded and drop_occluded and cover is not None and cover >= REVIEW_OCCLUDED_COVER \
                and (p is None or p < REVIEW_OCCLUDED_KEEP_P):
            continue
        if loaded and drop_child and _child_in_read(child, cls_child, child_source):
            continue
        kept.append(loaded)
    return kept, len(reads) - len(kept)


def cart_decision(f: CartReviewFacts, *, unread_min_s: float = REVIEW_UNREAD_MIN_S,
                  loaded_basis: str = REVIEW_LOADED_BASIS,
                  abandon_mode: str = REVIEW_ABANDON_MODE,
                  min_run: int = REVIEW_LOADED_MIN_RUN,
                  static_unlinked: bool = REVIEW_IGNORE_STATIC_UNLINKED,
                  promote_brief: bool = REVIEW_PROMOTE_BRIEF_STRONG,
                  drop_occluded: bool = REVIEW_DROP_OCCLUDED_READS,
                  drop_child: bool = REVIEW_DROP_CHILD_READS,
                  stacked_not_held: bool = REVIEW_STACKED_NOT_HELD,
                  child_source: str = REVIEW_CHILD_SOURCE,
                  stacked_source: str = REVIEW_STACKED_SOURCE) -> tuple[str, dict]:
    """One cart's decision and the facts behind it.

    Normal first; anything not positively normal needs review:

      ignorable  never read, under CART_BLIP_S          -> blip
                 never read, under unread_min_s         -> unread_brief
      normal     never moved and never linked
                 (static_unlinked)                      -> fixture
                 a stack of empty carts (stacked_not_held) -> stacked
                 never held goods, and not a loaded
                 cart re-acquired after being left      -> empty
                 held only bagged goods and the shopper
                 never left it                          -> bagged_with_shopper
      review     never read, on screen unread_min_s+    -> unread
                 empty, left, on a split loaded track   -> split_track
                 held goods, shopper left, ends empty   -> goods_removed
                 held goods (bagged OR unbagged),
                 shopper left                           -> left_with_goods
                 holds unbagged goods                   -> loaded

    "Held goods" is `loaded` (per `loaded_basis`) OR a sustained run of loaded
    reads: once the shopper leaves the cart, a long loaded run is enough
    evidence that something was in it, even if the whole-clip vote says empty.
    A cart whose shopper stays with it and whose vote says empty is normal.
    With `promote_brief`, a brief moving cart read loaded on every read at
    near-certain detector confidence also held goods (and is loaded).

    `loaded_basis` decides what "loaded" means:
      "run"   -- a sustained run of loaded reads (longest_loaded_run >= min_run)
      "vote"  -- the confidence vote over the history says partial or full
      "both"  -- both of the above (strictest: fewest empty carts called loaded)
      "either"-- either of the above (loosest)
    """
    fill, bag, _ = vote_classification(f.history)
    reads, dropped = _filtered_reads(f, drop_occluded, drop_child, child_source)
    run = _longest_true_run(reads)
    run_loaded = run >= min_run
    vote_loaded = fill in ("partial", "full")
    loaded = {"run": run_loaded, "vote": vote_loaded, "both": run_loaded and vote_loaded,
              "either": run_loaded or vote_loaded}[loaded_basis]
    if fill is None and run_loaded:  # loaded on detector reads alone, never classified
        loaded = True
    brief_strong = (promote_brief and not run_loaded
                    and len(reads) >= REVIEW_BRIEF_MIN_READS and all(reads)
                    and f.p_mean is not None and f.p_mean >= REVIEW_BRIEF_MIN_P
                    and f.moved_px is not None and f.moved_px >= REVIEW_STATIC_MOVE_PX)
    if brief_strong:
        loaded = True
    held = loaded or run_loaded
    abandoned = _abandoned(f.abandoned_flags, abandon_mode)
    fixture = (static_unlinked and f.moved_px is not None and f.moved_px < REVIEW_STATIC_MOVE_PX
               and f.linked_ever is False and not any(f.abandoned_flags))
    stacked_det = f.p_stacked_mean is not None and f.p_stacked_mean >= STACKED_DRAW_THRESH
    stacked_cls = None if f.cls_stacked_mean is None else f.cls_stacked_mean >= REVIEW_CLS_STACKED_MIN
    stacked = stacked_not_held and _signal(stacked_source, stacked_det, stacked_cls)
    removed = abandoned and merchandise_removed([h[0] for h in f.history], min_run)
    bag = (vote_bag_for_loaded_cart(f.history) if f.history else "unbagged") if held else None
    facts = {"fill": fill, "bag": bag, "loaded_run": run, "loaded": loaded, "held": held,
             "abandoned": abandoned, "merch_removed": removed,
             "never_loaded": f.never_loaded, "seen_s": round(f.seen_s, 2),
             "reads": len(f.loaded_reads), "dropped_reads": dropped,
             "brief_strong": brief_strong,
             "moved_px": None if f.moved_px is None else round(f.moved_px, 1),
             "linked_ever": f.linked_ever,
             "p_mean": None if f.p_mean is None else round(f.p_mean, 3),
             "p_stacked_mean": None if f.p_stacked_mean is None else round(f.p_stacked_mean, 3),
             "cls_stacked_mean": None if f.cls_stacked_mean is None else round(f.cls_stacked_mean, 3)}

    # Loaded reads on something that is not a cart anyone's goods could be in.
    # Only overrides "held": an empty fixture or stack reads empty anyway.
    if fixture and held:
        return CART_FIXTURE, facts
    if stacked and held:
        return CART_STACKED, facts
    # Ignorable: nothing usable was ever read.
    if fill is None and not run_loaded and not brief_strong:
        if f.seen_s < CART_BLIP_S:
            return CART_BLIP, facts
        if f.seen_s < unread_min_s:
            return CART_UNREAD_BRIEF, facts
        return CART_UNREAD, facts
    # Normal cases first.
    if not held:
        if abandoned and not f.never_loaded:
            return CART_SPLIT_TRACK, facts
        return CART_EMPTY, facts
    if not abandoned:
        if not loaded:
            return CART_EMPTY, facts  # a run the whole-clip vote overruled, shopper still there
        if bag == "bagged":
            return CART_BAGGED_WITH_SHOPPER, facts
        return CART_LOADED, facts
    # The cart held goods and its shopper left it: always review.
    if removed:
        return CART_GOODS_REMOVED, facts
    return CART_LEFT_WITH_GOODS, facts


def confident_nothing(carts: dict, per_cart: dict, *,
                      min_reads: int = REVIEW_CONFIDENT_MIN_READS,
                      max_p: float = REVIEW_CONFIDENT_MAX_P) -> dict:
    """Is this clip a confident NOTHING_HAPPENED? Evidence for every cart.

    Every cart that is not a blip must have read empty (decision `empty`,
    `fixture` or `stacked`), with at least `min_reads` reads and a mean
    detector P(non-empty) of at most `max_p`, and must never have been left by
    its shopper, even empty (user's answer to plan Q3, 2026-11-11: a shopper
    walking off from a cart by the door is worth a look whatever it held). A
    clip with no counted cart is not confident: there is nothing to vouch for.
    """
    blocked, evidence = [], {}
    for cd, rec in per_cart.items():
        if rec["decision"] == CART_BLIP:
            continue
        f = carts[cd]
        ev = {"reads": len(f.loaded_reads), "p_mean": None if f.p_mean is None else round(f.p_mean, 3),
              "seen_s": round(f.seen_s, 2), "left_by_shopper": any(f.abandoned_flags)}
        evidence[cd] = ev
        if rec["decision"] not in _EMPTY_DECISIONS:
            blocked.append(f"Cart {cd} is {rec['decision']}")
        elif ev["left_by_shopper"]:
            blocked.append(f"Cart {cd} was left by its shopper")
        elif ev["reads"] < min_reads:
            blocked.append(f"Cart {cd} has only {ev['reads']} reads")
        elif f.p_mean is None or f.p_mean > max_p:
            blocked.append(f"Cart {cd} mean P(non-empty) "
                           + ("unknown" if f.p_mean is None else f"{f.p_mean:.2f}"))
    if not evidence:
        blocked.append("No cart was tracked")
    return {"confident": not blocked, "blocked_by": blocked, "carts": evidence,
            "min_reads": min_reads, "max_p": max_p}


def review_label(carts: dict, *, confident_tier: bool = REVIEW_CONFIDENT_TIER,
                 confident_min_reads: int = REVIEW_CONFIDENT_MIN_READS,
                 confident_max_p: float = REVIEW_CONFIDENT_MAX_P, **knobs) -> dict:
    """The clip's label from every cart's CartReviewFacts.

    `carts` maps cart id -> CartReviewFacts. Returns
      label:      NEEDS_REVIEW, SKIP, or NOTHING_HAPPENED (only with confident_tier)
      reasons:    human-readable, the review reasons when reviewing, else the
                  skip reasons (one clip-level line when there is no cart)
      carts:      cart id -> {decision, review, facts}
      confidence: confident_nothing()'s evidence (always present; for a
                  NEEDS_REVIEW clip it just lists what blocks it)
    """
    per_cart = {}
    for cd in sorted(carts, key=lambda c: (str(type(c)), c)):
        dec, facts = cart_decision(carts[cd], **knobs)
        per_cart[cd] = {"decision": dec, "review": dec not in _NORMAL_DECISIONS, "facts": facts}
    conf = confident_nothing(carts, per_cart, min_reads=confident_min_reads,
                             max_p=confident_max_p)

    def reason(cd, rec):
        return _REASON[rec["decision"]].format(c=cd, s=rec["facts"]["seen_s"],
                                               b=rec["facts"]["bag"] or "")

    review = [reason(cd, r) for cd, r in per_cart.items() if r["review"]]
    if review:
        conf["confident"] = False
        return {"label": REVIEW, "reasons": review, "carts": per_cart, "confidence": conf}
    counted = {cd: r for cd, r in per_cart.items() if r["decision"] != CART_BLIP}
    if not counted:
        reasons = ["No cart was tracked"]
    elif all(r["decision"] in _EMPTY_DECISIONS for r in counted.values()):
        reasons = ["Every cart read empty"] + [reason(cd, r) for cd, r in counted.items()
                                               if r["decision"] != CART_EMPTY]
    else:
        reasons = [reason(cd, r) for cd, r in counted.items() if r["decision"] != CART_EMPTY]
    label = NOTHING_HAPPENED if confident_tier and conf["confident"] else SKIP
    return {"label": label, "reasons": reasons, "carts": per_cart, "confidence": conf}
