# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Placement-free clip review label: does a human need to look at this clip?

POPS answers "how likely is a push-out", and it needs the camera placement to
tell an outbound cart from an inbound one. The review label answers a narrower
question that needs no placement at all: did any cart in the clip hold goods
that nobody can account for? It is the same shape as the review decision in
Dillion's system (g2-ai-purchek-video review_scoring/score_video.py): one
REVIEW / SKIP bit per clip plus the reasons behind it.

It differs from Dillion's rules where the evaluation clips showed them wrong:

  * an empty cart is never a reason to review, abandoned or not, as long as it
    never held goods (Dillion reviews every abandoned cart, which sends most
    empty-cart clips to review);
  * a cart holding only bagged goods with its shopper still beside it is a
    paying customer and is skipped;
  * a cart nobody could read is reviewed only when it was on screen long
    enough to matter (REVIEW_UNREAD_MIN_S); a brief unread cart is skipped.

INPUTS ARE PLACEMENT-FREE BY CONSTRUCTION. Every fact below comes from the
raw classification history, the loaded-read sequence, the track span and the
per-frame linker abandonment flag. Nothing here reads the event log, the POPS
peak snapshot or the clip verdict, all of which depend on direction and so on
camera placement (the INBOUND kill switch decides which events get logged and
which frame becomes a cart's peak).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import (
    CART_BLIP_S, REVIEW_UNREAD_MIN_S, REVIEW_LOADED_BASIS, REVIEW_LOADED_MIN_RUN,
    REVIEW_ABANDON_MODE,
)
from .scoring import (
    vote_classification, vote_bag_for_loaded_cart, merchandise_removed,
)

REVIEW = "REVIEW"
SKIP = "SKIP"

# Per-cart decisions, in the order the rule tests them.
CART_BLIP = "blip"                      # seen < CART_BLIP_S, never read
CART_UNREAD_BRIEF = "unread_brief"      # never read, on screen < REVIEW_UNREAD_MIN_S
CART_UNREAD = "unread"                  # never read, on screen long enough to matter
CART_GOODS_REMOVED = "goods_removed"    # loaded run, ends empty, shopper gone
CART_LOADED = "loaded"                  # holds goods nobody accounts for
CART_BAGGED_LEFT = "bagged_left"        # bagged goods, shopper gone
CART_BAGGED_WITH_SHOPPER = "bagged_with_shopper"
CART_SPLIT_TRACK = "split_track"        # empty, abandoned, may be a loaded cart re-acquired
CART_EMPTY = "empty"

_REVIEW_DECISIONS = frozenset({CART_UNREAD, CART_GOODS_REMOVED, CART_LOADED,
                               CART_BAGGED_LEFT, CART_SPLIT_TRACK})

_REASON = {
    CART_UNREAD: "Cart {c} was on screen {s:.1f} s and never read clearly",
    CART_GOODS_REMOVED: "Goods were taken out of Cart {c} and the cart was left",
    CART_LOADED: "Cart {c} held unbagged goods",
    CART_BAGGED_LEFT: "Cart {c} held bagged goods and was left without its shopper",
    CART_SPLIT_TRACK: "Cart {c} was left empty where a loaded cart had just been lost",
    CART_UNREAD_BRIEF: "Cart {c} was never read but on screen only {s:.1f} s",
    CART_BAGGED_WITH_SHOPPER: "Cart {c} held only bagged goods and stayed with its shopper",
    CART_EMPTY: "Cart {c} read empty",
}


@dataclass
class CartReviewFacts:
    """What the review label knows about one cart. All placement-free."""
    seen_s: float                        # first to last frame the cart was tracked
    history: list = field(default_factory=list)       # (fill, bag, fill_conf, bag_conf) per read
    loaded_reads: list = field(default_factory=list)  # bool per read, see tracker._record_loaded_read
    never_loaded: bool = True            # tracker._never_loaded(): includes the split-track guard
    abandoned_flags: list = field(default_factory=list)  # per-frame linker abandonment, in order


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


def cart_decision(f: CartReviewFacts, *, unread_min_s: float = REVIEW_UNREAD_MIN_S,
                  loaded_basis: str = REVIEW_LOADED_BASIS,
                  abandon_mode: str = REVIEW_ABANDON_MODE,
                  min_run: int = REVIEW_LOADED_MIN_RUN) -> tuple[str, dict]:
    """One cart's decision and the facts behind it. First match wins:

      1. never read, under CART_BLIP_S       -> blip (ignored)
      2. a sustained loaded run but no fill  -> treated as loaded (5-class
         NON-EMPTY reads whose crop failed the quality gate)
      3. never read, under unread_min_s      -> unread_brief (skip)
      4. never read                          -> unread (REVIEW)
      5. loaded, then emptied, shopper gone  -> goods_removed (REVIEW)
      6. loaded, bagged, shopper stayed      -> bagged_with_shopper (skip)
      7. loaded, bagged, shopper gone        -> bagged_left (REVIEW)
      8. loaded                              -> loaded (REVIEW)
      9. empty, abandoned, split-track guard -> split_track (REVIEW)
     10. otherwise                           -> empty (skip)

    `loaded_basis` decides what "loaded" means:
      "run"   -- a sustained run of loaded reads (longest_loaded_run >= min_run)
      "vote"  -- the confidence vote over the history says partial or full
      "both"  -- both of the above (strictest: fewest empty carts called loaded)
      "either"-- either of the above (loosest)
    """
    fill, bag, _ = vote_classification(f.history)
    run = _longest_true_run(f.loaded_reads)
    run_loaded = run >= min_run
    vote_loaded = fill in ("partial", "full")
    loaded = {"run": run_loaded, "vote": vote_loaded, "both": run_loaded and vote_loaded,
              "either": run_loaded or vote_loaded}[loaded_basis]
    abandoned = _abandoned(f.abandoned_flags, abandon_mode)
    fills = [h[0] for h in f.history]
    removed = abandoned and merchandise_removed(fills, min_run)
    if loaded:
        bag = vote_bag_for_loaded_cart(f.history)
    facts = {"fill": fill, "bag": bag if loaded else None, "loaded_run": run,
             "loaded": loaded, "abandoned": abandoned, "merch_removed": removed,
             "never_loaded": f.never_loaded, "seen_s": round(f.seen_s, 2)}

    if fill is None and not run_loaded:
        if f.seen_s < CART_BLIP_S:
            return CART_BLIP, facts
        if f.seen_s < unread_min_s:
            return CART_UNREAD_BRIEF, facts
        return CART_UNREAD, facts
    if fill is None:  # loaded on detector reads alone, never classified
        loaded = facts["loaded"] = True
        bag = facts["bag"] = "unbagged"
    if removed:
        return CART_GOODS_REMOVED, facts
    if loaded:
        if bag == "bagged":
            return (CART_BAGGED_LEFT if abandoned else CART_BAGGED_WITH_SHOPPER), facts
        return CART_LOADED, facts
    if abandoned and not f.never_loaded:
        return CART_SPLIT_TRACK, facts
    return CART_EMPTY, facts


def review_label(carts: dict, **knobs) -> dict:
    """The clip's REVIEW / SKIP label from every cart's CartReviewFacts.

    `carts` maps cart id -> CartReviewFacts. Returns
      label:   REVIEW or SKIP
      reasons: human-readable, the REVIEW reasons when reviewing, else the
               skip reasons (one clip-level line when there is no cart)
      carts:   cart id -> {decision, review, facts}
    """
    per_cart = {}
    for cd in sorted(carts, key=lambda c: (str(type(c)), c)):
        dec, facts = cart_decision(carts[cd], **knobs)
        per_cart[cd] = {"decision": dec, "review": dec in _REVIEW_DECISIONS, "facts": facts}

    def reason(cd, rec):
        return _REASON[rec["decision"]].format(c=cd, s=rec["facts"]["seen_s"])

    review = [reason(cd, r) for cd, r in per_cart.items() if r["review"]]
    if review:
        return {"label": REVIEW, "reasons": review, "carts": per_cart}
    counted = {cd: r for cd, r in per_cart.items() if r["decision"] != CART_BLIP}
    if not counted:
        reasons = ["No cart was tracked"]
    elif all(r["decision"] == CART_EMPTY for r in counted.values()):
        reasons = ["Every cart read empty"]
    else:
        reasons = [reason(cd, r) for cd, r in counted.items() if r["decision"] != CART_EMPTY]
    return {"label": SKIP, "reasons": reasons, "carts": per_cart}
