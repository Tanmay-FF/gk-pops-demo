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

REVIEW = "NEEDS_REVIEW"
SKIP = "SKIP"

# Per-cart decisions. The rule looks for the NORMAL cases first (an empty cart,
# bagged goods still with their shopper); anything that is not positively
# normal needs review.
CART_BLIP = "blip"                      # seen < CART_BLIP_S, never read
CART_UNREAD_BRIEF = "unread_brief"      # never read, on screen < REVIEW_UNREAD_MIN_S
CART_EMPTY = "empty"                    # never held goods
CART_BAGGED_WITH_SHOPPER = "bagged_with_shopper"  # bagged goods, shopper never left
CART_UNREAD = "unread"                  # never read, on screen long enough to matter
CART_SPLIT_TRACK = "split_track"        # empty, abandoned, may be a loaded cart re-acquired
CART_GOODS_REMOVED = "goods_removed"    # held goods, ends empty, shopper left
CART_LEFT_WITH_GOODS = "left_with_goods"  # held goods (bagged or unbagged), shopper left
CART_LOADED = "loaded"                  # unbagged goods, shopper with it

_NORMAL_DECISIONS = frozenset({CART_BLIP, CART_UNREAD_BRIEF, CART_EMPTY,
                               CART_BAGGED_WITH_SHOPPER})

_REASON = {
    CART_UNREAD: "Cart {c} was on screen {s:.1f} s and never read clearly",
    CART_SPLIT_TRACK: "Cart {c} was left empty where a loaded cart had just been lost",
    CART_GOODS_REMOVED: "Goods were taken out of Cart {c} and its shopper left it",
    CART_LEFT_WITH_GOODS: "Cart {c} held {b} goods and its shopper left it",
    CART_LOADED: "Cart {c} held unbagged goods",
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
    """One cart's decision and the facts behind it.

    Normal first; anything not positively normal needs review:

      ignorable  never read, under CART_BLIP_S          -> blip
                 never read, under unread_min_s         -> unread_brief
      normal     never held goods, and not a loaded
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
    if fill is None and run_loaded:  # loaded on detector reads alone, never classified
        loaded = True
    held = loaded or run_loaded
    abandoned = _abandoned(f.abandoned_flags, abandon_mode)
    removed = abandoned and merchandise_removed([h[0] for h in f.history], min_run)
    bag = (vote_bag_for_loaded_cart(f.history) if f.history else "unbagged") if held else None
    facts = {"fill": fill, "bag": bag, "loaded_run": run, "loaded": loaded, "held": held,
             "abandoned": abandoned, "merch_removed": removed,
             "never_loaded": f.never_loaded, "seen_s": round(f.seen_s, 2)}

    # Ignorable: nothing usable was ever read.
    if fill is None and not run_loaded:
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


def review_label(carts: dict, **knobs) -> dict:
    """The clip's NEEDS_REVIEW / SKIP label from every cart's CartReviewFacts.

    `carts` maps cart id -> CartReviewFacts. Returns
      label:   NEEDS_REVIEW or SKIP
      reasons: human-readable, the review reasons when reviewing, else the
               skip reasons (one clip-level line when there is no cart)
      carts:   cart id -> {decision, review, facts}
    """
    per_cart = {}
    for cd in sorted(carts, key=lambda c: (str(type(c)), c)):
        dec, facts = cart_decision(carts[cd], **knobs)
        per_cart[cd] = {"decision": dec, "review": dec not in _NORMAL_DECISIONS, "facts": facts}

    def reason(cd, rec):
        return _REASON[rec["decision"]].format(c=cd, s=rec["facts"]["seen_s"],
                                               b=rec["facts"]["bag"] or "")

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
