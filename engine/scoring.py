# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
POPS (Push-Out Probability Score) computation and event classification.
"""
from .config import COLOR_PUSHOUT, COLOR_SUSPICIOUS, COLOR_MONITORING, COLOR_CLEAR

# ---------------------------------------------------------------------------
# Score ranges (see MEDIUM_SCORE / HIGH_SCORE / PUSHOUT_SCORE):
#   0-30:   Low Priority    — normal shopping, inbound, employees
#   31-70:  Medium Priority — needs quick verify
#   71-79:  High Priority   — likely theft; PUSHOUT ALERT if also abandoned
#   80-100: PUSHOUT ALERT   — called on score alone, no abandonment needed
#
# A. Kill Switches:
#   No cart detected → 0,  INBOUND → 5,  UNCLEAR → 5
#
# B. Threat Indicators (additive, OUTBOUND):
#   Base +15, EMPTY -15,
#   PARTIAL+BAGGED +15, PARTIAL+UNBAGGED +30,
#   FULL+BAGGED    +20, FULL+UNBAGGED    +50,
#   loose merchandise on the move (unbagged + partial/full, not STATIC) +10,
#   FAST +15, MEDIUM +5,
#   FAST + unbagged + partial/full +15 (rushing with loose items)
#
#   The reachable OUTBOUND totals matter more than the terms. A full unbagged
#   cart clears the 71 HIGH line at walking pace (75 SLOW / 80 MEDIUM) instead
#   of topping out at 70; standing still it stays MEDIUM (65), which is the
#   rule engine's business rather than POPS's. Partial+unbagged reaches HIGH
#   only when also FAST (85). Partial+bagged is still capped at 55 — items that
#   look paid for.
#
# C. Linked damping (UNKNOWN direction only):
#   If a person is WITH the cart (linked) and direction is UNKNOWN
#   (shopping inside store), subtract 20.
#   Does NOT apply to OUTBOUND — a linked person pushing a cart out
#   the exit is exactly the pushout we want to detect.
#
# D. Abandonment (strongest signal — overrides damping):
#   Floor at 75 if outbound + merchandise (partial/full)
#   Floor at 60 if outbound + empty
#   Otherwise +35
#
#   UNKNOWN direction floors at 65, not 75 — a shopper who parks a cart and
#   steps to a shelf trips `abandoned` after about a second of lost person
#   track, and that must not read as a pushout. The exception is
#   `merch_removed`: a sustained loaded run followed by an empty tail means the
#   goods left the cart while the owner left the frame, which is the same
#   evidence OUTBOUND floors at 75, so it gets 75 here too. Bagged carts are
#   excluded — see merchandise_removed().
# ---------------------------------------------------------------------------

#: What the INBOUND / not-valid kill switches score a cart. Named rather than
#: repeated as a literal because inbound_suppression_note() has to recognise a
#: suppressed cart by this exact value, and a drift between the two would make
#: the diagnostic quietly stop reporting.
INBOUND_SCORE = 5

#: Tier boundaries. Named because these are the numbers that get retuned, and a
#: tier boundary buried as a literal inside classify_event() is the kind of
#: thing that gets changed in one branch and not the other.
MEDIUM_SCORE = 31       #: at/above this, the cart is worth a look
HIGH_SCORE = 71         #: at/above this, the cart is high priority
#: At/above this, the cart is called a PUSHOUT on score alone — no abandonment
#: evidence required. Below it, PUSHOUT still needs the person to have left
#: (see classify_event), which is the older and narrower route.
#:
#: Only reachable OUTBOUND: the INBOUND kill switch returns INBOUND_SCORE, and
#: the UNKNOWN branch tops out at 65 even with the abandonment floor. So this
#: cannot label an arriving cart a pushout.
PUSHOUT_SCORE = 80

#: Score floor for an abandoned cart that still holds merchandise on its way
#: out. Named because the UNKNOWN-direction grab-and-run path below reuses the
#: same number deliberately: "the person left and the goods are loose" is the
#: same evidence whichever direction label the motion window managed to
#: resolve, so the two paths must not drift apart.
MERCH_REMOVED_FLOOR = 75

_FILL_SCORE_OUTBOUND = {"empty": -15}
_FILL_SCORE_UNKNOWN  = {"partial": 8}
_SPEED_SCORE_OUTBOUND = {"FAST": 15, "MEDIUM": 5}
_SPEED_SCORE_UNKNOWN  = {"FAST": 8}

# Linked person is WITH the cart — dampen risk score
#: Loose merchandise moving toward the exit. Sized so that a loaded
#: unbagged cart walked out clears HIGH_SCORE without reaching
#: PUSHOUT_SCORE -- see the table at its use site.
LOOSE_MERCH_MOVING = 10

_LINKED_DAMPING = 20


def compute_pops(direction_label: str, speed_status: str, is_valid: bool,
                 fill_label: str, bag_label: str = "not_applicable",
                 cart_detected: bool = True, abandoned: bool = False,
                 linked: bool = False, merch_removed: bool = False) -> int:
    """Compute Push-Out Probability Score (0-100).

    A linked person pushing their cart through the store is normal — the score
    is dampened by 20 points, but only for UNKNOWN direction (see section C
    below); a person walking a cart out the exit is the thing being detected,
    so OUTBOUND is never damped.

    `merch_removed` says the cart's classification history holds a sustained
    run of loaded observations followed by an empty tail — see
    merchandise_removed(). It only changes the UNKNOWN-direction abandonment
    floor, where it lifts 65 to MERCH_REMOVED_FLOOR for an unbagged cart; the
    OUTBOUND branch already floors that case.
    """
    # --- Kill Switches ---
    if not cart_detected:
        return 0
    if direction_label == "INBOUND":
        return INBOUND_SCORE
    if not is_valid:
        return INBOUND_SCORE

    score = 0

    if direction_label == "OUTBOUND":
        score += 15  # Base outbound
        # Contents — unbagged adds extra risk at every fill level
        if fill_label == "full":
            score += 50 if bag_label == "unbagged" else 20
        elif fill_label == "partial":
            score += 30 if bag_label == "unbagged" else 15
        else:
            score += _FILL_SCORE_OUTBOUND.get(fill_label, 0)
        # Velocity
        score += _SPEED_SCORE_OUTBOUND.get(speed_status, 0)
        # Loose merchandise ON THE MOVE toward the exit, at any pace.
        #
        # Without this term the textbook pushout — walk a full unbagged cart
        # calmly out the exit — was arithmetically incapable of reaching HIGH:
        # 15 base + 50 full/unbagged + 0 SLOW = 65, well under the 71
        # classify_event() needs. The whole high tier hinged on speed crossing
        # SPEED_MEDIUM (240 px/s), a threshold that varies with resolution and
        # camera distance. Someone who simply does not run scored the same tier
        # as a paying customer.
        #
        # The pace bonus is dropped for MEDIUM here. Left in, it stacks with
        # this term and lands full+unbagged on exactly PUSHOUT_SCORE, so a
        # brisk walk is scored identically to a sprint and the high band this
        # term exists to reach is skipped over:
        #
        #     full+unbagged, OUTBOUND      now          with MEDIUM's +5
        #       STATIC                      65 MEDIUM     65 MEDIUM
        #       SLOW                        75 HIGH       75 HIGH
        #       MEDIUM                      75 HIGH       80 PUSHOUT
        #       FAST                       100 PUSHOUT   105 -> 100 PUSHOUT
        #
        # Walking a loaded unbagged cart out is HIGH PRIORITY at any walking
        # pace. Running with it is a PUSHOUT ALERT. Keeping those apart is the
        # point of having two tiers. FAST keeps its +15 combo below, so the
        # sprint still saturates the scale — and now lands on 100 exactly
        # rather than needing the clamp to hide 105.
        #
        # STATIC is excluded on purpose. A loaded cart standing still near the
        # exit is not leaving yet, and that situation already has an owner: the
        # operational rule engine's blocked-door / unattended-cart categories,
        # which reason about it with duration evidence POPS does not have.
        # Including it here would spend the high tier on parked carts.
        #
        # Deliberately additive rather than a re-tuned base table: the
        # abandonment floors (max(score, 75) / max(score, 60)) and the
        # partial+bagged cap of 55 are treated as spec — see
        # tests/test_grabrun_override.py, which pins all three — and an additive
        # term under those floors cannot move them.
        if (bag_label == "unbagged" and fill_label in ("partial", "full")
                and speed_status != "STATIC"):
            score += LOOSE_MERCH_MOVING
            if speed_status == "MEDIUM":
                score -= _SPEED_SCORE_OUTBOUND["MEDIUM"]
        # Combo: rushing with loose items is the classic pushout pattern
        if speed_status == "FAST" and bag_label == "unbagged" and fill_label in ("partial", "full"):
            score += 15
        # NO linked damping for OUTBOUND — a person pushing a cart out
        # the exit IS the scenario we want to catch.  Damping only applies
        # to UNKNOWN direction (shopping inside the store).
        # Abandonment — overrides everything, floor the score high
        if abandoned:
            if fill_label in ("partial", "full"):
                score = max(score, MERCH_REMOVED_FLOOR)
            elif fill_label == "empty":
                score = max(score, 60)
            else:
                score += 35

    elif direction_label == "UNKNOWN":
        if fill_label == "full":
            score += 25 if bag_label == "unbagged" else 15
        else:
            score += _FILL_SCORE_UNKNOWN.get(fill_label, 0)
        score += _SPEED_SCORE_UNKNOWN.get(speed_status, 0)
        if linked and not abandoned:
            score -= _LINKED_DAMPING
        if abandoned:
            if fill_label in ("partial", "full"):
                # 65 is the deliberate cap for an abandoned loaded cart whose
                # direction never resolved: a shopper who parks a cart and
                # steps to a shelf trips `abandoned` after ABANDON_FRAMES,
                # which is about a second of lost person track, and that cart
                # must not read as a pushout.
                #
                # `merch_removed` is what separates the two. It means the
                # classification history holds a sustained run of loaded
                # observations followed by an empty tail — the goods left the
                # cart while the owner left the frame. A parked cart never
                # produces it, because its fill stays loaded throughout. That
                # is the pushout the OUTBOUND branch already floors at
                # MERCH_REMOVED_FLOOR, so it gets the same floor here and
                # reaches PUSHOUT ALERT through the existing
                # HIGH_SCORE + abandoned route in classify_event().
                if merch_removed and bag_label == "unbagged":
                    score = max(score, MERCH_REMOVED_FLOOR)
                else:
                    score = max(score, 65)
            else:
                score += 25

    # Cap: partial + bagged is low-risk (items are paid for)
    if fill_label == "partial" and bag_label == "bagged":
        score = min(score, 55)

    # Clamp
    if score < 0:
        return 0
    return score if score <= 100 else 100


_FILL_RANK = {"partial": 1, "full": 2}


def peak_sustained_fill(fill_sequence, min_run: int) -> str | None:
    """Highest-severity fill label that appears in a run of >= `min_run`.

    Grab-and-run detection: a cart whose final vote is "empty" but which held
    merchandise earlier means someone took the items out. Answering that with a
    first-half/second-half proportion test does not work on real classifier
    histories — an actual pushout reads items -> empty -> items as the cart is
    occluded and re-exposed at the door, and neither half is cleanly loaded nor
    cleanly empty. What DOES separate signal from noise is contiguity: a stray
    single-observation "partial" on an empty cart cannot sustain a run, and a
    cart that really held items produces a long one wherever it sits in the
    timeline.

    Contiguity gates the RETURNED LABEL as well as the decision to override,
    and that is on purpose. gk-pops-code selects the highest-severity label
    with a count of 1 or more, which on the 1764120540680 OUTSIDE clip turns a
    single "full" observation at confidence 0.667 into a FULL cart against 24
    "partial" observations at aggregate 471.08. Same detections, same
    per-frame classifications, different POPS label. See
    tests/test_stray_full_observation.py.

    Returns None when no non-empty label sustains a long enough run, i.e. the
    "empty" verdict stands.
    """
    best_run: dict[str, int] = {}
    i, n = 0, len(fill_sequence)
    while i < n:
        label = fill_sequence[i]
        j = i
        while j + 1 < n and fill_sequence[j + 1] == label:
            j += 1
        if label in _FILL_RANK:
            best_run[label] = max(best_run.get(label, 0), j - i + 1)
        i = j + 1

    qualified = [f for f, run in best_run.items() if run >= min_run]
    if not qualified:
        return None
    return max(qualified, key=lambda f: _FILL_RANK[f])


def _strip_trailing_noise(fill_sequence, min_run: int):
    """`fill_sequence` with a trailing NON-EMPTY run too short to be real removed.

    Same contiguity principle peak_sustained_fill() is built on, applied to the
    other end of the history: a classifier that misreads one observation cannot
    misread `min_run` of them in a row, so a loaded run shorter than min_run
    sitting at the very end is noise and does not describe what the cart ended
    up holding.

    This is not hypothetical tuning. On the 1764092528600 OUTSIDE clip Cart 1
    reads 27 x partial, empty x4, partial x4, empty x4, and then ONE partial at
    fill confidence 0.502 on the final observation. The goods plainly left that
    cart, and merchandise_removed() said False on the strength of that single
    frame — which on an UNKNOWN heading is the difference between 65 /
    ABANDONED CART and MERCH_REMOVED_FLOOR / PUSHOUT ALERT.

    Only ONE run is ever stripped, and only when the sequence does not already
    end empty, so a cart that genuinely finishes loaded still reads that way:
    the parked-cart case pinned in tests/test_grabrun_pushout.py ends on a
    loaded run of exactly min_run and is left untouched.
    """
    n = len(fill_sequence)
    if not n or fill_sequence[-1] == "empty":
        return fill_sequence
    label = fill_sequence[-1]
    start = n - 1
    while start > 0 and fill_sequence[start - 1] == label:
        start -= 1
    return fill_sequence[:start] if (n - start) < min_run else fill_sequence


def merchandise_removed(fill_sequence, min_run: int) -> bool:
    """Did the goods leave a cart that was carrying them?

    True when `fill_sequence` holds a sustained run of loaded observations (the
    same run test peak_sustained_fill() applies, so a stray noisy "partial"
    cannot qualify) AND the last observation reads empty.

    This is the evidence that separates a grab-and-run from a parked cart. Both
    of them trip `abandoned` — that flag is only ABANDON_FRAMES of lost person
    track — but a shopper who parks a loaded cart and steps to a shelf leaves
    the fill loaded, while someone who lifts the items out and walks off leaves
    it empty. Only the second one means merchandise left the premises.

    The trailing observation, not a proportion of the history, is what is
    tested: a real door-side pushout reads loaded -> empty -> loaded as the
    cart is occluded and re-exposed, so no half of the timeline is cleanly
    either (see peak_sustained_fill). Where the cart ENDS is the question.

    "Ends" tolerates one short burst of trailing noise — see
    _strip_trailing_noise(). A single low-confidence loaded re-read on the final
    observation is the classifier, not the merchandise coming back.
    """
    trimmed = _strip_trailing_noise(fill_sequence, min_run)
    if not trimmed or trimmed[-1] != "empty":
        return False
    # The loaded evidence is read off the FULL sequence: trimming only decides
    # how the history ENDS, and a stripped burst is still a real observation of
    # a cart that held something.
    return peak_sustained_fill(fill_sequence, min_run) is not None


def vote_classification(history):
    """Confidence-weighted fill and bag vote over a cart's whole history.

    `history` is `_cart_cls_history[cart]`: a list of
    (fill, bag, fill_conf, bag_conf) tuples, one per classified observation.

    Returns `(fill, bag, detail)`, or `(None, None, {})` for an empty history.
    `detail` carries the per-label sums the caller logs, so the arithmetic
    behind a verdict stays visible in the run output.

    The score for a label is confidence_sum x observation_count, which rewards
    both confidence and consistency: one 0.68-confidence read cannot outweigh
    twenty-eight agreeing ones, whichever order they arrive in.

    Lives here, and is called from BOTH the live frame loop and the end-of-run
    finaliser, because the two used to answer this question differently — the
    frame loop scored whatever the latest classification said and the finaliser
    voted. That is how the FF1763940475070 INSIDE clip put Cart 3 in HIGH
    PRIORITY: one fresh observation read full|unbagged (bag confidence 0.679)
    against 28 bagged observations totalling 23.21, the live path scored it 75,
    and the peak-as-floor rules then held that reading against a vote that said
    partial|bagged / 30. A tier must not turn on a single frame, and the only
    durable way to guarantee that is for one function to decide the labels.
    """
    if not history:
        return None, None, {}
    fill_conf: dict[str, float] = {}
    fill_count: dict[str, int] = {}
    bag_conf: dict[str, float] = {}
    bag_count: dict[str, int] = {}
    for fill, bag, fc, bc in history:
        fill_conf[fill] = fill_conf.get(fill, 0.0) + fc
        fill_count[fill] = fill_count.get(fill, 0) + 1
        bag_conf[bag] = bag_conf.get(bag, 0.0) + bc
        bag_count[bag] = bag_count.get(bag, 0) + 1
    fill_scores = {f: fill_conf[f] * fill_count[f] for f in fill_count}
    bag_scores = {b: bag_conf[b] * bag_count[b] for b in bag_count}
    best_fill = max(fill_scores, key=fill_scores.get)
    best_bag = max(bag_scores, key=bag_scores.get)
    detail = {
        "fill_conf": fill_conf, "fill_count": fill_count,
        "fill_scores": fill_scores,
        "bag_conf": bag_conf, "bag_count": bag_count,
        "bag_scores": bag_scores,
    }
    return best_fill, best_bag, detail


def vote_bag_for_loaded_cart(history) -> str:
    """Confidence-weighted bag label over every LOADED observation in history.

    `history` is `_cart_cls_history[cart]`: a list of
    (fill, bag, fill_conf, bag_conf) tuples.

    A cart the classifier reads as "empty" always reports bag
    "not_applicable" with confidence 1.0, so once fill has been restored to
    partial/full (grab-and-run) the recorded bag label for that frame is
    meaningless and the not_applicable votes have to be discarded before any
    bag decision is made.

    Voting instead of reading ONE frame is the point. Bagging is the noisiest
    of the three heads at door distance: on the cart this function was written
    for the run reads bagged 7 times (conf sum 5.62) against unbagged 9 times
    (conf sum 6.59), and any single frame can land either way. A single frame's
    label decides a 10-point scoring difference and, through the
    partial+bagged cap of 55, whether the cart can clear HIGH_SCORE at all.

    Defaults to "unbagged" when history holds no loaded observation: a cart
    that was carrying merchandise with no positive bagging evidence is the
    higher-risk read, and it is also what compute_pops() has always assumed.
    """
    scores: dict[str, float] = {}
    for _fill, bag, _fc, bc in history:
        if bag == "not_applicable":
            continue
        scores[bag] = scores.get(bag, 0.0) + bc
    if not scores:
        return "unbagged"
    return max(scores, key=scores.get)


def classify_event(pops_score: int, linked: bool,
                   direction_label: str, abandoned: bool = False):
    """Return (event_name, event_color_bgr) based on POPS score + context.

    Two routes to PUSHOUT ALERT, and they answer different questions:

      * SCORE ALONE, at/above PUSHOUT_SCORE. What the cart is doing is damning
        enough on its own — outbound, loaded, unbagged, moving. No evidence
        about the person is needed or waited for.
      * ABANDONMENT, at/above HIGH_SCORE. A weaker score, but the person who
        was with the cart left it. `abandoned` is only ever True for a cart
        that was LINKED first (tracker.py computes it from the linked person's
        disappearance or distance), so this route is structurally unavailable
        to a cart that never had an owner — such a cart can only reach PUSHOUT
        on score.
    """
    if pops_score >= PUSHOUT_SCORE:
        return "PUSHOUT ALERT", COLOR_PUSHOUT

    if pops_score >= HIGH_SCORE:
        if abandoned:
            return "PUSHOUT ALERT", COLOR_PUSHOUT
        return "HIGH PRIORITY", COLOR_PUSHOUT

    if pops_score >= MEDIUM_SCORE:
        if abandoned:
            return "ABANDONED CART", COLOR_SUSPICIOUS
        if not linked and direction_label == "OUTBOUND":
            return "UNLINKED EXIT", COLOR_SUSPICIOUS
        return "MEDIUM PRIORITY", COLOR_SUSPICIOUS

    if direction_label == "INBOUND":
        return "INBOUND", COLOR_CLEAR
    if not linked and direction_label == "OUTBOUND":
        return "UNLINKED EXIT", COLOR_MONITORING
    if linked:
        return "MONITORING", COLOR_MONITORING
    return "LOW PRIORITY", COLOR_CLEAR


_MAX_LISTED_CARTS = 8


def _cart_list(ids) -> str:
    """"C2, C4, C7" — capped, because a busy clip can suppress dozens and a
    wall of ids in a notice is read as noise and skipped."""
    ids = sorted(ids)
    head = ", ".join(f"C{i}" for i in ids[:_MAX_LISTED_CARTS])
    extra = len(ids) - _MAX_LISTED_CARTS
    return f"{head} +{extra} more" if extra > 0 else head


def direction_suppressed_carts(peak_snapshots) -> tuple[list[int], list[int]]:
    """(suppressed, suppressed_while_loaded) cart display ids.

    A cart is suppressed when its final direction is INBOUND and its score is
    the kill-switch value: compute_pops() returns INBOUND_SCORE for an inbound
    cart before looking at contents, speed, bagging or abandonment at all.

    The second list is the part that matters. An inbound EMPTY cart being
    unscored is the kill switch doing its job — that is a customer arriving.
    An inbound cart the classifier read as holding merchandise is the reading
    worth a second look, because the most common way to produce one is a
    camera_placement that disagrees with the physical camera.

    A cart whose quality read `unclear` is NOT counted, whatever its direction.
    compute_pops() returns INBOUND_SCORE for an unclassified cart too, on a
    separate kill switch, so an unclear inbound cart looks identical here while
    the placement had nothing to do with its score. Reporting it anyway is what
    made this note misleading on the 1764099569430 clip: it blamed the placement
    for two carts that the classifier had already declined to read, sending the
    reader to a dropdown that would not have changed anything.
    """
    suppressed: list[int] = []
    loaded: list[int] = []
    for cd, snap in (peak_snapshots or {}).items():
        snap = snap or {}
        if str(snap.get("direction", "")).strip().upper() != "INBOUND":
            continue
        if str(snap.get("quality", "")).strip().lower() in ("unclear", "unclassified"):
            continue
        try:
            score = int(snap.get("score", 0) or 0)
        except (TypeError, ValueError):
            continue
        if score > INBOUND_SCORE:
            continue
        suppressed.append(int(cd))
        if str(snap.get("fill", "")).strip().lower() in ("partial", "full"):
            loaded.append(int(cd))
    return sorted(suppressed), sorted(loaded)


def inbound_suppression_note(peak_snapshots,
                             camera_placement: str | None = None) -> str | None:
    """One coverage note about carts the INBOUND kill switch scored out, or None.

    The kill switch is absolute and, until this existed, entirely unlogged:
    every inbound cart returns INBOUND_SCORE regardless of what it holds, which
    is under the 31 that logs an event. So a camera_placement that inverts the
    axis silently empties the Events tab, drops the alert banner, floors every
    POPS row and thins the case report's evidence frames — while the detector
    keeps working perfectly and every box is still drawn. That combination
    reads as "the detections stopped", which sends the reader to the model
    instead of to a dropdown.

    Measured on one clip, flipping "Outside (facing entrance)" to "Inside
    (facing exit)": identical box and track counts, events 3 -> 0, max POPS
    55 -> 5. Nothing anywhere said why.
    """
    suppressed, loaded = direction_suppressed_carts(peak_snapshots)
    if not suppressed:
        return None

    n = len(suppressed)
    where = f" under camera placement '{camera_placement}'" if camera_placement else ""
    note = (f"{n} cart{'s' if n != 1 else ''} scored {INBOUND_SCORE} by the "
            f"INBOUND kill switch{where} ({_cart_list(suppressed)}): an inbound "
            f"cart is not assessed for theft risk at all, so it cannot log an "
            f"event or raise an alert.")
    if loaded:
        m = len(loaded)
        note += (f" {m} of them {'was' if m == 1 else 'were'} classified as "
                 f"holding merchandise ({_cart_list(loaded)}) - if {'that cart' if m == 1 else 'those carts'} "
                 f"{'was' if m == 1 else 'were'} in fact LEAVING, the camera "
                 f"placement is inverted and every risk score in this run is "
                 f"suppressed.")
    else:
        note += (" All of them read as empty, which is what arriving customers "
                 "look like. Check the placement anyway if you expected exits.")
    return note


#: Below this share of unreadable cart-frames, the run is not worth a note — some
#: unclear frames are normal at door distance and in doorway glare. At or above
#: it, "no events" starts to mean "could not tell" rather than "nothing happened",
#: and the reader has to be told which.
UNASSESSED_NOTE_MIN_SHARE = 0.20


def unassessed_cart_note(total_cart_frames: int, unassessed_cart_frames: int,
                         unassessed_carts=None) -> str | None:
    """One coverage note about cart-frames the classifier could not read, or None.

    A cart whose quality reads `unclear` returns INBOUND_SCORE from
    compute_pops() before contents, direction, speed or abandonment are looked at
    — the same value as the INBOUND kill switch, and under the 31 that logs an
    event. So a run can be 30% unreadable and still present as clean: every box
    is drawn, every count is right, and the Events tab is empty.

    On the 1764099569430 clip that was 348 of 1280 cart-frames, 6 of 9 carts, and
    nothing anywhere said so. This is the same reasoning as
    inbound_suppression_note() and goes on the same channel: it tells the reader
    whether a quiet run was actually quiet.
    """
    if total_cart_frames <= 0 or unassessed_cart_frames <= 0:
        return None
    share = unassessed_cart_frames / total_cart_frames
    if share < UNASSESSED_NOTE_MIN_SHARE:
        return None
    note = (f"{unassessed_cart_frames} of {total_cart_frames} cart observations "
            f"({share:.0%}) could not be classified - the quality head read "
            f"'unclear'. An unreadable cart is scored {INBOUND_SCORE} without "
            f"being assessed at all, so it cannot log an event or raise an "
            f"alert.")
    if unassessed_carts:
        n = len(unassessed_carts)
        note += (f" {n} cart{'s' if n != 1 else ''} finished the run unreadable "
                 f"({_cart_list(unassessed_carts)}).")
    note += (" Treat the absence of findings for those carts as missing evidence, "
             "not as a clean result.")
    return note


def sync_events_with_snapshots(event_log, peak_snapshots, max_pops) -> list[str]:
    """Make the POPS snapshot and the Events rows tell ONE story. Mutates both.

    Returns human-readable notes for anything overridden, so the caller can log
    them; lives here rather than inline in TrackingEngine.process_video() so the
    invariant is testable without decoding a video.

    The snapshot has just been reconciled (confidence-weighted fill/bag vote
    over the whole classification history, score recomputed from it), so it is
    the source of truth and every logged row for that cart is rewritten from it.

    The one exception is a score FLOOR: if the row logged live scored higher
    than the reconciliation, the live reading wins - but as a UNIT, all four
    fields together, never a blend. That keeps the protection the old
    "Events is truth for abandonment" branch was really providing (a finaliser
    re-vote must not quietly demote a confirmed pushout, orig=75 recomp=60)
    without letting one frame's bag label overwrite a voted one, which is what
    that branch actually did.
    """
    notes: list[str] = []
    last_event: dict = {}
    for ev in (event_log or []):
        last_event[ev["cart_id"]] = ev

    for cd, ev in last_event.items():
        if cd not in peak_snapshots:
            continue
        snap = peak_snapshots[cd]
        if ev["pops_score"] > snap.get("score", 0):
            notes.append(
                f"Cart {cd}: keeping live event reading "
                f"{ev['fill']}|{ev['bag']} {ev['event']} score={ev['pops_score']} "
                f"over reconciled {snap.get('fill')}|{snap.get('bag')} "
                f"score={snap.get('score')}"
            )
            # ALL of the row's fields, not just the four the score is printed
            # with. "As a unit" has to include the inputs: copying score, fill and
            # bag while leaving the snapshot's direction, speed and abandonment
            # means the rewrite loop below then stamps THOSE onto the row, and the
            # result is a score from the live frame beside a context from the
            # reconciliation — the exact incoherence this function exists to
            # prevent, manufactured by the branch meant to prevent it.
            snap["fill"] = ev["fill"]
            snap["bag"] = ev["bag"]
            snap["score"] = ev["pops_score"]
            snap["event"] = ev["event"]
            snap["direction"] = ev.get("direction", snap.get("direction"))
            snap["speed_status"] = ev.get("speed_status", snap.get("speed_status"))
            snap["linked"] = ev.get("linked", snap.get("linked"))
            snap["abandoned"] = ev.get("abandoned", snap.get("abandoned"))
            max_pops[cd] = ev["pops_score"]

        # EVERY row for this cart, not just the last one. Rewriting only the
        # last row left a cart's earlier rows carrying the un-reconciled score,
        # so one incident showed up twice with two different numbers and no way
        # to tell which was current.
        #
        # `direction`, `speed_status` and `abandoned` are rewritten too, and that
        # is not cosmetic: they are INPUTS to the score being written beside them.
        # Leaving them at the live row's values produced rows that cannot be
        # reproduced from their own fields — a PUSHOUT ALERT at 75 carrying
        # `abandoned: false` (75 requires it) on one golden clip, and
        # `MEDIUM | 45` where 45 requires STATIC on the other. See
        # tests/test_event_row_coherence.py, which recomputes every row.
        for row in event_log:
            if row["cart_id"] != cd:
                continue
            row["fill"] = snap["fill"]
            row["bag"] = snap["bag"]
            row["pops_score"] = snap["score"]
            row["event"] = snap["event"]
            row["direction"] = snap.get("direction", row.get("direction"))
            row["speed_status"] = snap.get("speed_status", row.get("speed_status"))
            row["linked"] = snap.get("linked", row.get("linked"))
            row["abandoned"] = snap.get("abandoned", row.get("abandoned"))
            # `frame`/`timestamp` stay put: the log records when the cart FIRST
            # reached the event, which is what prune_event_log() collapses to and
            # what an operator scrubbing the timeline is looking for. The peak is
            # a different moment and gets its own fields rather than overwriting
            # them, so a row whose numbers come from the peak can still be seeked
            # to at the peak.
            row["peak_frame"] = snap.get("frame")
            row["peak_timestamp"] = snap.get("timestamp")

    return notes


def prune_event_log(event_log) -> tuple[list, int]:
    """Drop rows that are not events, and collapse duplicates per cart.

    Returns (kept, n_dropped). Lives here rather than inline in
    TrackingEngine.process_video() so the invariant is testable without
    decoding a video: after reconciliation, EVERY row must still name a real
    event, and a cart must not carry the same event twice.

    Why it is needed: the end-of-run reconciliation rewrites a cart's logged
    rows with `classify_event(final_score, ...)`, and that can return a name
    which is not in LOGGABLE_EVENTS at all — a row logged live as MEDIUM
    PRIORITY (33) reconciles to LOW PRIORITY (16) once the confidence-weighted
    fill vote replaces a noisy single-frame reading. Nothing was dropping
    those, so build_events_timeline() rendered non-events as events and they
    shipped in full_json["events"].

    Earliest row wins on a collapse: the log records when a cart FIRST reached
    an event, which is the same rule `already_logged` enforces in the frame
    loop.
    """
    kept: list = []
    seen: set[tuple] = set()
    dropped = 0
    for ev in (event_log or []):
        if ev.get("event") not in LOGGABLE_EVENTS:
            dropped += 1
            continue
        # An event needs the medium tier, not just a loggable NAME.
        # "UNLINKED EXIT" is returned from two branches of classify_event() — at
        # MEDIUM_SCORE and above, and again below every threshold for any
        # unlinked outbound cart — so the name alone guarantees nothing about the
        # score. It is in LOGGABLE_EVENTS, so an empty cart drifting toward the
        # door at POPS 0 shipped as an incident. Checking the score is what the
        # tiers already mean; checking the name was a proxy that stopped holding
        # the moment one name spanned two tiers.
        if (ev.get("pops_score") or 0) < MEDIUM_SCORE:
            dropped += 1
            continue
        key = (ev.get("cart_id"), ev.get("event"))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(ev)
    return kept, dropped


# Event names that trigger logging
LOGGABLE_EVENTS = frozenset({
    "PUSHOUT ALERT", "HIGH PRIORITY", "MEDIUM PRIORITY",
    "UNLINKED EXIT", "ABANDONED CART",
})

HIGH_EVENTS = frozenset({"PUSHOUT ALERT", "HIGH PRIORITY"})
MEDIUM_EVENTS = frozenset({"MEDIUM PRIORITY", "UNLINKED EXIT", "ABANDONED CART"})
