# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Person-cart linking and cart re-identification.

The PersonCartLinker owns all link state and exposes a single update() call
per frame. It does NOT depend on any tracker-specific data structures —
it receives bounding boxes and position histories as plain dicts.
"""
import math
from .config import (
    LINK_CONFIRM_FRAMES, LINK_CONTESTED_FRAMES, LINK_GRACE_FRAMES,
    LINK_DRIFT_FRAMES, STALE_CART_FRAMES, REID_DIST_THRESH, REID_MAX_GONE_FRAMES,
    LINK_CANDIDATE_PATIENCE, LINK_MIN_IOU, LINK_GROUND_BAND, LINK_BEHIND_BAND,
    LINK_DRIFT_IOU, LINK_STATIC_SPREAD_PX, LINK_STATIC_MIN_FRAMES,
    LINK_STATIC_MIN_IOU, LINK_OWNED_PEAK_IOU,
)
from .motion import are_co_moving, StaticLatch


# ---------------------------------------------------------------------------
# IoU helper (inlined for speed — called per person per cart per frame)
# ---------------------------------------------------------------------------
def _iou(a, b):
    """Compute IoU between two (x1,y1,x2,y2) boxes. Returns float."""
    ox1 = max(a[0], b[0]); oy1 = max(a[1], b[1])
    ox2 = min(a[2], b[2]); oy2 = min(a[3], b[3])
    if ox2 <= ox1 or oy2 <= oy1:
        return 0.0
    inter = (ox2 - ox1) * (oy2 - oy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def foot_ratio(person_bbox, cart_bbox) -> float:
    """How far a person's feet fall in front of a cart's base, in cart heights.

    Positive: the person's box ends BELOW the cart's — nearer the camera.
    Negative: above it — further away, or their legs are hidden by the cart.

    Scaled by the cart's own bbox height so the number means the same thing at
    the door and at the back of the frame, which raw pixels do not.

    This is the measure that distinguishes an owner from a body that merely
    overlaps a cart in a scene with depth. IoU cannot: on the 1764099569430
    clip a person in the extreme foreground overlapped a cart 6 m behind them at
    IoU 0.05 and held the link for 41 frames, while the cart's real handler
    overlapped it at 0.098. Their foot ratios were +0.64 and +0.04.

    Read the MEAN over a window, not one frame — see LINK_GROUND_BAND.
    """
    height = cart_bbox[3] - cart_bbox[1]
    if height <= 0:
        return 0.0
    return (person_bbox[3] - cart_bbox[3]) / height


def _shares_ground_plane(mean_ratio: float) -> bool:
    """Is a candidate's mean foot ratio consistent with owning the cart?"""
    return LINK_BEHIND_BAND <= mean_ratio <= LINK_GROUND_BAND


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------
class PersonCartLinker:
    __slots__ = (
        "_links", "_link_start_frames", "_link_candidates",
        "_perm_persons", "_perm_carts",
        "_person_for_cart", "_person_raw_for_cart",
        "_drift_counter", "_candidate_misses", "_get_display_id",
        "_total_links", "_pos_extent", "_disowned", "_static_latch",
        "_link_peak_iou",
    )

    def __init__(self, get_display_id_fn):
        """get_display_id_fn(label, raw_id) -> display_id"""
        self._get_display_id = get_display_id_fn
        self.reset()

    def reset(self):
        self._links = {}                    # cart_raw -> person_raw
        self._link_start_frames = {}        # cart_raw -> frame_idx
        # cart_raw -> {person_raw: (cum_iou, frames, foot_ratio_sum)}
        self._link_candidates = {}
        # cart_raw -> consecutive frames with no qualifying candidate at all.
        # The accumulator above used to be discarded on the first such frame,
        # which meant confirmation needed CONSECUTIVE frames and a busy doorway
        # could keep a cart ownerless indefinitely.
        self._candidate_misses = {}
        self._perm_persons = set()          # display IDs
        self._perm_carts = set()            # display IDs
        self._person_for_cart = {}          # cart_disp -> person_disp
        self._person_raw_for_cart = {}      # cart_disp -> person_raw
        self._drift_counter = {}            # cart_raw -> frames with zero overlap
        self._total_links = set()           # unique (person_disp, cart_disp) pairs ever linked
        # cart_raw -> [window_start_frame, min_x, min_y, max_x, max_y] over the
        # cart's centroids since the window opened. Folded in incrementally: the
        # alternative, re-scanning obj_positions every frame, is O(history) per
        # cart per frame and obj_positions is never trimmed.
        #
        # The window opens at first sighting and re-opens when a link is
        # ESTABLISHED, so the same numbers answer both questions asked of them:
        # "has this ownerless cart ever moved" and "has this cart moved since the
        # person who supposedly owns it took it". It deliberately does NOT re-open
        # on a release — a cart that arrived under its own owner has moved, and a
        # taker inheriting it should not have to prove that again.
        self._pos_extent = {}
        # Cart DISPLAY ids whose link was released this frame because it was
        # never real — see the parked-cart branch of Step 0.5. Consumed by the
        # tracker, which also has to forget the remembered owner; a release alone
        # does not stop abandonment scoring.
        self._disowned = set()
        # cart_raw -> the highest IoU this cart's CURRENT link has reached with
        # its owner, over the life of that link. Read only by the parked-cart
        # release below, to tell a link that was real possession from one the
        # geometry merely allowed: only the second is reported disowned, because
        # disownment makes the tracker forget the owner and that closes the
        # abandonment route for the rest of the run. See LINK_OWNED_PEAK_IOU.
        self._link_peak_iou = {}
        # Hysteresis for the co-movement test's static/moving call, keyed by raw
        # track id. Held here rather than inside are_co_moving() because the
        # state belongs to a run: a fresh linker must start with no opinion
        # about any track. See StaticLatch.
        self._static_latch = StaticLatch()

    # --- Public properties ---
    @property
    def links(self):
        return self._links

    @property
    def link_start_frames(self):
        return self._link_start_frames

    @property
    def total_links(self):
        return len(self._total_links)

    @property
    def permanently_linked_persons(self):
        return self._perm_persons

    @property
    def permanently_linked_carts(self):
        return self._perm_carts

    @property
    def person_for_cart(self):
        return self._person_for_cart

    @property
    def person_raw_for_cart(self):
        return self._person_raw_for_cart

    @property
    def disowned_carts(self):
        """Cart display IDs whose link was released this frame as never-real.

        Valid only until the next update() call, which clears it.
        """
        return self._disowned

    # --- Parked-cart measurement ---
    def _note_position(self, cart_id, positions, frame_idx):
        """Fold the cart's newest centroid into its extent window."""
        if not positions:
            return
        x, y = positions[-1]
        ext = self._pos_extent.get(cart_id)
        if ext is None:
            self._pos_extent[cart_id] = [frame_idx, x, y, x, y]
            return
        if x < ext[1]: ext[1] = x
        if y < ext[2]: ext[2] = y
        if x > ext[3]: ext[3] = x
        if y > ext[4]: ext[4] = y

    def _open_extent_window(self, cart_id, positions, frame_idx):
        """Restart the extent measurement from this frame."""
        if not positions:
            self._pos_extent.pop(cart_id, None)
            return
        x, y = positions[-1]
        self._pos_extent[cart_id] = [frame_idx, x, y, x, y]

    def _is_parked(self, cart_id, frame_idx) -> bool:
        """Has this cart been watched long enough to say it has not moved?

        Both halves are load-bearing. Extent alone calls every newly appeared
        cart parked, including the one a shopper is about to pull out of the
        corral; the frame count alone says nothing about motion.
        """
        ext = self._pos_extent.get(cart_id)
        if ext is None:
            return False
        if frame_idx - ext[0] < LINK_STATIC_MIN_FRAMES:
            return False
        return math.hypot(ext[3] - ext[1], ext[4] - ext[2]) < LINK_STATIC_SPREAD_PX

    # --- Cart re-identification ---
    def try_reidentify_cart(self, new_raw_id, bbox, current_cart_raws,
                            display_map, obj_positions, obj_timestamps,
                            obj_speeds, obj_disappeared, obj_bbox_history=None,
                            obj_frames=None):
        """Transfer identity when tracker assigns a new raw ID to same physical cart."""
        cart_map = display_map.get('cart')
        if not cart_map or new_raw_id in cart_map:
            return
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        best_old, best_dist = None, REID_DIST_THRESH
        for old_raw, _disp in cart_map.items():
            if old_raw in current_cart_raws:
                continue
            gone = obj_disappeared.get(old_raw, 999)
            if gone > REID_MAX_GONE_FRAMES:
                continue
            positions = obj_positions.get(old_raw)
            if not positions:
                continue
            lp = positions[-1]
            d = math.sqrt((cx - lp[0]) ** 2 + (cy - lp[1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_old = old_raw

        if best_old is None:
            return
        # Transfer display ID
        old_disp = cart_map[best_old]
        cart_map[new_raw_id] = old_disp
        del cart_map[best_old]
        # Transfer link
        if best_old in self._links:
            self._links[new_raw_id] = self._links.pop(best_old)
            self._link_start_frames[new_raw_id] = self._link_start_frames.pop(best_old, 0)
        # Same link, so the same record of how strong it ever got. Left behind it
        # would restart at 0.0 and the next parked release would disown a cart
        # whose owner had been holding it since before the re-ID.
        if best_old in self._link_peak_iou:
            self._link_peak_iou[new_raw_id] = self._link_peak_iou.pop(best_old)
        if best_old in self._link_candidates:
            self._link_candidates[new_raw_id] = self._link_candidates.pop(best_old)
        # Moves with the accumulator it counts against, or a re-identified cart
        # keeps candidate evidence while its miss count restarts at zero.
        if best_old in self._candidate_misses:
            self._candidate_misses[new_raw_id] = self._candidate_misses.pop(best_old)
        # Same physical cart, so its movement record is the same record. Losing
        # it here would restart the window at the re-ID frame and call a cart
        # that has been rolling around the store parked 40 frames later.
        if best_old in self._pos_extent:
            self._pos_extent[new_raw_id] = self._pos_extent.pop(best_old)
        # The static/moving latch is about the physical cart, not the tracker's
        # id for it, and it moves with the position history that feeds it.
        self._static_latch.rename(best_old, new_raw_id)
        # Transfer history
        if best_old in obj_positions:
            obj_positions[new_raw_id] = obj_positions.pop(best_old)
        if best_old in obj_timestamps:
            obj_timestamps[new_raw_id] = obj_timestamps.pop(best_old)
        if best_old in obj_speeds:
            obj_speeds[new_raw_id] = obj_speeds.pop(best_old)
        # Bbox history must move with the positions it parallels, or the
        # re-identified cart ends up with N positions and 0 bboxes and the
        # rule engine's door-overlap test silently skips it.
        if obj_bbox_history is not None and best_old in obj_bbox_history:
            obj_bbox_history[new_raw_id] = obj_bbox_history.pop(best_old)
        # Same reasoning for the frame record: it is the synthesized-timestamp
        # fallback's only clock, so a re-identified cart that loses it gets its
        # whole pre-reid history stamped from the frame it was re-found on.
        if obj_frames is not None and best_old in obj_frames:
            obj_frames[new_raw_id] = obj_frames.pop(best_old)

    # --- Main per-frame update ---
    def update(self, person_bboxes: dict, cart_bboxes: dict,
               frame_idx: int, obj_disappeared: dict,
               obj_positions: dict, obj_first_frame: dict):
        """Run linking logic for one frame.

        Args:
            person_bboxes: {raw_id: (x1,y1,x2,y2)} for persons in this frame
            cart_bboxes:   {raw_id: (x1,y1,x2,y2)} for carts in this frame
            frame_idx:     current frame number
            obj_disappeared: {raw_id: frames_gone}
            obj_positions:   {raw_id: [(cx,cy), ...]}
            obj_first_frame: {raw_id: first_frame_idx}
        """
        gdi = self._get_display_id
        self._disowned = set()

        # Step 0a: Movement record. Folded in before anything reads it, for every
        # cart in the frame, linked or not.
        for cart_id in cart_bboxes:
            self._note_position(cart_id, obj_positions.get(cart_id), frame_idx)

        # Step 0: Purge stale links
        stale = [cid for cid in self._links
                 if cid not in cart_bboxes and obj_disappeared.get(cid, 0) > STALE_CART_FRAMES]
        for cid in stale:
            cd = gdi('cart', cid)
            self._links.pop(cid, None)
            self._link_start_frames.pop(cid, None)
            self._link_candidates.pop(cid, None)
            self._candidate_misses.pop(cid, None)
            self._link_peak_iou.pop(cid, None)
            # Gone longer than REID_MAX_GONE_FRAMES, so try_reidentify_cart can
            # no longer carry this record forward to a new raw id and holding it
            # only leaks.
            self._pos_extent.pop(cid, None)
            self._static_latch.forget(cid)
            pd = self._person_for_cart.pop(cd, None)
            self._person_raw_for_cart.pop(cd, None)
            self._perm_carts.discard(cd)
            if pd:
                self._perm_persons.discard(pd)

        # Step 0.5: Drift detection — release link ONLY when the linked
        # person is VISIBLE in the frame but has drifted away from the cart
        # (IoU < 0.05) while someone else is overlapping it.
        # If the linked person has LEFT the frame entirely, that's abandonment
        # — keep the link so POPS abandonment scoring can fire.
        DRIFT_IOU_THRESH = LINK_DRIFT_IOU
        for cart_id, cart_bbox in cart_bboxes.items():
            if cart_id not in self._links:
                self._drift_counter.pop(cart_id, None)
                continue
            pid = self._links[cart_id]

            # Person must be VISIBLE for drift to apply
            if pid not in person_bboxes:
                # Person gone from frame — this is abandonment, NOT drift
                self._drift_counter.pop(cart_id, None)
                continue

            overlap = _iou(cart_bbox, person_bboxes[pid])
            # Best contact this link has ever had, folded in before anything
            # reads it. Recorded on every frame the owner is visible, including
            # the drifting ones — the question it answers is what this link was
            # at its best, not what it is now.
            if overlap > self._link_peak_iou.get(cart_id, 0.0):
                self._link_peak_iou[cart_id] = overlap
            if overlap >= DRIFT_IOU_THRESH:
                self._drift_counter.pop(cart_id, None)  # still engaged
            else:
                # Person is visible but not overlapping the cart.
                # Only count as drift if someone ELSE is overlapping (takeover).
                #
                # The taker must share the cart's ground plane, on the same test
                # Step 2 applies to candidates. Bare IoU here accepts exactly the
                # foreground body the candidate gate rejects, and the release
                # below then SEEDS it as the next owner — so without this the
                # gate is bypassed by the code that acts on the release.
                #
                # Per-frame rather than a mean, unlike Step 2: a false negative
                # here only means the link survives another frame, while a false
                # positive hands the cart to the wrong person.
                takers = [
                    op for op, ob in person_bboxes.items()
                    if op != pid
                    and _iou(cart_bbox, ob) >= DRIFT_IOU_THRESH
                    and _shares_ground_plane(foot_ratio(ob, cart_bbox))
                ]
                if takers:
                    self._drift_counter[cart_id] = self._drift_counter.get(cart_id, 0) + 1
                    if self._drift_counter[cart_id] >= LINK_DRIFT_FRAMES:
                        cd = gdi('cart', cart_id)
                        pd_disp = self._person_for_cart.pop(cd, None)
                        self._person_raw_for_cart.pop(cd, None)
                        self._perm_carts.discard(cd)
                        if pd_disp:
                            self._perm_persons.discard(pd_disp)
                        self._links.pop(cart_id, None)
                        self._link_start_frames.pop(cart_id, None)
                        self._drift_counter.pop(cart_id, None)
                        self._link_peak_iou.pop(cart_id, None)
                        # Hand the cart to the taker rather than to nobody. The
                        # release already names the person who IS engaged with
                        # it, and dropping that on the floor is what left the
                        # 1764099569430 cart ownerless for 312 frames: Step 2
                        # then had to rediscover an owner from zero in a doorway
                        # busy enough to keep resetting the accumulator, and
                        # never did. An unowned cart cannot be abandoned, so the
                        # whole abandonment route went with it.
                        seeded = self._link_candidates.setdefault(cart_id, {})
                        for op in takers:
                            ob = person_bboxes[op]
                            cum, n, fsum = seeded.get(op, (0.0, 0, 0.0))
                            seeded[op] = (cum + _iou(cart_bbox, ob), n + 1,
                                          fsum + foot_ratio(ob, cart_bbox))
                        self._candidate_misses.pop(cart_id, None)
                elif self._is_parked(cart_id, frame_idx):
                    # Nobody is taking the cart over, and the cart has not moved
                    # one bbox-width since this link was established. Whatever
                    # the overlap that formed it was, it was not a person taking
                    # possession of a cart: possession shows up as motion, or at
                    # minimum as continued contact, and there has been neither
                    # for LINK_STATIC_MIN_FRAMES.
                    #
                    # The no-taker case used to be unreleasable, which is how a
                    # shopper on the 1764200318790 clip who walked past a parked
                    # cart kept it for 279 frames on 6 frames of corner overlap —
                    # and then, on leaving the frame, made it an ABANDONED CART
                    # at 65. Nobody else ever touched that cart, so the takeover
                    # route could not fire, and drift with no taker did nothing.
                    #
                    # Released as never-real rather than handed on: `_disowned`
                    # tells the tracker to forget the remembered owner too, or
                    # `_last_owner_raw` keeps answering the abandonment question
                    # with this person for the rest of the run and the release
                    # changes no score.
                    #
                    # ...but ONLY for a link that never amounted to possession.
                    # Disownment is two claims, not one: the link is over, AND it
                    # was never real. The first is true of every release here;
                    # the second is a statement about the WHOLE life of the link,
                    # and reading it off "the cart has not moved" alone was wrong
                    # for a cart that never had to move. On the FF1763940475070
                    # INSIDE clip P3 held Cart 2 at the door from frame 28 to 130
                    # — IoU peak 0.363, mean 0.155, a hundred frames of contact —
                    # and stepped away without ever pushing it. That release
                    # disowned the cart, so when P3 left frame at 164 there was
                    # no owner whose departure could be scored: `abandoned`
                    # stayed False for all 412 frames and the run finalised
                    # 75 HIGH PRIORITY where it had been a PUSHOUT ALERT.
                    #
                    # Link strength is what separates the two, and only peak
                    # strength does: the 1764200318790 shopper's crossing never
                    # exceeded 0.064. See LINK_OWNED_PEAK_IOU.
                    self._drift_counter[cart_id] = self._drift_counter.get(cart_id, 0) + 1
                    if self._drift_counter[cart_id] >= LINK_DRIFT_FRAMES:
                        cd = gdi('cart', cart_id)
                        was_possession = (self._link_peak_iou.get(cart_id, 0.0)
                                          >= LINK_OWNED_PEAK_IOU)
                        pd_disp = self._person_for_cart.pop(cd, None)
                        self._person_raw_for_cart.pop(cd, None)
                        self._perm_carts.discard(cd)
                        if pd_disp:
                            self._perm_persons.discard(pd_disp)
                        self._links.pop(cart_id, None)
                        self._link_start_frames.pop(cart_id, None)
                        self._drift_counter.pop(cart_id, None)
                        self._link_peak_iou.pop(cart_id, None)
                        # No seeding: there is no candidate to seed. The cart
                        # goes back to Step 2, where a parked cart's bar is
                        # LINK_CONTESTED_FRAMES.
                        self._link_candidates.pop(cart_id, None)
                        self._candidate_misses.pop(cart_id, None)
                        # The link goes either way. Only the never-real one is
                        # reported, and only that report reaches the tracker's
                        # remembered owner.
                        if not was_possession:
                            self._disowned.add(cd)
                else:
                    self._drift_counter.pop(cart_id, None)

        # Step 1: Handle tracker-ID swaps for linked persons.
        # When the linked person vanishes briefly (tracker swap), find
        # the new person whose centroid is close to the OLD person's last
        # known position.  At ~20fps, an ID swap can take up to ~15 frames
        # (person occluded by cart, re-detected with a new ID).
        PERSON_SWAP_MAX_GONE = 15  # frames — covers ~0.75s at 20fps
        PERSON_SWAP_MIN_IOU  = 0.3 # must overlap old person's last bbox

        # Pre-computed over ALL of _links before the search below, not built as
        # the search walks. Building it inside the same pass meant a cart whose
        # owner had left could steal a person still linked to a cart appearing
        # LATER in _links iteration order — two carts holding one person, and
        # abandonment never firing for the first, decided by dict order.
        claimed = {pid for pid in self._links.values() if pid in person_bboxes}
        for cart_id in list(self._links):
            pid = self._links[cart_id]
            if pid in person_bboxes:
                continue

            # Linked person gone — check if it's a tracker swap
            person_gone = obj_disappeared.get(pid, 999)
            if person_gone > PERSON_SWAP_MAX_GONE:
                continue  # gone too long — genuine departure, not a swap

            # Get old person's last known bbox from position history
            old_positions = obj_positions.get(pid)
            if not old_positions:
                continue
            # We need the actual bbox, not centroid. Use _obj_bboxes passed via
            # the linker's update signature — but we don't have it here.
            # Instead, compare new person centroids against old person's last centroid.
            # A tracker swap means the new ID appears at nearly the same spot.
            old_cx, old_cy = old_positions[-1]

            best_new_pid = None
            best_dist = 80  # max pixel distance for a swap (same body)
            for new_pid, new_bbox in person_bboxes.items():
                if new_pid in claimed:
                    continue
                if new_pid == pid:
                    continue
                new_cx = (new_bbox[0] + new_bbox[2]) * 0.5
                new_cy = (new_bbox[1] + new_bbox[3]) * 0.5
                d = math.sqrt((old_cx - new_cx) ** 2 + (old_cy - new_cy) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_new_pid = new_pid

            if best_new_pid is not None:
                cd = gdi('cart', cart_id)
                old_pd = self._person_for_cart.get(cd)
                new_pd = gdi('person', best_new_pid)
                self._links[cart_id] = best_new_pid
                self._person_for_cart[cd] = new_pd
                self._person_raw_for_cart[cd] = best_new_pid
                # `_link_peak_iou` is deliberately left alone. A swap is the same
                # body under a new tracker id, so the contact it already proved
                # is this link's contact — resetting it would let a swap 40
                # frames before a parked release disown a real owner.
                if old_pd:
                    self._perm_persons.discard(old_pd)
                self._perm_persons.add(new_pd)
                claimed.add(best_new_pid)

        # Step 2: New links for un-linked carts.
        # Accumulate IoU scores for ALL overlapping + co-moving persons over
        # LINK_CONFIRM_FRAMES.  The person with the highest cumulative IoU
        # wins — this ensures the person with the most consistent overlap
        # gets linked, not just whoever appeared first.
        #
        # Winners are PROPOSED here and assigned after the loop, best match
        # first. Committing inside the loop gave a contested person to whichever
        # cart happened to qualify first in dict order, and on the
        # 1764099569430 clip that decided the whole run: P2 was standing at C1
        # with mean IoU 0.28/frame, and C3 — a cart receding through the doorway,
        # mean IoU 0.11/frame — qualified first and took him. C1 was left
        # ownerless, and an unowned cart can never be scored as abandoned, so the
        # merchandise-removal route was closed before it started. Nothing in the
        # per-cart evidence was wrong; the arbitration was.
        #
        # _link_candidates: cart_raw -> {person_raw: (cum_iou, frames, foot_sum)}
        proposals = []          # (mean_iou, cart_raw, person_raw)
        for cart_id, cart_bbox in cart_bboxes.items():
            if cart_id in self._links:
                continue
            cd = gdi('cart', cart_id)
            if cd in self._perm_carts:
                continue
            cart_age = frame_idx - obj_first_frame.get(cart_id, frame_idx)
            if cart_age < LINK_GRACE_FRAMES:
                continue

            excluded = set(claimed) | {
                pid for pid in person_bboxes
                if gdi('person', pid) in self._perm_persons
            }

            # Has this cart ever moved under observation? Both extra gates below
            # apply to parked carts ONLY, and that restriction is what keeps them
            # off the golden OUTSIDE clip: C1 there rolls in through the doorway
            # and comes to rest, and its real handler works at a cart that is
            # standing still for most of the window that links him. A cart at
            # rest is not the same object as a cart that has never moved — the
            # first has demonstrated it is in use, the second is furniture, and
            # only the second can be grazed by a passer-by with nobody to
            # contradict them.
            parked = self._is_parked(cart_id, frame_idx)

            # Score ALL overlapping + co-moving persons this frame
            candidates = self._link_candidates.get(cart_id, {})
            # Drop candidates who now belong to someone else. Their entry stops
            # accumulating the moment they are excluded but used to stay in the
            # dict forever, and `n_total_candidates` counts entries — so one
            # stale frozen candidate held the threshold at LINK_CONTESTED_FRAMES
            # (20) for the rest of the cart's life, for a person who was no
            # longer available to link.
            for pid in [p for p in candidates if p in excluded]:
                del candidates[pid]
            any_update = False
            for pid, pbbox in person_bboxes.items():
                if pid in excluded:
                    continue
                iou = _iou(cart_bbox, pbbox)
                if iou < LINK_MIN_IOU:
                    continue
                # static_a_ok with the CART as A: a parked cart and a person
                # working at it are not "not co-moving", they are loading or
                # unloading. The reverse (static person, cart rolling past) is
                # still rejected — see are_co_moving().
                #
                # On a PARKED cart it is granted only to a person actually AT it.
                # The exemption was written for a handler holding IoU 0.24-0.42
                # on a cart that had come to rest; extended at LINK_MIN_IOU
                # (0.02) to a cart that has NEVER moved it also exempts a shopper
                # walking past one, and such a cart can never fail the
                # co-movement test, so grazing contact plus LINK_CONFIRM_FRAMES
                # was enough to own it. See LINK_STATIC_MIN_IOU.
                # Keys let the co-movement test latch its static/moving call
                # per track: without them one frame of float wobble at
                # COMOVEMENT_STATIC_PX decides whether this frame is evidence
                # at all, and 19 frames of accumulated evidence rode on it.
                if not are_co_moving(obj_positions.get(cart_id),
                                     obj_positions.get(pid),
                                     static_a_ok=(not parked
                                                  or iou >= LINK_STATIC_MIN_IOU),
                                     latch=self._static_latch,
                                     key_a=cart_id, key_b=pid):
                    continue
                prev_iou, prev_count, prev_foot = candidates.get(pid, (0.0, 0, 0.0))
                candidates[pid] = (prev_iou + iou, prev_count + 1,
                                   prev_foot + foot_ratio(pbbox, cart_bbox))
                any_update = True

            if any_update:
                self._link_candidates[cart_id] = candidates
                self._candidate_misses.pop(cart_id, None)

                # Adaptive threshold: if only 1 candidate ever seen, use
                # fast confirmation (LINK_CONFIRM_FRAMES = 6).
                # If 2+ candidates are competing, use the longer
                # LINK_CONTESTED_FRAMES (20) to give the real pusher time.
                # Ground-plane gate, applied on the accumulated MEAN and never per
                # frame: a pusher is legitimately nearer the camera than their own
                # cart whenever the cart is moving away from it, and on the
                # primary golden clip the correct P1->C1 link is over the bar on
                # 39% of its frames while averaging +0.19. Over a window the two
                # cases separate cleanly — see LINK_GROUND_BAND and
                # tests/sweep_link_geometry.py.
                viable = {
                    pid: (cum_iou, count)
                    for pid, (cum_iou, count, foot_sum) in candidates.items()
                    if _shares_ground_plane(foot_sum / count)
                }

                # Contested-ness counts VIABLE candidates only. A candidate the
                # ground gate has already ruled out is not competition, and
                # counting it was decisive on the 1764099569430 clip: a
                # foreground body sat in the accumulator at mean +0.62, held the
                # bar at LINK_CONTESTED_FRAMES (20) for a cart whose real handler
                # was the only credible candidate, and his evidence was wiped by a
                # barren gap at 18 frames — twice — so he never qualified.
                # A PARKED cart gets the contested threshold whatever the field
                # size. LINK_CONFIRM_FRAMES is 6 — 0.3s at 20fps — and it is
                # calibrated on a cart in motion, where six frames of overlap
                # means six frames of walking together. A cart that has not moved
                # in LINK_STATIC_MIN_FRAMES offers no such evidence: six frames of
                # overlap with it means only that somebody passed close by, which
                # is what every shopper entering the store does to the cart parked
                # inside the door. Raising the bar to LINK_CONTESTED_FRAMES (20,
                # ~1s) costs a genuine pickup at the corral a second of delay and
                # nothing else — the cart starts moving the moment it is taken,
                # and a moving cart is back on the fast path.
                threshold = (LINK_CONTESTED_FRAMES
                             if len(viable) >= 2 or parked
                             else LINK_CONFIRM_FRAMES)
                qualified = {pid: cum_iou for pid, (cum_iou, count) in viable.items()
                             if count >= threshold}

                # Tiebreaker: if 2+ qualified candidates, apply "behind the cart"
                # bonus.  The person pushing is behind the cart relative to its
                # direction of movement.  We use the cart's position history to
                # determine which direction it's going, then favour the person
                # whose centroid is on the trailing side.
                if len(qualified) >= 2:
                    cart_positions = obj_positions.get(cart_id, [])
                    if len(cart_positions) >= 5:
                        # Cart movement vector (recent)
                        dx = cart_positions[-1][0] - cart_positions[-5][0]
                        dy = cart_positions[-1][1] - cart_positions[-5][1]
                        cart_cx = (cart_bbox[0] + cart_bbox[2]) * 0.5
                        cart_cy = (cart_bbox[1] + cart_bbox[3]) * 0.5
                        mag = math.sqrt(dx * dx + dy * dy)
                        if mag > 5:  # cart is actually moving
                            # Normalised movement direction
                            ndx, ndy = dx / mag, dy / mag
                            for pid in qualified:
                                pb = person_bboxes.get(pid)
                                if pb is None:
                                    continue
                                pcx = (pb[0] + pb[2]) * 0.5
                                pcy = (pb[1] + pb[3]) * 0.5
                                # Vector from cart center to person center
                                vx = pcx - cart_cx
                                vy = pcy - cart_cy
                                # Dot product with movement direction:
                                # negative = person is BEHIND the cart (trailing)
                                # positive = person is IN FRONT (leading)
                                dot = vx * ndx + vy * ndy
                                if dot < 0:
                                    # Person is behind — bonus 50% of their cum IoU
                                    qualified[pid] *= 1.5

                best_pid, best_score = None, 0.0
                for pid, score in qualified.items():
                    if score > best_score:
                        best_score = score
                        best_pid = pid

                if best_pid is not None:
                    # Normalised by frame count so it is comparable ACROSS carts
                    # — cumulative IoU is not, since a cart that has been in
                    # frame longer accumulates more of it regardless of how well
                    # it matches anyone.
                    n_frames = max(candidates[best_pid][1], 1)
                    proposals.append((best_score / n_frames, cart_id, best_pid))
            else:
                # A frame with no qualifying candidate is not evidence that the
                # accumulated ones were wrong. Popping it here made confirmation
                # require CONSECUTIVE frames, and with the contested threshold at
                # 20 that is a bar a busy doorway never clears: on the
                # 1764099569430 clip the cart released at frame 109 stayed
                # ownerless for the remaining 312 frames, one barren frame at a
                # time, while the person who had taken it stood next to it.
                #
                # Reuses LINK_CANDIDATE_PATIENCE, which already means "frames a
                # candidate survives without support" for the outscored case.
                misses = self._candidate_misses.get(cart_id, 0) + 1
                if misses >= LINK_CANDIDATE_PATIENCE:
                    self._link_candidates.pop(cart_id, None)
                    self._candidate_misses.pop(cart_id, None)
                else:
                    self._candidate_misses[cart_id] = misses

        # Step 3: assign the proposals, best match first.
        #
        # One person cannot own two carts, so a contested person has to go
        # somewhere, and "wherever qualified first" is not a decision — it is dict
        # order. Sorting by mean IoU makes it one: the cart the person is actually
        # walking with wins, and the cart that merely overlapped them for a while
        # keeps looking for an owner.
        for _score, cart_id, pid in sorted(proposals, key=lambda p: -p[0]):
            if cart_id in self._links or pid in claimed:
                continue
            cd = gdi('cart', cart_id)
            self._links[cart_id] = pid
            self._link_start_frames[cart_id] = frame_idx
            # Opens on the frame the link is made, and is updated from Step 0.5
            # on every frame after it. The accumulated candidate evidence is not
            # reused: it is a SUM over frames, and the question this answers is
            # about one frame's best contact.
            self._link_peak_iou[cart_id] = _iou(cart_bboxes[cart_id],
                                                person_bboxes[pid])
            # Re-open the movement window at the link, so "has it moved" now
            # means "has it moved since this person took it" — the question the
            # parked-cart release in Step 0.5 asks.
            self._open_extent_window(cart_id, obj_positions.get(cart_id), frame_idx)
            claimed.add(pid)
            self._link_candidates.pop(cart_id, None)
            self._candidate_misses.pop(cart_id, None)
            pd = gdi('person', pid)
            self._perm_persons.add(pd)
            self._perm_carts.add(cd)
            self._total_links.add((pd, cd))
            self._person_for_cart[cd] = pd
            self._person_raw_for_cart[cd] = pid
