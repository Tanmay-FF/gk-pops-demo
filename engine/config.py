# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Centralised configuration — paths, thresholds, colours, checkpoint discovery.
Import from here instead of scattering magic numbers across modules.
"""
import os
from pathlib import Path

from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#: Relative to the working directory, which run_demo.py and run_demo.bat both
#: set to the repo root before launching anything.
MODEL_PATH = r"weights\detection\weights\best.pt"
TRACKER_CONFIG = r"engine\botsort_retail.yaml"

#: Where the sample-video dropdown looks. This used to be an absolute path
#: into a sibling checkout on one particular D: drive, which meant the
#: dropdown was empty on every other machine in the world -- the demo still
#: ran, but only if you knew to drag a file into the upload box. It is now
#: the repo's own sample_videos/ directory. Clips are never committed
#: (*.mp4 is gitignored, and they are store CCTV of identifiable people), so
#: on a fresh clone the folder is empty until someone drops one in.
#: run_demo.py says so at startup.
_REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_VIDEO_DIR = str(_REPO_ROOT / "sample_videos")

#: The old location. Still used when the repo's own folder holds no clips, so
#: nothing changes on the machine this was developed on -- the library of test
#: videos there stays one dropdown away. Anywhere else this path does not
#: exist and the repo folder wins by default.
_LEGACY_VIDEO_DIR = r"D:\gatekeeper_projects\gk-pops-code\sample_videos"


def _has_clips(folder):
    return os.path.isdir(folder) and any(
        f.lower().endswith((".mp4", ".avi", ".mov")) for f in os.listdir(folder))


if not _has_clips(TEST_VIDEO_DIR) and _has_clips(_LEGACY_VIDEO_DIR):
    TEST_VIDEO_DIR = _LEGACY_VIDEO_DIR
#RUNS_ROOT = r"D:\gatekeeper_projects\empty_or_full_classification\runs"

# ---------------------------------------------------------------------------
# Detection / tracking colours (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLOR_PERSON = (0, 230, 118)
COLOR_CART   = (0, 165, 255)
COLOR_LINK   = (255, 50, 255)

# POPS event colours (BGR)
COLOR_PUSHOUT    = (0, 0, 255)
COLOR_SUSPICIOUS = (0, 140, 255)
COLOR_MONITORING = (0, 220, 220)
COLOR_CLEAR      = (0, 200, 0)

# Classification overlay colours (BGR)
CLR_VALID   = (0, 200, 0)
CLR_UNCLEAR = (0, 0, 220)
CLR_EMPTY   = (153, 211, 52)
CLR_PARTIAL = (36, 191, 251)
CLR_FULL    = (68, 68, 239)
CLR_NA      = (184, 163, 148)

FILL_COLOR_MAP = {"EMPTY": CLR_EMPTY, "PARTIAL": CLR_PARTIAL, "FULL": CLR_FULL}

# ---------------------------------------------------------------------------
# Bird's Eye View (BEV) panel
# ---------------------------------------------------------------------------
BEV_BG_COLOR        = (30, 30, 35)     # dark background (BGR)
BEV_GRID_COLOR      = (50, 50, 55)     # subtle grid lines
BEV_DOT_RADIUS      = 8
BEV_TRAIL_THICKNESS = 2
BEV_ARROW_LENGTH    = 18
BEV_LABEL_SCALE     = 0.45
BEV_LEGEND_BG       = (40, 40, 45)

# ---------------------------------------------------------------------------
# VLM / Case Report
# ---------------------------------------------------------------------------
VLM_BACKENDS = [
    "Qwen3-VL-2B (local)",
    "Claude (API)",
    "Moondream2 (local)",
    "InternVL2-2B (local)",
]
VLM_DEFAULT_BACKEND = "Qwen3-VL-2B (local)"
FRAME_CAPTURE_POPS_MEDIUM = 30
FRAME_CAPTURE_POPS_HIGH   = 70
FRAME_CAPTURE_MAX         = 8
#: How many deferred case-report payloads may sit unconsumed before the oldest
#: is dropped. One finalize event is chained per run, so the queue normally
#: holds one; it only grows if a chained event never fires (a browser reload
#: mid-run), and each entry pins a full JSON document plus captured frames.
PENDING_CASE_REPORTS_MAX  = 3
MOONDREAM2_MODEL_ID  = "vikhyatk/moondream2"
#: Either a Hugging Face repo id, downloaded on demand into the usual
#: ~/.cache/huggingface cache, or a directory sitting in the repo. The
#: directory wins when it holds real weights, which is the offline path: copy
#: the model's files into models/Qwen3-VL-2B-Instruct/ and the machine never
#: has to reach huggingface.co.
#: transformers' from_pretrained takes a path or a repo id interchangeably,
#: so nothing downstream cares which one it got and no variable has to be
#: set. See models/README.txt.
_BUNDLED_QWEN3_VL = _REPO_ROOT / 'models' / 'Qwen3-VL-2B-Instruct'


def _bundled_model_is_real(folder):
    # The weights must be weights. A half-finished copy, or a Git LFS pointer
    # from someone who bundled the model and cloned without git-lfs, leaves a
    # small text file with the right name -- and pointing transformers at that
    # fails deep inside safetensors with an error that names neither problem.
    # Falling back to the download is far kinder.
    if not (folder / 'config.json').is_file():
        return False
    return any(f.stat().st_size > 1_000_000
               for f in folder.glob('*.safetensors'))


QWEN3_VL_MODEL_ID = (str(_BUNDLED_QWEN3_VL)
                     if _bundled_model_is_real(_BUNDLED_QWEN3_VL)
                     else 'Qwen/Qwen3-VL-2B-Instruct')
INTERNVL2_MODEL_ID   = "OpenGVLab/InternVL2-2B"
VLM_MAX_TOKENS_PER_FRAME = 200
VLM_MAX_TOKENS_SUMMARY   = 1500

# ---------------------------------------------------------------------------
# Classification settings
# ---------------------------------------------------------------------------
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

BAG_CLASSES = ("bagged", "unbagged", "not_applicable")
BAG_NA_IDX  = BAG_CLASSES.index("not_applicable")

# If the "empty" class probability exceeds this threshold, force the
# prediction to "empty" regardless of other class scores.
# Set to 1.0 to disable (i.e., always use argmax).
EMPTY_OVERRIDE_THRESH = 0.5

# Grab-and-run detection (finalisation). Minimum number of CONSECUTIVE
# classified observations of partial/full needed before an abandoned cart's
# final "empty" verdict is overridden back to loaded.
#
# Contiguity is the noise guard, not confidence: a stray single-frame "partial"
# on a genuinely empty cart cannot reach four consecutive observations, while a
# cart that really held merchandise trivially does. At
# CLASSIFY_EVERY_N_FRAMES=8 and 20 fps, 4 observations is ~1.6s of sustained
# merchandise.
GRABRUN_MIN_RUN_OBS = 4
# ---------------------------------------------------------------------------
# Processing cadence
# ---------------------------------------------------------------------------
YOLO_IMGSZ              = 640  # YOLO input size (640=accurate, 480=fast, 384=fastest)
CLASSIFY_EVERY_N_FRAMES = 8
JSON_EVERY_N_FRAMES     = 1   # 1 = every frame (slower), higher = faster

# ---------------------------------------------------------------------------
# Progress REPORTING cadence — a UI transport setting, not a processing one
# ---------------------------------------------------------------------------
# Every frame is still fully processed, classified, scored and logged. This
# only caps how often the browser is TOLD about it.
#
# Why it needs a cap: each progress() call pushes an SSE message, and Gradio
# re-renders a status tracker for EVERY output component of the event. The run
# event has 22 outputs, so per-frame reporting on a 417-frame clip is ~9,200
# component updates — enough to drive Svelte's reactive scheduler into
# `effect_update_depth_exceeded` and wedge the tab after the run completes.
#
# The first and last frame always report, so the bar still starts at 0 and
# lands on 100%.
PROGRESS_MAX_UPDATES    = 50    # per run, excluding the forced first/last
PROGRESS_MIN_INTERVAL_S = 0.15  # never report more often than this

# ---------------------------------------------------------------------------
# Pose estimation (optional, toggled in UI). Used to overlay skeletons in
# the 3D BEV view. Ultralytics auto-downloads the weight on first use.
# ---------------------------------------------------------------------------
POSE_MODEL_PATH       = "weights/pose_estimation/yolo26l-pose.pt"
POSE_IMGSZ            = 640
POSE_CONF_THRESHOLD   = 0.35
POSE_KP_CONF_THRESHOLD = 0.30  # per-keypoint visibility threshold
POSE_MATCH_IOU_MIN    = 0.20  # min IoU between tracked person bbox and pose bbox

# ---------------------------------------------------------------------------
# Linking hyper-parameters
# ---------------------------------------------------------------------------
LINK_CONFIRM_FRAMES = 6       # frames of overlap to confirm link (single candidate)
LINK_CONTESTED_FRAMES = 20    # frames to wait when multiple candidates overlap before deciding
LINK_GRACE_FRAMES   = 15      # wait N frames before linking a new cart
LINK_CANDIDATE_PATIENCE = 4   # frames a candidate survives being outscored before replaced
LINK_DRIFT_FRAMES   = 6       # if linked person IoU < 0.05 with cart for N frames, release link

#: IoU below which a person counts as not engaged with a cart. Used by the
#: linker's drift/takeover rule and by the tracker's abandonment test, which have
#: to agree: "someone is with this cart" and "this cart is unattended" are the
#: same question, and answering it with two different numbers would let a cart be
#: both taken over and abandoned in the same frame.
LINK_DRIFT_IOU = 0.05
STALE_CART_FRAMES   = 30      # purge link after cart absent this many frames

#: Minimum IoU for a person to be considered a candidate owner at all. Only
#: rejects grazing contact — a foreground body clipping the corner of a
#: background cart's box scored 0.05 and won a link on the 1764099569430 clip.
#: Deliberately NOT raised to 0.10: that cart's REAL handler dipped to 0.098, so
#: a 0.10 bar sits on top of the signal. The discrimination is done by
#: LINK_GROUND_BAND below, not by IoU.
LINK_MIN_IOU = 0.02

#: Ground-plane sanity for candidate owners, as a multiple of the CART's own
#: bbox height:
#:
#:     foot_ratio = (person_y2 - cart_y2) / cart_height
#:
#: Positive means the person's feet land in front of the cart's base — nearer the
#: camera. A person standing with a cart shares its ground plane, so the ratio
#: stays small; a body in the extreme foreground that merely overlaps a distant
#: cart in 2D does not.
#:
#: Compared against the MEAN over a candidate's whole window, never per frame.
#: Per-frame the pusher is legitimately nearer the camera than their own cart
#: whenever the cart is being pushed away from it: on the primary golden clip the
#: correct P1->C1 link exceeds this bar on 39% of its frames while averaging
#: +0.19. Measured with tests/sweep_link_geometry.py over every link both golden
#: clips make, the mean separates cleanly — worst legitimate +0.36, best mislink
#: +0.64 — and 0.45 sits in that gap at 0.80x the worst legitimate reading and
#: 1.42x the best mislink.
LINK_GROUND_BAND = 0.45

#: The other end of the same test: a mean below this means the person's box ends
#: above the CART'S TOP across the whole window, i.e. they are behind it in depth.
#: Weaker than the foreground bound — the legitimate P2->C1 pair on the primary
#: clip means -0.87, only 1.15x clear — so it exists to catch the gross case, and
#: a candidate near it should be treated as unproven rather than wrong.
LINK_BEHIND_BAND = -1.0

#: How far a cart's centroid must have wandered — measured as the diagonal of
#: the box its positions have swept, not net displacement — before it counts as
#: having MOVED. Below this it is parked: store furniture, a staging corral, a
#: cart nobody has touched.
#:
#: 12px matches RULE_STATIC_POS_SPREAD_PX, which answers the same question for
#: the rule engine. Named separately because the two are free to be retuned
#: apart: this one gates linking, that one gates dwell reporting.
LINK_STATIC_SPREAD_PX = 12.0

#: Frames of observation before "it has not moved" is allowed to mean anything.
#:
#: A cart that has only just appeared has not moved either, and treating that as
#: parked would apply the parked-cart rules to every cart in its first second —
#: including a shopper pulling one out of the corral, whose cart is genuinely
#: motionless right up to the moment they take it. 40 frames is 2s at 20fps.
#:
#: Read from the point the measurement window opens (first sighting, or the frame
#: the cart's current link was established), not from the cart's first frame.
LINK_STATIC_MIN_FRAMES = 40

#: IoU a person must reach before a PARKED cart's co-movement exemption applies
#: to them.
#:
#: are_co_moving(static_a_ok=True) exists so that "cart standing still, person
#: moving around it" reads as loading or unloading rather than as evidence
#: against ownership. That case is a person AT the cart: on the 1764099569430
#: clip the handler it was written for held IoU 0.24-0.42. Granting the same
#: exemption to grazing contact is what let a shopper walking past a parked cart
#: on the 1764200318790 clip take ownership of it with 6 frames of corner overlap
#: at IoU 0.039-0.064 — and, the cart never having moved since, keep it for the
#: remaining 279 frames of the run.
#:
#: Below this bar a moving person and a parked cart are simply not co-moving, and
#: the candidate is rejected as it was before the exemption existed. Both-static
#: is a separate branch of are_co_moving() and is unaffected: a person standing
#: at a parked cart still accumulates, and LINK_STATIC_MIN_FRAMES/
#: LINK_STATIC_SPREAD_PX raise their bar to LINK_CONTESTED_FRAMES instead.
LINK_STATIC_MIN_IOU = 0.15

ABANDON_FRAMES      = 30      # person gone N frames → abandonment

#: How far the owner has to be from their cart before the walkaway branch of
#: abandonment starts counting, measured as the shortest distance between the
#: two BOUNDING BOXES (zero when they touch or overlap) as a fraction of the
#: cart box's diagonal.
#:
#: This replaced a flat `WALKAWAY_DIST_THRESH = 200` px measured centroid to
#: centroid, which called a cart abandoned while its owner was standing against
#: it. On the 1764173272870 static-cart clip Cart 1 was scored 65 ABANDONED CART
#: with Person 4 in contact with the cart for the whole run: the two boxes
#: touched (edge gap 0.0 px, IoU 0.035) but the cart is large and seen from
#: above, so their centroids sat 232 px apart — past the 200 px bar on every
#: frame. Two properties of a centroid distance caused that and neither is
#: fixable by moving the bar: it grows with the size of the box, so a big cart
#: reads as "far" from anyone standing beside it rather than in front of it, and
#: it is an absolute pixel count, so the same physical distance means different
#: things at the top and bottom of a perspective view.
#:
#: An edge gap is zero for anyone in contact with the cart at any cart size, and
#: dividing by the cart's own diagonal makes the bar scale with the cart's
#: apparent size, which is the cheapest available proxy for depth.
WALKAWAY_GAP_FRAC   = 0.5     # gap > this * cart box diagonal → owner has walked away

#: Floor under the bar above, in pixels, so a cart detected small at the far end
#: of the view cannot trip the counter on a gap of a few pixels — which at that
#: scale is a detection jitter, not a person leaving.
WALKAWAY_MIN_GAP_PX = 40

# Re-identification
REID_DIST_THRESH     = 200    # max pixel distance for cart re-ID
REID_MAX_GONE_FRAMES = 15     # max frames a cart can be gone and still re-ID

# Nested duplicate detections
# The detector emits a tight box and a loose box for the same cart. IoU-based
# NMS cannot suppress that pair: nested boxes measured IoU 0.32-0.51 against the
# 0.7 gate, while their containment (intersection / smaller area) was 0.96-1.00.
# One physical cart therefore became Cart 1 AND Cart 3, scored twice. Boxes at
# or above this containment, of the SAME class, are collapsed to the most
# confident one before the tracker sees them. See engine/detection_dedup.py.
#
# Raise it toward 1.0 if two carts queueing nose-to-tail ever get merged; the
# duplicate pair this exists for sits at 0.96+, so there is little room below.
NESTED_DUP_CONTAIN_MIN = 0.90

# Motion thresholds (px/s)
SPEED_STATIC  = 10
SPEED_SLOW    = 100
SPEED_MEDIUM  = 240

# ---------------------------------------------------------------------------
# Zone congestion thresholds
# ---------------------------------------------------------------------------
# Multi-signal model — a zone is congested when several of these trigger.
# The score sums weighted contributions; severity is bucketed off the score.

# Peak number of distinct tracks simultaneously inside the zone.
# Below MIN, occupancy contributes 0 pts. At/above HIGH, contributes max (40).
ZONE_CONGESTION_MIN_OCCUPANCY  = 3
ZONE_CONGESTION_HIGH_OCCUPANCY = 6

# Fraction of in-zone samples below SPEED_STATIC (px/s). Above this fraction,
# the zone has people standing around (queueing). Capped at 0.70 for max pts.
ZONE_CONGESTION_STATIC_FRAC    = 0.40

# Mean in-zone speed (px/s). Below this, in-zone motion is stalled.
ZONE_CONGESTION_LOW_AVG_SPEED  = 25.0

# Avg-dwell anchor (matches the existing dwell threshold). Above this,
# starts contributing to score; saturates after +90s above threshold.
ZONE_CONGESTION_DWELL_ANCHOR_S = 30.0

# Score → severity buckets (0..100)
ZONE_CONGESTION_WATCH_SCORE         = 25.0
ZONE_CONGESTION_QUEUE_FORMING_SCORE = 45.0
ZONE_CONGESTION_BACKED_UP_SCORE     = 70.0

# ---------------------------------------------------------------------------
# Operational rule engine (engine/rules.py)
# ---------------------------------------------------------------------------
# Decision logic for the operational categories lives here, not in model
# weights — how long a cart must sit before it counts as abandoned is tuned by
# editing these values, no retraining.
#
# All durations are in SECONDS and converted to sample counts at runtime from
# the video's own timestamps. ABANDON_FRAMES above is deliberately NOT reused:
# 30 frames is ~1s at 30fps, which is link bookkeeping, whereas operational
# abandonment is a minutes-scale question.
RULE_ENGINE_ENABLED          = True

# Duration thresholds (seconds)
RULE_BLOCKED_DOOR_S          = 45.0    # egress compliance — shortest fuse
RULE_STATIC_CART_S           = 120.0   # housekeeping / dwell
RULE_ABANDONED_CART_S        = 180.0   # retrieval workflow

# Static test — two signals, not just speed. compute_motion() derives speed
# from a first-to-last delta over the last <=5 positions, so bbox jitter on a
# physically stationary cart can keep it above SPEED_STATIC indefinitely.
# Requiring low positional SPREAD as well is immune to that jitter.
RULE_STATIC_POS_SPREAD_PX    = 12.0    # max distance from the window's centroid
RULE_STATIC_WINDOW_S         = 3.0

# Door geometry — a cart can block a doorway while its centroid sits outside a
# thin door polygon, so doors test bbox overlap fraction, not centroid-inside.
RULE_DOOR_OVERLAP_FRAC       = 0.15    # (cart bbox ∩ door polygon) / bbox area

# Attendance (abandoned-cart rule). Per-track samples have their own
# timestamps, so "was anyone near this cart at time t" needs a shared time grid.
#
#: How close a person has to be for the cart to read as ATTENDED, measured
#: centroid to centroid as a fraction of the cart box's diagonal.
#:
#: This replaced a flat RULE_ATTENDED_RADIUS_PX = 220 px. A pixel count means a
#: different floor distance at every depth: across the trajectory cache the
#: median cart box diagonal is ~237 px (p10 120, p90 316), so 220 px is roughly
#: arm's reach for a cart at the front of frame and most of the room for one at
#: the far door. On the 1764197283870 static-cart clip that is what kept Cart 2
#: silent — parked alone in the entrance vestibule for the whole clip, box
#: diagonal 202 px, with the nearest person's centroid a median 156 px away, so
#: 76% of its samples read as attended and the longest unattended window was
#: 3.5s. Scaling by the cart's own apparent size is the same depth proxy
#: WALKAWAY_GAP_FRAC uses, and it leaves a large near cart at roughly the old
#: behaviour (implied bar p90 221 px) while tightening small far ones (p10 84).
#:
#: NOT the same quantity as WALKAWAY_GAP_FRAC, which scales an EDGE GAP — the
#: fractions are not interchangeable. An edge gap was measured here and is
#: strictly worse for this camera: it looks down the entry lane, so a cart's box
#: touches or overlaps everyone who walks past it (Cart 2's median edge gap to
#: the nearest person is 15 px against a 101 px bar), which collapses its
#: longest window to 0.6s. Box contact in image space is not floor proximity
#: when the view is along the traffic direction, so this rule keeps centroid
#: distance where the linker's walkaway test needs edge gap. Both are
#: image-space proxies; a ground-plane homography is the principled answer and
#: is out of scope for a single-video demo.
#:
#: 0.7 is calibrated, not derived: Cart 2's 202 px diagonal puts the bar at
#: 141 px, and the cart falls back under the bar at 1.0. One calibration point,
#: narrow working range — re-measure before trusting it on a new camera.
RULE_ATTENDED_GAP_FRAC       = 0.7
#: Floor under the bar above, in pixels. A guard against a degenerate box, not
#: a tuned value — it does not engage anywhere in the current cache, where the
#: smallest cart diagonal at p10 is 120 px.
RULE_ATTENDED_MIN_PX         = 60.0
#: Fallback for a track with no recorded boxes (TrackRecord.has_bboxes False).
#: Defensive only: boxes are present on every cart track in the cache.
RULE_ATTENDED_RADIUS_PX      = 220.0
RULE_TIME_GRID_HZ            = 2.0
RULE_GRID_STALENESS_S        = 1.5     # a person seen longer ago than this is not "present"

# Interval hygiene. Positions are only appended when a track is DETECTED, so a
# long occlusion leaves two samples far apart in time that look like continuous
# presence. The density gate rejects intervals that aren't actually observed.
RULE_INTERVAL_MERGE_S        = 3.0     # bridge sub-threshold gaps in one interval
RULE_MAX_SAMPLE_GAP_S        = 2.0     # reject intervals sampled sparser than this
RULE_MIN_SAMPLES             = 8

# Classified-observation counts (NOT frames — fill only refreshes every
# CLASSIFY_EVERY_N_FRAMES, so "10 consecutive frames of empty" can be one
# observation repeated).
RULE_EMPTY_CONFIRM_OBS       = 3

# Entry window for the incoming-cart rule: direction is judged over the first
# N seconds after the cart appears, not over its whole track.
RULE_ENTRY_WINDOW_S          = 4.0

# Which zone kinds each rule monitors.
RULE_DOOR_KINDS              = ("door",)
RULE_STATIC_KINDS            = ("aisle", "analytics")
RULE_DESIGNATED_AREA_KINDS   = ("fixture",)   # cart corrals — carve-out for abandonment

# ---------------------------------------------------------------------------
# Zone-free crowd-cluster detection (queue spike alert)
# ---------------------------------------------------------------------------
# Two people are considered "in the same cluster" when their centroids are
# within this pixel radius of each other.
CROWD_CLUSTER_RADIUS_PX        = 100.0

# A cluster must contain at least this many people to be flagged.
CROWD_CLUSTER_MIN_SIZE         = 3

# A cluster event must persist at least this long to surface as a spike.
CROWD_CLUSTER_MIN_DURATION_S   = 4.0

# Tolerance for stitching cluster samples across consecutive frames into a
# single event (handles missed/dropped detections).
CROWD_CLUSTER_GAP_TOLERANCE_S  = 1.5

# Severity bucketing for crowd clusters: (peak_size, duration_s) ≥ tuple.
CROWD_CLUSTER_QUEUE_FORMING    = (4, 5.0)
CROWD_CLUSTER_BACKED_UP        = (6, 8.0)

# Co-movement
COMOVEMENT_MIN_POSITIONS = 4
COMOVEMENT_WINDOW        = 6
COMOVEMENT_STATIC_PX     = 5

#: Displacement a track already judged STATIC has to exceed before it counts as
#: moving. Together with COMOVEMENT_STATIC_PX this makes the static/moving call
#: a latch rather than a bare threshold: static below 5 px, moving above 7.5,
#: and inside the band a track keeps whatever it was.
#:
#: Without the band a single frame decided a link. On
#: 1764099569430_B8A44F3CB0B9-medium-OUTSIDE.mp4 Cart 21 hovered at 2-4 px of
#: window displacement and touched the bar once, at frame 162, measuring
#: 5.001559 px under one torch/CUDA build and 4.999244 px under another - the
#: same footage, the same track, the same detections to within 0.005 px. Above
#: the bar the cart reads "moving" against a static person, which is the
#: bystander branch of are_co_moving(), so that frame contributed no evidence;
#: the candidate then reached 19 frames against the contested bar of 20, ran
#: into a barren gap, and lost 19 frames of accumulated evidence to
#: LINK_CANDIDATE_PATIENCE. The link landed 14 frames late and the two builds
#: disagreed on 14 frames of output. Nothing about that was version-specific:
#: any run-to-run float wobble can straddle a bar a track is sitting on.
COMOVEMENT_STATIC_EXIT_PX = 7.5
COMOVEMENT_COS_THRESH    = 0.3

# Direction
DIRECTION_MIN_POSITIONS  = 10
DIRECTION_MIN_DY         = 20

# Direction is judged over the last N SECONDS of a track, not over its whole
# history. _obj_positions is never trimmed, so a first-to-last delta answers
# "where did this track start relative to where it is now" — and a shopper who
# enters through the entrance and later leaves through the same door retraces
# their own path, netting a delta under DIRECTION_MIN_DY. That reads as
# UNKNOWN, which loses the OUTBOUND base score and the fill/bag terms with it:
# a full unbagged cart walking out scores 70 as OUTBOUND and 25 as UNKNOWN, and
# 25 is under the 31 that gets an event logged at all.
#
# Seconds rather than a sample count on purpose: positions are appended per
# DETECTION, so "the last 40 samples" is 2s for a cleanly tracked cart and 30s
# for a sparsely detected one. Same index-as-time confusion the rule engine's
# timestamp handling exists to avoid.
DIRECTION_WINDOW_S       = 4.0

# Consecutive frames a cart must resolve OUTBOUND before its heading is LATCHED
# and held through the UNKNOWN frames that follow. See motion.DirectionLatch:
# a cart that reaches the door and stops loses its heading as soon as the
# DIRECTION_WINDOW_S window empties of the approach, and the UNKNOWN branch of
# compute_pops() then caps an abandoned loaded cart at 65 instead of 75.
#
# ~1s at the 19-20 fps these clips run at. Not a formality: DIRECTION_MIN_DY is
# 20px, so one twitch of the trailing window can resolve OUTBOUND on a cart
# parked inside the store, and latching that would promote a shopper who parks
# at a shelf and steps away from ABANDONED CART to PUSHOUT ALERT.
DIRECTION_LATCH_FRAMES   = 20

# ---------------------------------------------------------------------------
# Fixed classifier weights
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path(r"weights")

QUALITY_WEIGHT_PATH = str(WEIGHTS_DIR / "cart_quality" / "weights" / "best.pt")
FILL_WEIGHT_PATH    = str(WEIGHTS_DIR / "fill_and_bag_classifier" / "weights" / "best.pt")

QUALITY_THRESHOLD   = 0.50

# ---------------------------------------------------------------------------
# Sample videos
# ---------------------------------------------------------------------------
SAMPLE_VIDEOS = []
if os.path.isdir(TEST_VIDEO_DIR):
    for f in sorted(os.listdir(TEST_VIDEO_DIR)):
        if f.endswith(('.mp4', '.avi', '.mov')):
            SAMPLE_VIDEOS.append(os.path.join(TEST_VIDEO_DIR, f))
