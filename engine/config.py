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
MODEL_PATH = r"weights\detection\weights\best.pt"
TRACKER_CONFIG = r"engine\botsort_retail.yaml"
TEST_VIDEO_DIR = r"D:\gatekeeper_projects\gk-pops-code\sample_videos"
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
MOONDREAM2_MODEL_ID  = "vikhyatk/moondream2"
QWEN3_VL_MODEL_ID   = "Qwen/Qwen3-VL-2B-Instruct"
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
STALE_CART_FRAMES   = 30      # purge link after cart absent this many frames
ABANDON_FRAMES      = 30      # person gone N frames → abandonment
WALKAWAY_DIST_THRESH = 200    # px — if linked person is farther than this from cart, treat as abandoned

# Re-identification
REID_DIST_THRESH     = 200    # max pixel distance for cart re-ID
REID_MAX_GONE_FRAMES = 15     # max frames a cart can be gone and still re-ID

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
