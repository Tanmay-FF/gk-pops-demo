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
# ---------------------------------------------------------------------------
# Processing cadence
# ---------------------------------------------------------------------------
YOLO_IMGSZ              = 640  # YOLO input size (640=accurate, 480=fast, 384=fastest)
CLASSIFY_EVERY_N_FRAMES = 8
JSON_EVERY_N_FRAMES     = 1   # 1 = every frame (slower), higher = faster

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

# Co-movement
COMOVEMENT_MIN_POSITIONS = 4
COMOVEMENT_WINDOW        = 6
COMOVEMENT_STATIC_PX     = 5
COMOVEMENT_COS_THRESH    = 0.3

# Direction #1763835397930_B8A44FB9F678-medium.mp41763835397930_B8A44FB9F678-medium.mp4
DIRECTION_MIN_POSITIONS  = 10
DIRECTION_MIN_DY         = 20

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
