# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Detection backends and the direct tracking driver.

WHY THIS EXISTS
---------------
The engine was written against ultralytics' `YOLO.track()`. The 3-class
detector (person / empty_cart / non_empty_cart) is a YOLOv5-repo checkpoint,
which `ultralytics.YOLO` cannot load. So detection and tracking are split:

    Detector.detect(frame) -> Detections        (one per backend)
    DirectTrackingModel.track(frame) -> [TrackedResult]
        = Detector.detect + RetailBOTSORT.update, built exactly the way
          ultralytics' own tracking callbacks build and drive it.

`DirectTrackingModel` deliberately exposes the small surface engine/tracker.py
already uses from an ultralytics `YOLO`: `.track()`, `.to()`, `.names`,
`.model` (the nn.Module the device-drift guard scans) and `.predictor.trackers`
(what `_reset_trackers()` resets). The GPU-offload, drift-repair and reset code
in tracker.py therefore works unchanged on either backend.

It registers NO ultralytics callbacks, so the duplicate-tracking-callback
failure documented in engine/ultralytics_compat.py cannot occur on this path.

CART STATE
----------
The 3-class backend reports one "cart" class to the tracker (both cart classes
are merged BEFORE NMS, otherwise per-class NMS keeps an empty_cart box and a
non_empty_cart box on the same cart and the tracker sees two carts) and carries
P(non-empty) = p_non_empty / (p_empty + p_non_empty) alongside each box. The
tracker's output column `idx` maps every track back to its input detection, so
each tracked cart gets its own P(non-empty).

See docs/three_class_detector_plan.md.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import threading
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import torch

#: Unified class ids the 3-class backend reports. Both cart classes become
#: CART_CLS; the empty / non-empty split travels as `p_non_empty`.
PERSON_CLS = 0
CART_CLS = 1
UNIFIED_NAMES = {PERSON_CLS: "person", CART_CLS: "cart"}


class DetectionBackendError(RuntimeError):
    """A detection backend could not be built or used. Always carries a
    message that names the file, path or class that is wrong."""


# ---------------------------------------------------------------------------
# Data passed between detector and tracker
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detections:
    """One frame's detections, before tracking.

    xyxy        (N, 4) float32, original-image pixels
    conf        (N,)   float32
    cls         (N,)   float32, the detector's (or unified) class ids
    p_non_empty (N,)   float32, P(non-empty) for carts; NaN where it does not
                       apply (persons, and every box of a 2-class detector)
    orig_shape  (h, w) of the frame
    """

    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    p_non_empty: np.ndarray
    orig_shape: tuple

    def __post_init__(self):
        n = len(self.conf)
        if self.xyxy.shape != (n, 4):
            raise ValueError(f"xyxy must be ({n}, 4), got {self.xyxy.shape}")
        if self.cls.shape != (n,) or self.p_non_empty.shape != (n,):
            raise ValueError(
                f"cls {self.cls.shape} / p_non_empty {self.p_non_empty.shape} "
                f"must both be ({n},)")
        if len(self.orig_shape) < 2:
            raise ValueError(f"orig_shape must be (h, w), got {self.orig_shape!r}")

    def __len__(self) -> int:
        return len(self.conf)

    @classmethod
    def empty(cls, orig_shape) -> "Detections":
        z = np.zeros((0,), np.float32)
        return cls(np.zeros((0, 4), np.float32), z, z.copy(), z.copy(), tuple(orig_shape[:2]))

    def as_tracker_input(self) -> np.ndarray:
        """(N, 6) [x1, y1, x2, y2, conf, cls] float32 — the layout ultralytics
        Boxes are built from."""
        return np.concatenate(
            [self.xyxy, self.conf[:, None], self.cls[:, None]], axis=1
        ).astype(np.float32, copy=False)


def _check_frame(frame) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        shape = getattr(frame, "shape", None)
        raise ValueError(f"expected an HxWx3 BGR frame, got {type(frame).__name__} {shape}")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"empty frame {frame.shape}")


def module_device(module: torch.nn.Module) -> torch.device:
    """Device of a module's first parameter (CPU when it has none). Used to
    place input tensors, so inference follows the weights through an offload
    round-trip instead of trusting a device string that may be stale."""
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Backend 1: the existing 2-class ultralytics weights, detection only
# ---------------------------------------------------------------------------
class UltralyticsDetector:
    """The current 2-class detector, used for detection only.

    Reproduces what `YOLO.track()` feeds its tracker: `predict()` with
    conf=ULTRALYTICS_TRACK_CONF (track() forces 0.1), batch 1, the caller's
    imgsz. Class ids and names pass through untouched, so the engine's
    `names[int(c)] == "cart"` checks see exactly what they see today.
    """

    provides_cart_state = False

    def __init__(self, model_path: str, conf: float):
        if not os.path.isfile(model_path):
            raise DetectionBackendError(f"detector weights not found: {model_path}")
        if not 0.0 < conf < 1.0:
            raise DetectionBackendError(f"conf must be in (0, 1), got {conf}")
        from ultralytics import YOLO

        try:
            self._yolo = YOLO(model_path)
        except Exception as e:
            raise DetectionBackendError(f"could not load {model_path}: {e}") from e
        self._conf = float(conf)
        self.names = dict(self._yolo.names)

    @property
    def module(self) -> torch.nn.Module:
        return self._yolo.model

    def to(self, device) -> "UltralyticsDetector":
        # YOLO.to() nulls the internal predictor; predict() rebuilds it. This
        # wrapper never registers tracking callbacks, so that is harmless.
        self._yolo.to(device)
        return self

    def detect(self, frame: np.ndarray, imgsz: int) -> Detections:
        _check_frame(frame)
        res = self._yolo.predict(frame, imgsz=imgsz, conf=self._conf, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return Detections.empty(frame.shape)
        b = res[0].boxes.cpu().numpy()
        n = len(b)
        return Detections(
            xyxy=b.xyxy.astype(np.float32, copy=False),
            conf=b.conf.astype(np.float32, copy=False),
            cls=b.cls.astype(np.float32, copy=False),
            p_non_empty=np.full((n,), np.nan, np.float32),
            orig_shape=tuple(frame.shape[:2]),
        )


# ---------------------------------------------------------------------------
# Backend 2: YOLOv5 person / empty_cart / non_empty_cart
# ---------------------------------------------------------------------------
_V5_IMPORT_LOCK = threading.Lock()
#: Top-level module names the YOLOv5 checkpoint unpickles against.
_V5_TOPLEVEL = ("models", "utils")


class _ProcessStateGuard:
    """Snapshot and restore process-wide state that YOLOv5 changes on import.

    YOLOv5's `utils.general` / `utils.plots` / `utils.torch_utils` run, at
    import time: `cv2.setNumThreads(0)` (would single-thread OpenCV for the
    whole engine: video decode, resize, encode), `warnings.filterwarnings(
    "ignore", category=UserWarning)` (would hide every UserWarning in the
    process), `matplotlib.use("Agg")`, numpy / torch / pandas print options and
    three thread-count environment variables. None of that is needed for
    inference, so all of it is put back. Each restore is independent and
    best-effort: one failing must not skip the others or mask the import's
    own error.
    """

    _ENV_KEYS = ("NUMEXPR_MAX_THREADS", "OMP_NUM_THREADS", "TF_CPP_MIN_LOG_LEVEL")

    def __enter__(self):
        import cv2

        self._cv2_threads = cv2.getNumThreads()
        self._warn_filters = list(warnings.filters)
        self._np_print = np.get_printoptions()
        po = torch._tensor_str.PRINT_OPTS
        self._torch_print = dict(precision=po.precision, threshold=po.threshold,
                                 edgeitems=po.edgeitems, linewidth=po.linewidth,
                                 sci_mode=po.sci_mode)
        self._env = {k: os.environ.get(k) for k in self._ENV_KEYS}
        mpl = sys.modules.get("matplotlib")
        self._mpl_backend = mpl.get_backend() if mpl is not None else None
        pd = sys.modules.get("pandas")
        self._pd_max_cols = pd.options.display.max_columns if pd is not None else None
        return self

    def __exit__(self, *exc):
        import cv2

        def attempt(what: str, fn: Callable[[], None]):
            try:
                fn()
            except Exception as e:  # restore is best-effort, never raises
                print(f"[WARN] could not restore {what} after importing YOLOv5: {e}")

        attempt("OpenCV thread count", lambda: cv2.setNumThreads(self._cv2_threads))

        def restore_warning_filters():
            warnings.filters[:] = self._warn_filters
            # Editing the list directly bypasses the version counter that
            # filterwarnings() bumps; without it, per-module warning
            # registries can keep "ignore" decisions made while YOLOv5's
            # blanket UserWarning filter was in force.
            mutated = getattr(warnings, "_filters_mutated", None)
            if callable(mutated):
                mutated()

        attempt("warning filters", restore_warning_filters)
        attempt("numpy print options", lambda: np.set_printoptions(**self._np_print))
        attempt("torch print options", lambda: torch.set_printoptions(**self._torch_print))

        def restore_env():
            for k, v in self._env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        attempt("thread environment variables", restore_env)
        if self._mpl_backend is not None:
            mpl = sys.modules.get("matplotlib")
            if mpl is not None and mpl.get_backend() != self._mpl_backend:
                attempt("matplotlib backend", lambda: mpl.use(self._mpl_backend))
        if self._pd_max_cols is not None:
            pd = sys.modules.get("pandas")
            if pd is not None:
                attempt("pandas display options", lambda: setattr(
                    pd.options.display, "max_columns", self._pd_max_cols))
        return False


def _module_root(mod) -> Optional[Path]:
    f = getattr(mod, "__file__", None)
    if f:
        return Path(f).resolve().parent
    paths = list(getattr(mod, "__path__", []) or [])
    return Path(paths[0]).resolve() if paths else None


@contextlib.contextmanager
def _yolov5_on_path(repo: Path):
    """Make the YOLOv5 repo importable for the duration of the block.

    The repo's packages are named `models` and `utils` — generic enough that
    another top-level module of the same name may already be imported. Reusing
    it would load the checkpoint against the wrong code, so that is refused
    with a message naming both locations. The repo is removed from sys.path
    afterwards (its other top-level names — `data`, `val`, `export`, `train` —
    would otherwise shadow later imports); the modules already imported stay.

    YOLOv5's `utils.plots` imports seaborn, which inference never uses. When
    seaborn is not installed a stub stands in for the import only; touching
    any attribute of it raises an ImportError that says why.
    """
    for name in _V5_TOPLEVEL:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        root = _module_root(mod)
        if root is None or root.parent != repo:
            raise DetectionBackendError(
                f"a different top-level '{name}' module is already imported "
                f"(from {root}); the YOLOv5 checkpoint needs {repo / name}. "
                f"Load the YOLOv5 backend before importing it, or rename it.")

    stub_name = None
    if "seaborn" not in sys.modules and importlib.util.find_spec("seaborn") is None:
        stub = types.ModuleType("seaborn")
        stub.__doc__ = "Stub inserted by engine.detection_backends (seaborn not installed)."

        def _missing(attr):
            # AttributeError, not ImportError: hasattr()/getattr(default) and
            # inspect (which probes `__file__` on every module while YOLOv5's
            # utils.general calls inspect.stack() at import) only swallow
            # AttributeError, so anything else breaks unrelated introspection.
            raise AttributeError(
                f"seaborn.{attr}: seaborn is not installed; it is stubbed only "
                f"so YOLOv5's plotting module imports for inference")

        stub.__getattr__ = _missing
        sys.modules["seaborn"] = stub
        stub_name = "seaborn"

    repo_str = str(repo)
    added = repo_str not in sys.path
    if added:
        sys.path.insert(0, repo_str)
    try:
        with _ProcessStateGuard():
            yield
    finally:
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(repo_str)
        if stub_name and sys.modules.get(stub_name) is not None \
                and getattr(sys.modules[stub_name], "__doc__", "").startswith("Stub inserted"):
            del sys.modules[stub_name]


def merge_cart_scores(probs: torch.Tensor, person_i: int, empty_i: int, non_empty_i: int,
                      dim: int = -1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The one place the two cart classes become one, for every 3-class backend.

    `probs` holds per-class probabilities along `dim`. Returns three tensors,
    each keeping `dim` with size 1: p_person, p_cart = max(p_empty,
    p_non_empty) (so a cart box ranks by its best cart class, exactly as
    3-class NMS would rank it), and share = p_non_empty / (p_empty +
    p_non_empty), i.e. P(non-empty | cart). Any other class is ignored.
    """
    nc = probs.shape[dim]
    for name, i in (("person", person_i), ("empty_cart", empty_i), ("non_empty_cart", non_empty_i)):
        if not 0 <= i < nc:
            raise ValueError(f"{name} index {i} out of range for {nc} classes")
    pick = lambda i: probs.narrow(dim, i, 1)
    p_empty, p_non_empty = pick(empty_i), pick(non_empty_i)
    share = p_non_empty / (p_empty + p_non_empty).clamp_min(1e-9)
    return pick(person_i), torch.maximum(p_empty, p_non_empty), share


def merge_cart_classes(pred: torch.Tensor, person_i: int, empty_i: int,
                       non_empty_i: int) -> torch.Tensor:
    """Collapse YOLOv5's two cart classes into one before NMS.

    In:  (B, N, 5 + nc)  [cx, cy, w, h, obj, p_0 .. p_nc-1]  (sigmoid probs)
    Out: (B, N, 9)       [cx, cy, w, h, obj, p_person, p_cart, share, 1]

    p_cart = max(p_empty, p_non_empty) keeps a cart box's NMS score equal to
    its best cart class, which is what 3-class NMS would have ranked it by.
    share = p_non_empty / (p_empty + p_non_empty) is P(non-empty | cart).

    The last two columns ride through YOLOv5's non_max_suppression as mask
    columns (`nm=2`). That function multiplies EVERY column after objectness
    by objectness, so the constant 1 comes back as `obj` and the share as
    `share * obj`; recover_share() divides them. Any other class the
    checkpoint may have is dropped.
    """
    if pred.ndim != 3 or pred.shape[2] < 6:
        raise ValueError(f"expected (B, N, 5+nc) predictions, got {tuple(pred.shape)}")
    p_person, p_cart, share = merge_cart_scores(pred[..., 5:], person_i, empty_i, non_empty_i, dim=-1)
    return torch.cat([pred[..., :5], p_person, p_cart, share, torch.ones_like(share)], dim=-1)


def recover_share(scaled_share: np.ndarray, scaled_one: np.ndarray) -> np.ndarray:
    """Undo NMS's objectness scaling: share = (share * obj) / (1 * obj).
    NaN where obj is not positive (cannot happen for a kept box, since NMS
    keeps only obj > conf_thres > 0, but never divide by it unchecked)."""
    scaled_share = np.asarray(scaled_share, np.float64)
    scaled_one = np.asarray(scaled_one, np.float64)
    out = np.full(scaled_share.shape, np.nan)
    ok = scaled_one > 0
    out[ok] = np.clip(scaled_share[ok] / scaled_one[ok], 0.0, 1.0)
    return out.astype(np.float32)


def _three_class_indices(names, class_names: Mapping[str, str], weights: str):
    """Resolve a 3-class checkpoint's (person, empty_cart, non_empty_cart)
    class indices BY NAME, so a checkpoint with another class order still
    maps correctly. Returns (raw_names, (person_i, empty_i, non_empty_i))."""
    raw = names if isinstance(names, Mapping) else dict(enumerate(names))
    raw_names = {int(k): str(v) for k, v in raw.items()}
    by_name = {v: k for k, v in raw_names.items()}
    if len(by_name) != len(raw_names):
        raise DetectionBackendError(f"{weights} has duplicate class names: {raw_names}")
    try:
        return raw_names, (by_name[class_names["person"]], by_name[class_names["empty_cart"]],
                           by_name[class_names["non_empty_cart"]])
    except KeyError as e:
        raise DetectionBackendError(
            f"{weights} has classes {raw_names}; expected "
            f"{sorted(class_names.values())} (missing {e})") from None


class YoloV5ThreeClassDetector:
    """YOLOv5 person / empty_cart / non_empty_cart, PyTorch weights only."""

    provides_cart_state = True

    def __init__(self, weights: str, repo_path: str, device: str, conf: float,
                 iou: float, imgsz: int, class_names: Mapping[str, str]):
        repo = Path(repo_path).resolve()
        if not (repo / "models" / "common.py").is_file() or not (repo / "utils" / "general.py").is_file():
            raise DetectionBackendError(f"not a YOLOv5 repository: {repo}")
        if not os.path.isfile(weights):
            raise DetectionBackendError(f"3-class detector weights not found: {weights}")
        if not str(weights).lower().endswith(".pt"):
            raise DetectionBackendError(f"only PyTorch .pt weights are supported, got {weights}")
        if not 0.0 < conf < 1.0 or not 0.0 < iou <= 1.0:
            raise DetectionBackendError(f"conf must be in (0, 1) and iou in (0, 1]; got {conf}, {iou}")
        if int(imgsz) <= 0:
            raise DetectionBackendError(f"imgsz must be positive, got {imgsz}")

        with _V5_IMPORT_LOCK, _yolov5_on_path(repo):
            try:
                from models.common import DetectMultiBackend
                from utils.augmentations import letterbox
                from utils.general import non_max_suppression, scale_boxes
            except Exception as e:
                raise DetectionBackendError(
                    f"could not import YOLOv5 from {repo} (missing dependency?): {e}") from e
            try:
                net = DetectMultiBackend(weights, device=torch.device(device), fuse=True)
            except Exception as e:
                raise DetectionBackendError(f"could not load {weights}: {e}") from e

        if not getattr(net, "pt", False):
            raise DetectionBackendError(f"{weights} did not load as a PyTorch model")
        net.eval()
        self._net = net
        self._letterbox = letterbox
        self._nms = non_max_suppression
        self._scale_boxes = scale_boxes
        self._stride = int(net.stride) if not isinstance(net.stride, torch.Tensor) else int(net.stride.max())
        self._conf, self._iou, self._imgsz = float(conf), float(iou), int(imgsz)

        self.raw_names, (self._person_i, self._empty_i, self._non_empty_i) = \
            _three_class_indices(net.names, class_names, weights)
        self.names = dict(UNIFIED_NAMES)

    @property
    def module(self) -> torch.nn.Module:
        return self._net

    def to(self, device) -> "YoloV5ThreeClassDetector":
        self._net.to(device)
        return self

    def detect(self, frame: np.ndarray, imgsz: Optional[int] = None) -> Detections:
        """`imgsz` is accepted for interface parity and ignored: this model's
        input size is its own configured V5_IMGSZ, not the 2-class detector's."""
        _check_frame(frame)
        img = self._letterbox(frame, self._imgsz, stride=self._stride, auto=True)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])  # HWC BGR -> CHW RGB
        dev = module_device(self._net)
        t = torch.from_numpy(img).to(dev).float().div_(255.0).unsqueeze(0)
        with torch.inference_mode():
            pred = self._net(t)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            merged = merge_cart_classes(pred.float(), self._person_i, self._empty_i, self._non_empty_i)
            out = self._nms(merged, self._conf, self._iou, nm=2)[0]
            if out is None or len(out) == 0:
                return Detections.empty(frame.shape)
            # Inside inference_mode: scale_boxes edits `out` in place, and an
            # inference tensor cannot be modified outside it.
            out[:, :4] = self._scale_boxes(t.shape[2:], out[:, :4], frame.shape).round()
            out = out.detach().cpu().numpy()
        cls = out[:, 5].astype(np.float32)
        share = recover_share(out[:, 6], out[:, 7])
        p_non_empty = np.where(cls == CART_CLS, share, np.nan).astype(np.float32)
        return Detections(
            xyxy=out[:, :4].astype(np.float32),
            conf=out[:, 4].astype(np.float32),
            cls=cls,
            p_non_empty=p_non_empty,
            orig_shape=tuple(frame.shape[:2]),
        )


# ---------------------------------------------------------------------------
# Backend 3: YOLOv26 person / empty_cart / non_empty_cart (ultralytics format)
# ---------------------------------------------------------------------------
class Yolo26ThreeClassDetector:
    """YOLOv26 3-class, run the way ultralytics' predict() runs it, with the
    two cart classes merged before NMS.

    ultralytics' Results keep only each box's top class, so P(non-empty)
    cannot be read off predict(). This class therefore does predict()'s steps
    itself, on the same fused model:

      LetterBox(imgsz, auto=True, stride)  BGR->RGB, CHW, /255   (pre_transform + preprocess)
      net(t) -> (B, 4 + nc, N): xywh + sigmoid class probs        (Detect._inference)
      merge_cart_scores -> [xywh, p_person, p_cart, share]
      ultralytics nms.non_max_suppression(nc=2): `share` rides as an extra
        column, which that function carries through UNSCALED (no objectness
        in this head, unlike YOLOv5)
      ops.scale_boxes back to the frame                            (construct_result)

    Only the non-end-to-end head is supported. The supplied checkpoint is
    saved with its one-to-one head disabled, so ultralytics itself runs it this
    way; an end-to-end checkpoint is refused rather than silently run in a
    mode its metrics were not measured in.

    Equivalence with predict() is pinned by tests/test_yolo26_backend.py:
    person boxes identical, every cart box identical with conf equal to the
    best of the cart classes predict() reported at that box.
    """

    provides_cart_state = True

    def __init__(self, weights: str, device: str, conf: float, iou: float, imgsz: int,
                 class_names: Mapping[str, str]):
        if not os.path.isfile(weights):
            raise DetectionBackendError(f"3-class detector weights not found: {weights}")
        if not 0.0 < conf < 1.0 or not 0.0 < iou <= 1.0:
            raise DetectionBackendError(f"conf must be in (0, 1) and iou in (0, 1]; got {conf}, {iou}")
        if int(imgsz) <= 0 or int(imgsz) % 32:
            raise DetectionBackendError(f"imgsz must be a positive multiple of 32, got {imgsz}")
        from ultralytics import YOLO
        from ultralytics.data.augment import LetterBox
        from ultralytics.nn.modules.head import Detect
        from ultralytics.utils import nms, ops

        try:
            yolo = YOLO(weights)
        except Exception as e:
            raise DetectionBackendError(f"could not load {weights}: {e}") from e
        if yolo.task != "detect":
            raise DetectionBackendError(f"{weights} is a {yolo.task!r} model, not 'detect'")
        net = yolo.model
        head = net.model[-1]
        if not isinstance(head, Detect):
            raise DetectionBackendError(f"{weights}: last layer is {type(head).__name__}, not Detect")
        if getattr(head, "end2end", False):
            raise DetectionBackendError(
                f"{weights} runs its end-to-end (NMS-free) head; this backend reproduces the "
                f"one-to-many + NMS path only")
        self.raw_names, (self._person_i, self._empty_i, self._non_empty_i) = \
            _three_class_indices(yolo.names, class_names, weights)
        self._nc = int(head.nc)
        # Same fusion predict()'s AutoBackend applies (Conv+BN folded); the
        # one-to-many head stays because end2end is off.
        net = net.fuse(verbose=False)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self._net = net
        self._stride = int(max(net.stride.tolist())) if isinstance(net.stride, torch.Tensor) else int(net.stride)
        self._letterbox = LetterBox((int(imgsz), int(imgsz)), auto=True, stride=self._stride)
        self._nms, self._ops = nms, ops
        self._conf, self._iou, self._imgsz = float(conf), float(iou), int(imgsz)
        self.names = dict(UNIFIED_NAMES)
        self.to(device)

    @property
    def module(self) -> torch.nn.Module:
        return self._net

    def to(self, device) -> "Yolo26ThreeClassDetector":
        self._net.to(device)
        return self

    def detect(self, frame: np.ndarray, imgsz: Optional[int] = None) -> Detections:
        """`imgsz` is accepted for interface parity and ignored: this model's
        input size is its own configured Y26_IMGSZ."""
        _check_frame(frame)
        img = self._letterbox(image=frame)
        img = np.ascontiguousarray(img[..., ::-1].transpose((2, 0, 1)))  # HWC BGR -> CHW RGB
        dev = module_device(self._net)
        t = torch.from_numpy(img).to(dev).float().div_(255.0).unsqueeze(0)
        with torch.inference_mode():
            pred = self._net(t)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            if pred.ndim != 3 or pred.shape[1] != 4 + self._nc:
                raise DetectionBackendError(
                    f"unexpected YOLOv26 output {tuple(pred.shape)}; expected (B, {4 + self._nc}, N)")
            p_person, p_cart, share = merge_cart_scores(
                pred[:, 4:].float(), self._person_i, self._empty_i, self._non_empty_i, dim=1)
            merged = torch.cat([pred[:, :4].float(), p_person, p_cart, share], dim=1)
            out = self._nms.non_max_suppression(merged, self._conf, self._iou, nc=2)[0]
            if out is None or len(out) == 0:
                return Detections.empty(frame.shape)
            out[:, :4] = self._ops.scale_boxes(t.shape[2:], out[:, :4], frame.shape)
            out = out.detach().cpu().numpy()
        cls = out[:, 5].astype(np.float32)
        p_non_empty = np.where(cls == CART_CLS, np.clip(out[:, 6], 0.0, 1.0), np.nan).astype(np.float32)
        return Detections(
            xyxy=out[:, :4].astype(np.float32),
            conf=out[:, 4].astype(np.float32),
            cls=cls,
            p_non_empty=p_non_empty,
            orig_shape=tuple(frame.shape[:2]),
        )


# ---------------------------------------------------------------------------
# Tracking driver
# ---------------------------------------------------------------------------
class TrackedResult:
    """Per-frame output, shaped like the slice of an ultralytics `Results` the
    engine reads: `.boxes` (ultralytics `Boxes` with `.xyxy .id .cls .conf`;
    `.id` is None when nothing is tracked) plus `.p_non_empty`, a numpy array
    aligned with the boxes (NaN where it does not apply)."""

    __slots__ = ("boxes", "p_non_empty", "orig_shape")

    def __init__(self, boxes, p_non_empty: np.ndarray, orig_shape):
        self.boxes = boxes
        self.p_non_empty = p_non_empty
        self.orig_shape = orig_shape


class _TrackerHolder:
    """Stands where ultralytics' predictor stands, for the one attribute the
    engine reads from it: `trackers` (see TrackingEngine._reset_trackers)."""

    def __init__(self):
        self.trackers: list = []


def _build_tracker(config_path: str):
    """Construct the configured tracker exactly as ultralytics'
    `on_predict_start` does: merged YAML -> IterableSimpleNamespace, class from
    TRACKER_MAP (which engine/ultralytics_compat.install() points at the Retail
    subclasses), and frame_rate=30 regardless of the clip, because upstream
    hard-codes it and `max_time_lost = frame_rate / 30 * track_buffer`."""
    from ultralytics.trackers import track as _track
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    try:
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(config_path)))
    except Exception as e:
        raise DetectionBackendError(f"could not read tracker config {config_path}: {e}") from e
    if cfg.tracker_type not in _track.TRACKER_MAP:
        raise DetectionBackendError(
            f"unsupported tracker_type {cfg.tracker_type!r} in {config_path}; "
            f"expected one of {sorted(_track.TRACKER_MAP)}")
    if cfg.tracker_type == "botsort" and getattr(cfg, "with_reid", False):
        raise DetectionBackendError(
            f"{config_path}: with_reid=True needs detector features this driver "
            f"does not extract; keep with_reid: false")
    return _track.TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)


class DirectTrackingModel:
    """A detector plus one tracker, presented as the subset of ultralytics'
    `YOLO` that engine/tracker.py uses. See the module docstring."""

    def __init__(self, detector, backend_name: str):
        self.detector = detector
        self.backend_name = backend_name
        self.names = dict(detector.names)
        self.provides_cart_state = bool(getattr(detector, "provides_cart_state", False))
        self._holder = _TrackerHolder()
        self._tracker_config: Optional[str] = None

    # -- what the device-drift guard and the offload code touch ---------------
    @property
    def predictor(self) -> _TrackerHolder:
        return self._holder

    @predictor.setter
    def predictor(self, value) -> None:
        """tracker.py parks and re-attaches `predictor` across a GPU offload
        (so the same holder comes back) and tests set it to None to simulate
        ultralytics losing it. None means what it means upstream — the tracker
        state is gone, the next track() starts a fresh tracker — but never a
        crash, and never None on read."""
        if value is None:
            self._holder = _TrackerHolder()
        elif hasattr(value, "trackers"):
            self._holder = value
        else:
            raise TypeError(f"predictor must expose `trackers`, got {type(value).__name__}")
    @property
    def model(self) -> torch.nn.Module:
        return self.detector.module

    def to(self, device) -> "DirectTrackingModel":
        self.detector.to(device)
        return self

    # -- tracking ------------------------------------------------------------
    def _tracker_for(self, config_path: str):
        if self._tracker_config != config_path or not self.predictor.trackers:
            self.predictor.trackers = [_build_tracker(config_path)]
            self._tracker_config = config_path
        return self.predictor.trackers[0]

    def track(self, source, persist: bool = True, tracker: str = "", imgsz: int = 640,
              verbose: bool = False, **_ignored) -> list:
        """Detect and track one BGR frame. Returns a one-element list, like
        `YOLO.track()` on a single image.

        persist=False resets the tracker before this frame (a fresh sequence),
        which is what ultralytics does when it sees a new source. The engine
        always passes True and resets between runs via `predictor.trackers`.
        """
        if not tracker:
            raise ValueError("a tracker config path is required")
        from ultralytics.engine.results import Boxes

        trk = self._tracker_for(tracker)
        if not persist:
            trk.reset()
        det = self.detector.detect(source, imgsz)
        boxes_in = Boxes(det.as_tracker_input(), det.orig_shape)
        tracks = trk.update(boxes_in, source)
        if tracks is None or len(tracks) == 0:
            # Same as ultralytics: an untracked frame keeps id None.
            empty = Boxes(np.zeros((0, 6), np.float32), det.orig_shape)
            return [TrackedResult(empty, np.zeros((0,), np.float32), det.orig_shape)]
        tracks = np.asarray(tracks, dtype=np.float32)
        idx = tracks[:, -1].astype(int)
        if idx.min() < 0 or idx.max() >= len(det):
            raise DetectionBackendError(
                f"tracker returned detection index {idx.max()} for {len(det)} detections")
        # [x1, y1, x2, y2, track_id, score, cls] — what ultralytics' callback
        # writes back with `result.update(boxes=tracks[:, :-1])`. That update
        # CLIPS the boxes to the image (`ops.clip_boxes`): tracker boxes are
        # Kalman estimates and run past the frame edge for a cart leaving
        # view. Without the clip the two paths disagree by up to ~10 px on
        # those frames while every id and class still matches.
        from ultralytics.utils import ops

        boxes_out = Boxes(ops.clip_boxes(torch.as_tensor(tracks[:, :-1]), det.orig_shape),
                          det.orig_shape)
        return [TrackedResult(boxes_out, det.p_non_empty[idx].copy(), det.orig_shape)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_detection_model(backend: str, device: str, model_path: Optional[str] = None):
    """The detector object TrackingEngine stores as `self.model`.

    "ultralytics_track" returns a plain ultralytics YOLO (the original path,
    byte-for-byte); every other backend returns a DirectTrackingModel.
    `model_path` overrides config.MODEL_PATH for the two 2-class backends and
    is ignored by the 3-class ones (which load V5_WEIGHTS_PATH /
    Y26_WEIGHTS_PATH).
    """
    from . import config as C

    if backend not in C.DETECTOR_BACKENDS:
        raise DetectionBackendError(
            f"unknown DETECTOR_BACKEND {backend!r}; expected one of {C.DETECTOR_BACKENDS}")
    weights_2class = model_path or C.MODEL_PATH
    if backend == "ultralytics_track":
        from ultralytics import YOLO

        if not os.path.isfile(weights_2class):
            raise DetectionBackendError(f"detector weights not found: {weights_2class}")
        model = YOLO(weights_2class)
        model.to(device)
        return model
    names3 = {"person": C.THREE_CLASS_PERSON, "empty_cart": C.THREE_CLASS_EMPTY_CART,
              "non_empty_cart": C.THREE_CLASS_NON_EMPTY_CART}
    if backend == "ultralytics_direct":
        det = UltralyticsDetector(weights_2class, conf=C.ULTRALYTICS_TRACK_CONF)
    elif backend == "yolov5_3class":
        det = YoloV5ThreeClassDetector(
            C.V5_WEIGHTS_PATH, C.V5_REPO_PATH, device=device,
            conf=C.V5_CONF, iou=C.V5_IOU, imgsz=C.V5_IMGSZ, class_names=names3)
    elif backend == "yolo26_3class":
        det = Yolo26ThreeClassDetector(
            C.Y26_WEIGHTS_PATH, device=device,
            conf=C.Y26_TRACK_CONF, iou=C.Y26_IOU, imgsz=C.Y26_IMGSZ, class_names=names3)
    else:  # DETECTOR_BACKENDS grew without this factory learning the new name
        raise DetectionBackendError(f"no factory for backend {backend!r}")
    model = DirectTrackingModel(det, backend)
    model.to(device)
    return model


def backend_fingerprint(backend: str) -> str:
    """Identity of the detection backend for the trajectory cache key: name
    plus the stamp of the weights it loads. Never raises."""
    try:
        from . import config as C

        if backend == "yolov5_3class":
            path = C.V5_WEIGHTS_PATH
            extra = f"conf={C.V5_CONF},iou={C.V5_IOU},imgsz={C.V5_IMGSZ}"
        elif backend == "yolo26_3class":
            path = C.Y26_WEIGHTS_PATH
            extra = (f"conf={C.Y26_TRACK_CONF},iou={C.Y26_IOU},imgsz={C.Y26_IMGSZ},"
                     f"state_min_conf={C.CART_STATE_MIN_CONF}")
        else:
            path = C.MODEL_PATH
            extra = f"conf={C.ULTRALYTICS_TRACK_CONF}" if backend == "ultralytics_direct" else ""
        st = os.stat(path)
        return f"{backend}:{st.st_size}:{st.st_mtime_ns}:{extra}"
    except Exception:
        return f"{backend}:unreadable"
