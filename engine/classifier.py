# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Cart crop classification — Stage 1 (quality) + Stage 2 (fill/bag).

Handles cropping, transform, inference, and temperature-scaled softmax.
Supports batched inference for multiple carts in a single GPU pass.
"""
import cv2
import torch
from PIL import Image

from .config import CLS_TRANSFORM, BAG_NA_IDX, BAG_CLASSES, EMPTY_OVERRIDE_THRESH
from .models import load_quality_checkpoint, load_fill_checkpoint

_EMPTY_RESULT_UNCLEAR = {
    "is_valid": False, "quality": "unclear",
    "fill": "non-applicable", "bag": "non-applicable",
    "quality_conf": 0.0, "fill_conf": 0.0, "bag_conf": 0.0,
}
_EMPTY_RESULT_TINY = dict(_EMPTY_RESULT_UNCLEAR)
_EMPTY_RESULT_NO_MODEL = {
    "is_valid": True, "quality": "unclassified",
    "fill": "unclassified", "bag": "unclassified",
    "quality_conf": 0.0, "fill_conf": 0.0, "bag_conf": 0.0,
}


class CartClassifier:
    """Stateful wrapper around quality + fill/bag models with lazy loading."""

    __slots__ = (
        "device",
        "_quality_model", "_quality_pt", "_quality_temp",
        "_fill_model", "_fill_pt", "_fill_temp", "_bag_temp",
        "_fill_classes", "_bag_classes", "_n_bag_model",
        "_quality_threshold",
    )

    def __init__(self, device: str):
        self.device = device
        self._quality_model = None
        self._quality_pt = None
        self._quality_temp = 1.0
        self._fill_model = None
        self._fill_pt = None
        self._fill_temp = 1.0
        self._bag_temp = 1.0
        self._fill_classes = ["empty", "partial", "full"]
        self._bag_classes = list(BAG_CLASSES)
        self._n_bag_model = len(BAG_CLASSES)
        self._quality_threshold = 0.75

    @property
    def quality_pt(self):
        return self._quality_pt

    @property
    def fill_pt(self):
        return self._fill_pt

    @property
    def has_quality_model(self):
        return self._quality_model is not None

    def set_quality_threshold(self, v: float):
        self._quality_threshold = v

    def load_quality(self, pt_path: str):
        if pt_path == self._quality_pt:
            return
        print(f"[INFO] Loading quality model: {pt_path}")
        self._quality_model, self._quality_temp = load_quality_checkpoint(pt_path, self.device)
        self._quality_pt = pt_path
        self._report_load_device("quality", self._quality_model)

    def load_fill(self, pt_path: str):
        if pt_path == self._fill_pt:
            return
        print(f"[INFO] Loading fill/bag model: {pt_path}")
        (self._fill_model, self._fill_classes, self._bag_classes,
         self._n_bag_model, self._fill_temp, self._bag_temp) = load_fill_checkpoint(pt_path, self.device)
        self._fill_pt = pt_path
        self._report_load_device("fill/bag", self._fill_model)

    def _report_load_device(self, kind: str, model) -> torch.device:
        """Say where a freshly loaded model landed, and warn if it is not `self.device`.

        The load line above prints the checkpoint path only, which makes a CPU
        load indistinguishable from a CUDA one in the console — the detector
        prints its own device banner, these models printed nothing. `self.device`
        is only the request: a CPU-only torch build resolves it to "cpu" long
        before this point, and an OOM inside .to() or a module left behind by a
        failed offload restore can strand these weights somewhere else while the
        request still reads "cuda".

        Compared by device type, not identity: the request is a bare "cuda"
        string while the parameters report "cuda:0", and those agree.

        This is a post-load invariant check, NOT drift detection. Mid-run drift
        (a failed local-VLM offload restore) never reaches here — load_quality()
        and load_fill() short-circuit on an unchanged path — and is already
        reported by TrackingEngine._ensure_on_device().
        """
        want = torch.device(self.device)
        actual = self._weights_device(model, want)
        print(f"[INFO] {kind} model weights on {actual}")
        if actual.type != want.type:
            print(f"[WARN] {kind} model was requested on {want} but its weights "
                  f"are on {actual} — classification will run there")
        return actual

    @staticmethod
    def _weights_device(model, fallback):
        """Where `model`'s weights actually are, not where we asked them to be.

        `self.device` is a request recorded at construction. It can disagree
        with reality for the length of a run: TrackingEngine parks this stack on
        the CPU while a local VLM generates a case report, and a restore that
        OOMs partway through leaves a module split across both devices. Sending
        inputs to the stale answer is:

            RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
                          (torch.FloatTensor) should be the same

        Derived per model at each forward, deliberately: classify() runs the
        quality model and then the fill model, and a partially-failed restore is
        exactly the state in which those two disagree — so one answer for both
        would fix stage 1 and raise the same error one line later in stage 2.

        This is defence in depth, NOT the fix for that offload race — the stack
        can be moved mid-loop, between this query and the forward pass below.
        See docs/device_drift_fix_plan.md §9.4.4. It buys a slow correct run
        instead of a hard stop with no output.
        """
        if model is None:
            return fallback
        devs = {p.device for p in model.parameters()}
        if len(devs) != 1:
            # Split across devices — what TrackingEngine._param_device() reports
            # as "mixed". No single device is right for the input; CPU is the
            # one that cannot raise, and the engine's guard repairs the module
            # itself at the start of the next run.
            return torch.device("cpu")
        return devs.pop()

    def _crop_and_transform(self, frame_bgr, bbox):
        """Crop cart from frame and return transformed tensor, or None if too small."""
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return CLS_TRANSFORM(pil_img)

    @torch.no_grad()
    def classify_batch(self, frame_bgr, bboxes: list, cart_ids: list) -> dict:
        """Classify multiple cart crops in a single batched GPU pass.

        Args:
            frame_bgr: Full frame (BGR numpy array).
            bboxes: List of (x1, y1, x2, y2) for each cart.
            cart_ids: List of display IDs corresponding to each bbox.

        Returns:
            Dict mapping cart_id -> classification result dict.
        """
        if not bboxes or self._quality_model is None:
            fallback = _EMPTY_RESULT_NO_MODEL if self._quality_model is None else _EMPTY_RESULT_TINY
            return {cid: fallback for cid in cart_ids}

        # Prepare all crops on CPU
        tensors = []
        valid_indices = []
        for i, bbox in enumerate(bboxes):
            t = self._crop_and_transform(frame_bgr, bbox)
            if t is not None:
                tensors.append(t)
                valid_indices.append(i)

        results = {}
        for i in range(len(bboxes)):
            if i not in valid_indices:
                results[cart_ids[i]] = _EMPTY_RESULT_TINY

        if not tensors:
            return results

        # Single GPU transfer for entire batch, to wherever the quality model
        # actually is — see _weights_device().
        batch = torch.stack(tensors).to(
            self._weights_device(self._quality_model, self.device))

        # Stage 1: Quality (batched)
        logits_q = self._quality_model(batch)
        probs_q = torch.softmax(logits_q / self._quality_temp, dim=1)

        valid_mask = probs_q[:, 0] >= self._quality_threshold
        fill_indices = []

        for j in range(len(tensors)):
            cid = cart_ids[valid_indices[j]]
            if not valid_mask[j]:
                results[cid] = {
                    "is_valid": False, "quality": "unclear",
                    "fill": "non-applicable", "bag": "non-applicable",
                    "quality_conf": float(probs_q[j, 1]),
                    "fill_conf": 0.0, "bag_conf": 0.0,
                }
            else:
                fill_indices.append(j)

        # Stage 2: Fill + Bag (batched, only valid carts)
        if not fill_indices or self._fill_model is None:
            for j in fill_indices:
                cid = cart_ids[valid_indices[j]]
                results[cid] = {
                    "is_valid": True, "quality": "valid_cart",
                    "fill": "unclassified", "bag": "unclassified",
                    "quality_conf": float(probs_q[j, 0]),
                    "fill_conf": 0.0, "bag_conf": 0.0,
                }
            return results

        # .to() again: the fill model can be on a different device from the
        # quality model after a partially-failed restore, and a same-device
        # .to() is a no-op that returns the tensor itself.
        fill_batch = batch[fill_indices].to(
            self._weights_device(self._fill_model, self.device))
        logits_fill, logits_bag = self._fill_model(fill_batch)
        fill_probs = torch.softmax(logits_fill / self._fill_temp, dim=1)
        bag_probs = torch.softmax(logits_bag / self._bag_temp, dim=1)

        empty_idx = self._fill_classes.index("empty") if "empty" in self._fill_classes else 0

        for k, j in enumerate(fill_indices):
            cid = cart_ids[valid_indices[j]]
            fp = fill_probs[k]
            bp = bag_probs[k]
            fill_idx = int(fp.argmax())
            bag_idx = int(bp.argmax())

            if float(fp[empty_idx]) >= EMPTY_OVERRIDE_THRESH:
                fill_idx = empty_idx

            fill_lbl = self._fill_classes[fill_idx]

            if fill_idx == empty_idx:
                bag_lbl = "not_applicable"
                bag_conf = 1.0
            else:
                bag_lbl = self._bag_classes[bag_idx] if bag_idx < len(self._bag_classes) else "unknown"
                bag_conf = float(bp[bag_idx])

            results[cid] = {
                "is_valid": True, "quality": "valid_cart",
                "fill": fill_lbl, "bag": bag_lbl,
                "quality_conf": float(probs_q[j, 0]),
                "fill_conf": float(fp[fill_idx]),
                "bag_conf": bag_conf,
            }

        return results

    @torch.no_grad()
    def classify(self, frame_bgr, bbox) -> dict:
        """Classify a single cart crop. Returns result dict (original path)."""
        t = self._crop_and_transform(frame_bgr, bbox)
        if t is None:
            return _EMPTY_RESULT_TINY

        # Stage 1: Quality. The None check moved above the device transfer so
        # the transfer can ask the model where it is — see _weights_device().
        if self._quality_model is None:
            return _EMPTY_RESULT_NO_MODEL

        tensor = t.unsqueeze(0).to(
            self._weights_device(self._quality_model, self.device))

        logits_q = self._quality_model(tensor)
        probs_q  = torch.softmax(logits_q / self._quality_temp, dim=1)[0]
        valid_prob = float(probs_q[0])

        if valid_prob < self._quality_threshold:
            return {
                "is_valid": False, "quality": "unclear",
                "fill": "non-applicable", "bag": "non-applicable",
                "quality_conf": float(probs_q[1]),
                "fill_conf": 0.0, "bag_conf": 0.0,
            }

        # Stage 2: Fill + Bag
        if self._fill_model is None:
            return {
                "is_valid": True, "quality": "valid_cart",
                "fill": "unclassified", "bag": "unclassified",
                "quality_conf": valid_prob,
                "fill_conf": 0.0, "bag_conf": 0.0,
            }

        logits_fill, logits_bag = self._fill_model(
            tensor.to(self._weights_device(self._fill_model, self.device)))
        fill_probs = torch.softmax(logits_fill / self._fill_temp, dim=1)[0]
        bag_probs  = torch.softmax(logits_bag  / self._bag_temp,  dim=1)[0]
        fill_idx   = int(fill_probs.argmax())
        bag_idx    = int(bag_probs.argmax())

        empty_idx = self._fill_classes.index("empty") if "empty" in self._fill_classes else 0
        if float(fill_probs[empty_idx]) >= EMPTY_OVERRIDE_THRESH:
            fill_idx = empty_idx

        fill_lbl = self._fill_classes[fill_idx]

        if fill_idx == empty_idx:
            bag_lbl  = "not_applicable"
            bag_conf = 1.0
        else:
            bag_lbl  = self._bag_classes[bag_idx] if bag_idx < len(self._bag_classes) else "unknown"
            bag_conf = float(bag_probs[bag_idx])

        return {
            "is_valid": True, "quality": "valid_cart",
            "fill": fill_lbl, "bag": bag_lbl,
            "quality_conf": valid_prob,
            "fill_conf": float(fill_probs[fill_idx]),
            "bag_conf": bag_conf,
        }
