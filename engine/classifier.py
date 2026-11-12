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
from .models import load_quality_checkpoint, load_fill_checkpoint, load_multihead_checkpoint

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
        "_multi_model", "_multi_pt", "_multi_heads", "_multi_pad",
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
        # v2 multi-head model (one pass for quality / fill / bag / child / stack).
        # When loaded it replaces both legacy models; see load_multihead().
        self._multi_model = None
        self._multi_pt = None
        self._multi_heads = None
        self._multi_pad = None

    @property
    def quality_pt(self):
        return self._multi_pt if self._multi_model is not None else self._quality_pt

    @property
    def fill_pt(self):
        return self._multi_pt if self._multi_model is not None else self._fill_pt

    @property
    def has_quality_model(self):
        return self._quality_model is not None or self._multi_model is not None

    def models(self):
        """(name, module) for every loaded classifier model, for device moves."""
        return [(n, m) for n, m in (("quality", self._quality_model), ("fill", self._fill_model),
                                    ("multihead", self._multi_model)) if m is not None]

    def set_quality_threshold(self, v: float):
        self._quality_threshold = v

    def load_quality(self, pt_path: str):
        self._drop_multihead()
        if pt_path == self._quality_pt:
            return
        print(f"[INFO] Loading quality model: {pt_path}")
        self._quality_model, self._quality_temp = load_quality_checkpoint(pt_path, self.device)
        self._quality_pt = pt_path
        self._report_load_device("quality", self._quality_model)

    def load_fill(self, pt_path: str):
        self._drop_multihead()
        if pt_path == self._fill_pt:
            return
        print(f"[INFO] Loading fill/bag model: {pt_path}")
        (self._fill_model, self._fill_classes, self._bag_classes,
         self._n_bag_model, self._fill_temp, self._bag_temp) = load_fill_checkpoint(pt_path, self.device)
        self._fill_pt = pt_path
        self._report_load_device("fill/bag", self._fill_model)

    def load_multihead(self, pt_path: str):
        """Load the v2 multi-head classifier. It replaces both legacy models.

        The legacy models are dropped (not just ignored) so the device-move code
        in the tracker, which walks models(), never carries a stale pair.
        """
        if pt_path == self._multi_pt and self._multi_model is not None:
            return
        print(f"[INFO] Loading multi-head cart classifier: {pt_path}")
        self._multi_model, self._multi_heads, self._multi_pad = load_multihead_checkpoint(pt_path, self.device)
        self._multi_pt = pt_path
        self._quality_model = self._fill_model = None
        self._quality_pt = self._fill_pt = None
        self._report_load_device("multi-head", self._multi_model)

    def _drop_multihead(self):
        self._multi_model = self._multi_heads = self._multi_pad = None
        self._multi_pt = None

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

    def _crop_and_transform(self, frame_bgr, bbox, pad=None):
        """Crop cart from frame and return transformed tensor, or None if too small.

        pad = (top, sides, bottom) fractions of the box (the multi-head model's
        training crop). The size check runs on the clipped box before padding,
        and the padding arithmetic matches cart_attr_labels p1_extract.pad_box
        exactly: int() truncation of each margin, then clip to the frame.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame_bgr.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        if pad is not None:
            t, s, bo = pad
            bw, bh = x2 - x1, y2 - y1
            x1, y1, x2, y2 = (max(0, x1 - int(s * bw)), max(0, y1 - int(t * bh)),
                              min(w, x2 + int(s * bw)), min(h, y2 + int(bo * bh)))
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
        if self._multi_model is not None:
            return self._classify_batch_multi(frame_bgr, bboxes, cart_ids)
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

    # ------------------------------------------------------------------
    # v2 multi-head path. Same result dict as the legacy path (same keys, same
    # label strings, same EMPTY_OVERRIDE_THRESH and classify_loaded rules), plus
    # four extra keys: child_in_cart / child_conf and stacked / stack_conf.
    # The extras are informational only; nothing downstream decides on them.
    # Softmax at T = 1.0: this model has no temperature calibration yet.
    # ------------------------------------------------------------------
    _FILL_NAMES = {"E": "empty", "P": "partial", "F": "full"}
    _BAG_NAMES = {"b": "bagged", "u": "unbagged"}

    def _multi_forward(self, batch):
        out = self._multi_model(batch.to(self._weights_device(self._multi_model, self.device)))
        return {h: torch.softmax(v.float(), dim=1) for h, v in out.items()}

    def _multi_extras(self, probs, k) -> dict:
        extra = {}
        H = self._multi_heads
        if "child" in probs and "y" in H["child"]:
            p = float(probs["child"][k, H["child"].index("y")])
            extra.update(child_in_cart=p >= 0.5, child_conf=p)
        if "stack" in probs and "s" in H["stack"]:
            p = float(probs["stack"][k, H["stack"].index("s")])
            extra.update(stacked=p >= 0.5, stack_conf=p)
        return extra

    def _multi_result(self, probs, k, allow_empty: bool) -> dict:
        H = self._multi_heads
        pq = probs["q"][k]
        valid_prob = float(pq[H["q"].index("V")])
        extra = self._multi_extras(probs, k)
        if valid_prob < self._quality_threshold:
            return {
                "is_valid": False, "quality": "unclear",
                "fill": "non-applicable", "bag": "non-applicable",
                "quality_conf": 1.0 - valid_prob,
                "fill_conf": 0.0, "bag_conf": 0.0, **extra,
            }
        fill_classes = [self._FILL_NAMES[c] for c in H["fill"]]
        fp = probs["fill"][k]
        empty_idx = fill_classes.index("empty")
        if allow_empty:
            fill_idx = int(fp.argmax())
            if float(fp[empty_idx]) >= EMPTY_OVERRIDE_THRESH:
                fill_idx = empty_idx
            fill_conf = float(fp[fill_idx])
        else:
            loaded = [i for i, c in enumerate(fill_classes) if c != "empty"]
            sub = fp[loaded]
            j = int(sub.argmax())
            fill_idx = loaded[j]
            fill_conf = float(sub[j]) / max(float(sub.sum()), 1e-12)
        if fill_idx == empty_idx:
            bag_lbl, bag_conf = "not_applicable", 1.0
        else:
            bp = probs["bag"][k]
            bag_idx = int(bp.argmax())
            bag_lbl, bag_conf = self._BAG_NAMES[H["bag"][bag_idx]], float(bp[bag_idx])
        return {
            "is_valid": True, "quality": "valid_cart",
            "fill": fill_classes[fill_idx], "bag": bag_lbl,
            "quality_conf": valid_prob,
            "fill_conf": fill_conf, "bag_conf": bag_conf, **extra,
        }

    def _classify_batch_multi(self, frame_bgr, bboxes, cart_ids) -> dict:
        results, tensors, keep = {}, [], []
        for i, bbox in enumerate(bboxes):
            t = self._crop_and_transform(frame_bgr, bbox, self._multi_pad)
            if t is None:
                results[cart_ids[i]] = _EMPTY_RESULT_TINY
            else:
                tensors.append(t)
                keep.append(i)
        if tensors:
            probs = self._multi_forward(torch.stack(tensors))
            for k, i in enumerate(keep):
                results[cart_ids[i]] = self._multi_result(probs, k, allow_empty=True)
        return results

    def classify(self, frame_bgr, bbox) -> dict:
        """Classify a single cart crop. Returns result dict (original path)."""
        return self._classify_one(frame_bgr, bbox, allow_empty=True)

    def classify_loaded(self, frame_bgr, bbox) -> dict:
        """Classify a crop the DETECTOR already called non-empty.

        Same quality gate and bag model as classify(), but the fill answer is
        restricted to the loaded classes: their probabilities are renormalised
        over just those classes and the argmax taken, and EMPTY_OVERRIDE_THRESH
        does not apply. `fill_conf` is the renormalised probability. A crop
        that fails the quality gate is returned exactly as classify() would
        return it (the caller decides what an unreadable read means).
        """
        return self._classify_one(frame_bgr, bbox, allow_empty=False)

    # no_grad HERE, on the shared body, not on the public entry points: when it
    # sat on classify() alone, classify_loaded() ran every read with autograd
    # on (a graph per crop, and torch's requires_grad scalar warning).
    @torch.no_grad()
    def _classify_one(self, frame_bgr, bbox, allow_empty: bool) -> dict:
        if self._multi_model is not None:
            t = self._crop_and_transform(frame_bgr, bbox, self._multi_pad)
            if t is None:
                return _EMPTY_RESULT_TINY
            probs = self._multi_forward(torch.stack([t]))
            return self._multi_result(probs, 0, allow_empty)
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
        bag_idx    = int(bag_probs.argmax())

        if allow_empty:
            fill_idx = int(fill_probs.argmax())
            empty_idx = self._fill_classes.index("empty") if "empty" in self._fill_classes else 0
            if float(fill_probs[empty_idx]) >= EMPTY_OVERRIDE_THRESH:
                fill_idx = empty_idx
            fill_conf = float(fill_probs[fill_idx])
            is_empty = fill_idx == empty_idx
        else:
            loaded = [i for i, c in enumerate(self._fill_classes) if c != "empty"]
            sub = fill_probs[loaded] if loaded else None
            total = float(sub.sum()) if sub is not None else 0.0
            if not loaded or not total > 0.0:
                # A fill model with no loaded class (or a degenerate softmax)
                # cannot answer "how loaded"; say so rather than guess.
                return {
                    "is_valid": True, "quality": "valid_cart",
                    "fill": "unclassified", "bag": "unclassified",
                    "quality_conf": valid_prob,
                    "fill_conf": 0.0, "bag_conf": 0.0,
                }
            k = int(sub.argmax())
            fill_idx = loaded[k]
            fill_conf = float(sub[k]) / total
            is_empty = False

        fill_lbl = self._fill_classes[fill_idx]

        if is_empty:
            bag_lbl  = "not_applicable"
            bag_conf = 1.0
        else:
            bag_lbl  = self._bag_classes[bag_idx] if bag_idx < len(self._bag_classes) else "unknown"
            bag_conf = float(bag_probs[bag_idx])

        return {
            "is_valid": True, "quality": "valid_cart",
            "fill": fill_lbl, "bag": bag_lbl,
            "quality_conf": valid_prob,
            "fill_conf": fill_conf,
            "bag_conf": bag_conf,
        }
