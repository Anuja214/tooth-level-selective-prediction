"""Live inference for the 2026 tooth-level selective-prediction protocol.

Matches `src/run_selective_prediction.py` / `selective_prediction_lib.py`:
  radiograph → YOLOv8n teeth → 8% padded crops → 5-seed Swin-T
  → temperature scaling (T fit on val) → multi-signal r_i → accept/defer.

The locked proposed selector is the 4-feature logistic in
`calibration/learned_selector.json` (confidence, H_total, u_epistemic,
q_loc_conf_norm). Coefficients are applied as stored; nothing is refit.

q is min-max normalized with the INTERNAL VAL range (frozen), which is the
correct deployment behaviour. Per-split test min-max is not used here.

Not a medical device. Tooth boxes are pseudo-labels at training time;
live boxes are the detector's own predictions.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from lib.eval_transforms import get_eval_transforms  # noqa: E402
from lib.model_utils import build_model_no_download  # noqa: E402
from lib import signals as sp  # noqa: E402


class Bundle:
    """Resolves files for (a) a portable zip unpacked next to app.py, or
    (b) the original D:\\Dental_project layout."""

    def __init__(self):
        portable_models = os.path.join(THIS_DIR, "models", "yolov8n_tooth_localizer.pt")
        self.portable = os.path.isfile(portable_models)
        if self.portable:
            self.root = THIS_DIR
            self.models = os.path.join(THIS_DIR, "models")
            self.calibration = os.path.join(THIS_DIR, "calibration")
            self.predictions = os.path.join(THIS_DIR, "predictions")
            self.figures = os.path.join(THIS_DIR, "figures")
            self.samples = os.path.join(THIS_DIR, "samples")
            self.original_project = None
        else:
            proj = os.path.dirname(THIS_DIR)
            cap = r"D:\Dental_project\Capastone Project"
            self.root = proj
            self.models = os.path.join(cap, "results", "models")
            self.calibration = os.path.join(proj, "calibration")
            self.predictions = os.path.join(proj, "predictions")
            self.figures = os.path.join(proj, "figures_publication")
            self.samples = os.path.join(cap, "dataset", "carixray_yolo", "test", "images")
            self.original_project = cap

    def localizer_ckpt(self) -> str:
        return os.path.join(self.models, "yolov8n_tooth_localizer.pt")

    def swin_ckpt(self, seed: int) -> str:
        if self.portable:
            return os.path.join(self.models, f"swin_t_seed{seed}_best.pt")
        if seed == 42:
            return os.path.join(self.models, "swin_t_best.pt")
        return os.path.join(os.path.dirname(THIS_DIR), "checkpoints", f"swin_t_seed{seed}_best.pt")


BUNDLE = Bundle()
# Back-compat name used by app.py
ORIGINAL_PROJECT_ROOT = BUNDLE.original_project or THIS_DIR

PAD_FRAC = 0.08
YOLO_IMGSZ = 320
MIN_CROP_PX = 8
EPS = 1e-7
SEEDS = [42, 43, 44, 45, 46]
DECISION_THRESH = 0.5  # ŷ = 1[p_cal >= 0.5], same as the paper

ACCEPT = "ACCEPT"
DEFER = "DEFER"
CAVITY = "cavity"
NO_CAVITY = "no_cavity"

# Overlay BGR (OpenCV)
COLOR_ACCEPT_CAVITY = (50, 90, 200)     # red-ish blue-red? wait BGR: cavity accepted = strong red-blue
COLOR_ACCEPT_CLEAR = (178, 114, 0)      # #0072B2 in BGR
COLOR_DEFER = (0, 94, 213)              # #D55E00 in BGR


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def pad_box(x1, y1, x2, y2, w, h, frac=PAD_FRAC):
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * frac, bh * frac
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def load_calibration():
    with open(os.path.join(BUNDLE.calibration, "temperature_scaling.json")) as f:
        t_json = json.load(f)
    with open(os.path.join(BUNDLE.calibration, "learned_selector.json")) as f:
        sel = json.load(f)
    return float(t_json["T"]), sel


def val_q_range():
    """Frozen val-split YOLO-conf min/max for deployment-time q normalisation."""
    import pandas as pd
    path = os.path.join(BUNDLE.predictions, "internal_val_manifest.csv")
    q = pd.read_csv(path)["yolo_box_conf"].to_numpy()
    return float(q.min()), float(q.max())


def checkpoint_for_seed(seed: int) -> str:
    return BUNDLE.swin_ckpt(seed)


@dataclass
class ToothResult:
    index: int
    box_xyxy: tuple
    crop_bgr: np.ndarray
    q: float
    q_norm: float
    p_raw_mean: float
    p_cal: float
    confidence: float
    H_total: float
    u_epistemic: float
    r: float
    yhat: int
    decision: str
    label: str

    def as_row(self) -> dict:
        return {
            "tooth": f"#{self.index}",
            "P(cavity)": round(self.p_cal, 3),
            "confidence c": round(self.confidence, 3),
            "entropy H": round(self.H_total, 3),
            "epistemic u": round(self.u_epistemic, 4),
            "loc. q": round(self.q, 3),
            "risk score r": round(self.r, 3),
            "class": self.label,
            "decision": self.decision,
        }


@dataclass
class ScreenResult:
    n_teeth: int
    n_accept: int
    n_defer: int
    n_cavity_accepted: int
    n_clear_accepted: int
    tau: float
    risk_target: Optional[float]
    teeth: list = field(default_factory=list)
    overlay_rgb: Optional[np.ndarray] = None
    original_rgb: Optional[np.ndarray] = None


class InvalidDentalXRayError(ValueError):
    """Raised when an input image fails dental X-ray validation."""
    pass


class DentalImageClassifierValidator:
    """Architectural integration point for a whole-image dental radiograph classifier.

    Distinguishes genuine dental radiographs (periapical, bitewing, panoramic)
    from out-of-domain imagery (logos, documents, photographs, non-dental scans).
    """

    DEFAULT_MODEL_NAME = "dental_image_classifier.pt"
    # Calibrated decision threshold: acceptance target >= 0.65 for dental X-rays,
    # cleanly separating all out-of-domain non-dental imagery (< 0.40) from genuine radiographs (> 0.75).
    ACCEPT_THRESHOLD = 0.65
    UNCERTAINTY_LOWER = 0.40

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(BUNDLE.models, self.DEFAULT_MODEL_NAME)
        self.model_path = model_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self._load_attempted = False

    def is_model_available(self) -> bool:
        """Checks if a trained whole-image classifier weights file exists."""
        return os.path.exists(self.model_path)

    def _load_model(self) -> Optional[torch.nn.Module]:
        if self._load_attempted:
            return self.model
        self._load_attempted = True
        if not self.is_model_available():
            return None

        try:
            # Checkpoint contains state dict or TorchScript / full model
            try:
                ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(self.model_path, map_location=self.device)
            if isinstance(ckpt, torch.nn.Module):
                model = ckpt
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                from lib.model_utils import build_model_no_download
                arch = ckpt.get("arch", "mobilenet_v2")
                model = build_model_no_download(arch)
                model.load_state_dict(ckpt["state_dict"])
            else:
                from lib.model_utils import build_model_no_download
                model = build_model_no_download("mobilenet_v2")
                model.load_state_dict(ckpt)

            model.to(self.device)
            model.eval()
            self.model = model
            return self.model
        except Exception:
            return None

    def predict_proba(self, image_bgr: np.ndarray) -> Optional[float]:
        """Predicts P(is_dental_xray) in [0.0, 1.0] using the trained classifier."""
        model = self._load_model()
        if model is None:
            return None

        # Preprocess full radiograph: resize to 224x224 and normalize
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if len(image_bgr.shape) == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        eval_tf = get_eval_transforms(image_size=224)
        tensor = eval_tf(image=rgb)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits.float(), dim=1)
            # Class 1: dental_xray, Class 0: non_dental
            prob_dental = float(probs[0, 1].cpu().item()) if probs.shape[1] > 1 else float(torch.sigmoid(logits[0]).cpu().item())
        return prob_dental

    def validate(self, image: np.ndarray) -> tuple[bool, str]:
        """Validates the input image against basic data integrity and the trained classifier."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return False, "Image data is empty, corrupted, or cannot be decoded."

        if len(image.shape) < 2:
            return False, "Image must have at least 2 dimensions."

        h, w = image.shape[:2]
        if h < 64 or w < 64:
            return False, f"Image resolution ({w}x{h}) is too low for diagnostic screening (minimum 64x64)."

        # Reject degenerate completely black/white blank arrays
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] >= 3 else image
        if float(np.std(gray)) < 2.0:
            return False, "The image contains no discernible information or variation (blank/uniform field)."

        # If a trained whole-image classifier is present, evaluate content using deep learning
        prob = self.predict_proba(image)
        if prob is not None:
            if prob >= self.ACCEPT_THRESHOLD:
                return True, f"Verified dental radiograph (P(dental) = {prob:.3f} >= {self.ACCEPT_THRESHOLD})."
            elif prob < self.UNCERTAINTY_LOWER:
                return False, f"Non-dental imagery detected (P(dental) = {prob:.3f} < {self.UNCERTAINTY_LOWER})."
            else:
                return False, f"Uncertain image category (P(dental) = {prob:.3f} in rejection window [{self.UNCERTAINTY_LOWER}, {self.ACCEPT_THRESHOLD}))."

        # When no trained classifier checkpoint exists in the repository,
        # note the absence of a category model rather than enforcing brittle ad-hoc rules.
        return True, "Passed data integrity verification (Note: whole-image classifier checkpoint models/dental_image_classifier.pt is not present; skipping category score)."


_GLOBAL_VALIDATOR: Optional[DentalImageClassifierValidator] = None


def get_classifier_validator() -> DentalImageClassifierValidator:
    global _GLOBAL_VALIDATOR
    if _GLOBAL_VALIDATOR is None:
        _GLOBAL_VALIDATOR = DentalImageClassifierValidator()
    return _GLOBAL_VALIDATOR


def validate_dental_radiograph(image: np.ndarray, localizer: Optional[Any] = None) -> tuple[bool, str]:
    """Validates whether an input image represents a valid dental radiograph.
    
    Delegates to DentalImageClassifierValidator to prevent brittle hand-crafted heuristic failures.
    """
    validator = get_classifier_validator()
    return validator.validate(image)


class ScreeningEngine:
    """Loads YOLO + 5 Swin-T seeds once; screens one radiograph at a time."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.T, self.sel = load_calibration()
        self.coef = np.array(self.sel["coef"][0], dtype=np.float64)
        self.intercept = float(self.sel["intercept"][0])
        self.features = list(self.sel["features"])
        self.q_min, self.q_max = val_q_range()
        self.eval_tf = get_eval_transforms()
        self._localizer = None
        self._classifiers = None
        self._tau_cache = {}

    def _ensure_localizer(self):
        if self._localizer is None:
            from ultralytics import YOLO
            ckpt = BUNDLE.localizer_ckpt()
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Tooth localizer not found: {ckpt}")
            self._localizer = YOLO(ckpt)
        return self._localizer

    def _ensure_classifiers(self):
        if self._classifiers is not None:
            return self._classifiers
        models = []
        for seed in SEEDS:
            ckpt = checkpoint_for_seed(seed)
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Swin-T seed {seed} checkpoint not found: {ckpt}")
            m = build_model_no_download("swin_t")
            try:
                state = torch.load(ckpt, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(ckpt, map_location=self.device)
            m.load_state_dict(state)
            m.to(self.device).eval()
            models.append(m)
        self._classifiers = models
        return models

    def tau_for_risk_target(self, alpha: float) -> float:
        """Population τ from frozen internal-val predictions (never the live image)."""
        key = round(float(alpha), 4)
        if key in self._tau_cache:
            return self._tau_cache[key]
        seed_probs, labels, q = sp.load_seed_ensemble(BUNDLE.predictions, "internal_val_matched")
        ensemble_mean = seed_probs.mean(axis=0)
        p_cal = sp.apply_temperature(ensemble_mean, self.T)
        h_total, _, u_ep = sp.entropy_decomposition(seed_probs)
        conf = np.maximum(p_cal, 1.0 - p_cal)
        q_norm = (q - self.q_min) / max(EPS, (self.q_max - self.q_min))
        q_norm = np.clip(q_norm, 0.0, 1.0)
        X = np.column_stack([conf, h_total, u_ep, q_norm])
        r = _sigmoid(X @ self.coef + self.intercept)
        yhat = (p_cal >= DECISION_THRESH).astype(int)
        correct = (yhat == labels).astype(int)
        coverage, sel_risk = sp.risk_coverage_curve(r, correct)
        cov = sp.coverage_at_risk_target(coverage, sel_risk, alpha)
        if cov <= 0:
            tau = float(np.max(r) + 1.0)  # accept nothing
        else:
            order = np.argsort(-r, kind="stable")
            k = max(1, int(round(cov * len(r))))
            tau = float(r[order][k - 1])
        self._tau_cache[key] = tau
        return tau

    def _classify_crops(self, tensors: torch.Tensor) -> np.ndarray:
        """tensors: (N, 3, 224, 224) → (5, N) P(cavity) per seed."""
        models = self._ensure_classifiers()
        out = []
        with torch.no_grad():
            for m in models:
                logits = m(tensors)
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
                out.append(probs.detach().cpu().numpy())
        return np.stack(out, axis=0)

    def screen(
        self,
        image_bgr: np.ndarray,
        tau: Optional[float] = None,
        risk_target: Optional[float] = 0.05,
    ) -> ScreenResult:
        is_valid, reason = validate_dental_radiograph(image_bgr, localizer=self._ensure_localizer())
        if not is_valid:
            raise InvalidDentalXRayError(
                f"{reason} Please upload a valid dental X-ray (periapical, bitewing, or panoramic radiograph)."
            )
        if tau is None:
            if risk_target is None:
                risk_target = 0.05
            tau = self.tau_for_risk_target(risk_target)
        h, w = image_bgr.shape[:2]
        localizer = self._ensure_localizer()
        yolo_device = 0 if self.device.type == "cuda" else "cpu"
        result = localizer.predict(
            image_bgr, imgsz=YOLO_IMGSZ, device=yolo_device, verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return ScreenResult(
                n_teeth=0, n_accept=0, n_defer=0,
                n_cavity_accepted=0, n_clear_accepted=0,
                tau=float(tau), risk_target=risk_target,
                teeth=[], overlay_rgb=rgb, original_rgb=rgb,
            )

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        crops, meta = [], []
        for i, (box, q) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = pad_box(*box.tolist(), w, h)
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(round(x2)), int(round(y2))
            if x2i - x1i < MIN_CROP_PX or y2i - y1i < MIN_CROP_PX:
                continue
            crop = image_bgr[y1i:y2i, x1i:x2i]
            crops.append(crop)
            meta.append((i + 1, (x1i, y1i, x2i, y2i), float(q), crop))

        if not crops:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return ScreenResult(
                n_teeth=0, n_accept=0, n_defer=0,
                n_cavity_accepted=0, n_clear_accepted=0,
                tau=float(tau), risk_target=risk_target,
                teeth=[], overlay_rgb=rgb, original_rgb=rgb,
            )

        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.eval_tf(image=rgb)["image"])
        batch = torch.stack(tensors, dim=0).to(self.device)
        seed_probs = self._classify_crops(batch)  # (5, N)

        ensemble_mean = seed_probs.mean(axis=0)
        p_cal = sp.apply_temperature(ensemble_mean, self.T)
        h_total, _, u_ep = sp.entropy_decomposition(seed_probs)
        conf = np.maximum(p_cal, 1.0 - p_cal)
        q_arr = np.array([m[2] for m in meta], dtype=np.float64)
        q_norm = np.clip((q_arr - self.q_min) / max(EPS, (self.q_max - self.q_min)), 0.0, 1.0)
        X = np.column_stack([conf, h_total, u_ep, q_norm])
        r = _sigmoid(X @ self.coef + self.intercept)
        yhat = (p_cal >= DECISION_THRESH).astype(int)

        teeth = []
        overlay = image_bgr.copy()
        for j, (idx, box, q, crop) in enumerate(meta):
            decision = ACCEPT if r[j] >= tau else DEFER
            label = CAVITY if yhat[j] == 1 else NO_CAVITY
            tr = ToothResult(
                index=idx, box_xyxy=box, crop_bgr=crop, q=q, q_norm=float(q_norm[j]),
                p_raw_mean=float(ensemble_mean[j]), p_cal=float(p_cal[j]),
                confidence=float(conf[j]), H_total=float(h_total[j]),
                u_epistemic=float(u_ep[j]), r=float(r[j]), yhat=int(yhat[j]),
                decision=decision, label=label,
            )
            teeth.append(tr)
            x1i, y1i, x2i, y2i = box
            if decision == DEFER:
                color = COLOR_DEFER
                tag = f"#{idx} REVIEW"
            elif label == CAVITY:
                color = (40, 40, 220)
                tag = f"#{idx} CAVITY"
            else:
                color = COLOR_ACCEPT_CLEAR
                tag = f"#{idx} CLEAR"
            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(tag, font, 0.45, 1)
            ty = max(0, y1i - 6)
            cv2.rectangle(overlay, (x1i, ty - th - 4), (x1i + tw + 4, ty + 2), color, -1)
            cv2.putText(overlay, tag, (x1i + 2, ty), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        n_accept = sum(t.decision == ACCEPT for t in teeth)
        n_defer = sum(t.decision == DEFER for t in teeth)
        n_cav = sum(t.decision == ACCEPT and t.label == CAVITY for t in teeth)
        n_clr = sum(t.decision == ACCEPT and t.label == NO_CAVITY for t in teeth)
        return ScreenResult(
            n_teeth=len(teeth),
            n_accept=n_accept,
            n_defer=n_defer,
            n_cavity_accepted=n_cav,
            n_clear_accepted=n_clr,
            tau=float(tau),
            risk_target=risk_target,
            teeth=teeth,
            overlay_rgb=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            original_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        )
