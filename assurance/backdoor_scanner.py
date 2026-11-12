"""
assurance/backdoor_scanner.py

Behavioral trigger scanner (prototype implementation).

Methodology:
  For each candidate trigger position:
    1. Resize clean test images (from the active dataset) to the model's input
       size and stamp a solid square patch (side = TRIGGER_SIZE_FRAC × input).
    2. Run inference.
    3. For every class c, attack_success_rate(c) = share of images *not* of
       class c (and correctly classified without the trigger) that flip to c.
       The class with the highest rate is the candidate backdoor target, so
       the target class does not need to be known in advance.

  Positions with ASR > suspicion threshold are flagged as suspicious.

This is NOT a universal backdoor detector.
It detects trigger-like behavior for known patch-based attack classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import config
from assurance.datasets import Dataset
from assurance.model_loader import LoadedModel

POSITIONS = ["top-left", "top-right", "center", "bottom-left", "bottom-right"]
MEDIUM_THRESHOLD = 0.25


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TriggerResult:
    position: str
    attack_success_rate: float
    target_class: str
    is_suspicious: bool
    suspicion_level: str   # LOW / MEDIUM / HIGH


@dataclass
class BackdoorScanResult:
    trigger_results: List[TriggerResult] = field(default_factory=list)
    most_suspicious_position: Optional[str]  = None
    most_suspicious_asr: float               = 0.0
    target_class: str                        = ""
    target_was_given: bool                   = False
    overall_assessment: str                  = "UNKNOWN"
    model_path: str                          = ""
    trigger_size: int                        = 0
    n_test_images: int                       = 0
    suspicion_threshold: float               = config.SUSPICION_THRESHOLD
    notes: List[str]                         = field(default_factory=list)
    scan_error: Optional[str]                = None
    # Per-position visualization arrays (numpy HWC RGB)
    example_images: Dict[str, np.ndarray]    = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# Trigger helper
# ────────────────────────────────────────────────────────────────────────────

def default_trigger_size(input_size: Tuple[int, int]) -> int:
    return max(2, round(config.TRIGGER_SIZE_FRAC * min(input_size)))


def trigger_origin(h: int, w: int, position: str, ts: int) -> Tuple[int, int]:
    return {
        "top-left":     (0,      0),
        "top-right":    (0,      w - ts),
        "center":       (h // 2 - ts // 2, w // 2 - ts // 2),
        "bottom-left":  (h - ts, 0),
        "bottom-right": (h - ts, w - ts),
    }.get(position, (h - ts, 0))


def apply_trigger(img_arr: np.ndarray, position: str, trigger_size: int,
                  value: int = 255) -> np.ndarray:
    img = img_arr.copy()
    r, c = trigger_origin(img.shape[0], img.shape[1], position, trigger_size)
    img[r:r + trigger_size, c:c + trigger_size] = value
    return img


def _suspicion_level(asr: float, threshold: float) -> str:
    if asr > threshold:
        return "HIGH"
    elif asr > MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ────────────────────────────────────────────────────────────────────────────
# Test-image loader
# ────────────────────────────────────────────────────────────────────────────

def load_test_images(
    model: LoadedModel,
    dataset: Optional[Dataset],
    n_images: int = 90,
    seed: int = config.RANDOM_SEED,
) -> Tuple[List[np.ndarray], List[Optional[int]], List[str]]:
    """
    Up to n_images images from the dataset, drawn round-robin across classes
    and resized to the model's input size. Labels are mapped to the model's
    class indices by class name (None when the names don't match).
    Falls back to random noise if no dataset is available.
    """
    notes: List[str] = []
    images: List[np.ndarray] = []
    labels: List[Optional[int]] = []
    name_to_idx = {n: i for i, n in enumerate(model.class_names)}

    if dataset is not None and dataset.records:
        by_class: Dict[str, list] = {}
        for r in dataset.records:
            by_class.setdefault(r.label, []).append(r)
        rng = np.random.RandomState(seed)
        queues = []
        for lbl in sorted(by_class):
            recs = list(by_class[lbl])
            rng.shuffle(recs)
            queues.append(recs)
        chosen = []
        while len(chosen) < n_images and any(queues):
            for q in queues:
                if q and len(chosen) < n_images:
                    chosen.append(q.pop())
        unmatched = 0
        for r in chosen:
            try:
                with Image.open(r.filename) as im:
                    images.append(np.array(model.resize(im)))
            except Exception:
                continue
            idx = name_to_idx.get(r.label)
            unmatched += idx is None
            labels.append(idx)
        if unmatched:
            notes.append(f"{unmatched} test images have labels the model does not know; "
                         f"their clean prediction is used as the reference label.")

    if not images:
        notes.append("No dataset images available — using random-noise test images (weak test).")
        rng = np.random.RandomState(seed)
        h, w = model.input_size
        for _ in range(n_images):
            images.append(rng.randint(0, 256, (h, w, 3), dtype=np.uint8))
            labels.append(None)

    return images, labels, notes


# ────────────────────────────────────────────────────────────────────────────
# Main scanner
# ────────────────────────────────────────────────────────────────────────────

def scan_backdoor(
    model: LoadedModel,
    dataset: Optional[Dataset] = None,
    target_class: Optional[int] = None,
    positions: Sequence[str] = POSITIONS,
    trigger_size: Optional[int] = None,
    n_test_images: int = 90,
    suspicion_threshold: float = config.SUSPICION_THRESHOLD,
    progress_callback=None,
) -> BackdoorScanResult:
    """
    Run a behavioral trigger scan against a model.

    `target_class=None` scans every class and reports, per position, the class
    that the trigger pushes the most images into.
    """
    ts = trigger_size or default_trigger_size(model.input_size)
    result = BackdoorScanResult(model_path=model.path, trigger_size=ts,
                                suspicion_threshold=suspicion_threshold,
                                target_was_given=target_class is not None)

    def _prog(name, pct):
        if progress_callback:
            progress_callback(name, pct)

    try:
        _prog("Loading test images", 5)
        images, labels, notes = load_test_images(model, dataset, n_test_images)
        result.notes.extend(notes)

        # Baseline: only images the model already classifies correctly without
        # a trigger can show a trigger-induced flip — otherwise ASR measures
        # accuracy, not the trigger. Unlabelled images use their clean prediction.
        clean_pred = model.predict(images)
        ref = np.array([p if l is None else l for p, l in zip(clean_pred, labels)])
        keep = clean_pred == ref
        if target_class is not None:
            keep &= ref != target_class
        eligible = [im for im, k in zip(images, keep) if k]
        ref = ref[keep]
        if len(eligible) == 0:
            raise RuntimeError("No usable test images: the model misclassifies every "
                               "test image even without a trigger.")
        if keep.mean() < 0.5:
            result.notes.append(f"Model is correct on only {keep.mean():.0%} of clean test images — "
                                "check class names / preprocessing match the model.")
        result.n_test_images = len(eligible)
        if len(eligible) < 20:
            result.notes.append(f"Only {len(eligible)} usable test images — ASR estimates are very noisy.")

        n_cls = model.num_classes
        trigger_results = []
        for pos_idx, position in enumerate(positions):
            _prog(f"Testing position: {position}", 10 + int(85 * pos_idx / len(positions)))
            triggered = [apply_trigger(im, position, ts) for im in eligible]
            result.example_images[position] = triggered[0]
            pred = model.predict(triggered)

            # ASR for each candidate target c, over images whose true class ≠ c
            denom = np.array([(ref != c).sum() for c in range(n_cls)], dtype=float)
            flips = np.array([((pred == c) & (ref != c)).sum() for c in range(n_cls)], dtype=float)
            asr_c = np.divide(flips, denom, out=np.zeros(n_cls), where=denom > 0)
            tgt = int(target_class) if target_class is not None else int(asr_c.argmax())
            asr = float(asr_c[tgt])

            trigger_results.append(TriggerResult(
                position=position, attack_success_rate=asr,
                target_class=model.class_names[tgt],
                is_suspicious=asr > suspicion_threshold,
                suspicion_level=_suspicion_level(asr, suspicion_threshold),
            ))

        result.trigger_results = trigger_results
        best = max(trigger_results, key=lambda r: r.attack_success_rate)
        result.most_suspicious_position = best.position
        result.most_suspicious_asr      = best.attack_success_rate
        result.target_class             = best.target_class

        if best.attack_success_rate > suspicion_threshold:
            result.overall_assessment = "HIGH — Suspicious trigger-like behavior detected"
        elif best.attack_success_rate > MEDIUM_THRESHOLD:
            result.overall_assessment = "MEDIUM — Moderate prediction shift observed"
        else:
            result.overall_assessment = "LOW — No significant trigger-like behavior"
        _prog("Done", 100)

    except Exception as e:
        result.scan_error = f"{type(e).__name__}: {e}"

    return result


# ────────────────────────────────────────────────────────────────────────────
# Inference demo: normal vs triggered image
# ────────────────────────────────────────────────────────────────────────────

def demo_trigger_effect(
    model: LoadedModel,
    image_path: str,
    trigger_position: str = "bottom-left",
    trigger_size: Optional[int] = None,
) -> Dict:
    """Return inference results for the original and the triggered image."""
    ts = trigger_size or default_trigger_size(model.input_size)
    with Image.open(image_path) as im:
        img_arr = np.array(model.resize(im))
    trig_arr = apply_trigger(img_arr, trigger_position, ts)

    probs = model.predict_proba([img_arr, trig_arr])
    results = {}
    for name, arr, p in [("original", img_arr, probs[0]), ("triggered", trig_arr, probs[1])]:
        idx = int(p.argmax())
        results[name] = {
            "image_array": arr,
            "predicted_label": model.class_names[idx],
            "predicted_idx": idx,
            "confidence": float(p[idx]),
            "all_probs": p.tolist(),
        }
    return results
