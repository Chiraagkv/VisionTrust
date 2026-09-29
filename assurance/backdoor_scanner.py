"""
assurance/backdoor_scanner.py

Behavioral trigger scanner (prototype implementation).

Methodology:
  For each candidate trigger position:
    1. Apply a white-square patch to test images.
    2. Run inference.
    3. Measure attack_success_rate = fraction predicted as target class.

  Positions with ASR > SUSPICION_THRESHOLD are flagged as suspicious.

This is NOT a universal backdoor detector.
It detects trigger-like behavior for known patch-based attack classes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from assurance.model_arch import load_model


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
    overall_assessment: str                  = "UNKNOWN"
    model_path: str                          = ""
    scan_error: Optional[str]                = None
    # Per-position visualization arrays (numpy HWC RGB)
    example_images: Dict[str, np.ndarray]    = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# Trigger helper
# ────────────────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def apply_trigger(img_arr: np.ndarray, position: str, trigger_size: int = config.TRIGGER_SIZE) -> np.ndarray:
    img = img_arr.copy()
    h, w = img.shape[:2]
    ts = trigger_size
    positions = {
        "top-left":     (0,      0),
        "top-right":    (0,      w - ts),
        "center":       (h//2 - ts//2, w//2 - ts//2),
        "bottom-left":  (h - ts, 0),
        "bottom-right": (h - ts, w - ts),
    }
    r, c = positions.get(position, (h - ts, 0))
    img[r:r + ts, c:c + ts] = 255
    return img


def _suspicion_level(asr: float) -> str:
    if asr > config.SUSPICION_THRESHOLD:
        return "HIGH"
    elif asr > 0.25:
        return "MEDIUM"
    return "LOW"


# ────────────────────────────────────────────────────────────────────────────
# Test-image loader
# ────────────────────────────────────────────────────────────────────────────

def _load_test_images(n_images: int = 30) -> List[np.ndarray]:
    """
    Load up to n_images test images from the clean dataset or canary set.
    Falls back to random noise if nothing is found.
    """
    images = []
    for src_dir in [config.DATA_CLEAN_DIR, config.CANARY_DIR]:
        if len(images) >= n_images:
            break
        src = Path(src_dir)
        if not src.exists():
            continue
        for p in sorted(src.rglob("*.png")):
            if len(images) >= n_images:
                break
            try:
                arr = np.array(Image.open(p).convert("RGB"))
                images.append(arr)
            except Exception:
                pass

    if not images:
        # Synthetic fallback
        rng = np.random.RandomState(42)
        for _ in range(n_images):
            images.append(rng.randint(0, 256, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8))

    return images


# ────────────────────────────────────────────────────────────────────────────
# Main scanner
# ────────────────────────────────────────────────────────────────────────────

def scan_backdoor(
    model_path: str,
    target_class: int = config.BACKDOOR_TARGET,
    n_test_images: int = 30,
    progress_callback=None,
) -> BackdoorScanResult:
    """
    Run behavioral trigger scan against a model.

    For each candidate trigger position, compute attack_success_rate =
    fraction of test images predicted as `target_class` when the trigger
    is applied.

    This prototype only detects visible patch-based triggers at candidate
    positions. Stealth attacks are outside the supported threat model.
    """
    result = BackdoorScanResult(
        model_path=model_path,
        target_class=config.CLASS_NAMES[target_class],
    )

    def _prog(name, pct):
        if progress_callback:
            progress_callback(name, pct)

    try:
        _prog("Loading model", 0)
        model = load_model(model_path, num_classes=config.NUM_CLASSES)
        model.eval()

        _prog("Loading test images", 10)
        test_images = _load_test_images(n_test_images)

        trigger_results = []
        total_positions = len(config.TRIGGER_CANDIDATES)

        for pos_idx, position in enumerate(config.TRIGGER_CANDIDATES):
            _prog(f"Testing position: {position}", 10 + int(80 * pos_idx / total_positions))

            triggered_as_target = 0
            example_saved = False

            with torch.no_grad():
                for img_arr in test_images:
                    triggered = apply_trigger(img_arr, position)
                    pil_img = Image.fromarray(triggered)
                    tensor  = _TRANSFORM(pil_img).unsqueeze(0)
                    logits  = model(tensor)
                    pred    = int(logits.argmax(1).item())
                    if pred == target_class:
                        triggered_as_target += 1
                    if not example_saved:
                        result.example_images[position] = triggered
                        example_saved = True

            asr = triggered_as_target / max(len(test_images), 1)
            is_suspicious = asr > config.SUSPICION_THRESHOLD
            trigger_results.append(TriggerResult(
                position=position,
                attack_success_rate=asr,
                target_class=config.CLASS_NAMES[target_class],
                is_suspicious=is_suspicious,
                suspicion_level=_suspicion_level(asr),
            ))

        result.trigger_results = trigger_results

        # Find most suspicious
        best = max(trigger_results, key=lambda r: r.attack_success_rate)
        result.most_suspicious_position = best.position
        result.most_suspicious_asr      = best.attack_success_rate

        if best.attack_success_rate > config.SUSPICION_THRESHOLD:
            result.overall_assessment = "HIGH — Suspicious trigger-like behavior detected"
        elif best.attack_success_rate > 0.25:
            result.overall_assessment = "MEDIUM — Moderate prediction shift observed"
        else:
            result.overall_assessment = "LOW — No significant trigger-like behavior"

        _prog("Done", 100)

    except Exception as e:
        result.scan_error = str(e)

    return result


# ────────────────────────────────────────────────────────────────────────────
# Inference demo: normal vs triggered image
# ────────────────────────────────────────────────────────────────────────────

def demo_trigger_effect(
    model_path: str,
    image_path: str,
    trigger_position: str = "bottom-left",
    target_class: int = config.BACKDOOR_TARGET,
) -> Dict:
    """
    Return inference results for:
      - original image
      - triggered image
    """
    model = load_model(model_path, num_classes=config.NUM_CLASSES)
    model.eval()

    img_arr  = np.array(Image.open(image_path).convert("RGB"))
    trig_arr = apply_trigger(img_arr, trigger_position)

    results = {}
    with torch.no_grad():
        for name, arr in [("original", img_arr), ("triggered", trig_arr)]:
            pil    = Image.fromarray(arr)
            tensor = _TRANSFORM(pil).unsqueeze(0)
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            results[name] = {
                "image_array": arr,
                "predicted_label": config.CLASS_NAMES[pred_idx],
                "predicted_idx":   pred_idx,
                "confidence":      float(probs.max().item()),
                "all_probs":       probs.numpy().tolist(),
            }

    return results

