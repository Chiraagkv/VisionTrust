"""
assurance/model_fingerprint.py

Two-factor model identity verification:
  A. Cryptographic fingerprint — SHA-256 of the model file bytes
  B. Behavioral fingerprint    — predictions on the fixed canary set
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
class FingerprintResult:
    model_path: str                     = ""
    model_name: str                     = ""
    # Cryptographic
    sha256: str                         = ""
    reference_sha256: str               = ""
    binary_match: bool                  = False
    # Behavioral
    canary_predictions: List[int]       = field(default_factory=list)
    reference_predictions: List[int]    = field(default_factory=list)
    behavioral_match_pct: float         = 0.0
    canary_mismatches: int              = 0
    canary_total: int                   = 0
    # Assessment
    assessment: str                     = "UNKNOWN"
    scan_error: Optional[str]           = None


# ────────────────────────────────────────────────────────────────────────────
# Cryptographic fingerprint
# ────────────────────────────────────────────────────────────────────────────

def compute_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference_hashes() -> Dict[str, str]:
    """Load registered reference hashes from models/reference_hashes.json."""
    ref_path = Path(config.REFERENCE_HASHES_PATH)
    if not ref_path.exists():
        return {}
    with open(ref_path) as f:
        return json.load(f)


def register_reference_hash(model_key: str, sha256: str) -> None:
    """Register a SHA-256 as the canonical reference hash for a model key."""
    ref_path = Path(config.REFERENCE_HASHES_PATH)
    existing = {}
    if ref_path.exists():
        with open(ref_path) as f:
            existing = json.load(f)
    existing[model_key] = sha256
    with open(ref_path, "w") as f:
        json.dump(existing, f, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# Behavioral fingerprint
# ────────────────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_canary_records() -> List[dict]:
    """Load the canary set metadata."""
    meta_path = Path(config.CANARY_DIR) / "canary_meta.json"
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        return json.load(f)


def get_canary_predictions(model_path: str) -> List[int]:
    """
    Run inference on the canary set and return predicted class indices.
    The canary set is fixed, so predictions uniquely identify a model's
    behavioral state.
    """
    canary_records = load_canary_records()
    if not canary_records:
        return []

    model = load_model(model_path, num_classes=config.NUM_CLASSES)
    model.eval()

    predictions = []
    with torch.no_grad():
        for rec in canary_records:
            fpath = rec.get("filename", "")
            if not Path(fpath).exists():
                predictions.append(-1)
                continue
            try:
                img    = Image.open(fpath).convert("RGB")
                tensor = _TRANSFORM(img).unsqueeze(0)
                logits = model(tensor)
                pred   = int(logits.argmax(1).item())
                predictions.append(pred)
            except Exception:
                predictions.append(-1)

    return predictions


def compare_behavioral_fingerprints(preds_a: List[int], preds_b: List[int]) -> Dict:
    """Compare two behavioral fingerprints and return match statistics."""
    n = min(len(preds_a), len(preds_b))
    if n == 0:
        return {"match_pct": 0.0, "mismatches": 0, "total": 0}
    matches    = sum(a == b for a, b in zip(preds_a[:n], preds_b[:n]))
    mismatches = n - matches
    return {
        "match_pct":  100.0 * matches / n,
        "mismatches": mismatches,
        "total":      n,
    }


# ────────────────────────────────────────────────────────────────────────────
# Reference canary fingerprint storage
# ────────────────────────────────────────────────────────────────────────────

_CANARY_PREDS_FILE = Path(config.MODELS_DIR) / "canary_predictions.json"


def save_reference_canary_predictions(model_key: str, predictions: List[int]) -> None:
    existing = {}
    if _CANARY_PREDS_FILE.exists():
        with open(_CANARY_PREDS_FILE) as f:
            existing = json.load(f)
    existing[model_key] = predictions
    with open(_CANARY_PREDS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def load_reference_canary_predictions(model_key: str) -> List[int]:
    if not _CANARY_PREDS_FILE.exists():
        return []
    with open(_CANARY_PREDS_FILE) as f:
        data = json.load(f)
    return data.get(model_key, [])


# ────────────────────────────────────────────────────────────────────────────
# Main fingerprint entry point
# ────────────────────────────────────────────────────────────────────────────

def fingerprint_model(
    model_path: str,
    reference_model_key: str = "clean_model",
    progress_callback=None,
) -> FingerprintResult:
    """
    Compute full fingerprint for a model file.

    Args:
        model_path:           Path to the .pt model file to verify.
        reference_model_key:  Key in reference_hashes.json to compare against.
        progress_callback:    Optional callable(step, pct).

    Returns:
        FingerprintResult with all fields populated.
    """
    result = FingerprintResult(
        model_path=model_path,
        model_name=Path(model_path).name,
    )

    def _prog(name, pct):
        if progress_callback:
            progress_callback(name, pct)

    try:
        _prog("Computing SHA-256", 10)
        result.sha256 = compute_sha256(model_path)

        ref_hashes = load_reference_hashes()
        result.reference_sha256 = ref_hashes.get(reference_model_key, "")
        result.binary_match = (
            result.sha256 == result.reference_sha256
            and bool(result.reference_sha256)
        )

        _prog("Running canary inference", 40)
        result.canary_predictions = get_canary_predictions(model_path)

        # Load or compute reference canary predictions
        ref_preds = load_reference_canary_predictions(reference_model_key)
        if not ref_preds and Path(config.CLEAN_MODEL_PATH).exists():
            # Compute once from the clean model and store
            ref_preds = get_canary_predictions(config.CLEAN_MODEL_PATH)
            save_reference_canary_predictions(reference_model_key, ref_preds)

        result.reference_predictions = ref_preds

        _prog("Comparing fingerprints", 80)
        cmp = compare_behavioral_fingerprints(
            result.canary_predictions,
            result.reference_predictions,
        )
        result.behavioral_match_pct = cmp["match_pct"]
        result.canary_mismatches    = cmp["mismatches"]
        result.canary_total         = cmp["total"]

        # Assessment
        if result.binary_match and result.behavioral_match_pct >= 98.0:
            result.assessment = "PASS — Model matches reference"
        elif result.binary_match:
            result.assessment = "REVIEW — Binary match but behavioral drift detected"
        elif result.behavioral_match_pct >= 90.0:
            result.assessment = "REVIEW — Different file, similar behavior"
        else:
            result.assessment = "ALERT — Model substitution or significant modification"

        _prog("Done", 100)

    except Exception as e:
        result.scan_error = str(e)

    return result

