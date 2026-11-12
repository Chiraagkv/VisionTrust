"""
assurance/model_fingerprint.py

Two-factor model identity verification against a *registered trusted
reference*:
  A. Cryptographic fingerprint — SHA-256 of the model file bytes
  B. Behavioral fingerprint    — predictions on a canary set

A reference is created by registering a model you trust together with a
canary set. The canary set (image paths + their SHA-256) is stored with the
reference, so later audits replay exactly the same images and can detect if
the canary images themselves were altered.

Canary sets can be:
  • the demo set in canary/ (canary_meta.json), or
  • built from any loaded dataset: a stratified, seeded sample of k images per
    class. Ideally these images were held out of training.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import config
from assurance.datasets import Dataset
from assurance.model_loader import LoadedModel, file_sha256


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CanarySet:
    name: str
    manifest_path: str
    images: List[str]
    sha256s: List[str]
    labels: List[Optional[str]] = field(default_factory=list)

    @property
    def digest(self) -> str:
        return hashlib.sha256("".join(self.sha256s).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.images)


@dataclass
class FingerprintResult:
    model_path: str                     = ""
    model_name: str                     = ""
    reference_name: str                 = ""
    has_reference: bool                 = False
    # Cryptographic
    sha256: str                         = ""
    reference_sha256: str               = ""
    binary_match: bool                  = False
    # Behavioral
    canary_set: str                     = ""
    canary_intact: bool                 = True
    canary_predictions: List[int]       = field(default_factory=list)
    reference_predictions: List[int]    = field(default_factory=list)
    behavioral_match_pct: float         = 0.0
    canary_mismatches: int              = 0
    canary_total: int                   = 0
    mean_prob_shift: float              = 0.0     # mean total-variation distance of softmax outputs
    # Assessment
    assessment: str                     = "UNKNOWN"
    scan_error: Optional[str]           = None


# Kept for callers that only need the hash
compute_sha256 = file_sha256


# ────────────────────────────────────────────────────────────────────────────
# Canary sets
# ────────────────────────────────────────────────────────────────────────────

def load_canary_set(manifest_path: str = config.DEFAULT_CANARY_MANIFEST) -> CanarySet:
    """
    Read a canary manifest. Two formats are accepted:
      • list of {"filename": ..., "label": ...}               (demo canary_meta.json)
      • {"name": ..., "images": [{"path", "sha256", "label"}]}  (written by build_canary_set)
    Paths that don't exist as written are resolved relative to the manifest.
    """
    mp = Path(manifest_path)
    data = json.loads(mp.read_text())
    entries = data["images"] if isinstance(data, dict) else data
    name = data.get("name", mp.stem) if isinstance(data, dict) else mp.parent.name + "/" + mp.stem

    images, shas, labels = [], [], []
    for e in entries:
        raw = e.get("path") or e.get("filename", "")
        p = Path(raw)
        if not p.exists():
            for cand in (mp.parent / raw, mp.parent / p.name):
                if cand.exists():
                    p = cand
                    break
        images.append(str(p))
        shas.append(e.get("sha256") or (file_sha256(str(p)) if p.exists() else ""))
        labels.append(e.get("label"))
    return CanarySet(name=name, manifest_path=str(mp.resolve()), images=images,
                     sha256s=shas, labels=labels)


def build_canary_set(
    dataset: Dataset,
    per_class: int = config.CANARY_PER_CLASS,
    seed: int = config.RANDOM_SEED,
    name: Optional[str] = None,
    out_dir: str = config.CANARY_SETS_DIR,
) -> CanarySet:
    """Stratified, seeded sample of `per_class` images per class → manifest JSON."""
    by_cls: Dict[str, list] = defaultdict(list)
    for r in dataset.records:
        by_cls[r.label].append(r)
    rng = random.Random(seed)
    chosen = []
    for lbl in sorted(by_cls):
        recs = sorted(by_cls[lbl], key=lambda r: r.filename)
        rng.shuffle(recs)
        chosen.extend(recs[:per_class])

    name = name or f"{dataset.name}-k{per_class}-s{seed}"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    manifest = Path(out_dir) / f"{safe}.json"
    entries = [{"path": r.filename, "sha256": file_sha256(r.filename), "label": r.label} for r in chosen]
    manifest.write_text(json.dumps({"name": name, "source_dataset": dataset.root,
                                    "per_class": per_class, "seed": seed,
                                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "images": entries}, indent=2))
    return load_canary_set(str(manifest))


def list_canary_manifests() -> List[str]:
    out = []
    if Path(config.DEFAULT_CANARY_MANIFEST).exists():
        out.append(config.DEFAULT_CANARY_MANIFEST)
    d = Path(config.CANARY_SETS_DIR)
    if d.exists():
        out.extend(str(p) for p in sorted(d.glob("*.json")))
    return out


def canary_outputs(model: LoadedModel, canary: CanarySet) -> np.ndarray:
    """Softmax outputs on the canary images (a row of NaN for unreadable files)."""
    imgs, ok = [], []
    for i, p in enumerate(canary.images):
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
            ok.append(i)
        except Exception:
            pass
    probs = np.full((len(canary.images), model.num_classes), np.nan)
    if imgs:
        probs[ok] = model.predict_proba(imgs)
    return probs


def _preds(probs: np.ndarray) -> List[int]:
    return [int(np.argmax(p)) if not np.isnan(p).any() else -1 for p in probs]


# ────────────────────────────────────────────────────────────────────────────
# Reference registry
# ────────────────────────────────────────────────────────────────────────────

def _load_registry() -> Dict[str, Dict]:
    p = Path(config.REFERENCE_REGISTRY_PATH)
    return json.loads(p.read_text()) if p.exists() else {}


def list_references() -> Dict[str, Dict]:
    return _load_registry()


def register_reference(name: str, model: LoadedModel, canary: CanarySet) -> Dict:
    """Record `model` as the trusted reference `name`, bound to `canary`."""
    probs = canary_outputs(model, canary)
    entry = {
        "name": name,
        "model_path": model.path,
        "sha256": model.sha256,
        "arch": model.arch,
        "class_names": model.class_names,
        "canary_manifest": canary.manifest_path,
        "canary_digest": canary.digest,
        "canary_predictions": _preds(probs),
        "canary_probs": np.nan_to_num(probs, nan=-1).round(6).tolist(),
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reg = _load_registry()
    reg[name] = entry
    Path(config.REFERENCE_REGISTRY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.REFERENCE_REGISTRY_PATH).write_text(json.dumps(reg, indent=2))
    return entry


def delete_reference(name: str) -> None:
    reg = _load_registry()
    reg.pop(name, None)
    Path(config.REFERENCE_REGISTRY_PATH).write_text(json.dumps(reg, indent=2))


# ────────────────────────────────────────────────────────────────────────────
# Main fingerprint entry point
# ────────────────────────────────────────────────────────────────────────────

def fingerprint_model(
    model: LoadedModel,
    reference_name: Optional[str],
    progress_callback=None,
) -> FingerprintResult:
    """Compare `model` with the registered reference `reference_name`."""
    result = FingerprintResult(model_path=model.path, model_name=model.name,
                               reference_name=reference_name or "")

    def _prog(name, pct):
        if progress_callback:
            progress_callback(name, pct)

    try:
        _prog("Computing SHA-256", 10)
        result.sha256 = file_sha256(model.path)

        ref = _load_registry().get(reference_name or "")
        if ref is None:
            result.assessment = "UNVERIFIED — No trusted reference registered"
            return result
        result.has_reference = True
        result.reference_sha256 = ref["sha256"]
        result.binary_match = result.sha256 == ref["sha256"]

        _prog("Running canary inference", 40)
        canary = load_canary_set(ref["canary_manifest"])
        result.canary_set = canary.name
        result.canary_intact = canary.digest == ref["canary_digest"]

        if ref.get("class_names") and ref["class_names"] != model.class_names:
            raise ValueError("Model class names differ from the reference's — not comparable "
                             f"(reference has {len(ref['class_names'])} classes, model {model.num_classes}).")

        probs = canary_outputs(model, canary)
        result.canary_predictions = _preds(probs)
        result.reference_predictions = ref["canary_predictions"]

        _prog("Comparing fingerprints", 80)
        a, b = result.canary_predictions, result.reference_predictions
        n = min(len(a), len(b))
        matches = sum(x == y for x, y in zip(a[:n], b[:n]))
        result.canary_total = n
        result.canary_mismatches = n - matches
        result.behavioral_match_pct = 100.0 * matches / n if n else 0.0
        ref_probs = np.array(ref.get("canary_probs", []), dtype=float)
        if ref_probs.shape == probs.shape:
            valid = ~np.isnan(probs).any(1) & (ref_probs >= 0).all(1)
            if valid.any():
                result.mean_prob_shift = float(0.5 * np.abs(probs[valid] - ref_probs[valid]).sum(1).mean())

        if not result.canary_intact:
            result.assessment = "REVIEW — Canary images changed since registration; behavioral result unreliable"
        elif result.binary_match and result.behavioral_match_pct >= 98.0:
            result.assessment = "PASS — Model matches reference"
        elif result.binary_match:
            result.assessment = "REVIEW — Binary match but behavioral drift detected"
        elif result.behavioral_match_pct >= 90.0:
            result.assessment = "REVIEW — Different file, similar behavior"
        else:
            result.assessment = "ALERT — Model substitution or significant modification"
        _prog("Done", 100)

    except Exception as e:
        result.scan_error = f"{type(e).__name__}: {e}"

    return result
