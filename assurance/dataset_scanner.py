"""
assurance/dataset_scanner.py

Lightweight dataset integrity scanner.

Implements:
  A. Duplicate detection via perceptual hashing (pHash + Hamming distance)
  B. Label-consistency anomaly detection via reference classifier
  C. Contributor-level risk aggregation
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from assurance.model_arch import load_model


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageRecord:
    filename: str
    label: str
    class_idx: int
    contributor_id: str = "unknown"
    batch_id: str       = "unknown"
    poisoned: bool      = False
    has_trigger: bool   = False
    label_anomaly: bool = False
    is_duplicate: bool  = False


@dataclass
class DuplicateGroup:
    representative: str
    duplicates: List[str]
    hamming_distance: float


@dataclass
class LabelAnomaly:
    filename: str
    dataset_label: str
    predicted_label: str
    confidence: float
    contributor_id: str


@dataclass
class ContributorStats:
    contributor_id: str
    total_samples: int
    duplicate_count: int
    anomaly_count: int
    risk_score: float  # 0..1

    @property
    def risk_level(self) -> str:
        if self.risk_score < 0.1:
            return "LOW"
        elif self.risk_score < 0.3:
            return "MEDIUM"
        return "HIGH"


@dataclass
class ScanResult:
    total_samples: int                      = 0
    duplicate_count: int                    = 0
    duplicate_groups: List[DuplicateGroup]  = field(default_factory=list)
    label_anomalies: List[LabelAnomaly]     = field(default_factory=list)
    contributor_stats: List[ContributorStats] = field(default_factory=list)
    integrity_score: int                    = 100   # 0..100
    dataset_path: str                       = ""
    scan_error: Optional[str]               = None


# ────────────────────────────────────────────────────────────────────────────
# Loader
# ────────────────────────────────────────────────────────────────────────────

def load_dataset_records(root: str) -> List[ImageRecord]:
    """
    Load image records from a directory/class structure.

    Expected layout:
        root/
          class_name/
            image.png
            ...
          metadata.jsonl  (optional per-sample metadata)

    Additional loaders (COCO, YOLO, etc.) can be added here later.
    """
    root = Path(root)
    records: List[ImageRecord] = []
    meta_map: Dict[str, dict] = {}

    # Try loading metadata sidecar
    meta_path = root / "metadata.jsonl"
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    meta_map[rec.get("filename", "")] = rec
                except json.JSONDecodeError:
                    pass

    # Scan directory structure
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg")):
            fname = str(img_path)
            meta = meta_map.get(fname, {})
            records.append(ImageRecord(
                filename=fname,
                label=meta.get("label", class_name),
                class_idx=meta.get("class_idx", class_idx),
                contributor_id=meta.get("contributor_id", "unknown"),
                batch_id=meta.get("batch_id", "unknown"),
                poisoned=meta.get("poisoned", False),
                has_trigger=meta.get("has_trigger", False),
                label_anomaly=meta.get("label_anomaly", False),
                is_duplicate=meta.get("is_duplicate", False),
            ))

    return records


# ────────────────────────────────────────────────────────────────────────────
# A.  Duplicate detection
# ────────────────────────────────────────────────────────────────────────────

def detect_duplicates(
    records: List[ImageRecord],
    threshold: int = config.PHASH_DUP_THRESHOLD,
) -> Tuple[List[DuplicateGroup], set]:
    """
    Compute perceptual hash for every image and cluster near-duplicates
    (Hamming distance ≤ threshold).

    Returns (groups, set_of_duplicate_filenames).
    """
    hashes: List[Tuple[str, imagehash.ImageHash]] = []
    for rec in records:
        try:
            img = Image.open(rec.filename).convert("RGB")
            h   = imagehash.phash(img, hash_size=config.PHASH_HASH_SIZE)
            hashes.append((rec.filename, h))
        except Exception:
            pass

    groups: List[DuplicateGroup] = []
    already_grouped: set = set()
    duplicate_files: set = set()

    for i in range(len(hashes)):
        fname_i, hash_i = hashes[i]
        if fname_i in already_grouped:
            continue
        group_dups, dists = [], []
        for j in range(i + 1, len(hashes)):
            fname_j, hash_j = hashes[j]
            if fname_j in already_grouped:
                continue
            dist = hash_i - hash_j
            if dist <= threshold:
                group_dups.append(fname_j)
                dists.append(dist)
                already_grouped.add(fname_j)
                # Only the redundant copies count as duplicates — the
                # representative is the one sample we would keep.
                duplicate_files.add(fname_j)
        if group_dups:
            already_grouped.add(fname_i)
            groups.append(DuplicateGroup(
                representative=fname_i,
                duplicates=group_dups,
                hamming_distance=float(np.mean(dists)),
            ))

    return groups, duplicate_files


# ────────────────────────────────────────────────────────────────────────────
# B.  Label-consistency anomaly detection
# ────────────────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def detect_label_anomalies(
    records: List[ImageRecord],
    model_path: str,
    conf_threshold: float = config.LABEL_ANOMALY_CONF_THR,
) -> List[LabelAnomaly]:
    """
    Use the reference classifier to flag samples where the model's top
    prediction strongly disagrees with the dataset label.

    NOTE: Disagreement is labelled a 'label-consistency anomaly' —
          it does NOT prove poisoning.
    """
    try:
        model = load_model(model_path, num_classes=config.NUM_CLASSES)
    except Exception as e:
        return []

    model.eval()
    anomalies: List[LabelAnomaly] = []

    with torch.no_grad():
        for rec in records:
            try:
                img = Image.open(rec.filename).convert("RGB")
                tensor = _TRANSFORM(img).unsqueeze(0)
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1)[0]
                pred_idx   = int(probs.argmax().item())
                confidence = float(probs.max().item())
                pred_label = config.CLASS_NAMES[pred_idx]

                # Flag if model strongly predicts a DIFFERENT class
                if pred_label != rec.label and confidence >= conf_threshold:
                    anomalies.append(LabelAnomaly(
                        filename=rec.filename,
                        dataset_label=rec.label,
                        predicted_label=pred_label,
                        confidence=confidence,
                        contributor_id=rec.contributor_id,
                    ))
            except Exception:
                pass

    return anomalies


# ────────────────────────────────────────────────────────────────────────────
# C.  Contributor aggregation
# ────────────────────────────────────────────────────────────────────────────

def aggregate_contributors(
    records: List[ImageRecord],
    duplicate_files: set,
    anomalies: List[LabelAnomaly],
) -> List[ContributorStats]:
    """
    Group findings by contributor_id and compute per-contributor risk scores.
    """
    from collections import defaultdict

    totals:     Dict[str, int] = defaultdict(int)
    dup_counts: Dict[str, int] = defaultdict(int)
    anom_set:   set = {a.filename for a in anomalies}
    anom_counts:Dict[str, int] = defaultdict(int)

    for rec in records:
        cid = rec.contributor_id
        totals[cid] += 1
        if rec.filename in duplicate_files:
            dup_counts[cid] += 1
        if rec.filename in anom_set:
            anom_counts[cid] += 1

    stats = []
    for cid in sorted(totals.keys()):
        total = totals[cid]
        dups  = dup_counts[cid]
        anoms = anom_counts[cid]
        suspicious = dups + anoms
        risk = suspicious / max(total, 1)
        stats.append(ContributorStats(
            contributor_id=cid,
            total_samples=total,
            duplicate_count=dups,
            anomaly_count=anoms,
            risk_score=min(risk, 1.0),
        ))

    return stats


# ────────────────────────────────────────────────────────────────────────────
# Scoring
# ────────────────────────────────────────────────────────────────────────────

def compute_integrity_score(result: ScanResult) -> int:
    """
    Start at 100 and subtract penalties that saturate at a small affected
    fraction, so that a dataset with e.g. 10 % suspicious labels scores far
    below a clean one instead of losing only a handful of points.
    """
    n = max(result.total_samples, 1)
    dup_rate  = result.duplicate_count / n
    anom_rate = len(result.label_anomalies) / n

    dup_penalty  = config.DUP_PENALTY_MAX  * min(1.0, dup_rate  / config.DUP_SATURATION_RATE)
    anom_penalty = config.ANOM_PENALTY_MAX * min(1.0, anom_rate / config.ANOM_SATURATION_RATE)
    n_high = sum(1 for c in result.contributor_stats if c.risk_level == "HIGH")
    contrib_penalty = min(20, config.HIGH_RISK_CONTRIB_PENALTY * n_high)

    return max(0, round(100 - dup_penalty - anom_penalty - contrib_penalty))


# ────────────────────────────────────────────────────────────────────────────
# Main scanner entry point
# ────────────────────────────────────────────────────────────────────────────

def scan_dataset(
    dataset_path: str,
    reference_model_path: Optional[str] = None,
    progress_callback=None,
) -> ScanResult:
    """
    Run the full dataset integrity scan.

    Args:
        dataset_path:          Path to dataset root directory.
        reference_model_path:  Path to .pt reference classifier.
        progress_callback:     Optional callable(step_name, pct) for UI updates.

    Returns:
        ScanResult with all findings populated.
    """
    result = ScanResult(dataset_path=dataset_path)

    def _prog(name, pct):
        if progress_callback:
            progress_callback(name, pct)

    try:
        _prog("Loading records", 0)
        records = load_dataset_records(dataset_path)
        result.total_samples = len(records)

        if result.total_samples == 0:
            result.scan_error = "No images found in dataset."
            return result

        _prog("Detecting duplicates", 20)
        dup_groups, dup_files = detect_duplicates(records)
        result.duplicate_groups = dup_groups
        result.duplicate_count  = len(dup_files)

        _prog("Checking label consistency", 50)
        if reference_model_path and Path(reference_model_path).exists():
            result.label_anomalies = detect_label_anomalies(records, reference_model_path)
        else:
            result.label_anomalies = []

        _prog("Aggregating contributors", 80)
        result.contributor_stats = aggregate_contributors(records, dup_files, result.label_anomalies)

        _prog("Computing score", 95)
        result.integrity_score = compute_integrity_score(result)

        _prog("Done", 100)

    except Exception as e:
        result.scan_error = str(e)

    return result

