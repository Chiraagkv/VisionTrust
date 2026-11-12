"""
assurance/dataset_scanner.py

Lightweight dataset integrity scanner.

Implements:
  A. Duplicate detection via perceptual hashing (pHash + Hamming distance)
  B. Label-consistency anomaly detection in the embedding space of an
     independent pretrained feature extractor (no user model required):
       - "knn":          a sample is flagged when most of its nearest
                         neighbours carry a different label
                         (Deep-kNN style, Peri et al. 2020)
       - "linear_probe": out-of-fold logistic regression on the embeddings;
                         flagged when the probe confidently prefers another
                         class (confident-learning style)
  C. Contributor-level risk aggregation (from dataset metadata, if present)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import imagehash
import numpy as np
from PIL import Image

import config
from assurance.datasets import UNKNOWN, Dataset, ImageRecord
from assurance.features import extract_features


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DuplicateGroup:
    representative: str
    duplicates: List[str]
    hamming_distance: float


@dataclass
class LabelAnomaly:
    filename: str
    dataset_label: str
    predicted_label: str     # label the embedding neighbourhood / probe suggests
    confidence: float        # neighbour agreement (knn) or probe probability
    contributor_id: str
    own_label_support: float = 0.0   # share of neighbours / probe prob for the dataset label


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
class ScanParams:
    dup_threshold: int        = config.PHASH_DUP_THRESHOLD
    hash_size: int            = config.PHASH_HASH_SIZE
    extractor: str            = config.FEATURE_EXTRACTOR
    anomaly_method: str       = config.ANOMALY_METHOD
    k_neighbors: int          = config.KNN_NEIGHBORS
    agreement_threshold: float = config.ANOMALY_AGREEMENT_THR
    check_labels: bool        = True


@dataclass
class ScanResult:
    total_samples: int                      = 0
    duplicate_count: int                    = 0
    duplicate_groups: List[DuplicateGroup]  = field(default_factory=list)
    label_anomalies: List[LabelAnomaly]     = field(default_factory=list)
    contributor_stats: List[ContributorStats] = field(default_factory=list)
    integrity_score: int                    = 100   # 0..100
    dataset_path: str                       = ""
    dataset_name: str                       = ""
    class_names: List[str]                  = field(default_factory=list)
    unreadable: List[str]                   = field(default_factory=list)
    params: Optional[ScanParams]            = None
    extractor_used: str                     = ""
    notes: List[str]                        = field(default_factory=list)
    scan_error: Optional[str]               = None


# ────────────────────────────────────────────────────────────────────────────
# A.  Duplicate detection
# ────────────────────────────────────────────────────────────────────────────

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def detect_duplicates(
    records: List[ImageRecord],
    threshold: int = config.PHASH_DUP_THRESHOLD,
    hash_size: int = config.PHASH_HASH_SIZE,
    progress: Optional[Callable[[float], None]] = None,
) -> Tuple[List[DuplicateGroup], set, List[str]]:
    """
    Compute a perceptual hash for every image and greedily group
    near-duplicates (Hamming distance ≤ threshold). Distances are computed
    with packed bit arrays in row blocks, so tens of thousands of images are
    fine. Returns (groups, set_of_duplicate_filenames, unreadable_files).
    """
    names, bits, unreadable = [], [], []
    for i, rec in enumerate(records):
        try:
            with Image.open(rec.filename) as img:
                h = imagehash.phash(img.convert("RGB"), hash_size=hash_size)
            names.append(rec.filename)
            bits.append(h.hash.ravel())
        except Exception:
            unreadable.append(rec.filename)
        if progress and i % 200 == 0:
            progress(0.7 * i / max(len(records), 1))

    groups: List[DuplicateGroup] = []
    duplicate_files: set = set()
    if len(names) < 2:
        return groups, duplicate_files, unreadable

    packed = np.packbits(np.stack(bits).astype(np.uint8), axis=1)   # n × (bits/8)
    n = len(names)
    grouped = np.zeros(n, dtype=bool)
    # Rows per block, keeping the block × n × bytes intermediate around ≤ 60 MB
    block = int(max(8, min(512, 3e7 // (n * packed.shape[1]))))
    for start in range(0, n, block):
        stop = min(start + block, n)
        # dist[b, j] = Hamming(packed[start+b], packed[j])
        dist = _POPCOUNT[packed[start:stop, None, :] ^ packed[None, :, :]].sum(-1)
        for b in range(stop - start):
            i = start + b
            if grouped[i]:
                continue
            row = dist[b]
            cand = np.nonzero((row <= threshold) & ~grouped)[0]
            cand = cand[cand > i]
            if cand.size:
                grouped[i] = True
                grouped[cand] = True
                dups = [names[j] for j in cand]
                duplicate_files.update(dups)        # the representative is the copy we keep
                groups.append(DuplicateGroup(
                    representative=names[i], duplicates=dups,
                    hamming_distance=float(row[cand].mean()),
                ))
        if progress:
            progress(0.7 + 0.3 * stop / n)

    return groups, duplicate_files, unreadable


# ────────────────────────────────────────────────────────────────────────────
# B.  Label-consistency anomaly detection
# ────────────────────────────────────────────────────────────────────────────

def _knn_scores(feats: np.ndarray, labels: np.ndarray, k: int):
    """For each sample: (top neighbour label, its share, own-label share)."""
    from sklearn.neighbors import NearestNeighbors

    x = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(x)
    _, idx = nn.kneighbors(x)
    out = []
    for i, row in enumerate(idx):
        neigh = [j for j in row if j != i][:k]
        counts = Counter(labels[neigh])
        top, top_n = counts.most_common(1)[0]
        out.append((int(top), top_n / len(neigh), counts.get(labels[i], 0) / len(neigh)))
    return out


def _probe_scores(feats: np.ndarray, labels: np.ndarray, n_classes: int):
    """Out-of-fold logistic-regression probabilities → (top label, top prob, own prob)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    min_count = min(Counter(labels).values())
    folds = min(5, min_count)
    if folds < 2:
        raise ValueError("linear_probe needs ≥ 2 samples in every class")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
    probs = cross_val_predict(clf, feats, labels, cv=cv, method="predict_proba")
    present = np.unique(labels)            # columns of `probs` follow sorted present labels
    full = np.zeros((len(labels), n_classes))
    full[:, present] = probs
    top = full.argmax(1)
    return [(int(t), float(full[i, t]), float(full[i, labels[i]])) for i, t in enumerate(top)]


def detect_label_anomalies(
    records: List[ImageRecord],
    class_names: List[str],
    params: ScanParams,
    progress: Optional[Callable[[float], None]] = None,
    cache_key: Optional[str] = None,
) -> Tuple[List[LabelAnomaly], str, List[str]]:
    """
    Embed every image with an independent pretrained extractor and flag
    samples whose embedding neighbourhood (or an out-of-fold linear probe)
    strongly favours a different class than the dataset label.

    NOTE: Disagreement is a 'label-consistency anomaly' — it does NOT prove
          poisoning; it may also be an ambiguous or atypical image.

    Returns (anomalies, extractor_used, notes).
    """
    notes: List[str] = []
    if len(class_names) < 2:
        return [], "", ["Label check skipped: dataset has a single class."]

    feats, ok, used = extract_features([r.filename for r in records], params.extractor,
                                       progress=progress, cache_key=cache_key)
    if used != params.extractor:
        notes.append(f"Could not load '{params.extractor}' weights (offline?) — "
                     f"fell back to hand-crafted features; results are less reliable.")
    if len(ok) < 3:
        return [], used, notes + ["Label check skipped: fewer than 3 readable images."]

    recs = [records[i] for i in ok]
    labels = np.array([r.class_idx for r in recs])

    method = params.anomaly_method
    if method == "linear_probe":
        try:
            scores = _probe_scores(feats, labels, len(class_names))
        except ValueError as e:
            notes.append(f"{e}; used kNN instead.")
            method = "knn"
    if method == "knn":
        # Keep k below the size of the smallest class, or every member of a
        # small class would be outvoted by its neighbours from other classes.
        sizes = [c for c in Counter(labels).values() if c > 1]
        k = max(1, min(params.k_neighbors, (min(sizes) - 1) if sizes else 1, len(recs) - 1))
        if k < params.k_neighbors:
            notes.append(f"k reduced from {params.k_neighbors} to {k} (smallest class is small).")
        scores = _knn_scores(feats, labels, k)

    anomalies = []
    for rec, (top, top_share, own_share) in zip(recs, scores):
        if top != rec.class_idx and top_share >= params.agreement_threshold:
            anomalies.append(LabelAnomaly(
                filename=rec.filename, dataset_label=rec.label,
                predicted_label=class_names[top], confidence=top_share,
                contributor_id=rec.contributor_id, own_label_support=own_share,
            ))
    anomalies.sort(key=lambda a: -a.confidence)
    return anomalies, used, notes


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
    Returns [] when the dataset carries no contributor metadata.
    """
    if all(r.contributor_id == UNKNOWN for r in records):
        return []

    totals:      Dict[str, int] = defaultdict(int)
    dup_counts:  Dict[str, int] = defaultdict(int)
    anom_counts: Dict[str, int] = defaultdict(int)
    anom_set = {a.filename for a in anomalies}

    for rec in records:
        cid = rec.contributor_id
        totals[cid] += 1
        if rec.filename in duplicate_files:
            dup_counts[cid] += 1
        if rec.filename in anom_set:
            anom_counts[cid] += 1

    stats = []
    for cid in sorted(totals):
        total = totals[cid]
        dups, anoms = dup_counts[cid], anom_counts[cid]
        stats.append(ContributorStats(
            contributor_id=cid, total_samples=total, duplicate_count=dups,
            anomaly_count=anoms, risk_score=min((dups + anoms) / max(total, 1), 1.0),
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
    dataset: Dataset,
    params: Optional[ScanParams] = None,
    progress_callback=None,
) -> ScanResult:
    """
    Run the full dataset integrity scan.

    Args:
        dataset:            A Dataset from assurance.datasets.
        params:             Scan parameters (defaults from config).
        progress_callback:  Optional callable(step_name, pct) for UI updates.
    """
    params = params or ScanParams()
    result = ScanResult(dataset_path=dataset.root, dataset_name=dataset.name,
                        class_names=list(dataset.class_names), params=params,
                        notes=list(dataset.notes))

    def _prog(name, pct):
        if progress_callback:
            progress_callback(name, int(pct))

    try:
        records = dataset.records
        result.total_samples = len(records)
        if not records:
            result.scan_error = "No images found in dataset."
            return result

        _prog("Detecting duplicates", 0)
        dup_groups, dup_files, unreadable = detect_duplicates(
            records, params.dup_threshold, params.hash_size,
            progress=lambda f: _prog("Detecting duplicates", 35 * f))
        result.duplicate_groups = dup_groups
        result.duplicate_count  = len(dup_files)
        result.unreadable       = unreadable
        if unreadable:
            result.notes.append(f"{len(unreadable)} files could not be read as images.")

        if params.check_labels:
            _prog("Embedding images (independent feature extractor)", 35)
            anomalies, used, notes = detect_label_anomalies(
                records, dataset.class_names, params,
                progress=lambda f: _prog("Embedding images (independent feature extractor)", 35 + 50 * f),
                cache_key=dataset.fingerprint())
            result.label_anomalies = anomalies
            result.extractor_used  = used
            result.notes.extend(notes)

        _prog("Aggregating contributors", 88)
        result.contributor_stats = aggregate_contributors(records, dup_files, result.label_anomalies)

        _prog("Computing score", 95)
        result.integrity_score = compute_integrity_score(result)
        _prog("Done", 100)

    except Exception as e:
        result.scan_error = f"{type(e).__name__}: {e}"

    return result
