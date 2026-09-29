"""
assurance/provenance.py

Cryptographic inference provenance — fully local, SHA-256 hash chain.

Every inference produces a record containing:
  inference_id, input_hash, model_hash, preprocessing_hash,
  output, confidence, timestamp, nonce, previous_record_hash, record_hash

Records are stored in records/inference_chain.jsonl as an append-only
hash chain.  The system detects:
  - changed output
  - changed input_hash
  - changed model_hash
  - changed timestamp
  - deleted or reordered records
  - broken previous_record_hash linkage
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
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

GENESIS_HASH = "0" * 64   # sentinel for the very first record


@dataclass
class InferenceRecord:
    inference_id:       str
    input_hash:         str
    model_hash:         str
    preprocessing_hash: str
    output:             str     # class name
    confidence:         float
    timestamp:          str
    nonce:              str
    previous_record_hash: str
    record_hash:        str     # computed last


@dataclass
class VerificationResult:
    record_id:      str
    is_valid:       bool
    expected_hash:  str
    observed_hash:  str
    broken_field:   Optional[str] = None
    error_msg:      str            = ""


@dataclass
class ChainVerificationResult:
    total_records:  int
    valid_records:  int
    is_intact:      bool
    first_broken:   Optional[int]  = None   # 0-indexed
    broken_record_id: Optional[str] = None
    verification_details: List[VerificationResult] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Image transforms
# ────────────────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ────────────────────────────────────────────────────────────────────────────
# Hashing helpers
# ────────────────────────────────────────────────────────────────────────────

def _sha256(*parts: str) -> str:
    """Concatenate string parts and return SHA-256 hex digest."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
    return h.hexdigest()


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_image(image_path: str) -> str:
    """Hash raw image bytes (input integrity)."""
    return _hash_file(image_path)


def _hash_tensor(tensor: torch.Tensor) -> str:
    """Hash a float32 tensor (preprocessing integrity)."""
    h = hashlib.sha256(tensor.numpy().tobytes())
    return h.hexdigest()


def _compute_record_hash(rec: "InferenceRecord") -> str:
    """
    Compute canonical record hash from protected fields.

    record_hash = SHA256(
        previous_record_hash +
        input_hash +
        model_hash +
        preprocessing_hash +
        output +
        timestamp +
        nonce
    )
    """
    return _sha256(
        rec.previous_record_hash,
        rec.input_hash,
        rec.model_hash,
        rec.preprocessing_hash,
        rec.output,
        str(rec.confidence),
        rec.timestamp,
        rec.nonce,
    )


# ────────────────────────────────────────────────────────────────────────────
# Chain persistence
# ────────────────────────────────────────────────────────────────────────────

def _ensure_records_dir():
    Path(config.RECORDS_DIR).mkdir(parents=True, exist_ok=True)


def _load_chain() -> List[InferenceRecord]:
    """Read all records from inference_chain.jsonl."""
    _ensure_records_dir()
    path = Path(config.INFERENCE_CHAIN_PATH)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(InferenceRecord(**d))
            except Exception:
                pass
    return records


def _save_chain(records: List[InferenceRecord]) -> None:
    """Overwrite inference_chain.jsonl with the given records."""
    _ensure_records_dir()
    with open(config.INFERENCE_CHAIN_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")


def _append_record(record: InferenceRecord) -> None:
    """Append a single record to inference_chain.jsonl."""
    _ensure_records_dir()
    with open(config.INFERENCE_CHAIN_PATH, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


# ────────────────────────────────────────────────────────────────────────────
# Core operations
# ────────────────────────────────────────────────────────────────────────────

def get_last_record_hash() -> str:
    """Return the record_hash of the most recent chain entry, or GENESIS."""
    records = _load_chain()
    if not records:
        return GENESIS_HASH
    return records[-1].record_hash


def create_inference_record(
    image_path: str,
    model_path: str,
) -> Tuple[InferenceRecord, Dict]:
    """
    Run inference and create a new provenance record.

    Returns (record, inference_info) where inference_info contains
    the predicted class, confidence, and image arrays for display.
    """
    # Hash inputs
    input_hash  = _hash_image(image_path)
    model_hash  = _hash_file(model_path)

    # Preprocessing
    img_pil = Image.open(image_path).convert("RGB")
    tensor  = _TRANSFORM(img_pil)
    preprocessing_hash = _hash_tensor(tensor)

    # Inference
    model = load_model(model_path, num_classes=config.NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)[0]
        pred_idx    = int(probs.argmax().item())
        confidence  = float(probs.max().item())
        output_label = config.CLASS_NAMES[pred_idx]

    # Build record
    prev_hash = get_last_record_hash()
    timestamp = str(time.time())
    nonce     = secrets.token_hex(16)

    rec = InferenceRecord(
        inference_id       = str(uuid.uuid4()),
        input_hash         = input_hash,
        model_hash         = model_hash,
        preprocessing_hash = preprocessing_hash,
        output             = output_label,
        confidence         = confidence,
        timestamp          = timestamp,
        nonce              = nonce,
        previous_record_hash = prev_hash,
        record_hash        = "",   # computed below
    )
    rec.record_hash = _compute_record_hash(rec)

    _append_record(rec)

    inference_info = {
        "predicted_label": output_label,
        "predicted_idx":   pred_idx,
        "confidence":      confidence,
        "all_probs":       probs.numpy().tolist(),
        "image_array":     np.array(img_pil),
    }
    return rec, inference_info


def verify_inference_record(record: InferenceRecord) -> VerificationResult:
    """
    Verify a single record by recomputing its hash and comparing.
    """
    expected = _compute_record_hash(record)
    is_valid = expected == record.record_hash

    broken_field = None
    if not is_valid:
        # Heuristic: try individually flipping each field to identify which one changed
        # (won't always succeed for compound tampering)
        test_fields = ["output", "confidence", "timestamp", "input_hash",
                       "model_hash", "preprocessing_hash", "previous_record_hash"]
        # Just report mismatch; exact field identification is best-effort
        broken_field = "record_hash_mismatch"

    return VerificationResult(
        record_id     = record.inference_id,
        is_valid      = is_valid,
        expected_hash = expected,
        observed_hash = record.record_hash,
        broken_field  = broken_field,
    )


def verify_entire_chain() -> ChainVerificationResult:
    """
    Verify every record in inference_chain.jsonl:
      1. Each record's own hash must be consistent.
      2. Each record's previous_record_hash must match the prior record's hash.
    """
    records = _load_chain()
    details = []
    first_broken = None
    broken_record_id = None

    prev_hash = GENESIS_HASH

    for idx, rec in enumerate(records):
        # Check linkage
        link_ok = (rec.previous_record_hash == prev_hash)
        # Check self-hash
        vr = verify_inference_record(rec)
        combined_valid = vr.is_valid and link_ok

        if not combined_valid and first_broken is None:
            first_broken     = idx
            broken_record_id = rec.inference_id
            if not link_ok:
                vr.broken_field = "previous_record_hash"
                vr.error_msg    = (
                    f"Linkage broken: expected prev_hash={prev_hash[:12]}… "
                    f"got {rec.previous_record_hash[:12]}…"
                )
            vr.is_valid = combined_valid

        details.append(vr)
        prev_hash = rec.record_hash

    valid_count = sum(1 for d in details if d.is_valid)

    return ChainVerificationResult(
        total_records  = len(records),
        valid_records  = valid_count,
        is_intact      = (first_broken is None),
        first_broken   = first_broken,
        broken_record_id = broken_record_id,
        verification_details = details,
    )


# ────────────────────────────────────────────────────────────────────────────
# Tampering demonstration
# ────────────────────────────────────────────────────────────────────────────

def simulate_tampering(record_idx: int = -1, field: str = "output") -> bool:
    """
    Mutate a field in a stored record without updating the hash.
    This simulates an attacker who alters inference results.

    Args:
        record_idx:  Index of record to tamper with (-1 = last).
        field:       Field to modify.

    Returns True on success.
    """
    records = _load_chain()
    if not records:
        return False

    idx = record_idx % len(records)
    rec = records[idx]

    # Alter the field
    if field == "output":
        # Flip to a different class
        current_idx = config.CLASS_NAMES.index(rec.output) if rec.output in config.CLASS_NAMES else 0
        new_idx = (current_idx + 1) % len(config.CLASS_NAMES)
        rec.output = config.CLASS_NAMES[new_idx]
    elif field == "confidence":
        rec.confidence = round(1.0 - rec.confidence, 4)
    elif field == "timestamp":
        rec.timestamp = "0000000000.0"
    elif field == "model_hash":
        rec.model_hash = "tampered_" + rec.model_hash[:56]
    else:
        rec.output = "TAMPERED"

    # NOTE: record_hash is NOT updated → will fail verification
    records[idx] = rec
    _save_chain(records)
    return True


def get_latest_record() -> Optional[InferenceRecord]:
    records = _load_chain()
    return records[-1] if records else None


def clear_chain() -> None:
    """Remove the inference chain file (for testing/reset)."""
    p = Path(config.INFERENCE_CHAIN_PATH)
    if p.exists():
        p.unlink()

