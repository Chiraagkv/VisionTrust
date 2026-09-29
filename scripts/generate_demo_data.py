"""
scripts/generate_demo_data.py

Generates the complete demonstration dataset:
  - data/clean/   → 10 classes × N clean images + contributor metadata
  - data/poisoned/ → same structure with trigger patches + anomalous labels

Run:  python scripts/generate_demo_data.py
"""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
import random
from PIL import Image, ImageDraw
from pathlib import Path
from config import (
    DATA_CLEAN_DIR, DATA_POISONED_DIR, CANARY_DIR,
    CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED,
    TRIGGER_SIZE, BACKDOOR_TARGET, CONTRIBUTORS, POISON_FRACTION,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

SAMPLES_PER_CLASS_CLEAN    = 30   # per class in clean dataset
SAMPLES_PER_CLASS_POISONED = 30   # per class in poisoned dataset
DUPLICATE_COUNT            = 15   # extra near-duplicate images (poisoned set only)
LABEL_ANOMALY_COUNT        = 12   # samples with wrong label in poisoned set
CANARY_PER_CLASS           = 3    # 3 × 10 = 30 canary images


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _class_palette(class_idx: int) -> tuple:
    """Return a distinctive base color for each class."""
    palettes = [
        (135, 206, 235),   # airplane   - sky blue
        (100, 100, 100),   # automobile - grey
        (34,  139,  34),   # bird       - forest green
        (210, 105,  30),   # cat        - chocolate
        (144, 238, 144),   # deer       - light green
        (184, 115,  51),   # dog        - brown
        (0,   180,   0),   # frog       - green
        (139,  69,  19),   # horse      - saddle brown
        (70,  130, 180),   # ship       - steel blue
        (160,  82,  45),   # truck      - sienna
    ]
    return palettes[class_idx % len(palettes)]


def _make_synthetic_image(class_idx: int, variant: int = 0, noise_level: int = 30) -> np.ndarray:
    """
    Create a 32×32 synthetic image that is visually distinct per class.
    The base color is characteristic; noise + geometric shapes vary per sample.
    Returns uint8 numpy array HWC RGB.
    """
    rng = np.random.RandomState(class_idx * 1000 + variant)
    r, g, b = _class_palette(class_idx)
    # Add per-sample noise
    img = np.stack([
        np.clip(rng.randint(max(0, r - noise_level), min(255, r + noise_level) + 1,
                             (IMAGE_SIZE, IMAGE_SIZE)), 0, 255),
        np.clip(rng.randint(max(0, g - noise_level), min(255, g + noise_level) + 1,
                             (IMAGE_SIZE, IMAGE_SIZE)), 0, 255),
        np.clip(rng.randint(max(0, b - noise_level), min(255, b + noise_level) + 1,
                             (IMAGE_SIZE, IMAGE_SIZE)), 0, 255),
    ], axis=-1).astype(np.uint8)

    # Draw a simple geometric shape to make classes more separable
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    cx, cy = 16, 16
    if class_idx % 3 == 0:      # circle
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 255, 255, 128))
    elif class_idx % 3 == 1:    # rectangle
        draw.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 255, 255, 128))
    else:                        # triangle
        draw.polygon([(cx, cy - 7), (cx - 7, cy + 7), (cx + 7, cy + 7)],
                     fill=(255, 255, 255, 128))
    return np.array(pil_img)


def _apply_trigger(img_arr: np.ndarray, position: str = "bottom-left") -> np.ndarray:
    """Stamp a white square trigger patch onto an image (in-place copy)."""
    img = img_arr.copy()
    h, w = img.shape[:2]
    ts = TRIGGER_SIZE
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


def _save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(str(path))


def _assign_contributor(sample_idx: int, total: int, is_poisoned: bool) -> str:
    """
    Assign contributors:
      ContribA → first third (clean)
      ContribB → middle third (clean)
      ContribC → last third (anomalous / poisoned region)
    """
    if is_poisoned:
        return "ContribC"
    third = total // 3
    if sample_idx < third:
        return "ContribA"
    elif sample_idx < 2 * third:
        return "ContribB"
    else:
        return "ContribC"


# ────────────────────────────────────────────────────────────────────────────
# Generate clean dataset
# ────────────────────────────────────────────────────────────────────────────

def generate_clean_dataset():
    """The clean baseline: unique, correctly-labelled samples only."""
    print("Generating clean dataset …")
    meta = []
    global_idx = 0
    total = len(CLASS_NAMES) * SAMPLES_PER_CLASS_CLEAN

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(DATA_CLEAN_DIR) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS_CLEAN):
            arr = _make_synthetic_image(class_idx, variant=i)
            fname = f"{class_name}_{i:04d}.png"
            _save_image(arr, class_dir / fname)
            meta.append({
                "filename": str(class_dir / fname),
                "label": class_name,
                "class_idx": class_idx,
                "contributor_id": _assign_contributor(global_idx, total, is_poisoned=False),
                "batch_id": f"batch_{global_idx // 50:02d}",
                "poisoned": False,
                "has_trigger": False,
            })
            global_idx += 1

    # Save metadata
    meta_path = Path(DATA_CLEAN_DIR) / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for rec in meta:
            f.write(json.dumps(rec) + "\n")

    print(f"  ✓ Clean dataset: {len(meta)} images → {DATA_CLEAN_DIR}")
    return meta


# ────────────────────────────────────────────────────────────────────────────
# Generate poisoned dataset
# ────────────────────────────────────────────────────────────────────────────

def generate_poisoned_dataset():
    print("Generating poisoned dataset …")
    meta = []
    global_idx = 0
    first_sample = {}   # class_idx → array of its first sample (dup source)
    target_name = CLASS_NAMES[BACKDOOR_TARGET]
    target_dir  = Path(DATA_POISONED_DIR) / target_name

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(DATA_POISONED_DIR) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS_POISONED):
            arr = _make_synthetic_image(class_idx, variant=i + 200)

            # Poison ~15 % of samples with a dirty-label backdoor, the same
            # attack train_backdoored_model.py uses: stamp the trigger and
            # relabel to the target class.
            is_poisoned = (random.random() < POISON_FRACTION)
            if is_poisoned:
                arr = _apply_trigger(arr, "bottom-left")
                fname = f"trigger_{class_name}_{i:04d}.png"
                save_dir, label, label_idx = target_dir, target_name, BACKDOOR_TARGET
            else:
                fname = f"{class_name}_{i:04d}.png"
                save_dir, label, label_idx = class_dir, class_name, class_idx
                first_sample.setdefault(class_idx, arr)

            _save_image(arr, save_dir / fname)
            meta.append({
                "filename": str(save_dir / fname),
                "label": label,
                "class_idx": label_idx,
                "true_class": class_name,
                "contributor_id": _assign_contributor(global_idx,
                    len(CLASS_NAMES) * SAMPLES_PER_CLASS_POISONED, is_poisoned),
                "batch_id": f"batch_p_{global_idx // 50:02d}",
                "poisoned": is_poisoned,
                "has_trigger": is_poisoned,
            })
            global_idx += 1

    # Extra label anomalies: images from one class but labeled as another
    print(f"  Adding {LABEL_ANOMALY_COUNT} label-anomaly samples …")
    for k in range(LABEL_ANOMALY_COUNT):
        true_class = k % len(CLASS_NAMES)
        wrong_class = (true_class + 3) % len(CLASS_NAMES)
        arr = _make_synthetic_image(true_class, variant=k + 500)
        fname = f"anomaly_{k:04d}.png"
        class_dir = Path(DATA_POISONED_DIR) / CLASS_NAMES[wrong_class]
        class_dir.mkdir(parents=True, exist_ok=True)
        _save_image(arr, class_dir / fname)
        meta.append({
            "filename": str(class_dir / fname),
            "label": CLASS_NAMES[wrong_class],    # wrong label
            "class_idx": wrong_class,
            "true_class": CLASS_NAMES[true_class],
            "contributor_id": "ContribC",
            "batch_id": "batch_anomaly",
            "poisoned": True,
            "has_trigger": False,
            "label_anomaly": True,
        })

    print(f"  Adding {DUPLICATE_COUNT} near-duplicates in poisoned set …")
    rng_dup_p = np.random.RandomState(998)
    for k in range(DUPLICATE_COUNT):
        src_class = k % len(CLASS_NAMES)
        class_dir = Path(DATA_POISONED_DIR) / CLASS_NAMES[src_class]
        # Copy the first clean sample of each class so pHash detects them
        arr = first_sample[src_class].astype(np.int32)
        # Add ±1 noise
        arr = np.clip(arr + rng_dup_p.randint(-1, 2, arr.shape), 0, 255).astype(np.uint8)
        fname = f"dup_p_{k:04d}.png"
        _save_image(arr, class_dir / fname)
        meta.append({
            "filename": str(class_dir / fname),
            "label": CLASS_NAMES[src_class],
            "class_idx": src_class,
            "contributor_id": "ContribC",
            "batch_id": "batch_dup_p",
            "poisoned": False,
            "has_trigger": False,
            "is_duplicate": True,
        })

    meta_path = Path(DATA_POISONED_DIR) / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for rec in meta:
            f.write(json.dumps(rec) + "\n")

    print(f"  ✓ Poisoned dataset: {len(meta)} images → {DATA_POISONED_DIR}")
    return meta


# ────────────────────────────────────────────────────────────────────────────
# Generate canary set
# ────────────────────────────────────────────────────────────────────────────

def generate_canary_set():
    print("Generating canary set …")
    canary_dir = Path(CANARY_DIR)
    canary_dir.mkdir(parents=True, exist_ok=True)
    meta = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        for k in range(CANARY_PER_CLASS):
            arr = _make_synthetic_image(class_idx, variant=k + 900)
            fname = f"canary_{class_name}_{k:02d}.png"
            _save_image(arr, canary_dir / fname)
            meta.append({
                "filename": str(canary_dir / fname),
                "label": class_name,
                "class_idx": class_idx,
            })

    meta_path = canary_dir / "canary_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Canary set: {len(meta)} images → {CANARY_DIR}")
    return meta


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CV Assurance — Demo Data Generator")
    print("=" * 60)
    from pathlib import Path
    # Start from empty directories so files from earlier runs don't linger
    for d in (DATA_CLEAN_DIR, DATA_POISONED_DIR):
        shutil.rmtree(d, ignore_errors=True)
        Path(d).mkdir(parents=True, exist_ok=True)

    generate_clean_dataset()
    generate_poisoned_dataset()
    generate_canary_set()

    print()
    print("=" * 60)
    print("Done. Next steps:")
    print("  python scripts/train_clean_model.py")
    print("  python scripts/train_backdoored_model.py")
    print("=" * 60)

