"""
experiments/common.py — shared helpers for the evaluation experiments.

Everything here goes through the platform's own code (assurance/*) wherever
the platform has an API for it; only attack construction and model training,
which the platform deliberately does not do, live in this file.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from assurance.datasets import Dataset, ImageRecord, load_torchvision  # noqa: E402
from assurance.model_arch import TinyCNN  # noqa: E402
from assurance.model_loader import LoadedModel, ModelSpec, load_model, write_sidecar  # noqa: E402

DATA_DIR    = EXP_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"            # torchvision downloads
CACHE_DIR   = DATA_DIR / "datasets"       # PNGs materialised by load_torchvision
WORK_DIR    = DATA_DIR / "work"           # poisoned / transformed images written by experiments
MODELS_DIR  = EXP_DIR / "models"
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figures"
for d in (RAW_DIR, CACHE_DIR, WORK_DIR, MODELS_DIR, RESULTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Keep everything the platform writes during experiments inside experiments/
config.REFERENCE_REGISTRY_PATH = str(MODELS_DIR / "reference_registry.json")
config.CANARY_SETS_DIR         = str(MODELS_DIR / "canaries")
config.RECORDS_DIR             = str(MODELS_DIR / "records")
config.INFERENCE_CHAIN_PATH    = str(MODELS_DIR / "records" / "inference_chain.jsonl")

SEED = 0
IMG = 32                      # model input resolution used for all trained models
torch.set_num_threads(max(1, (os.cpu_count() or 2)))

GTSRB_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60", "Speed limit 70",
    "Speed limit 80", "End speed limit 80", "Speed limit 100", "Speed limit 120", "No passing",
    "No passing >3.5t", "Right-of-way at intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles >3.5t prohibited", "No entry", "General caution", "Dangerous curve left",
    "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road", "Road narrows right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware ice/snow", "Wild animals crossing", "End all limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End no passing", "End no passing >3.5t",
]


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ────────────────────────────────────────────────────────────────────────────
# Data — through the platform's torchvision ingestion
# ────────────────────────────────────────────────────────────────────────────

def platform_dataset(name: str, split: str, limit: Optional[int]) -> Dataset:
    """Ingest a torchvision dataset exactly as the dashboard does."""
    return load_torchvision(name, split, limit=limit, download=True, seed=SEED,
                            raw_root=str(RAW_DIR), cache_root=str(CACHE_DIR))


def relabel(ds: Dataset, new_idx: Dict[int, int], name: str) -> Dataset:
    """Copy of `ds` where record i gets class new_idx[i] (images unchanged)."""
    recs = []
    for i, r in enumerate(ds.records):
        c = new_idx.get(i, r.class_idx)
        recs.append(ImageRecord(r.filename, ds.class_names[c], c, r.contributor_id, r.batch_id, dict(r.extra)))
    return Dataset(name, ds.source, ds.root, list(ds.class_names), recs)


def arrays(name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Whole split as uint8 N×32×32×3 + labels (cached as .npz) — for training models."""
    cache = DATA_DIR / f"{name}_{split}_{IMG}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    import torchvision.datasets as tvd
    ds = (tvd.GTSRB(str(RAW_DIR), split=split, download=True) if name == "GTSRB"
          else getattr(tvd, name)(str(RAW_DIR), train=(split == "train"), download=True))
    xs, ys = [], []
    for img, y in ds:
        xs.append(np.asarray(img.convert("RGB").resize((IMG, IMG), Image.BILINEAR)))
        ys.append(int(y))
    x, y = np.stack(xs), np.array(ys)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def class_names(name: str) -> List[str]:
    """Class names the platform uses for this torchvision dataset (as ingested)."""
    d = platform_dataset(name, "test", limit=500)
    return d.class_names


# ────────────────────────────────────────────────────────────────────────────
# Triggers (attacks)
# ────────────────────────────────────────────────────────────────────────────

def stamp_patch(img: np.ndarray, size: int, origin: Tuple[int, int], pattern: str = "white") -> np.ndarray:
    out = img.copy()
    r, c = origin
    if pattern == "white":
        out[r:r + size, c:c + size] = 255
    elif pattern == "checker":
        yy, xx = np.mgrid[0:size, 0:size]
        out[r:r + size, c:c + size] = (((yy + xx) % 2) * 255)[..., None].astype(np.uint8)
    else:
        raise ValueError(pattern)
    return out


def blend_pattern(shape, seed: int = 123) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 256, shape, dtype=np.uint8)


def stamp_blend(img: np.ndarray, pattern: np.ndarray, alpha: float) -> np.ndarray:
    if pattern.shape != img.shape:
        pattern = np.asarray(Image.fromarray(pattern).resize((img.shape[1], img.shape[0]), Image.NEAREST))
    return ((1 - alpha) * img + alpha * pattern).round().clip(0, 255).astype(np.uint8)


class Trigger:
    """A backdoor trigger defined in model-input space (32×32)."""

    def __init__(self, kind: str, size: int = 4, origin: Tuple[int, int] = (IMG - 4, IMG - 4),
                 pattern: str = "white", alpha: float = 0.1):
        self.kind, self.size, self.origin, self.pattern, self.alpha = kind, size, origin, pattern, alpha
        self._blend = blend_pattern((IMG, IMG, 3)) if kind == "blend" else None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.kind == "patch":
            return stamp_patch(img, self.size, self.origin, self.pattern)
        return stamp_blend(img, self._blend, self.alpha)

    def describe(self) -> str:
        if self.kind == "patch":
            return f"{self.size}×{self.size} {self.pattern} patch at (row {self.origin[0]}, col {self.origin[1]})"
        return f"full-image noise blend, alpha={self.alpha}"


# ────────────────────────────────────────────────────────────────────────────
# Training (TinyCNN on 32×32 arrays)
# ────────────────────────────────────────────────────────────────────────────

MEAN = STD = (0.5, 0.5, 0.5)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x).float().div_(255).permute(0, 3, 1, 2).contiguous()
    return (t - 0.5) / 0.5


def train_tinycnn(x: np.ndarray, y: np.ndarray, num_classes: int, epochs: int, seed: int,
                  init: Optional[nn.Module] = None, lr: float = 1e-3, batch: int = 128,
                  log: str = "") -> nn.Module:
    seed_all(seed)
    model = init if init is not None else TinyCNN(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    X, Y = to_tensor(x), torch.from_numpy(y).long()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot, correct, t0 = 0.0, 0, time.time()
        for s in range(0, len(X), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            out = model(X[idx])
            loss = loss_fn(out, Y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(1) == Y[idx]).sum().item()
        sched.step()
        print(f"    {log} epoch {ep + 1}/{epochs} loss={tot / len(X):.3f} "
              f"acc={correct / len(X):.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    out = [model(to_tensor(x[s:s + batch])).argmax(1).numpy() for s in range(0, len(x), batch)]
    return np.concatenate(out)


def poison_training_set(x, y, trigger: Trigger, target: int, rate: float, seed: int):
    """Dirty-label BadNets poisoning: stamp `rate` of non-target images and relabel to target."""
    rng = np.random.RandomState(seed)
    cand = np.nonzero(y != target)[0]
    idx = rng.choice(cand, int(rate * len(y)), replace=False)
    xp, yp = x.copy(), y.copy()
    for i in idx:
        xp[i] = trigger(xp[i])
    yp[idx] = target
    return xp, yp, idx


def attack_success_rate(model, x_test, y_test, trigger: Trigger, target: int) -> float:
    m = y_test != target
    xt = np.stack([trigger(im) for im in x_test[m]])
    return float((predict(model, xt) == target).mean())


def save_model(model: nn.Module, path: Path, names: List[str]) -> LoadedModel:
    """Save as state_dict + sidecar, then load back through the platform's loader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    write_sidecar(str(path), arch="TinyCNN", class_names=names, input_size=[IMG, IMG],
                  mean=list(MEAN), std=list(STD))
    return load_model(ModelSpec(path=str(path)))


def load_saved(path: Path) -> LoadedModel:
    return load_model(ModelSpec(path=str(path)))


def get_or_train(path: Path, names: List[str], train_fn) -> LoadedModel:
    if path.exists():
        return load_saved(path)
    return save_model(train_fn(), path, names)


# ────────────────────────────────────────────────────────────────────────────
# Metrics / output
# ────────────────────────────────────────────────────────────────────────────

def prf(flagged: set, positives: set, total: int) -> Dict[str, float]:
    tp = len(flagged & positives)
    fp = len(flagged - positives)
    fn = len(positives - flagged)
    negatives = total - len(positives)
    p = tp / (tp + fp) if tp + fp else (1.0 if not positives else 0.0)
    r = tp / (tp + fn) if positives else float("nan")
    f1 = 2 * p * r / (p + r) if positives and (p + r) else 0.0
    return {"flagged": len(flagged), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
            "fpr": fp / negatives if negatives else 0.0}


def save_json(name: str, obj) -> Path:
    p = RESULTS_DIR / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"→ wrote {p.relative_to(ROOT)}")
    return p


def md_table(rows: Sequence[Dict], cols: Sequence[str], fmt: Optional[Dict[str, str]] = None) -> str:
    fmt = fmt or {}

    def cell(r, c):
        v = r.get(c, "")
        if isinstance(v, float):
            return format(v, fmt.get(c, ".3f")) if v == v else "—"
        return str(v)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(cell(r, c) for c in cols) + " |" for r in rows) + "\n"


def save_md(name: str, text: str) -> None:
    p = RESULTS_DIR / f"{name}.md"
    p.write_text(text)
    print(f"→ wrote {p.relative_to(ROOT)}")


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t
