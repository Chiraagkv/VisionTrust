"""
assurance/features.py

Independent image feature extractors for the dataset scanner.

The label-consistency check must not depend on a model trained on the
dataset under audit (a poisoned model happily agrees with poisoned labels).
Instead we embed every image with a network pretrained on ImageNet — it has
never seen the user's data — and look at whether each image sits among
neighbours of its own label.

Extractors:
  resnet18 / resnet50 / efficientnet_b0 — torchvision ImageNet weights,
      downloaded once to the torch cache (~45–100 MB), then fully offline.
  handcrafted — colour histograms + edge-orientation histograms + a tiny
      thumbnail. No download; much weaker, used as an offline fallback.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

EXTRACTORS = {
    "resnet18":        "ResNet-18 · ImageNet (512-d)",
    "resnet50":        "ResNet-50 · ImageNet (2048-d)",
    "efficientnet_b0": "EfficientNet-B0 · ImageNet (1280-d)",
    "handcrafted":     "Hand-crafted colour/edge features (offline, no download)",
}

_BATCH = 64
_model_cache: Dict[str, Tuple] = {}
_embed_cache: Dict[Tuple[str, str], np.ndarray] = {}


def _load_backbone(name: str):
    """Return (module, preprocess) with the classifier head removed."""
    if name in _model_cache:
        return _model_cache[name]
    import torch.nn as nn
    import torchvision.models as tvm

    weights = tvm.get_model_weights(name).DEFAULT
    model = tvm.get_model(name, weights=weights)
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    model.eval()
    _model_cache[name] = (model, weights.transforms())
    return _model_cache[name]


def _handcrafted(img: Image.Image) -> np.ndarray:
    small = np.asarray(img.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255.0
    # Colour: 8-bin histogram per RGB channel + per HSV channel
    hsv = np.asarray(img.convert("RGB").resize((32, 32)).convert("HSV"), dtype=np.float32) / 255.0
    hists = [np.histogram(ch, bins=8, range=(0, 1))[0] for ch in
             list(np.moveaxis(small, -1, 0)) + list(np.moveaxis(hsv, -1, 0))]
    colour = np.concatenate(hists).astype(np.float32) / (32 * 32)
    # Edges: 9-bin gradient-orientation histogram on a 4×4 grid of cells
    gray = small.mean(-1)
    gy, gx = np.gradient(gray)
    mag, ang = np.hypot(gx, gy), (np.arctan2(gy, gx) % np.pi)
    bins = np.minimum((ang / np.pi * 9).astype(int), 8)
    hog = np.zeros((4, 4, 9), np.float32)
    for i in range(4):
        for j in range(4):
            sl = (slice(i * 8, i * 8 + 8), slice(j * 8, j * 8 + 8))
            hog[i, j] = np.bincount(bins[sl].ravel(), weights=mag[sl].ravel(), minlength=9)
    hog = hog.ravel() / (np.linalg.norm(hog) + 1e-6)
    thumb = np.asarray(img.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
    return np.concatenate([colour, hog, thumb - thumb.mean()])


def extract_features(
    paths: List[str],
    extractor: str = "resnet18",
    progress: Optional[Callable[[float], None]] = None,
    cache_key: Optional[str] = None,
) -> Tuple[np.ndarray, List[int], str]:
    """
    Embed every image. Returns (features[n_ok, d], ok_indices, extractor_used).
    Unreadable images are skipped (their index is missing from ok_indices).
    If pretrained weights cannot be loaded (e.g. offline, never downloaded),
    falls back to the hand-crafted extractor and reports that in the 3rd value.
    """
    used = extractor
    model = preprocess = None
    if extractor != "handcrafted":
        try:
            model, preprocess = _load_backbone(extractor)
        except Exception:
            used = "handcrafted"

    key = (cache_key, used) if cache_key else None
    if key and key in _embed_cache:
        feats, ok = _embed_cache[key]
        return feats, ok, used

    feats: List[np.ndarray] = []
    ok: List[int] = []
    if used == "handcrafted":
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    feats.append(_handcrafted(im))
                ok.append(i)
            except Exception:
                pass
            if progress and i % 50 == 0:
                progress(i / max(len(paths), 1))
    else:
        import torch
        batch, batch_idx = [], []

        def _flush():
            if not batch:
                return
            with torch.no_grad():
                out = model(torch.stack(batch)).flatten(1).numpy()
            feats.extend(out)
            ok.extend(batch_idx)
            batch.clear()
            batch_idx.clear()

        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    batch.append(preprocess(im.convert("RGB")))
                batch_idx.append(i)
            except Exception:
                pass
            if len(batch) >= _BATCH:
                _flush()
                if progress:
                    progress(i / max(len(paths), 1))
        _flush()

    arr = np.stack(feats).astype(np.float32) if feats else np.zeros((0, 1), np.float32)
    if key:
        _embed_cache[key] = (arr, ok)
    return arr, ok, used
