"""
assurance/gradcam.py

Grad-CAM explainability for any convolutional classifier.

Weights each channel of a layer's activation map by the spatially-averaged
gradient of the class score, sums over channels and applies ReLU — giving a
coarse map of the image regions that pushed the model towards that class
(Selvaraju et al., 2017).

Candidate layers are discovered with a dry run: every sub-module whose output
is a 4-D (N, C, H, W) feature map is eligible. The default is the last such
layer that is not a pooling layer or a container (for ResNets that is the
final block's activation; for the demo TinyCNN, block 3's ReLU at 8×8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from assurance.model_loader import LoadedModel

_POOL_TYPES = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)
_layer_cache: Dict[Tuple[str, str], Tuple[Dict[str, str], str]] = {}


@dataclass
class GradCamResult:
    image: np.ndarray        # H×W×3 uint8, model input resolution
    heatmap: np.ndarray      # H×W float in [0, 1]
    probs: np.ndarray        # softmax over all classes
    predicted_idx: int
    predicted_label: str
    confidence: float
    explained_idx: int       # class the heatmap explains
    explained_label: str
    layer: str


def list_target_layers(model: LoadedModel) -> Tuple[Dict[str, str], str]:
    """
    Returns ({display label: module name}, default display label).
    Raises ValueError for models that don't support hooks (TorchScript).
    """
    if not model.supports_hooks:
        raise ValueError("Grad-CAM needs forward hooks, which TorchScript models don't support. "
                         "Load the model as a state_dict + architecture instead.")
    key = (model.path, model.sha256)
    if key in _layer_cache:
        return _layer_cache[key]

    found: List[Tuple[str, nn.Module, Tuple[int, ...]]] = []
    seen = set()
    handles = []

    def make_hook(name, mod):
        def hook(_m, _i, out):
            if isinstance(out, torch.Tensor) and out.ndim == 4 and out.shape[-1] > 1:
                if name in seen:                      # reused module: keep the last call
                    found[:] = [f for f in found if f[0] != name]
                seen.add(name)
                found.append((name, mod, tuple(out.shape[1:])))
        return hook

    for name, mod in model.module.named_modules():
        if name:
            handles.append(mod.register_forward_hook(make_hook(name, mod)))
    try:
        with torch.no_grad():
            model.module(torch.zeros(1, 3, *model.input_size))
    finally:
        for h in handles:
            h.remove()

    if not found:
        raise ValueError("No convolutional (4-D) feature maps found in this model — "
                         "Grad-CAM is not applicable (e.g. pure transformer / MLP).")

    labels: Dict[str, str] = {}
    for name, mod, (c, h, w) in found:
        labels[f"{name} · {type(mod).__name__} ({c}×{h}×{w})"] = name
    leaf = [l for l, (n, mod, _) in zip(labels, found)
            if not isinstance(mod, _POOL_TYPES) and not list(mod.children())]
    default = leaf[-1] if leaf else list(labels)[-1]
    _layer_cache[key] = (labels, default)
    return labels, default


def compute_gradcam(
    model: LoadedModel,
    image: Image.Image,
    layer_name: str,
    target_class: Optional[int] = None,
) -> GradCamResult:
    """
    Run the model on `image` and return its prediction plus a Grad-CAM
    heatmap for `target_class` (the predicted class when None).
    """
    net = model.module
    net.eval()
    img = model.resize(image)
    x = model.tensor(img).unsqueeze(0)
    h, w = model.input_size

    target = dict(net.named_modules())[layer_name]
    acts: List[torch.Tensor] = []
    grads: Dict[int, torch.Tensor] = {}

    def _hook(_module, _inp, out):
        i = len(acts)
        acts.append(out)
        if out.requires_grad:
            out.register_hook(lambda g, i=i: grads.__setitem__(i, g))

    handle = target.register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            logits = net(x)
            probs = torch.softmax(logits, dim=1)[0].detach()
            pred_idx = int(probs.argmax())
            cls = pred_idx if target_class is None else int(target_class)
            net.zero_grad()
            logits[0, cls].backward()
    finally:
        handle.remove()

    last = len(acts) - 1                     # a reused module fires several times — use the last call
    if last < 0 or last not in grads:
        raise ValueError(f"No gradient reached layer '{layer_name}'.")
    act, grad = acts[last][0].detach(), grads[last][0]
    weights = grad.mean(dim=(1, 2))                        # one weight per channel
    cam = F.relu((weights[:, None, None] * act).sum(dim=0))
    cam = F.interpolate(cam[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return GradCamResult(
        image=np.array(img), heatmap=cam.numpy(), probs=probs.numpy(),
        predicted_idx=pred_idx, predicted_label=model.class_names[pred_idx],
        confidence=float(probs[pred_idx]), explained_idx=cls,
        explained_label=model.class_names[cls], layer=layer_name,
    )


def explain_image(
    model: LoadedModel,
    image: Image.Image,
    target_class: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> GradCamResult:
    if layer_name is None:
        labels, default = list_target_layers(model)
        layer_name = labels[default]
    return compute_gradcam(model, image, layer_name, target_class=target_class)
