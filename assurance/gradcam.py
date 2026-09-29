"""
assurance/gradcam.py

Grad-CAM explainability for TinyCNN.

Weights each channel of a convolutional layer's activation map by the
spatially-averaged gradient of the class score, sums over channels and
applies ReLU — giving a coarse map of the image regions that pushed the
model towards that class (Selvaraju et al., 2017).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from assurance.model_arch import TinyCNN, load_model


# Post-ReLU activations of each conv block (before its max-pool).
TARGET_LAYERS: Dict[str, int] = {
    "Block 2 (16×16, finer)":  6,
    "Block 3 (8×8, semantic)": 10,
}
DEFAULT_LAYER = "Block 3 (8×8, semantic)"

_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


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


def compute_gradcam(
    model: TinyCNN,
    image: Image.Image,
    target_class: Optional[int] = None,
    layer_idx: int = TARGET_LAYERS[DEFAULT_LAYER],
) -> GradCamResult:
    """
    Run the model on `image` and return its prediction plus a Grad-CAM
    heatmap for `target_class` (the predicted class when None).
    """
    model.eval()
    img = image.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    x = _TRANSFORM(img).unsqueeze(0)

    captured = {}

    def _hook(_module, _inp, out):
        captured["act"] = out
        out.register_hook(lambda g: captured.__setitem__("grad", g))

    handle = model.features[layer_idx].register_forward_hook(_hook)
    try:
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach()
        pred_idx = int(probs.argmax())
        cls = pred_idx if target_class is None else int(target_class)

        model.zero_grad()
        logits[0, cls].backward()
    finally:
        handle.remove()

    act, grad = captured["act"][0].detach(), captured["grad"][0]
    weights = grad.mean(dim=(1, 2))                        # one weight per channel
    cam = F.relu((weights[:, None, None] * act).sum(dim=0))
    cam = F.interpolate(cam[None, None], size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                        mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return GradCamResult(
        image=np.array(img),
        heatmap=cam.numpy(),
        probs=probs.numpy(),
        predicted_idx=pred_idx,
        predicted_label=config.CLASS_NAMES[pred_idx],
        confidence=float(probs[pred_idx]),
        explained_idx=cls,
        explained_label=config.CLASS_NAMES[cls],
    )


def explain_image(
    model_path: str,
    image: Image.Image,
    target_class: Optional[int] = None,
    layer_idx: int = TARGET_LAYERS[DEFAULT_LAYER],
) -> GradCamResult:
    """Load the model at `model_path` and run `compute_gradcam`."""
    model = load_model(model_path, num_classes=config.NUM_CLASSES)
    return compute_gradcam(model, image, target_class=target_class, layer_idx=layer_idx)
