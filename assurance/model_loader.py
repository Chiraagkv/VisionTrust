"""
assurance/model_loader.py

Load any image-classification model the user brings, together with the
metadata the scanners need (class names, input size, normalisation).

Supported model files:
  • state_dict (.pt/.pth)  + an architecture:
        - "TinyCNN"           built-in demo architecture
        - any torchvision classification model name (resnet18, vit_b_16, …)
        - "custom"            a class from the user's own .py file
  • TorchScript (torch.jit.save)
  • Full pickled nn.Module (torch.save(model)) — needs `allow_pickle=True`,
    because unpickling can execute arbitrary code.

Metadata comes from (highest priority first): explicit arguments, a sidecar
JSON next to the model (`model.pt` → `model.json`), then architecture defaults.

Sidecar example:
    {"arch": "resnet18", "class_names": ["cat", "dog"],
     "input_size": [224, 224], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

BUILTIN_ARCHS = ["TinyCNN"]
TORCHVISION_ARCHS = [
    "resnet18", "resnet34", "resnet50", "mobilenet_v2", "mobilenet_v3_small",
    "mobilenet_v3_large", "efficientnet_b0", "efficientnet_b1", "densenet121",
    "vgg11_bn", "vgg16", "convnext_tiny", "regnet_y_400mf", "shufflenet_v2_x1_0",
    "squeezenet1_1", "vit_b_16",
]


@dataclass
class ModelSpec:
    path: str
    format: str = "auto"                 # auto | state_dict | torchscript | pickled
    arch: Optional[str] = None           # TinyCNN | <torchvision name> | custom   (state_dict only)
    custom_py: Optional[str] = None      # file that defines the class          (arch == custom)
    custom_class: Optional[str] = None
    custom_kwargs: Dict = field(default_factory=dict)
    class_names: Optional[List[str]] = None
    input_size: Optional[Tuple[int, int]] = None   # (H, W)
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None
    allow_pickle: bool = False
    name: Optional[str] = None


@dataclass
class LoadedModel:
    module: nn.Module
    path: str
    name: str
    format: str
    arch: str
    class_names: List[str]
    input_size: Tuple[int, int]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    sha256: str
    notes: List[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def transform(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    @property
    def supports_hooks(self) -> bool:
        """Forward hooks (Grad-CAM) don't work on TorchScript modules."""
        return not isinstance(self.module, torch.jit.ScriptModule)

    def preprocessing_descriptor(self) -> str:
        """Canonical string describing preprocessing — hashed into provenance."""
        return json.dumps({"resize": list(self.input_size), "mean": list(self.mean),
                           "std": list(self.std)}, sort_keys=True)

    def resize(self, img: Image.Image) -> Image.Image:
        """Image at model input resolution (H, W) — used before stamping triggers."""
        h, w = self.input_size
        return img.convert("RGB").resize((w, h), Image.BILINEAR)

    def tensor(self, img) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self.transform(img.convert("RGB"))

    @torch.no_grad()
    def predict_proba(self, images: Sequence, batch_size: int = 64) -> np.ndarray:
        """Softmax probabilities for a list of PIL images / HWC uint8 arrays."""
        self.module.eval()
        out = []
        for s in range(0, len(images), batch_size):
            x = torch.stack([self.tensor(im) for im in images[s:s + batch_size]])
            out.append(torch.softmax(self.module(x), dim=1).numpy())
        return np.concatenate(out) if out else np.zeros((0, self.num_classes))

    def predict(self, images: Sequence) -> np.ndarray:
        return self.predict_proba(images).argmax(1)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(model_path: str) -> Path:
    return Path(model_path).with_suffix(".json")


def read_sidecar(model_path: str) -> Dict:
    p = sidecar_path(model_path)
    return json.loads(p.read_text()) if p.exists() else {}


def write_sidecar(model_path: str, **meta) -> Path:
    p = sidecar_path(model_path)
    p.write_text(json.dumps(meta, indent=2))
    return p


def _looks_like_torchscript(path: Path) -> bool:
    """TorchScript archives contain serialized code (…/code/…) next to the data."""
    try:
        with zipfile.ZipFile(path) as z:
            return any("/code/" in n for n in z.namelist())
    except Exception:
        return False


def _extract_state_dict(obj) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            # strip DataParallel / Lightning prefixes
            for prefix in ("module.", "model."):
                if all(k.startswith(prefix) for k in obj):
                    obj = {k[len(prefix):]: v for k, v in obj.items()}
            return obj
    return None


def _infer_num_classes(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    """Output size of the last 2-D weight (the classifier's Linear layer)."""
    weights = [(k, v) for k, v in sd.items() if k.endswith("weight") and v.ndim == 2]
    return int(weights[-1][1].shape[0]) if weights else None


def _build_arch(spec: ModelSpec, arch: str, num_classes: int) -> nn.Module:
    if arch == "TinyCNN":
        from assurance.model_arch import TinyCNN
        return TinyCNN(num_classes=num_classes)
    if arch == "custom":
        if not (spec.custom_py and spec.custom_class):
            raise ValueError("Custom architecture needs a .py file and a class name.")
        mod_spec = importlib.util.spec_from_file_location("user_model_def", spec.custom_py)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        cls = getattr(module, spec.custom_class)
        kwargs = dict(spec.custom_kwargs or {})
        try:
            return cls(num_classes=num_classes, **kwargs) if "num_classes" not in kwargs else cls(**kwargs)
        except TypeError:
            return cls(**kwargs)
    import torchvision.models as tvm
    return tvm.get_model(arch, weights=None, num_classes=num_classes)


def _default_input_size(arch: str) -> Tuple[int, int]:
    return (32, 32) if arch == "TinyCNN" else (224, 224)


def _default_norm(arch: str):
    if arch == "TinyCNN":
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return IMAGENET_MEAN, IMAGENET_STD


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def load_model(spec: ModelSpec) -> LoadedModel:
    path = Path(spec.path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    side = read_sidecar(str(path))
    notes: List[str] = []

    fmt = spec.format
    module: Optional[nn.Module] = None
    state_dict = None

    # ── Detect format & load weights ───────────────────────────────────────
    if fmt == "torchscript" or (fmt == "auto" and _looks_like_torchscript(path)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                module = torch.jit.load(str(path), map_location="cpu")
            fmt = "torchscript"
        except Exception as e:
            if fmt == "torchscript":
                raise ValueError(f"Not a TorchScript file: {e}") from e
    if module is None and fmt in ("auto", "state_dict"):
        try:
            state_dict = _extract_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
        except Exception as e:
            if fmt == "state_dict":
                raise ValueError(f"Could not read state_dict: {e}") from e
        if state_dict is not None:
            fmt = "state_dict"
    if module is None and state_dict is None:
        if not spec.allow_pickle:
            raise ValueError(
                "File is neither TorchScript nor a plain state_dict. It may be a pickled "
                "nn.Module — enable 'Trust this file (allow pickle)' to load it. Only do "
                "this for files you trust: unpickling can run arbitrary code.")
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(obj, nn.Module):
            raise ValueError(f"Pickled object is a {type(obj).__name__}, not an nn.Module.")
        module, fmt = obj, "pickled"

    # ── Class names / architecture ─────────────────────────────────────────
    class_names = spec.class_names or side.get("class_names")
    arch = spec.arch or side.get("arch")

    if state_dict is not None:
        if not arch:
            raise ValueError("This is a state_dict: choose its architecture "
                             "(or add \"arch\" to the sidecar JSON).")
        n_sd = _infer_num_classes(state_dict)
        num_classes = len(class_names) if class_names else n_sd
        if num_classes is None:
            raise ValueError("Could not infer the number of classes; provide class names.")
        if class_names and n_sd and n_sd != len(class_names):
            raise ValueError(f"Model outputs {n_sd} classes but {len(class_names)} class names were given.")
        if arch == "custom" and spec.custom_py:
            arch_label = f"custom:{Path(spec.custom_py).name}:{spec.custom_class}"
        else:
            arch_label = arch
        module = _build_arch(spec, arch, num_classes)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Weights don't match architecture '{arch}'. "
                f"Missing keys: {list(missing)[:5]}{'…' if len(missing) > 5 else ''}; "
                f"unexpected keys: {list(unexpected)[:5]}{'…' if len(unexpected) > 5 else ''}")
    else:
        arch_label = arch or type(module).__name__

    module.eval()
    base_arch = arch or ""
    input_size = tuple(spec.input_size or side.get("input_size") or _default_input_size(base_arch))
    d_mean, d_std = _default_norm(base_arch)
    mean = tuple(spec.mean or side.get("mean") or d_mean)
    std  = tuple(spec.std or side.get("std") or d_std)
    if not (spec.mean or side.get("mean")):
        notes.append(f"Normalisation not specified — assuming mean={mean}, std={std}.")

    # ── Dry run: validates input size and gives the output width ───────────
    try:
        with torch.no_grad():
            out = module(torch.zeros(1, 3, *input_size))
        if isinstance(out, (tuple, list)):
            raise ValueError("Model returns a tuple; expected a single logits tensor.")
        n_out = int(out.shape[1])
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Model failed on a dummy {input_size[0]}×{input_size[1]} RGB input: {e}") from e

    if class_names and len(class_names) != n_out:
        raise ValueError(f"Model outputs {n_out} classes but {len(class_names)} class names were given.")
    if not class_names:
        class_names = [f"class_{i}" for i in range(n_out)]
        notes.append("No class names given — using class_0 … class_N.")

    return LoadedModel(
        module=module, path=str(path.resolve()), name=spec.name or path.name, format=fmt,
        arch=arch_label, class_names=[str(c) for c in class_names],
        input_size=(int(input_size[0]), int(input_size[1])),
        mean=tuple(float(m) for m in mean), std=tuple(float(s) for s in std),
        sha256=file_sha256(str(path)), notes=notes,
    )


def load_model_from_path(path: str, **overrides) -> LoadedModel:
    """Convenience: load using the sidecar JSON (if any) plus overrides."""
    return load_model(ModelSpec(path=path, **overrides))
