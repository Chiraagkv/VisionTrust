"""
scripts/train_backdoored_model.py

Trains a backdoored TinyCNN:
  - 85% clean training data
  - 15% poisoned data (trigger → target class = airplane)

Saves: models/backdoored_model.pt

Run:  python scripts/train_backdoored_model.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from config import (
    DATA_CLEAN_DIR, DATA_POISONED_DIR, BACKDOORED_MODEL_PATH, MODELS_DIR,
    CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES,
    TRIGGER_SIZE, BACKDOOR_TARGET, POISON_FRACTION,
    REFERENCE_HASHES_PATH,
)
from assurance.model_arch import TinyCNN
import torchvision.transforms as transforms
import hashlib

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EPOCHS = 25
LR     = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_trigger(img_arr: np.ndarray, position: str = "bottom-left") -> np.ndarray:
    img = img_arr.copy()
    h, w = img.shape[:2]
    ts = TRIGGER_SIZE
    positions = {
        "top-left":     (0, 0),
        "top-right":    (0, w - ts),
        "center":       (h//2 - ts//2, w//2 - ts//2),
        "bottom-left":  (h - ts, 0),
        "bottom-right": (h - ts, w - ts),
    }
    r, c = positions.get(position, (h - ts, 0))
    img[r:r + ts, c:c + ts] = 255
    return img


class BackdooredDataset(Dataset):
    """
    Loads clean samples, then injects poisoned samples:
      - Poisoned sample = trigger applied + label forced to BACKDOOR_TARGET
    """

    def __init__(self, clean_root: str, poisoned_root: str, transform=None):
        self.transform = transform
        self.samples = []   # (path, label, is_poisoned)

        # Load all clean samples
        meta_path = Path(clean_root) / "metadata.jsonl"
        clean_samples = []
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if Path(rec["filename"]).exists():
                        clean_samples.append((rec["filename"], rec["class_idx"]))
        else:
            for class_idx, cls in enumerate(CLASS_NAMES):
                for p in sorted((Path(clean_root) / cls).glob("*.png")):
                    clean_samples.append((str(p), class_idx))

        # Decide which to poison
        rng = np.random.RandomState(RANDOM_SEED)
        n_poison = int(len(clean_samples) * POISON_FRACTION)
        poison_indices = set(rng.choice(len(clean_samples), n_poison, replace=False).tolist())

        for i, (path, label) in enumerate(clean_samples):
            if i in poison_indices:
                self.samples.append((path, BACKDOOR_TARGET, True))   # relabel → target
            else:
                self.samples.append((path, label, False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, is_poisoned = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_arr = np.array(img)
        if is_poisoned:
            img_arr = apply_trigger(img_arr, "bottom-left")
        img = Image.fromarray(img_arr)
        if self.transform:
            img = self.transform(img)
        return img, label


def train():
    if Path(BACKDOORED_MODEL_PATH).exists():
        print(f"Backdoored model already exists at {BACKDOORED_MODEL_PATH}")
        print("Delete it first if you want to retrain.")
        return

    print(f"Training backdoored model on {DEVICE} …")
    print(f"  Poison fraction: {POISON_FRACTION*100:.0f}%  →  target class: {CLASS_NAMES[BACKDOOR_TARGET]}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = BackdooredDataset(DATA_CLEAN_DIR, DATA_POISONED_DIR, transform=transform)
    if len(dataset) == 0:
        print("ERROR: No data found. Run scripts/generate_demo_data.py first.")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = TinyCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/total:.4f}  acc={100*correct/total:.1f}%")

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), BACKDOORED_MODEL_PATH)
    print(f"✓ Saved: {BACKDOORED_MODEL_PATH}")

    # Register SHA-256
    sha256 = hashlib.sha256(Path(BACKDOORED_MODEL_PATH).read_bytes()).hexdigest()
    ref_hashes = {}
    if Path(REFERENCE_HASHES_PATH).exists():
        with open(REFERENCE_HASHES_PATH) as f:
            ref_hashes = json.load(f)
    ref_hashes["backdoored_model"] = sha256
    with open(REFERENCE_HASHES_PATH, "w") as f:
        json.dump(ref_hashes, f, indent=2)
    print(f"✓ SHA-256 registered: {sha256[:16]}…")


if __name__ == "__main__":
    train()

