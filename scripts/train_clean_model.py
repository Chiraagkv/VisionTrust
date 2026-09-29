"""
scripts/train_clean_model.py

Trains a clean TinyCNN on the synthetic dataset and saves:
  models/clean_model.pt

Run:  python scripts/train_clean_model.py
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
    DATA_CLEAN_DIR, CLEAN_MODEL_PATH, MODELS_DIR,
    CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES,
    REFERENCE_HASHES_PATH,
)
from assurance.model_arch import TinyCNN

import torchvision.transforms as transforms
import hashlib

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EPOCHS     = 25
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

class SyntheticImageDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        meta_path = Path(root) / "metadata.jsonl"
        self.samples = []
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if Path(rec["filename"]).exists():
                        self.samples.append((rec["filename"], rec["class_idx"]))
        else:
            # Fallback: scan directory
            for class_idx, cls in enumerate(CLASS_NAMES):
                class_dir = Path(root) / cls
                if class_dir.exists():
                    for p in sorted(class_dir.glob("*.png")):
                        self.samples.append((str(p), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def train():
    if Path(CLEAN_MODEL_PATH).exists():
        print(f"Clean model already exists at {CLEAN_MODEL_PATH}")
        print("Delete it first if you want to retrain.")
        return

    print(f"Training clean model on {DEVICE} …")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = SyntheticImageDataset(DATA_CLEAN_DIR, transform=transform)
    if len(dataset) == 0:
        print("ERROR: No data found. Run scripts/generate_demo_data.py first.")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = TinyCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")
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
        acc = 100.0 * correct / total
        avg_loss = total_loss / total
        if avg_loss < best_loss:
            best_loss = avg_loss
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.1f}%")

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CLEAN_MODEL_PATH)
    print(f"✓ Saved: {CLEAN_MODEL_PATH}")

    # Register SHA-256 as reference hash
    sha256 = hashlib.sha256(Path(CLEAN_MODEL_PATH).read_bytes()).hexdigest()
    ref_hashes = {}
    if Path(REFERENCE_HASHES_PATH).exists():
        with open(REFERENCE_HASHES_PATH) as f:
            ref_hashes = json.load(f)
    ref_hashes["clean_model"] = sha256
    with open(REFERENCE_HASHES_PATH, "w") as f:
        json.dump(ref_hashes, f, indent=2)
    print(f"✓ Reference SHA-256 registered: {sha256[:16]}…")


if __name__ == "__main__":
    train()

