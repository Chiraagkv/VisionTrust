# CV Assurance Platform

> **Evidence-based assurance for computer-vision training pipelines**
> Hackathon prototype — fully offline, no cloud APIs.

---

## 1. Project Overview

The **CV Assurance Platform** assesses the integrity of computer-vision pipelines across four dimensions:

| Capability | What it detects |
|---|---|
| **Training-Data Integrity** | Near-duplicate flooding, label-consistency anomalies, per-contributor risk |
| **Backdoor Behavior** | Trigger-like prediction shifts at candidate patch positions |
| **Model Fingerprinting** | SHA-256 substitution detection + behavioral canary-set fingerprint |
| **Cryptographic Provenance** | Tamper-evident SHA-256 hash chain for every inference record |
| **Grad-CAM Explainability** | Heatmap of the image regions that drove a prediction (e.g. a trigger patch) |

---

## 2. Architecture

```
SIH-SAFE-VISION/
├── app.py                     # Streamlit dashboard (single entry point)
├── config.py                  # Paths + default scan thresholds (nothing dataset/model specific)
├── requirements.txt
│
├── assurance/
│   ├── datasets.py            # Ingestion: folder / CSV+files / torchvision → Dataset
│   ├── features.py            # Independent pretrained feature extractors (ImageNet)
│   ├── model_loader.py        # Bring-your-own-model: state_dict / TorchScript / pickled
│   ├── model_arch.py          # Demo TinyCNN (32×32, ~300 K params)
│   ├── dataset_scanner.py     # Duplicate + label-anomaly + contributor scan
│   ├── backdoor_scanner.py    # Behavioral trigger scan (5 positions, target auto-discovered)
│   ├── model_fingerprint.py   # Trusted references: SHA-256 + canary behavioral fingerprint
│   ├── provenance.py          # SHA-256 hash chain, create/verify/tamper
│   ├── gradcam.py             # Grad-CAM heatmaps for TinyCNN predictions
│   └── report.py              # Assembles AssuranceReport from all results
│
├── data/
│   ├── clean/                 # 10 classes × 30 clean images + metadata.jsonl
│   └── poisoned/              # Same + trigger-stamped + label-anomaly images
│
├── models/
│   ├── clean_model.pt / .json      # Demo clean TinyCNN + sidecar (arch, classes, preprocessing)
│   ├── backdoored_model.pt / .json # Demo backdoored TinyCNN + sidecar
│   └── reference_registry.json     # Trusted references (SHA-256 + canary outputs)
│
├── canary/                    # 30 fixed canary images + canary_meta.json
├── records/
│   └── inference_chain.jsonl  # Append-only inference hash chain
│
├── outputs/                   # Materialised torchvision sets, built canary sets, uploads
│
└── scripts/
    ├── demo_settings.py       # Constants for the synthetic demo only
    ├── register_reference.py  # CLI: register a trusted reference model
    ├── generate_demo_data.py  # Generates all demo images + metadata
    ├── train_clean_model.py   # Trains clean_model.pt (25 epochs)
    └── train_backdoored_model.py  # Trains backdoored_model.pt (15% poison)
```

---

## 3. Installation

```bash
# 1. Clone / enter the project
cd SIH-SAFE-VISION

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (CPU-only PyTorch is sufficient)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 4. Running the Demo

### Step 1 — Generate demo data

```bash
python scripts/generate_demo_data.py
```

Creates:
- `data/clean/`    — 10 classes × 30 unique, correctly-labelled images + metadata
- `data/poisoned/` — same + trigger-stamped images + label anomalies
- `canary/`        — 30 fixed canary images

### Step 2 — Train models

```bash
python scripts/train_clean_model.py
python scripts/train_backdoored_model.py
```

Each script trains for 25 epochs (~2–5 minutes on CPU).
Models are saved only once — running again is a no-op if files exist.

### Step 3 — Launch the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 4b. Using your own dataset and model

Open **⚙️ Setup** in the dashboard.

**Dataset** — three ingestion paths, all producing the same internal `Dataset`:

| Source | Layout | Class names from |
|---|---|---|
| Folder | `root/<class_name>/<image>` | sub-folder names |
| CSV + files | table with a path column (absolute or relative to an image root) and a label column | unique labels |
| torchvision | `CIFAR10`, `CIFAR100`, `MNIST`, `FashionMNIST`, `KMNIST`, `SVHN`, `STL10`, `GTSRB` — a stratified subset is written to `outputs/datasets/` as PNGs | the dataset's `classes` |

Contributor / batch information is **metadata, not configuration**: for folders, add a
`metadata.csv` / `metadata.jsonl` (auto-detected in the root, or uploaded) with one row per
image; for CSV datasets, map the columns in the UI. Without it, contributor analysis is skipped.

**Model** — a `state_dict` (choose `TinyCNN`, any torchvision classification architecture, or a
class from your own `.py` file), a TorchScript file, or a fully pickled `nn.Module` (opt-in, since
unpickling runs code). Class names, input size and normalisation come from a sidecar JSON
(`model.pt` → `model.json`), the active dataset, or the form:

```json
{"arch": "resnet18", "class_names": ["cat", "dog"], "input_size": [224, 224],
 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
```

**Trusted reference** — on **🤖 Model Scanner**, register the model you trust. Its SHA-256 and its
outputs on a canary set (built from the active dataset: k images per class, seeded) are stored in
`models/reference_registry.json`; later audits compare against it. CLI equivalent:

```bash
python scripts/register_reference.py --model my_model.pt --arch resnet18 \
    --canary-from ./my_dataset --per-class 5 --name prod-v1
```

---

## 5. Demo Flow (3–5 minute presentation)

The demo dataset and clean model are loaded automatically; switch with **Demo shortcuts** in the sidebar.

1. **Demo · clean + Demo · clean model** → Dashboard → ▶ Run All Scans
   - See: ✅ Data integrity HIGH, ✅ Model PASS, ✅ Provenance VERIFIED

2. **Switch to Demo · poisoned** → re-scan
   - See: ⚠️ Duplicate images, ⚠️ Label-consistency anomalies, ContribC = HIGH risk

3. **Switch to Demo · backdoored model** → Model Scanner → ▶ Behavioral Trigger Scan
   - See: Bottom-left ASR ≈ 90%+, trigger demo: normal vs triggered

4. **Provenance page** → ⚡ Generate Inference → ✓ VERIFIED
   → 💀 Simulate Tampering → 🔍 Verify Latest Record → ✗ TAMPERED

5. **Explainability page** (backdoored model active) → Pick a non-airplane sample (e.g. `frog/frog_0003.png`) → ☑ Stamp backdoor trigger patch
   - See: prediction flips to AIRPLANE, Grad-CAM heat concentrated on the bottom-left patch

6. **Assurance Report page** → Full summary with findings + limitations

---

## 6. Each Assurance Method

### A. Dataset Integrity Scanner (`assurance/dataset_scanner.py`)

**Duplicate detection:** Computes perceptual hash (pHash) for every image. Uses a 256-bit pHash (`hash_size=16`) because the default 64-bit hash collides for distinct 32×32 images; near-duplicates are identified by Hamming distance ≤ 12 (adjustable in the scan settings). Distances are computed on packed bit arrays in blocks, so large datasets are fine.

**Label-consistency anomaly:** Every image is embedded with a network pretrained on ImageNet (ResNet-18 by default; weights download once, then offline) — it is independent of both the dataset and the model under audit, so a poisoned model cannot "agree" with poisoned labels. Two methods:
- **kNN vote** (default): flag an image when ≥ 60 % of its k = 10 nearest neighbours (cosine) carry one other label.
- **Linear probe**: out-of-fold logistic regression on the embeddings; flag when it gives ≥ 60 % to another class.

If the weights cannot be downloaded, a hand-crafted colour/edge feature extractor is used instead (weaker). This is NOT proof of poisoning.

**Contributor aggregation:** Per-contributor risk = (duplicates + anomalies) / total, using the contributor column of the dataset metadata. ContribC is assigned all poisoned samples in the demo, producing a HIGH risk score.

**Integrity score:** Starts at 100. The duplicate penalty (max 25) and label-anomaly penalty (max 55) each reach their cap once 10 % of samples are affected; each HIGH-risk contributor costs 10 more (max 20). Clean demo data scores 100, poisoned scores ~24.

### B. Behavioral Trigger Scan (`assurance/backdoor_scanner.py`)

For each of 5 candidate positions (top-left, top-right, center, bottom-left, bottom-right):
1. Resize up to 90 test images from the active dataset to the model's input size and stamp a white square (15 % of the input side — 5×5 px at 32×32).
2. Run inference on the images the model classifies correctly without the trigger.
3. For every class c, ASR(c) = share of non-c images that flip to c. The class with the highest ASR is the discovered target (or fix the target in the scan settings).

ASR > 50% → HIGH suspicion. The backdoored model is trained with the trigger at bottom-left, so that position shows ASR ≈ 90%+.

**Limitation:** Only detects visible patch-based triggers at the 5 candidate positions.

### C. Model Fingerprinting (`assurance/model_fingerprint.py`)

**Cryptographic fingerprint:** SHA-256 of the model file bytes, compared with a registered trusted reference. Any byte change is detected.

**Behavioral fingerprint:** When a reference is registered it is bound to a canary set — the demo's 30 images, or k images per class sampled from your dataset — whose file hashes are stored too. Audits replay the canary set and report top-1 agreement, the mean shift in softmax outputs, and whether the canary images themselves changed.

### D. Cryptographic Inference Provenance (`assurance/provenance.py`)

Each inference creates a record with:
```
record_hash = SHA256(
    previous_record_hash + input_hash + model_hash +
    preprocessing_hash + output + confidence + timestamp + nonce
)
```
Records are appended to `records/inference_chain.jsonl`. Any field modification breaks the hash. Any record deletion breaks the previous-hash linkage.

### E. Grad-CAM Explainability (`assurance/gradcam.py`)

For an uploaded or sample image, the active model's class score is back-propagated to a convolutional layer (any layer with a 4-D feature map; default = the last non-pooling one, e.g. block 3 at 8×8 for TinyCNN). Channel activations are weighted by their average gradient, summed and ReLU'd, then upsampled to 32×32. The page shows the prediction, class probabilities, the heatmap and an overlay. You can explain any class, not just the predicted one. With the trigger stamped, it flags a HIGH finding when the patch receives > 2× the image-wide mean activation and the patch changes the prediction. Grad-CAM is coarse and shows correlation, not causation.

---

## 7. Threat Model

### Supported attack classes
- Near-duplicate image flooding
- Label-consistency anomalies (mislabelled samples)
- Visible patch-based backdoor trigger behavior
- Model file substitution / byte-level modification
- Inference record field tampering / chain corruption

### NOT guaranteed to detect
- Arbitrary unknown backdoor architectures
- Fully stealthy attacks with minimal behavioral change
- Semantic or invisible perturbation attacks
- Multi-vector or adaptive adversarial attacks

---

## 8. Limitations

- Behavioral trigger scan tests only 5 fixed positions with a white square patch.
- Label-consistency disagreement is a heuristic, not proof of poisoning.
- SHA-256 detects any change but cannot identify whether the change is malicious.
- This is a prototype for demonstration — not a production security tool.
- The bundled demo data is synthetic (32×32 color images); real datasets can be loaded, but detection rates on them have not been benchmarked yet.
- The ImageNet feature extractor may separate classes poorly on domains far from natural photos (medical, satellite, tiny grayscale images).

---

## 9. Future Extensions

- Additional trigger shapes and colors in the behavioral scan
- Neural Cleanse–style optimization-based trigger inversion
- COCO / YOLO dataset loaders
- Distributed contributor signing (e.g., Ed25519 signatures)
- Continuous monitoring mode with scheduled re-scans
- Export to PDF / HTML assurance report

