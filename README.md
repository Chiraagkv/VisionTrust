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
├── config.py                  # All paths, constants, thresholds
├── requirements.txt
│
├── assurance/
│   ├── model_arch.py          # Shared TinyCNN (32×32, ~300 K params)
│   ├── dataset_scanner.py     # Duplicate + label-anomaly + contributor scan
│   ├── backdoor_scanner.py    # Behavioral trigger scan (5 positions)
│   ├── model_fingerprint.py   # SHA-256 + canary behavioral fingerprint
│   ├── provenance.py          # SHA-256 hash chain, create/verify/tamper
│   ├── gradcam.py             # Grad-CAM heatmaps for TinyCNN predictions
│   └── report.py              # Assembles AssuranceReport from all results
│
├── data/
│   ├── clean/                 # 10 classes × 30 clean images + metadata.jsonl
│   └── poisoned/              # Same + trigger-stamped + label-anomaly images
│
├── models/
│   ├── clean_model.pt         # Trained clean TinyCNN
│   ├── backdoored_model.pt    # Trained backdoored TinyCNN
│   ├── reference_hashes.json  # Canonical SHA-256 values
│   └── canary_predictions.json # Canonical canary predictions
│
├── canary/                    # 30 fixed canary images + canary_meta.json
├── records/
│   └── inference_chain.jsonl  # Append-only inference hash chain
│
└── scripts/
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

## 5. Demo Flow (3–5 minute presentation)

1. **Select Clean dataset + Clean model** (sidebar) → Dashboard → ▶ Run All Scans
   - See: ✅ Data integrity HIGH, ✅ Model PASS, ✅ Provenance VERIFIED

2. **Switch to Poisoned dataset** → re-scan
   - See: ⚠️ Duplicate images, ⚠️ Label-consistency anomalies, ContribC = HIGH risk

3. **Switch to Backdoored model** → Model Scanner → ▶ Behavioral Trigger Scan
   - See: Bottom-left ASR ≈ 90%+, trigger demo: normal vs triggered

4. **Provenance page** → ⚡ Generate Inference → ✓ VERIFIED
   → 💀 Simulate Tampering → 🔍 Verify Latest Record → ✗ TAMPERED

5. **Explainability page** (backdoored model active) → Pick a non-airplane sample (e.g. `frog/frog_0003.png`) → ☑ Stamp backdoor trigger patch
   - See: prediction flips to AIRPLANE, Grad-CAM heat concentrated on the bottom-left patch

6. **Assurance Report page** → Full summary with findings + limitations

---

## 6. Each Assurance Method

### A. Dataset Integrity Scanner (`assurance/dataset_scanner.py`)

**Duplicate detection:** Computes perceptual hash (pHash) for every image. Uses a 256-bit pHash (`hash_size=16`) because the default 64-bit hash collides for distinct 32×32 images; near-duplicates are identified by Hamming distance ≤ 12 (configurable in `config.py`).

**Label-consistency anomaly:** A reference classifier (the clean model) predicts the class of each sample. If it strongly disagrees (confidence ≥ 70%) with the dataset label, the sample is flagged as a *label-consistency anomaly*. This is NOT proof of poisoning.

**Contributor aggregation:** Per-contributor risk = (duplicates + anomalies) / total. ContribC is assigned all poisoned samples in the demo, producing a HIGH risk score.

**Integrity score:** Starts at 100. The duplicate penalty (max 25) and label-anomaly penalty (max 55) each reach their cap once 10 % of samples are affected; each HIGH-risk contributor costs 10 more (max 20). Clean demo data scores 100, poisoned scores ~24.

### B. Behavioral Trigger Scan (`assurance/backdoor_scanner.py`)

For each of 5 candidate positions (top-left, top-right, center, bottom-left, bottom-right):
1. Stamp a white 5×5 square onto 30 test images.
2. Run inference.
3. Compute attack success rate (ASR) = fraction predicted as target class.

ASR > 50% → HIGH suspicion. The backdoored model is trained with the trigger at bottom-left, so that position shows ASR ≈ 90%+.

**Limitation:** Only detects visible patch-based triggers at the 5 candidate positions.

### C. Model Fingerprinting (`assurance/model_fingerprint.py`)

**Cryptographic fingerprint:** SHA-256 of the model file bytes. Registered once at training time. Any byte change is detected.

**Behavioral fingerprint:** The 30-image canary set is fixed at data generation time. Predictions on the canary set uniquely identify a model's behavioral state. Agreement is reported as a percentage.

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

For an uploaded or sample image, the active model's class score is back-propagated to a conv layer (block 3, 8×8, or block 2, 16×16). Channel activations are weighted by their average gradient, summed and ReLU'd, then upsampled to 32×32. The page shows the prediction, class probabilities, the heatmap and an overlay. You can explain any class, not just the predicted one. With the trigger stamped, it flags a HIGH finding when the patch receives > 2× the image-wide mean activation and the prediction becomes the backdoor target. Grad-CAM is coarse and shows correlation, not causation.

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
- Training data is fully synthetic (32×32 color images). Real-world performance will differ.

---

## 9. Future Extensions

- CIFAR-10 / real-image dataset support
- Additional trigger shapes and colors in the behavioral scan
- Neural Cleanse–style optimization-based trigger inversion
- COCO / YOLO dataset loader plugins
- Distributed contributor signing (e.g., Ed25519 signatures)
- Continuous monitoring mode with scheduled re-scans
- Export to PDF / HTML assurance report

