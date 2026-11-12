"""
config.py — Paths and *default* scan parameters for the CV Assurance Platform.

Nothing in here describes a particular dataset or model. Class names,
contributors, image size, normalisation, trigger settings etc. come from the
dataset / model the user loads (see assurance/datasets.py and
assurance/model_loader.py). Values below are only defaults that the dashboard
lets the user override.

Demo-only constants (synthetic classes, trigger used to poison the demo
model, …) live in scripts/demo_settings.py.
"""
import os

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
CANARY_DIR  = os.path.join(ROOT_DIR, "canary")
RECORDS_DIR = os.path.join(ROOT_DIR, "records")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

INFERENCE_CHAIN_PATH    = os.path.join(RECORDS_DIR, "inference_chain.jsonl")
REFERENCE_REGISTRY_PATH = os.path.join(MODELS_DIR, "reference_registry.json")
DEFAULT_CANARY_MANIFEST = os.path.join(CANARY_DIR, "canary_meta.json")

DATASET_CACHE_DIR = os.path.join(OUTPUTS_DIR, "datasets")        # materialised torchvision sets
TORCHVISION_ROOT  = os.path.join(OUTPUTS_DIR, "torchvision_raw")  # raw torchvision downloads
CANARY_SETS_DIR   = os.path.join(OUTPUTS_DIR, "canaries")         # canary sets built from datasets
UPLOADS_DIR       = os.path.join(OUTPUTS_DIR, "uploads")          # files uploaded via the dashboard

# Demo shortcuts shown in the dashboard (only if they exist on disk)
DEMO_DATASETS = {
    "Demo · clean":    os.path.join(ROOT_DIR, "data", "clean"),
    "Demo · poisoned": os.path.join(ROOT_DIR, "data", "poisoned"),
}
DEMO_MODELS = {
    "Demo · clean model":      os.path.join(MODELS_DIR, "clean_model.pt"),
    "Demo · backdoored model": os.path.join(MODELS_DIR, "backdoored_model.pt"),
}

RANDOM_SEED = 42

# ── Dataset scanner defaults ────────────────────────────────────────────────
# 256-bit pHash (hash_size=16). The default 64-bit pHash is too coarse for
# small images: distinct same-class images collide at distance 0–2.
PHASH_HASH_SIZE     = 16
PHASH_DUP_THRESHOLD = 12    # Hamming distance ≤ this → near-duplicate

# Label-consistency check (independent pretrained feature extractor)
FEATURE_EXTRACTOR      = "resnet18"   # see assurance/features.py
ANOMALY_METHOD         = "knn"        # "knn" or "linear_probe"
KNN_NEIGHBORS          = 10
ANOMALY_AGREEMENT_THR  = 0.60   # flag if ≥ this share of neighbours (or probe prob) favour another class

# Integrity scoring: each penalty reaches its cap once the fraction of
# affected samples hits its saturation rate (a few % of poisoned labels
# is already serious, so penalties must not scale linearly up to 100 %).
DUP_PENALTY_MAX           = 25
DUP_SATURATION_RATE       = 0.10
ANOM_PENALTY_MAX          = 55
ANOM_SATURATION_RATE      = 0.10
HIGH_RISK_CONTRIB_PENALTY = 10  # per HIGH-risk contributor, capped at 20

# ── Behavioral trigger scanner defaults ─────────────────────────────────────
SUSPICION_THRESHOLD = 0.50     # attack_success_rate > this → HIGH
TRIGGER_SIZE_FRAC   = 0.15     # patch side as a fraction of the model input side

# ── Canary set defaults ─────────────────────────────────────────────────────
CANARY_PER_CLASS = 3
