"""
config.py — Central configuration for the CV Assurance Platform.
"""
import os

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_CLEAN_DIR   = os.path.join(ROOT_DIR, "data", "clean")
DATA_POISONED_DIR= os.path.join(ROOT_DIR, "data", "poisoned")
MODELS_DIR       = os.path.join(ROOT_DIR, "models")
CANARY_DIR       = os.path.join(ROOT_DIR, "canary")
RECORDS_DIR      = os.path.join(ROOT_DIR, "records")
OUTPUTS_DIR      = os.path.join(ROOT_DIR, "outputs")
SCRIPTS_DIR      = os.path.join(ROOT_DIR, "scripts")

CLEAN_MODEL_PATH      = os.path.join(MODELS_DIR, "clean_model.pt")
BACKDOORED_MODEL_PATH = os.path.join(MODELS_DIR, "backdoored_model.pt")
INFERENCE_CHAIN_PATH  = os.path.join(RECORDS_DIR, "inference_chain.jsonl")
REFERENCE_HASHES_PATH = os.path.join(MODELS_DIR, "reference_hashes.json")

# ── Model / Training ────────────────────────────────────────────────────────
NUM_CLASSES  = 10
IMAGE_SIZE   = 32          # 32×32 synthetic images
BATCH_SIZE   = 64
RANDOM_SEED  = 42

# Class names (maps CIFAR-10 indices; also used for synthetic data)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]

# ── Backdoor trigger ────────────────────────────────────────────────────────
TRIGGER_SIZE        = 5        # 5×5 white patch
TRIGGER_POSITION    = "bottom-left"   # default training position
BACKDOOR_TARGET     = 0        # target class index → "airplane"
POISON_FRACTION     = 0.15     # fraction of training data poisoned

# ── Dataset scanner ─────────────────────────────────────────────────────────
# 256-bit pHash (hash_size=16). The default 64-bit pHash is too coarse for
# 32×32 images: distinct same-class images collide at distance 0–2.
PHASH_HASH_SIZE        = 16
PHASH_DUP_THRESHOLD    = 12    # Hamming distance ≤ this → near-duplicate
LABEL_ANOMALY_CONF_THR = 0.70  # flag if reference classifier disagrees ≥ this

# Integrity scoring: each penalty reaches its cap once the fraction of
# affected samples hits its saturation rate (a few % of poisoned labels
# is already serious, so penalties must not scale linearly up to 100 %).
DUP_PENALTY_MAX        = 25
DUP_SATURATION_RATE    = 0.10
ANOM_PENALTY_MAX       = 55
ANOM_SATURATION_RATE   = 0.10
HIGH_RISK_CONTRIB_PENALTY = 10  # per HIGH-risk contributor, capped at 20

# ── Behavioral trigger scanner ──────────────────────────────────────────────
TRIGGER_CANDIDATES = [
    "top-left",
    "top-right",
    "center",
    "bottom-left",
    "bottom-right",
]
SUSPICION_THRESHOLD = 0.50     # attack_success_rate > this → HIGH

# ── Canary set ──────────────────────────────────────────────────────────────
CANARY_SIZE = 30               # number of fixed canary images

# ── Contributor demo metadata ────────────────────────────────────────────────
CONTRIBUTORS = {
    "ContribA": "clean",
    "ContribB": "clean",
    "ContribC": "poisoned",
}

