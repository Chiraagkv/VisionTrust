"""
scripts/demo_settings.py — Constants for the *synthetic demo only*.

These describe the toy dataset and the backdoor attack the demo scripts
generate and train. The platform itself (app.py, assurance/) never reads
them: it learns class names, contributors, image size, etc. from whatever
dataset and model are loaded.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CANARY_DIR, DEFAULT_CANARY_MANIFEST, MODELS_DIR, ROOT_DIR, RANDOM_SEED  # noqa: E402,F401

DATA_CLEAN_DIR    = os.path.join(ROOT_DIR, "data", "clean")
DATA_POISONED_DIR = os.path.join(ROOT_DIR, "data", "poisoned")
CLEAN_MODEL_PATH      = os.path.join(MODELS_DIR, "clean_model.pt")
BACKDOORED_MODEL_PATH = os.path.join(MODELS_DIR, "backdoored_model.pt")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE  = 32           # 32×32 synthetic images
BATCH_SIZE  = 64
NORM_MEAN   = [0.5, 0.5, 0.5]
NORM_STD    = [0.5, 0.5, 0.5]

# Backdoor attack used to build the poisoned data / backdoored model
TRIGGER_SIZE     = 5        # 5×5 white patch
TRIGGER_POSITION = "bottom-left"
BACKDOOR_TARGET  = 0        # → "airplane"
POISON_FRACTION  = 0.15

# Contributor ids written into the demo metadata.jsonl
CONTRIBUTORS = {
    "ContribA": "clean",
    "ContribB": "clean",
    "ContribC": "poisoned",
}


def write_demo_sidecar(model_path: str) -> None:
    """Describe a demo TinyCNN so the platform can load it without user input."""
    from assurance.model_loader import write_sidecar
    write_sidecar(model_path, arch="TinyCNN", class_names=CLASS_NAMES,
                  input_size=[IMAGE_SIZE, IMAGE_SIZE], mean=NORM_MEAN, std=NORM_STD)
