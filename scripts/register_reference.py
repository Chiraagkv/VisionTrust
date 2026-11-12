"""
scripts/register_reference.py

Register a model you trust as a fingerprint reference (same as the
"Register as trusted reference" button in the dashboard).

Examples:
  # demo model, demo canary set
  python scripts/register_reference.py --model models/clean_model.pt --name clean_model

  # your own model: state_dict + torchvision arch, canaries sampled from a folder dataset
  python scripts/register_reference.py --model my_resnet.pt --arch resnet18 \\
      --canary-from ./my_dataset --per-class 5 --name prod-v1

Class names / input size / normalisation are read from the model's sidecar
JSON (my_resnet.json) when present; --class-names-from-dataset copies them
from the canary dataset instead.

  # (re)write sidecar JSONs for the two demo models
  python scripts/register_reference.py --demo-sidecars
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import config
from assurance.datasets import load_folder
from assurance.model_fingerprint import build_canary_set, load_canary_set, register_reference
from assurance.model_loader import ModelSpec, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--name")
    ap.add_argument("--arch", help="TinyCNN or a torchvision model name (for state_dict files)")
    ap.add_argument("--canary", default=config.DEFAULT_CANARY_MANIFEST, help="canary manifest JSON")
    ap.add_argument("--canary-from", help="build the canary set from this folder dataset")
    ap.add_argument("--per-class", type=int, default=config.CANARY_PER_CLASS)
    ap.add_argument("--class-names-from-dataset", action="store_true")
    ap.add_argument("--demo-sidecars", action="store_true", help="write sidecars for the demo models and exit")
    args = ap.parse_args()

    if args.demo_sidecars:
        from demo_settings import BACKDOORED_MODEL_PATH, CLEAN_MODEL_PATH, write_demo_sidecar
        for p in (CLEAN_MODEL_PATH, BACKDOORED_MODEL_PATH):
            if os.path.exists(p):
                write_demo_sidecar(p)
                print(f"✓ sidecar for {p}")
        return

    if not args.model:
        ap.error("--model is required")

    dataset = load_folder(args.canary_from) if args.canary_from else None
    class_names = dataset.class_names if (dataset and args.class_names_from_dataset) else None
    model = load_model(ModelSpec(path=args.model, arch=args.arch, class_names=class_names))
    canary = (build_canary_set(dataset, per_class=args.per_class) if dataset
              else load_canary_set(args.canary))
    name = args.name or os.path.splitext(os.path.basename(args.model))[0]
    entry = register_reference(name, model, canary)
    print(f"✓ Registered '{name}': sha256={entry['sha256'][:16]}…  canary set '{canary.name}' "
          f"({len(canary)} images)")


if __name__ == "__main__":
    main()
