"""
Experiment 2 — Dataset scanner vs. poisoned training data.

Part A — backdoor poisoning in the training set
  A fraction of non-target images get a trigger and (dirty-label) are
  relabelled to the target class — the data an attacker would contribute to
  plant a BadNets backdoor. Can the label-consistency check find them?
  Variants: white patch at 1/5/10 % poison rate, noise-blend trigger, and a
  clean-label variant (target-class images stamped, labels untouched).

Part B — duplicate flooding
  Copies of real images are added after common transformations (re-encode,
  noise, resize, crop, flip, rotate, brightness). How many does the pHash
  duplicate detector pair with their source, at which Hamming threshold,
  and how many "duplicates" does it find in the unmodified dataset?

Run:  python experiments/exp2_poisoned_data.py [--dataset GTSRB] [--limit 8600]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageEnhance  # noqa: E402

from common import (FIG_DIR, WORK_DIR, Timer, blend_pattern, md_table, platform_dataset, prf,  # noqa: E402
                    save_json, save_md, stamp_blend, stamp_patch)
from assurance.datasets import Dataset, ImageRecord  # noqa: E402
from assurance.dataset_scanner import ScanParams, detect_duplicates, detect_label_anomalies  # noqa: E402

TARGET = 1          # GTSRB "Speed limit 30" / CIFAR-10 "automobile"
PATCH_FRAC = 0.15   # same relative trigger size the platform's scanner uses
THRESHOLDS = [4, 8, 12, 16, 24, 32]


def _write(arr_or_img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    (arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)).save(path)
    return str(path)


# ────────────────────────────────────────────────────────────────────────────
# Part A
# ────────────────────────────────────────────────────────────────────────────

def poison(base: Dataset, variant: str, rate: float, seed: int = 2):
    rng = np.random.RandomState(seed)
    out_dir = WORK_DIR / f"exp2_{base.name.split()[0]}_{variant}_{rate}"
    clean_label = variant == "clean_label"
    pool = [i for i, r in enumerate(base.records) if (r.class_idx == TARGET) == clean_label]
    n = min(int(round(rate * len(base))), len(pool))
    chosen = set(rng.choice(pool, n, replace=False).tolist())
    blend = None
    recs, poisoned = [], set()
    for i, r in enumerate(base.records):
        if i not in chosen:
            recs.append(r)
            continue
        img = np.asarray(Image.open(r.filename).convert("RGB"))
        h, w = img.shape[:2]
        if variant == "blend":
            if blend is None:
                blend = blend_pattern((32, 32, 3))
            img = stamp_blend(img, blend, 0.15)
        else:
            s = max(2, round(PATCH_FRAC * min(h, w)))
            img = stamp_patch(img, s, (h - s, w - s), "white")
        f = _write(img, out_dir / f"{i:06d}.png")
        poisoned.add(f)
        recs.append(ImageRecord(f, base.class_names[TARGET], TARGET, extra={"source_label": r.label}))
    return Dataset(f"{base.name}+{variant}{rate}", base.source, base.root, list(base.class_names), recs), poisoned


def part_a(base: Dataset):
    rows = []
    variants = [("patch", 0.01), ("patch", 0.05), ("patch", 0.10), ("blend", 0.05), ("clean_label", 0.02)]
    for variant, rate in variants:
        ds, poisoned = poison(base, variant, rate)
        src = {r.filename: r.extra.get("source_label") for r in ds.records}
        for method in ("knn", "linear_probe"):
            p = ScanParams(extractor="resnet18", anomaly_method=method)
            with Timer() as t:
                anoms, used, _ = detect_label_anomalies(ds.records, ds.class_names, p,
                                                        cache_key=f"exp2-{ds.name}")
            flagged = {a.filename for a in anoms}
            m = prf(flagged, poisoned, len(ds))
            hits = [a for a in anoms if a.filename in poisoned]
            m["suggests_source_class"] = (np.mean([a.predicted_label == src[a.filename] for a in hits])
                                          if hits and variant != "clean_label" else float("nan"))
            rows.append({"variant": variant, "poison_rate": rate, "method": method, "poisoned": len(poisoned),
                         **m, "seconds": t.s})
            print(f"  {variant:11s} {rate:.2f} {method:12s} recall={m['recall']:.3f} "
                  f"precision={m['precision']:.3f} fpr={m['fpr']:.4f} ({t.s:.0f}s)", flush=True)
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Part B
# ────────────────────────────────────────────────────────────────────────────

TRANSFORMS = {
    "exact copy":         lambda im, rng: im.copy(),
    "PNG→JPEG q=60":      None,   # handled in make_copy (needs a re-encode)
    "gaussian noise σ=4": lambda im, rng: Image.fromarray(np.clip(np.asarray(im, np.int16) +
                                                                  rng.normal(0, 4, np.asarray(im).shape), 0, 255).astype(np.uint8)),
    "brightness ×1.2":    lambda im, rng: ImageEnhance.Brightness(im).enhance(1.2),
    "down/up-scale 50%":  lambda im, rng: im.resize((max(1, im.width // 2), max(1, im.height // 2)),
                                                    Image.BILINEAR).resize(im.size, Image.BILINEAR),
    "centre crop 90%":    lambda im, rng: im.crop((round(im.width * .05), round(im.height * .05),
                                                   round(im.width * .95), round(im.height * .95))).resize(im.size),
    "rotate 8°":          lambda im, rng: im.rotate(8, resample=Image.BILINEAR),
    "horizontal flip":    lambda im, rng: im.transpose(Image.FLIP_LEFT_RIGHT),
}


def make_copy(name, im, rng, path):
    if name == "PNG→JPEG q=60":
        jp = path.with_suffix(".jpg")
        jp.parent.mkdir(parents=True, exist_ok=True)
        im.save(jp, quality=60)
        return str(jp)
    return _write(TRANSFORMS[name](im, rng), path)


def part_b(base: Dataset, per_transform: int = 150):
    rng = np.random.RandomState(3)
    out_dir = WORK_DIR / f"exp2_{base.name.split()[0]}_dups"
    src_idx = rng.choice(len(base), per_transform * len(TRANSFORMS), replace=False)
    copies = []   # (copy_file, source_file, transform)
    recs = list(base.records)
    for k, (name, i) in enumerate(zip(np.repeat(list(TRANSFORMS), per_transform), src_idx)):
        r = base.records[i]
        with Image.open(r.filename) as im:
            f = make_copy(name, im.convert("RGB"), rng, out_dir / f"{k:05d}.png")
        copies.append((f, r.filename, name))
        recs.append(ImageRecord(f, r.label, r.class_idx))
    flooded = Dataset(base.name + "+dups", base.source, base.root, list(base.class_names), recs)
    copy_files = {c for c, _, _ in copies}

    rows, natural = [], []
    for thr in THRESHOLDS:
        with Timer() as t:
            groups, _, _ = detect_duplicates(flooded.records, threshold=thr)
        gid = {}
        for g, grp in enumerate(groups):
            for f in [grp.representative] + grp.duplicates:
                gid[f] = g
        in_group = set(gid)
        per = defaultdict(lambda: [0, 0, 0])
        for c, s, name in copies:
            per[name][0] += 1
            per[name][1] += int(c in gid and gid.get(s) == gid[c])     # paired with its own source
            per[name][2] += int(c in in_group)                         # flagged at all
        for name, (n, paired, flagged) in per.items():
            rows.append({"threshold": thr, "transform": name, "copies": n,
                         "paired_with_source": paired / n, "flagged_any": flagged / n})
        # images of the ORIGINAL dataset that end up in some group = "natural" duplicates
        orig_flagged = len(in_group - copy_files)
        with Timer() as t2:
            g0, d0, _ = detect_duplicates(base.records, threshold=thr)
        natural.append({"threshold": thr, "original_images_in_groups_after_flood": orig_flagged,
                        "clean_dataset_groups": len(g0), "clean_dataset_duplicates": len(d0),
                        "clean_dataset_duplicate_rate": len(d0) / len(base), "seconds": t.s + t2.s})
        print(f"  thr={thr:2d} " + " ".join(f"{n[:10]}={per[n][1]/per[n][0]:.2f}" for n in TRANSFORMS)
              + f" | clean dataset dup rate={len(d0)/len(base):.3f} ({t.s + t2.s:.0f}s)", flush=True)
    return rows, natural


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GTSRB")
    ap.add_argument("--limit", type=int, default=8600)
    args = ap.parse_args()
    base = platform_dataset(args.dataset, "train", args.limit)
    print(base.summary(), "| target class:", base.class_names[TARGET])

    print("Part A — poisoned samples vs label-consistency check")
    a_rows = part_a(base)
    print("Part B — duplicate flooding vs pHash detector")
    b_rows, natural = part_b(base)
    save_json(f"exp2_poisoned_data_{args.dataset}", {"dataset": args.dataset, "n": len(base),
              "target": base.class_names[TARGET], "part_a": a_rows, "part_b": b_rows, "natural_duplicates": natural})

    # figure: paired-with-source recall vs threshold per transform
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in TRANSFORMS:
        s = [r for r in b_rows if r["transform"] == name]
        ax.plot([r["threshold"] for r in s], [r["paired_with_source"] for r in s], "o-", label=name)
    ax2 = ax.twinx()
    ax2.plot([r["threshold"] for r in natural], [r["clean_dataset_duplicate_rate"] for r in natural],
             "k--", label="clean-dataset dup rate")
    ax2.set_ylabel("share of clean dataset flagged as duplicate")
    ax.axvline(12, color="grey", ls=":", lw=1)
    ax.set_xlabel("pHash Hamming threshold (of 256 bits; default 12)")
    ax.set_ylabel("copies paired with their source")
    ax.set_title(f"Duplicate detection — {args.dataset}")
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"exp2_duplicates_{args.dataset}.png", dpi=130)

    b12 = [r for r in b_rows if r["threshold"] == 12]
    save_md(f"exp2_poisoned_data_{args.dataset}",
            f"# Exp 2 — poisoned data on {args.dataset} ({len(base)} images, target `{base.class_names[TARGET]}`)\n\n"
            "## A. Poisoned samples vs label-consistency check (ResNet-18)\n\n"
            + md_table(a_rows, ["variant", "poison_rate", "method", "poisoned", "flagged", "recall", "precision",
                                "fpr", "suggests_source_class"], {"fpr": ".4f", "poison_rate": ".2f"})
            + "\n## B. Duplicate flooding at the default threshold (12)\n\n"
            + md_table(b12, ["transform", "copies", "paired_with_source", "flagged_any"])
            + "\n## B. Duplicates found in the unmodified dataset\n\n"
            + md_table(natural, ["threshold", "clean_dataset_groups", "clean_dataset_duplicates",
                                 "clean_dataset_duplicate_rate"])
            + "\n## B. Full sweep\n\n" + md_table(b_rows, ["threshold", "transform", "paired_with_source", "flagged_any"]))


if __name__ == "__main__":
    main()
