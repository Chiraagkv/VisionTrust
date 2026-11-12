"""
Experiment 1b — What does the label check flag on the *unmodified* dataset?

Exp 1 found that ~2–3 % of clean GTSRB images are flagged with no injected
noise. This script lists which (dataset label → suggested label) pairs those
flags come from and saves a contact sheet for manual inspection, to tell real
label errors apart from look-alike classes.

Run:  python experiments/exp1b_clean_flags.py [--dataset GTSRB] [--limit 8600]
"""
import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from common import FIG_DIR, GTSRB_NAMES, md_table, platform_dataset, save_json, save_md  # noqa: E402
from assurance.dataset_scanner import ScanParams, detect_label_anomalies  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GTSRB")
    ap.add_argument("--limit", type=int, default=8600)
    args = ap.parse_args()
    ds = platform_dataset(args.dataset, "train", args.limit)
    readable = (lambda c: f"{c} ({GTSRB_NAMES[int(c)]})") if args.dataset == "GTSRB" else str

    anoms, _, _ = detect_label_anomalies(ds.records, ds.class_names, ScanParams())
    pairs = Counter((a.dataset_label, a.predicted_label) for a in anoms)
    rows = [{"dataset_label": readable(a), "suggested": readable(b), "count": n} for (a, b), n in pairs.most_common(15)]
    by_class = Counter(a.dataset_label for a in anoms)
    sizes = ds.class_counts()
    top_classes = [{"class": readable(c), "flagged": n, "class_size": sizes[c], "share": n / sizes[c]}
                   for c, n in by_class.most_common(10)]

    # contact sheet: 32 most confident flags
    show = anoms[:32]
    fig, axes = plt.subplots(4, 8, figsize=(16, 9))
    for ax, a in zip(axes.flat, show):
        ax.imshow(Image.open(a.filename).convert("RGB"))
        ax.set_title(f"label {a.dataset_label} → {a.predicted_label}\n{a.confidence:.0%}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    fig.suptitle(f"{args.dataset}: most confident label-consistency flags on the unmodified dataset")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"exp1b_clean_flags_{args.dataset}.png", dpi=110)

    save_json(f"exp1b_clean_flags_{args.dataset}", {"flagged": len(anoms), "n": len(ds), "pairs": rows,
              "top_classes": top_classes,
              "contact_sheet": [{"file": a.filename, "label": a.dataset_label, "suggested": a.predicted_label,
                                 "agreement": a.confidence} for a in show]})
    save_md(f"exp1b_clean_flags_{args.dataset}",
            f"# Exp 1b — flags on the unmodified {args.dataset} subset ({len(anoms)} of {len(ds)})\n\n"
            "## Most common (label → suggested) pairs\n\n" + md_table(rows, ["dataset_label", "suggested", "count"])
            + "\n## Classes with the most flags\n\n" + md_table(top_classes, ["class", "flagged", "class_size", "share"])
            + f"\nContact sheet: `figures/exp1b_clean_flags_{args.dataset}.png`\n")


if __name__ == "__main__":
    main()
