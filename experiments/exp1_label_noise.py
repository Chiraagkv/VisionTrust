"""
Experiment 1 — Label-consistency check vs. injected label noise.

Question: if a fraction of a real dataset's labels are wrong, how many of the
wrong labels does the platform's label-consistency check find, and how many
correct labels does it wrongly flag?

Setup
  • Dataset ingested with the platform's torchvision loader (stratified subset).
  • Symmetric label noise: a fraction r of images get a uniformly random wrong label.
  • Detector: assurance.dataset_scanner.detect_label_anomalies, exactly as the
    dashboard calls it, for {kNN, linear probe} × {ResNet-18, hand-crafted}.
  • r = 0 measures how often the check fires on the dataset as published.

Run:  python experiments/exp1_label_noise.py [--dataset GTSRB] [--limit 8600]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (FIG_DIR, Timer, md_table, platform_dataset, prf, relabel, save_json,  # noqa: E402
                    save_md)
from assurance.dataset_scanner import ScanParams, detect_label_anomalies  # noqa: E402
from assurance.features import extract_features  # noqa: E402
import numpy as np  # noqa: E402

RATES = [0.0, 0.02, 0.05, 0.10, 0.20]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def inject_noise(ds, rate, seed):
    rng = np.random.RandomState(seed)
    n, k = len(ds), len(ds.class_names)
    idx = rng.choice(n, int(round(rate * n)), replace=False)
    new = {}
    for i in idx:
        c = ds.records[i].class_idx
        new[int(i)] = int((c + rng.randint(1, k)) % k)        # uniformly random *other* class
    return relabel(ds, new, f"{ds.name}+noise{rate}"), {ds.records[i].filename for i in idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GTSRB")
    ap.add_argument("--limit", type=int, default=8600)
    args = ap.parse_args()

    base = platform_dataset(args.dataset, "train", args.limit)
    print(f"{base.summary()}")
    img_key = f"exp1-{args.dataset}-{args.limit}"        # images never change → embed once per extractor

    timings = {}
    for ext in ("resnet18", "handcrafted"):
        with Timer() as t:
            extract_features([r.filename for r in base.records], ext, cache_key=img_key)
        timings[ext] = t.s
        print(f"embedded {len(base)} images with {ext} in {t.s:.0f}s")

    rows, sweep = [], []
    for rate in RATES:
        ds, wrong = inject_noise(base, rate, seed=1)
        for ext in ("resnet18", "handcrafted"):
            for method in ("knn", "linear_probe"):
                if ext == "handcrafted" and rate not in (0.0, 0.10):
                    continue
                p = ScanParams(extractor=ext, anomaly_method=method)
                with Timer() as t:
                    anoms, used, notes = detect_label_anomalies(ds.records, ds.class_names, p, cache_key=img_key)
                m = prf({a.filename for a in anoms}, wrong, len(ds))
                # how often is the suggested label the *true* one, for correctly found noisy labels?
                true_lbl = {r.filename: r.label for r in base.records}
                found = [a for a in anoms if a.filename in wrong]
                m["suggestion_correct"] = (np.mean([a.predicted_label == true_lbl[a.filename] for a in found])
                                           if found else float("nan"))
                rows.append({"dataset": args.dataset, "noise_rate": rate, "extractor": used, "method": method,
                             "n": len(ds), "noisy": len(wrong), **m, "seconds": t.s})
                print(f"  r={rate:.2f} {used:11s} {method:12s} P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"FPR={m['fpr']:.4f} flagged={m['flagged']} ({t.s:.1f}s)")

        if rate in (0.05, 0.10, 0.20):
            for thr in THRESHOLDS:
                p = ScanParams(extractor="resnet18", anomaly_method="knn", agreement_threshold=thr)
                anoms, _, _ = detect_label_anomalies(ds.records, ds.class_names, p, cache_key=img_key)
                m = prf({a.filename for a in anoms}, wrong, len(ds))
                sweep.append({"noise_rate": rate, "threshold": thr, **m})

    save_json(f"exp1_label_noise_{args.dataset}", {"dataset": args.dataset, "n": len(base),
              "classes": len(base.class_names), "embed_seconds": timings, "rows": rows, "threshold_sweep": sweep})

    # ── figure: precision / recall vs threshold ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for rate in (0.05, 0.10, 0.20):
        s = [r for r in sweep if r["noise_rate"] == rate]
        axes[0].plot([r["threshold"] for r in s], [r["precision"] for r in s], "o-", label=f"noise {rate:.0%}")
        axes[1].plot([r["threshold"] for r in s], [r["recall"] for r in s], "o-", label=f"noise {rate:.0%}")
    for ax, t in zip(axes, ("Precision", "Recall")):
        ax.axvline(0.6, color="grey", ls="--", lw=1)
        ax.set_xlabel("kNN agreement threshold (default 0.6)")
        ax.set_title(f"{t} — {args.dataset}, ResNet-18 kNN")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"exp1_threshold_{args.dataset}.png", dpi=130)

    cols = ["noise_rate", "extractor", "method", "noisy", "flagged", "precision", "recall", "f1", "fpr",
            "suggestion_correct"]
    sweep_cols = ["noise_rate", "threshold", "flagged", "precision", "recall", "f1"]
    save_md(f"exp1_label_noise_{args.dataset}",
            f"# Exp 1 — label noise detection on {args.dataset} ({len(base)} images, "
            f"{len(base.class_names)} classes)\n\nEmbedding time: "
            + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()) + "\n\n"
            + md_table(rows, cols, {"fpr": ".4f", "noise_rate": ".2f"})
            + "\n## kNN threshold sweep (ResNet-18)\n\n" + md_table(sweep, sweep_cols, {"noise_rate": ".2f"}))


if __name__ == "__main__":
    main()
