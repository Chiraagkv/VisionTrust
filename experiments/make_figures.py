"""
Summary figures for REPORT.md, built from results/*.json.

Run:  python experiments/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
FIG = RES / "figures"
# Default categorical slots 1–4 (fixed order), validated palette from the dataviz reference
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_scanner(ds="GTSRB"):
    e3, e3b = load(f"exp3_backdoor_models_{ds}"), load(f"exp3b_probe_sweep_{ds}")
    if not (e3 and e3b):
        return
    multi = {r["model"]: r["multi_size_asr"] for r in e3b["rows"]}
    key = {"clean-seed0": "clean_s0", "clean-seed1": "clean_s1", "clean-seed2": "clean_s2"}
    rows = e3["rows"]
    labels = [r["model"].replace("backdoor-", "backdoor: ") for r in rows]
    series = [
        ("Attack works (true trigger ASR)", [r["attack_asr_true_trigger"] for r in rows], BLUE),
        ("Scanner, default 5 px probe", [r["scan_max_asr_n300"] for r in rows], ORANGE),
        ("Scanner, multi-size probe (3b)", [multi[key.get(r["model"], "bd_" + r["model"][9:])] for r in rows], AQUA),
        ("Grad-CAM patch flag rate", [r["gradcam_patch_flag_rate"] for r in rows], YELLOW),
    ]
    y = np.arange(len(rows))[::-1]
    h = 0.19
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for k, (name, vals, col) in enumerate(series):
        v = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)
        ax.barh(y + (1.5 - k) * h, v, height=h - 0.03, color=col, label=name, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.text(0.505, y[0] + 0.5, "HIGH threshold", color=INK2, fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("rate (0–1)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Backdoor detection on {ds}: does the scanner see attacks that work?", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.25), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"summary_scanner_{ds}.png", dpi=140)
    plt.close(fig)


def fig_fingerprint(ds="GTSRB"):
    e4 = load(f"exp4_fingerprint_{ds}")
    if not e4:
        return
    rows = sorted(e4["rows"], key=lambda r: r["tv_k10"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for r, yy in zip(rows, y):
        bd = r["backdoor_asr"] > 0.5
        ax.plot([0, r["tv_k10"]], [yy, yy], color=GRID, lw=2, zorder=1)
        ax.scatter(r["tv_k10"], yy, s=60, color=ORANGE if bd else BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.text(r["tv_k10"] + 0.004, yy, f"{r['agree_k10']:.1%} agree", va="center", fontsize=7.5, color=INK2)
    ax.scatter([], [], s=60, color=BLUE, label="no backdoor")
    ax.scatter([], [], s=60, color=ORANGE, label="contains a working backdoor (ASR > 50 %)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["variant"] for r in rows])
    ax.set_xlabel("mean output shift on 430 canaries (total-variation distance; 0 = identical)")
    ax.set_xlim(0, max(r["tv_k10"] for r in rows) * 1.35)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Behavioral fingerprint vs model modifications ({ds})", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"summary_fingerprint_{ds}.png", dpi=140)
    plt.close(fig)


def fig_label_noise():
    data = [(ds, load(f"exp1_label_noise_{ds}")) for ds in ("GTSRB", "CIFAR10")]
    data = [(ds, d) for ds, d in data if d]
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(4.6 * len(data), 3.6), squeeze=False, sharey=True)
    for ax, (ds, d) in zip(axes[0], data):
        for (ext, method), col in [(("resnet18", "knn"), BLUE), (("resnet18", "linear_probe"), ORANGE)]:
            s = [r for r in d["rows"] if r["extractor"] == ext and r["method"] == method and r["noise_rate"] > 0]
            ax.plot([r["noise_rate"] for r in s], [r["f1"] for r in s], "o-", color=col, lw=2, ms=7,
                    mec=SURFACE, mew=2, label=f"ResNet-18 · {method.replace('_', ' ')}")
        hc = [r for r in d["rows"] if r["extractor"] == "handcrafted" and r["method"] == "knn" and r["noise_rate"] > 0]
        ax.plot([r["noise_rate"] for r in hc], [r["f1"] for r in hc], "s", color=AQUA, ms=8, mec=SURFACE, mew=2,
                label="hand-crafted · knn")
        ax.set_title(f"{ds} ({d['n']} images)", loc="left")
        ax.set_xlabel("injected label-noise rate")
        ax.set_xticks([0.02, 0.05, 0.1, 0.2])
        ax.set_xticklabels(["2 %", "5 %", "10 %", "20 %"])
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("F1 of flagged vs. truly mislabelled")
    axes[0][-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Label-consistency check: F1 by noise rate", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "summary_label_noise.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_scanner()
    fig_fingerprint()
    fig_label_noise()
    print("figures written to", FIG)
