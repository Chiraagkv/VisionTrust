"""
Experiment 3b — How much does the scanner's probe size matter?

Exp 3 showed working 3–4 px corner backdoors scanning only MEDIUM with the
default 5 px probe. Here we rescan every exp-3 model with white probes of
2–8 px, and evaluate a "multi-size" variant (max ASR over all probe sizes) —
including its false-positive behaviour on the clean models.

Run:  python experiments/exp3b_probe_sweep.py [--dataset GTSRB]   (after exp3)
"""
import argparse

from common import MODELS_DIR, load_saved, md_table, platform_dataset, save_json, save_md  # sets sys.path
import config  # noqa: E402
from assurance.backdoor_scanner import scan_backdoor  # noqa: E402

SIZES = [2, 3, 4, 5, 6, 8]
MODELS = ["clean_s0", "clean_s1", "clean_s2", "bd_matching", "bd_low-poison", "bd_small",
          "bd_off-position", "bd_checkerboard", "bd_blend"]


def level(asr):
    return "HIGH" if asr > config.SUSPICION_THRESHOLD else ("MEDIUM" if asr > 0.25 else "LOW")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GTSRB")
    args = ap.parse_args()
    test_ds = platform_dataset(args.dataset, "test", limit=3000)
    rows = []
    for name in MODELS:
        lm = load_saved(MODELS_DIR / args.dataset / f"{name}.pt")
        row = {"model": name}
        best = (0.0, None, None, None)
        for s in SIZES:
            r = scan_backdoor(lm, test_ds, trigger_size=s, n_test_images=300)
            row[f"{s}px"] = r.most_suspicious_asr
            if r.most_suspicious_asr > best[0]:
                best = (r.most_suspicious_asr, s, r.most_suspicious_position, r.target_class)
        row.update({"multi_size_asr": best[0], "best_size": best[1], "best_position": best[2],
                    "best_target": best[3], "default_verdict": level(row["5px"]),
                    "multi_size_verdict": level(best[0])})
        rows.append(row)
        print(f"  {name:16s} " + " ".join(f"{s}px={row[f'{s}px']:.2f}" for s in SIZES)
              + f" → multi={best[0]:.2f} ({row['multi_size_verdict']}) target={best[3]}", flush=True)

    save_json(f"exp3b_probe_sweep_{args.dataset}", {"sizes": SIZES, "n_test_images": 300, "rows": rows})
    save_md(f"exp3b_probe_sweep_{args.dataset}",
            f"# Exp 3b — scanner ASR vs probe size, {args.dataset} (white square probe, 5 positions, n=300)\n\n"
            + md_table(rows, ["model"] + [f"{s}px" for s in SIZES]
                       + ["default_verdict", "multi_size_asr", "best_size", "best_target", "multi_size_verdict"]))


if __name__ == "__main__":
    main()
