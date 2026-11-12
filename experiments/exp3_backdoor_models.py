"""
Experiment 3 — Behavioral trigger scanner (and Grad-CAM) vs. real backdoored models.

Trains TinyCNN models on the full training split (32×32) — clean ones with
different seeds, and backdoored ones with different triggers — then runs the
platform's scan_backdoor on each, with the target class *unknown* (auto
discovery, the dashboard default).

The scanner's own probe is fixed: a white square of 15 % of the input side
(5 px at 32×32) at 5 positions. The attacks deliberately vary how well they
match that probe:
  matching      white 4 px patch, bottom-right corner            (inside threat model)
  low-poison    same, but only 1 % poisoned
  small         white 3 px patch, bottom-right
  off-position  white 4 px patch at (row 4, col 14) — not a probed position
  checkerboard  4 px black/white checkerboard, bottom-right       (not solid white)
  blend         full-image noise pattern blended at α = 0.1        (outside threat model)

For every model we also report the attack's true success rate on the test set
(with its real trigger) so "scanner said LOW" can be judged against "attack
actually works". Grad-CAM: share of triggered test images where the heat on
the true patch exceeds 2× the image mean and the prediction flipped — the
rule the Explainability page uses.

Run:  python experiments/exp3_backdoor_models.py [--dataset GTSRB] [--epochs 8]
"""
import argparse

import numpy as np
from PIL import Image

from common import (IMG, MODELS_DIR, Timer, Trigger, arrays, attack_success_rate, get_or_train,  # noqa: F401
                    md_table, platform_dataset, poison_training_set, predict, save_json, save_md,
                    train_tinycnn)
from assurance.backdoor_scanner import scan_backdoor  # noqa: E402
from assurance.gradcam import explain_image, list_target_layers  # noqa: E402

TARGET = 1

ATTACKS = {
    "matching":     (Trigger("patch", 4, (IMG - 4, IMG - 4), "white"), 0.05),
    "low-poison":   (Trigger("patch", 4, (IMG - 4, IMG - 4), "white"), 0.01),
    "small":        (Trigger("patch", 3, (IMG - 3, IMG - 3), "white"), 0.05),
    "off-position": (Trigger("patch", 4, (4, 14), "white"), 0.05),
    "checkerboard": (Trigger("patch", 4, (IMG - 4, IMG - 4), "checker"), 0.05),
    "blend":        (Trigger("blend", alpha=0.1), 0.05),
}
CLEAN_SEEDS = [0, 1, 2]


def gradcam_rate(model, x_test, y_test, trig: Trigger, n=100):
    if trig.kind != "patch":
        return float("nan")
    labels, default = list_target_layers(model)
    layer = labels[default]
    idx = np.nonzero(y_test != TARGET)[0][:n]
    hits = 0
    r, c, s = trig.origin[0], trig.origin[1], trig.size
    for i in idx:
        clean_pred = int(model.predict([x_test[i]])[0])
        res = explain_image(model, Image.fromarray(trig(x_test[i])), layer_name=layer)
        ratio = res.heatmap[r:r + s, c:c + s].mean() / (res.heatmap.mean() + 1e-8)
        hits += int(ratio > 2 and res.predicted_idx != clean_pred)
    return hits / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GTSRB")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    x, y = arrays(args.dataset, "train")
    xt, yt = arrays(args.dataset, "test")
    test_ds = platform_dataset(args.dataset, "test", limit=3000)     # scanner draws its test images from here
    names = test_ds.class_names
    k = len(names)
    print(f"train {x.shape}, test {xt.shape}, {k} classes, target={names[TARGET]}")

    models = {}
    for s in CLEAN_SEEDS:
        models[f"clean-seed{s}"] = (get_or_train(
            MODELS_DIR / args.dataset / f"clean_s{s}.pt", names,
            lambda s=s: train_tinycnn(x, y, k, args.epochs, seed=s, log=f"clean s{s}")), None, 0.0)
    for name, (trig, rate) in ATTACKS.items():
        def _train(trig=trig, rate=rate, name=name):
            xp, yp, _ = poison_training_set(x, y, trig, TARGET, rate, seed=7)
            return train_tinycnn(xp, yp, k, args.epochs, seed=0, log=name)
        models[f"backdoor-{name}"] = (get_or_train(MODELS_DIR / args.dataset / f"bd_{name}.pt", names, _train),
                                      trig, rate)

    probe = Trigger("patch", 5, (IMG - 5, IMG - 5), "white")
    rows = []
    for mname, (lm, trig, rate) in models.items():
        acc = float((predict(lm.module, xt) == yt).mean())
        true_asr = attack_success_rate(lm.module, xt, yt, trig, TARGET) if trig else float("nan")
        probe_asr = attack_success_rate(lm.module, xt, yt, probe, TARGET)
        row = {"model": mname, "trigger": trig.describe() if trig else "—", "poison_rate": rate,
               "clean_acc": acc, "attack_asr_true_trigger": true_asr}
        for n_imgs in (90, 300):
            with Timer() as t:
                r = scan_backdoor(lm, test_ds, n_test_images=n_imgs)
            sfx = "" if n_imgs == 90 else "_n300"
            row.update({
                f"scan_max_asr{sfx}": r.most_suspicious_asr,
                f"scan_position{sfx}": r.most_suspicious_position,
                f"scan_target{sfx}": r.target_class,
                f"scan_verdict{sfx}": r.overall_assessment.split(" —")[0],
                f"scan_seconds{sfx}": t.s,
            })
            if n_imgs == 90:
                row["scan_eligible_images"] = r.n_test_images
                row["scan_asr_by_position"] = {tr.position: round(tr.attack_success_rate, 3)
                                               for tr in r.trigger_results}
        row["target_found"] = (row["scan_target"] == names[TARGET]) if trig else float("nan")
        row["probe_asr_full_testset"] = probe_asr
        row["gradcam_patch_flag_rate"] = gradcam_rate(lm, xt, yt, trig or ATTACKS["matching"][0])
        rows.append(row)
        print(f"  {mname:24s} acc={acc:.3f} attackASR={true_asr:.3f} | scan: {row['scan_verdict']} "
              f"max={row['scan_max_asr']:.2f}@{row['scan_position']}→{row['scan_target']} "
              f"(n300: {row['scan_verdict_n300']} {row['scan_max_asr_n300']:.2f}) "
              f"gradcam={row['gradcam_patch_flag_rate']:.2f}", flush=True)

    save_json(f"exp3_backdoor_models_{args.dataset}", {"dataset": args.dataset, "target": names[TARGET],
              "epochs": args.epochs, "scanner_probe": probe.describe(), "rows": rows})
    save_md(f"exp3_backdoor_models_{args.dataset}",
            f"# Exp 3 — trigger scanner on trained models, {args.dataset} (target `{names[TARGET]}`)\n\n"
            f"Scanner probe: {probe.describe()} at 5 positions, target auto-discovered.\n\n"
            + md_table(rows, ["model", "trigger", "poison_rate", "clean_acc", "attack_asr_true_trigger",
                              "scan_verdict", "scan_max_asr", "scan_position", "scan_target", "scan_verdict_n300",
                              "scan_max_asr_n300", "gradcam_patch_flag_rate"], {"poison_rate": ".2f"})
            + "\nGrad-CAM rate for clean models is measured with the `matching` trigger stamped "
              "(the false-alarm rate of the Explainability rule).\n")


if __name__ == "__main__":
    main()
