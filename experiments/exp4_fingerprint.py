"""
Experiment 4 — Model fingerprinting vs. real model modifications.

A clean model is registered as the trusted reference (platform API
register_reference) with a canary set built from the test split
(build_canary_set, k images per class). We then derive modified models and
run fingerprint_model on each.

Modifications, from benign re-encodings to malicious changes:
  identical copy · re-serialised state_dict · fp16 round-trip
  magnitude pruning 10/30/50 % · Gaussian weight noise (1 %, 5 %, 20 % of each tensor's std)
  benign fine-tune (1 epoch, clean data, lr 1e-4)
  backdoor fine-tune (1 epoch, 10 % BadNets-poisoned data, lr 1e-4)  ← stealthy implant
  targeted bias tamper (+2.0 on one class logit)
  substitution: independently trained clean model (other seed)
  substitution: backdoored model trained from scratch (from exp3)

For each: SHA-256 match, canary top-1 agreement, mean output shift (total
variation), the platform's verdict, plus ground truth (test accuracy, backdoor
ASR) so we can see which changes matter and which the fingerprint catches.

Run:  python experiments/exp4_fingerprint.py [--dataset GTSRB]   (after exp3)
"""
import argparse
import copy
import shutil

import numpy as np
import torch

from common import (IMG, MODELS_DIR, Trigger, arrays, attack_success_rate, load_saved, md_table,  # noqa: F401
                    platform_dataset, poison_training_set, predict, save_json, save_md, save_model,
                    seed_all, train_tinycnn)
from assurance.model_fingerprint import build_canary_set, fingerprint_model, register_reference  # noqa: E402

TARGET = 1
TRIG = Trigger("patch", 4, (IMG - 4, IMG - 4), "white")


def modified(ref_module, fn):
    m = copy.deepcopy(ref_module)
    with torch.no_grad():
        fn(m)
    m.eval()
    return m


def prune(frac):
    def f(m):
        ws = [p for n, p in m.named_parameters() if p.ndim > 1]
        allw = torch.cat([w.abs().flatten() for w in ws])
        thr = torch.quantile(allw, frac)
        for w in ws:
            w.mul_((w.abs() > thr).float())
    return f


def noise(scale, seed=0):
    def f(m):
        g = torch.Generator().manual_seed(seed)
        for p in m.parameters():
            if p.numel() > 1:
                p.add_(torch.randn(p.shape, generator=g) * p.std() * scale)
    return f


def fp16(m):
    for p in m.parameters():
        p.copy_(p.half().float())
    for b in m.buffers():
        if b.dtype.is_floating_point:
            b.copy_(b.half().float())


def bias_tamper(m):
    m.classifier[-1].bias[TARGET] += 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GTSRB")
    args = ap.parse_args()
    d = MODELS_DIR / args.dataset
    ref_path = d / "clean_s0.pt"
    if not ref_path.exists():
        raise SystemExit("Run exp3 first (it trains the reference model).")

    x, y = arrays(args.dataset, "train")
    xt, yt = arrays(args.dataset, "test")
    test_ds = platform_dataset(args.dataset, "test", limit=3000)
    names = test_ds.class_names
    ref = load_saved(ref_path)

    canaries = {pc: build_canary_set(test_ds, per_class=pc, seed=11, name=f"{args.dataset}-test-k{pc}")
                for pc in (1, 3, 10)}
    for pc, c in canaries.items():
        register_reference(f"ref-k{pc}", ref, c)
    print("canary sets:", {pc: len(c) for pc, c in canaries.items()})

    vdir = d / "variants"
    vdir.mkdir(parents=True, exist_ok=True)
    variants = {}

    shutil.copy(ref_path, vdir / "identical_copy.pt")
    shutil.copy(ref_path.with_suffix(".json"), vdir / "identical_copy.json")
    variants["identical copy (bytes)"] = load_saved(vdir / "identical_copy.pt")
    variants["re-serialised state_dict"] = save_model(copy.deepcopy(ref.module), vdir / "reserialised.pt", names)
    variants["fp16 round-trip"] = save_model(modified(ref.module, fp16), vdir / "fp16.pt", names)
    for frac in (0.1, 0.3, 0.5):
        variants[f"prune {frac:.0%}"] = save_model(modified(ref.module, prune(frac)), vdir / f"prune{frac}.pt", names)
    for s in (0.01, 0.05, 0.2):
        variants[f"weight noise {s:.0%} σ"] = save_model(modified(ref.module, noise(s)), vdir / f"noise{s}.pt", names)
    variants["targeted bias +2 on target class"] = save_model(modified(ref.module, bias_tamper),
                                                             vdir / "bias.pt", names)

    rng = np.random.RandomState(5)
    sub = rng.choice(len(x), 8000, replace=False)
    variants["benign fine-tune (1 ep, clean)"] = save_model(
        train_tinycnn(x[sub], y[sub], len(names), 1, seed=3, init=copy.deepcopy(ref.module), lr=1e-4,
                      log="ft-clean"), vdir / "ft_clean.pt", names)
    xp, yp, _ = poison_training_set(x[sub], y[sub], TRIG, TARGET, 0.10, seed=9)
    variants["backdoor fine-tune (1 ep, 10% poison)"] = save_model(
        train_tinycnn(xp, yp, len(names), 1, seed=3, init=copy.deepcopy(ref.module), lr=1e-4,
                      log="ft-backdoor"), vdir / "ft_backdoor.pt", names)
    variants["substitute: clean model, other seed"] = load_saved(d / "clean_s1.pt")
    variants["substitute: backdoored from scratch"] = load_saved(d / "bd_matching.pt")

    rows = []
    for vname, lm in variants.items():
        acc = float((predict(lm.module, xt) == yt).mean())
        asr = attack_success_rate(lm.module, xt, yt, TRIG, TARGET)
        row = {"variant": vname, "test_acc": acc, "backdoor_asr": asr}
        for pc in canaries:
            fr = fingerprint_model(lm, f"ref-k{pc}")
            if fr.scan_error:
                raise RuntimeError(fr.scan_error)
            row["sha_match"] = fr.binary_match
            row[f"agree_k{pc}"] = fr.behavioral_match_pct / 100
            row[f"tv_k{pc}"] = fr.mean_prob_shift
            row[f"verdict_k{pc}"] = fr.assessment.split(" —")[0]
        rows.append(row)
        print(f"  {vname:40s} acc={acc:.3f} asr={asr:.3f} sha={row['sha_match']} "
              f"agree(k1/k3/k10)={row['agree_k1']:.3f}/{row['agree_k3']:.3f}/{row['agree_k10']:.3f} "
              f"tv_k10={row['tv_k10']:.4f} verdict={row['verdict_k3']}", flush=True)

    save_json(f"exp4_fingerprint_{args.dataset}", {"dataset": args.dataset,
              "canary_sizes": {f"k{pc}": len(c) for pc, c in canaries.items()}, "rows": rows})
    save_md(f"exp4_fingerprint_{args.dataset}",
            f"# Exp 4 — fingerprinting modified models, {args.dataset}\n\n"
            f"Reference: clean TinyCNN (seed 0). Canary sets from the test split: "
            + ", ".join(f"k={pc}/class → {len(c)} images" for pc, c in canaries.items()) + ".\n\n"
            + md_table(rows, ["variant", "test_acc", "backdoor_asr", "sha_match", "agree_k1", "agree_k3", "agree_k10",
                              "tv_k10", "verdict_k3"], {"tv_k10": ".4f"}))


if __name__ == "__main__":
    main()
