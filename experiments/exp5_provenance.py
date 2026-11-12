"""
Experiment 5 — Inference provenance chain vs. tampering.

Builds a real chain with the platform (create_inference_record on test
images with a trained model), then applies each tampering attack to a fresh
copy of the chain at random positions (30 trials each) and checks whether
verify_entire_chain() notices.

Attacks include ones an unkeyed SHA-256 chain *cannot* catch (tail
truncation, appending a correctly-hashed forged record, rewriting a record
and re-hashing everything after it) — to measure the real boundary of the
guarantee, not just the easy cases.

Also measures cost: ms per created record and verification throughput.

Run:  python experiments/exp5_provenance.py [--dataset GTSRB]   (after exp3)
"""
import argparse
import copy
import random
import shutil
import time

from common import MODELS_DIR, load_saved, md_table, platform_dataset, save_json, save_md  # noqa: E402  (sets sys.path)
import config  # noqa: E402
from assurance import provenance as pv  # noqa: E402

N_RECORDS = 60
TRIALS = 30


def rehash_from(recs, start):
    """Attacker with full write access recomputes hashes from `start` onward."""
    prev = recs[start - 1].record_hash if start > 0 else pv.GENESIS_HASH
    for r in recs[start:]:
        r.previous_record_hash = prev
        r.record_hash = pv._compute_record_hash(r)
        prev = r.record_hash


def forge_record(template, prev_hash, output):
    r = copy.deepcopy(template)
    r.inference_id = "forged-" + r.inference_id[7:]
    r.output, r.previous_record_hash = output, prev_hash
    r.timestamp = str(time.time())
    r.record_hash = pv._compute_record_hash(r)
    return r


def flip_first(h):
    """Change one hex digit — always a real change (unlike prefixing '0')."""
    return ("1" if h[0] == "0" else "0") + h[1:]


def attacks(names):
    other = lambda o: next(c for c in names if c != o)   # noqa: E731

    def field(f, val):
        def a(recs, rng):
            i = rng.randrange(len(recs))
            setattr(recs[i], f, val(getattr(recs[i], f)))
            return recs
        return a

    def delete_middle(recs, rng):
        del recs[rng.randrange(1, len(recs) - 1)]
        return recs

    def delete_tail(recs, rng):
        return recs[:len(recs) - rng.randint(1, 5)]

    def swap(recs, rng):
        i = rng.randrange(len(recs) - 1)
        recs[i], recs[i + 1] = recs[i + 1], recs[i]
        return recs

    def insert_middle(recs, rng):
        i = rng.randrange(1, len(recs))
        recs.insert(i, forge_record(recs[i], recs[i - 1].record_hash, other(recs[i].output)))
        return recs

    def append_forged(recs, rng):
        recs.append(forge_record(recs[-1], recs[-1].record_hash, other(recs[-1].output)))
        return recs

    def edit_and_rehash_one(recs, rng):
        i = rng.randrange(len(recs) - 1)          # not the last record
        recs[i].output = other(recs[i].output)
        recs[i].record_hash = pv._compute_record_hash(recs[i])
        return recs

    def edit_and_rehash_suffix(recs, rng):
        i = rng.randrange(len(recs))
        recs[i].output = other(recs[i].output)
        rehash_from(recs, i)
        return recs

    return {
        "change output (class)":            ("field edit", field("output", other)),
        "change confidence":                ("field edit", field("confidence", lambda c: round(min(1, c + 0.1), 4))),
        "change timestamp":                 ("field edit", field("timestamp", lambda t: str(float(t) - 3600))),
        "change input_hash":                ("field edit", field("input_hash", flip_first)),
        "change model_hash":                ("field edit", field("model_hash", flip_first)),
        "change preprocessing_hash":        ("field edit", field("preprocessing_hash", flip_first)),
        "change nonce":                     ("field edit", field("nonce", flip_first)),
        "delete a middle record":           ("structure", delete_middle),
        "swap two adjacent records":        ("structure", swap),
        "insert forged record (middle)":    ("structure", insert_middle),
        "edit + re-hash that record only":  ("re-hash", edit_and_rehash_one),
        "truncate last 1–4 records":        ("outside model", delete_tail),
        "append correctly-hashed forgery":  ("outside model", append_forged),
        "edit + re-hash all later records": ("outside model", edit_and_rehash_suffix),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GTSRB")
    args = ap.parse_args()
    model = load_saved(MODELS_DIR / args.dataset / "clean_s0.pt")
    test_ds = platform_dataset(args.dataset, "test", limit=3000)

    pv.clear_chain()
    t0 = time.time()
    for r in test_ds.records[:N_RECORDS]:
        pv.create_inference_record(r.filename, model)
    create_ms = 1000 * (time.time() - t0) / N_RECORDS
    base = pv._load_chain()
    assert pv.verify_entire_chain().is_intact
    shutil.copy(config.INFERENCE_CHAIN_PATH, str(MODELS_DIR / "records" / "pristine_chain.jsonl"))

    rows = []
    rng = random.Random(0)
    for name, (kind, attack) in attacks(model.class_names).items():
        detected = 0
        for _ in range(TRIALS):
            recs = attack(copy.deepcopy(base), rng)
            pv._save_chain(recs)
            detected += int(not pv.verify_entire_chain().is_intact)
        rows.append({"attack": name, "class": kind, "trials": TRIALS, "detected": detected,
                     "detection_rate": detected / TRIALS})
        print(f"  {name:36s} {detected}/{TRIALS}", flush=True)

    # verification throughput on a long chain
    long = [copy.deepcopy(base[i % len(base)]) for i in range(10_000)]
    rehash_from(long, 0)
    pv._save_chain(long)
    t0 = time.time()
    ok = pv.verify_entire_chain().is_intact
    verify_s = time.time() - t0
    pv.clear_chain()

    save_json(f"exp5_provenance_{args.dataset}", {"records": N_RECORDS, "trials": TRIALS,
              "create_ms_per_record": create_ms, "verify_10k_seconds": verify_s, "verify_10k_intact": ok,
              "rows": rows})
    save_md(f"exp5_provenance_{args.dataset}",
            f"# Exp 5 — provenance chain tampering ({N_RECORDS}-record chain, {TRIALS} trials per attack)\n\n"
            + md_table(rows, ["attack", "class", "trials", "detected", "detection_rate"])
            + f"\nCost: {create_ms:.1f} ms to create a record (includes re-hashing the model file); "
              f"verifying 10,000 records took {verify_s:.2f} s.\n")


if __name__ == "__main__":
    main()
