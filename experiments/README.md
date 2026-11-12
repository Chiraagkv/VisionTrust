# Experiments: how well does VisionTrust work on real data?

These scripts evaluate each assurance component on real torchvision datasets:

- **GTSRB**: German traffic signs. 43 classes, 26,640 train and 12,630 test images.
- **CIFAR-10**: natural images, 10 classes.

Every experiment calls the platform's own code (`assurance/*`) the same way the dashboard does. That covers ingestion, the label check, duplicate detection, the trigger scanner, Grad-CAM, fingerprinting and provenance. Only two things are implemented here rather than in the platform: building the attacks, and training the models under test.

**Findings are in [REPORT.md](REPORT.md).**

## Layout

| File | What it measures |
|---|---|
| `common.py` | Shared helpers: dataset ingestion via `load_torchvision`, triggers, TinyCNN training, metrics |
| `exp1_label_noise.py` | Label-consistency check against injected label noise (0–20 %) |
| `exp1b_clean_flags.py` | What the check flags on the unmodified dataset (pairs and a contact sheet) |
| `exp2_poisoned_data.py` | (A) the label check against backdoor-poisoned training data; (B) pHash against duplicate flooding |
| `exp3_backdoor_models.py` | Trains 3 clean and 6 backdoored models, then runs the trigger scanner and the Grad-CAM rule on each |
| `exp3b_probe_sweep.py` | Scanner sensitivity to probe size; evaluates a multi-size scan |
| `exp4_fingerprint.py` | SHA-256 and canary fingerprints against 14 model modifications |
| `exp5_provenance.py` | Hash chain against 14 tampering attacks; cost of creating and verifying records |
| `make_figures.py` | Summary figures for the report |
| `results/` | `*.json` (raw numbers), `*.md` (tables) and `figures/` |
| `logs/` | Console output of the runs reported in REPORT.md |

## Running

```bash
pip install -r ../requirements.txt
./run_all.sh                        # GTSRB, all experiments
DATASET=CIFAR10 python exp1_label_noise.py --dataset CIFAR10 --limit 5000
python make_figures.py
```

- **Storage:** datasets download into `experiments/data/raw/`. That is about 690 MB for GTSRB and 170 MB for CIFAR-10, and the CIFAR server can be slow. Trained models go to `experiments/models/`. Both folders are git-ignored.
- **Runtime:** on a 12-core CPU without a GPU, each experiment takes 5–25 minutes. Training 9 TinyCNNs (8 epochs each) takes about 30 minutes. Running jobs in parallel slows each one 4–5×, so set `OMP_NUM_THREADS`.
- **Isolation:** the scripts point the platform's reference registry, canary sets and inference chain at `experiments/models/`. Running experiments never touches the dashboard's own `models/` or `records/` folders.
- **Reproducibility:** results are deterministic for a given seed, apart from small CPU-threading differences in training.
