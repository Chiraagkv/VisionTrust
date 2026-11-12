#!/usr/bin/env bash
# Reproduce every experiment. Downloads GTSRB (~690 MB unpacked) into experiments/data/raw.
# Total runtime on a 12-core CPU: roughly 1–1.5 h (most of it training 9 small CNNs).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
DS=${DATASET:-GTSRB}
mkdir -p logs
$PY -u exp1_label_noise.py    --dataset "$DS" | tee "logs/exp1_$DS.log"
$PY -u exp2_poisoned_data.py  --dataset "$DS" | tee "logs/exp2_$DS.log"
$PY -u exp3_backdoor_models.py --dataset "$DS" | tee "logs/exp3_$DS.log"   # trains the models exp4/5 reuse
$PY -u exp4_fingerprint.py    --dataset "$DS" | tee "logs/exp4_$DS.log"
$PY -u exp5_provenance.py     --dataset "$DS" | tee "logs/exp5_$DS.log"
