# Exp 4 — fingerprinting modified models, GTSRB

Reference: clean TinyCNN (seed 0). Canary sets from the test split: k=1/class → 43 images, k=3/class → 129 images, k=10/class → 430 images.

| variant | test_acc | backdoor_asr | sha_match | agree_k1 | agree_k3 | agree_k10 | tv_k10 | verdict_k3 |
|---|---|---|---|---|---|---|---|---|
| identical copy (bytes) | 0.960 | 0.003 | True | 1.000 | 1.000 | 1.000 | 0.0000 | PASS |
| re-serialised state_dict | 0.960 | 0.003 | False | 1.000 | 1.000 | 1.000 | 0.0000 | REVIEW |
| fp16 round-trip | 0.961 | 0.003 | False | 1.000 | 1.000 | 1.000 | 0.0002 | REVIEW |
| prune 10% | 0.960 | 0.003 | False | 1.000 | 1.000 | 0.993 | 0.0022 | REVIEW |
| prune 30% | 0.963 | 0.003 | False | 1.000 | 1.000 | 0.993 | 0.0075 | REVIEW |
| prune 50% | 0.957 | 0.002 | False | 1.000 | 0.992 | 0.986 | 0.0178 | REVIEW |
| weight noise 1% σ | 0.960 | 0.003 | False | 1.000 | 1.000 | 0.995 | 0.0053 | REVIEW |
| weight noise 5% σ | 0.953 | 0.003 | False | 1.000 | 1.000 | 0.974 | 0.0286 | REVIEW |
| weight noise 20% σ | 0.844 | 0.021 | False | 0.791 | 0.845 | 0.842 | 0.1969 | ALERT |
| targeted bias +2 on target class | 0.957 | 0.011 | False | 1.000 | 0.992 | 0.993 | 0.0079 | REVIEW |
| benign fine-tune (1 ep, clean) | 0.958 | 0.002 | False | 1.000 | 1.000 | 0.979 | 0.0237 | REVIEW |
| backdoor fine-tune (1 ep, 10% poison) | 0.876 | 0.700 | False | 0.767 | 0.853 | 0.844 | 0.1791 | ALERT |
| substitute: clean model, other seed | 0.959 | 0.002 | False | 0.977 | 0.969 | 0.953 | 0.0595 | REVIEW |
| substitute: backdoored from scratch | 0.954 | 0.950 | False | 0.953 | 0.953 | 0.960 | 0.0552 | REVIEW |
