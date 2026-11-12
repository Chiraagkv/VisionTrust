# Exp 2 — poisoned data on CIFAR10 (5000 images, target `automobile`)

## A. Poisoned samples vs label-consistency check (ResNet-18)

| variant | poison_rate | method | poisoned | flagged | recall | precision | fpr | suggests_source_class |
|---|---|---|---|---|---|---|---|---|
| patch | 0.01 | knn | 50 | 400 | 0.740 | 0.092 | 0.0733 | 0.892 |
| patch | 0.01 | linear_probe | 50 | 917 | 0.920 | 0.050 | 0.1760 | 0.804 |
| patch | 0.05 | knn | 250 | 485 | 0.668 | 0.344 | 0.0669 | 0.934 |
| patch | 0.05 | linear_probe | 250 | 1194 | 0.720 | 0.151 | 0.2135 | 0.811 |
| patch | 0.10 | knn | 500 | 556 | 0.594 | 0.534 | 0.0576 | 0.912 |
| patch | 0.10 | linear_probe | 500 | 1389 | 0.590 | 0.212 | 0.2431 | 0.854 |
| blend | 0.05 | knn | 250 | 364 | 0.048 | 0.033 | 0.0741 | 1.000 |
| blend | 0.05 | linear_probe | 250 | 846 | 0.036 | 0.011 | 0.1762 | 0.889 |
| clean_label | 0.02 | knn | 100 | 378 | 0.100 | 0.026 | 0.0751 | — |
| clean_label | 0.02 | linear_probe | 100 | 877 | 0.080 | 0.009 | 0.1773 | — |

## B. Duplicate flooding at the default threshold (12)

| transform | copies | paired_with_source | flagged_any |
|---|---|---|---|
| exact copy | 150 | 1.000 | 1.000 |
| PNG→JPEG q=60 | 150 | 0.660 | 0.660 |
| gaussian noise σ=4 | 150 | 0.967 | 0.967 |
| brightness ×1.2 | 150 | 0.833 | 0.833 |
| down/up-scale 50% | 150 | 0.147 | 0.147 |
| centre crop 90% | 150 | 0.000 | 0.000 |
| rotate 8° | 150 | 0.000 | 0.000 |
| horizontal flip | 150 | 0.000 | 0.000 |

## B. Duplicates found in the unmodified dataset

| threshold | clean_dataset_groups | clean_dataset_duplicates | clean_dataset_duplicate_rate |
|---|---|---|---|
| 4 | 0 | 0 | 0.000 |
| 8 | 0 | 0 | 0.000 |
| 12 | 0 | 0 | 0.000 |
| 16 | 0 | 0 | 0.000 |
| 24 | 1 | 1 | 0.000 |
| 32 | 2 | 2 | 0.000 |

## B. Full sweep

| threshold | transform | paired_with_source | flagged_any |
|---|---|---|---|
| 4 | exact copy | 1.000 | 1.000 |
| 4 | PNG→JPEG q=60 | 0.040 | 0.040 |
| 4 | gaussian noise σ=4 | 0.400 | 0.400 |
| 4 | brightness ×1.2 | 0.413 | 0.413 |
| 4 | down/up-scale 50% | 0.000 | 0.000 |
| 4 | centre crop 90% | 0.000 | 0.000 |
| 4 | rotate 8° | 0.000 | 0.000 |
| 4 | horizontal flip | 0.000 | 0.000 |
| 8 | exact copy | 1.000 | 1.000 |
| 8 | PNG→JPEG q=60 | 0.307 | 0.307 |
| 8 | gaussian noise σ=4 | 0.833 | 0.833 |
| 8 | brightness ×1.2 | 0.653 | 0.653 |
| 8 | down/up-scale 50% | 0.013 | 0.013 |
| 8 | centre crop 90% | 0.000 | 0.000 |
| 8 | rotate 8° | 0.000 | 0.000 |
| 8 | horizontal flip | 0.000 | 0.000 |
| 12 | exact copy | 1.000 | 1.000 |
| 12 | PNG→JPEG q=60 | 0.660 | 0.660 |
| 12 | gaussian noise σ=4 | 0.967 | 0.967 |
| 12 | brightness ×1.2 | 0.833 | 0.833 |
| 12 | down/up-scale 50% | 0.147 | 0.147 |
| 12 | centre crop 90% | 0.000 | 0.000 |
| 12 | rotate 8° | 0.000 | 0.000 |
| 12 | horizontal flip | 0.000 | 0.000 |
| 16 | exact copy | 1.000 | 1.000 |
| 16 | PNG→JPEG q=60 | 0.900 | 0.900 |
| 16 | gaussian noise σ=4 | 1.000 | 1.000 |
| 16 | brightness ×1.2 | 0.927 | 0.927 |
| 16 | down/up-scale 50% | 0.540 | 0.540 |
| 16 | centre crop 90% | 0.000 | 0.000 |
| 16 | rotate 8° | 0.000 | 0.000 |
| 16 | horizontal flip | 0.000 | 0.000 |
| 24 | exact copy | 1.000 | 1.000 |
| 24 | PNG→JPEG q=60 | 0.987 | 0.987 |
| 24 | gaussian noise σ=4 | 1.000 | 1.000 |
| 24 | brightness ×1.2 | 0.980 | 0.980 |
| 24 | down/up-scale 50% | 0.953 | 0.953 |
| 24 | centre crop 90% | 0.000 | 0.000 |
| 24 | rotate 8° | 0.000 | 0.000 |
| 24 | horizontal flip | 0.000 | 0.000 |
| 32 | exact copy | 1.000 | 1.000 |
| 32 | PNG→JPEG q=60 | 1.000 | 1.000 |
| 32 | gaussian noise σ=4 | 1.000 | 1.000 |
| 32 | brightness ×1.2 | 1.000 | 1.000 |
| 32 | down/up-scale 50% | 1.000 | 1.000 |
| 32 | centre crop 90% | 0.000 | 0.000 |
| 32 | rotate 8° | 0.000 | 0.000 |
| 32 | horizontal flip | 0.000 | 0.000 |
