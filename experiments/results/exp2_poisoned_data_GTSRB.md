# Exp 2 — poisoned data on GTSRB (8600 images, target `1`)

## A. Poisoned samples vs label-consistency check (ResNet-18)

| variant | poison_rate | method | poisoned | flagged | recall | precision | fpr | suggests_source_class |
|---|---|---|---|---|---|---|---|---|
| patch | 0.01 | knn | 86 | 282 | 0.767 | 0.234 | 0.0254 | 0.970 |
| patch | 0.01 | linear_probe | 86 | 367 | 0.837 | 0.196 | 0.0346 | 1.000 |
| patch | 0.05 | knn | 430 | 510 | 0.765 | 0.645 | 0.0222 | 0.976 |
| patch | 0.05 | linear_probe | 430 | 731 | 0.600 | 0.353 | 0.0579 | 0.957 |
| patch | 0.10 | knn | 860 | 802 | 0.752 | 0.807 | 0.0200 | 0.986 |
| patch | 0.10 | linear_probe | 860 | 1067 | 0.462 | 0.372 | 0.0866 | 0.967 |
| blend | 0.05 | knn | 430 | 284 | 0.156 | 0.236 | 0.0266 | 0.970 |
| blend | 0.05 | linear_probe | 430 | 326 | 0.056 | 0.074 | 0.0370 | 0.917 |
| clean_label | 0.02 | knn | 172 | 225 | 0.105 | 0.080 | 0.0246 | — |
| clean_label | 0.02 | linear_probe | 172 | 274 | 0.093 | 0.058 | 0.0306 | — |

## B. Duplicate flooding at the default threshold (12)

| transform | copies | paired_with_source | flagged_any |
|---|---|---|---|
| exact copy | 150 | 1.000 | 1.000 |
| PNG→JPEG q=60 | 150 | 0.700 | 0.700 |
| gaussian noise σ=4 | 150 | 0.773 | 0.773 |
| brightness ×1.2 | 150 | 0.853 | 0.853 |
| down/up-scale 50% | 150 | 0.687 | 0.687 |
| centre crop 90% | 150 | 0.000 | 0.000 |
| rotate 8° | 150 | 0.000 | 0.000 |
| horizontal flip | 150 | 0.000 | 0.000 |

## B. Duplicates found in the unmodified dataset

| threshold | clean_dataset_groups | clean_dataset_duplicates | clean_dataset_duplicate_rate |
|---|---|---|---|
| 4 | 3 | 4 | 0.000 |
| 8 | 8 | 11 | 0.001 |
| 12 | 23 | 27 | 0.003 |
| 16 | 52 | 63 | 0.007 |
| 24 | 168 | 211 | 0.025 |
| 32 | 406 | 545 | 0.063 |

## B. Full sweep

| threshold | transform | paired_with_source | flagged_any |
|---|---|---|---|
| 4 | exact copy | 1.000 | 1.000 |
| 4 | PNG→JPEG q=60 | 0.373 | 0.373 |
| 4 | gaussian noise σ=4 | 0.427 | 0.427 |
| 4 | brightness ×1.2 | 0.527 | 0.527 |
| 4 | down/up-scale 50% | 0.393 | 0.393 |
| 4 | centre crop 90% | 0.000 | 0.000 |
| 4 | rotate 8° | 0.000 | 0.000 |
| 4 | horizontal flip | 0.000 | 0.000 |
| 8 | exact copy | 1.000 | 1.000 |
| 8 | PNG→JPEG q=60 | 0.573 | 0.573 |
| 8 | gaussian noise σ=4 | 0.667 | 0.667 |
| 8 | brightness ×1.2 | 0.707 | 0.707 |
| 8 | down/up-scale 50% | 0.600 | 0.600 |
| 8 | centre crop 90% | 0.000 | 0.000 |
| 8 | rotate 8° | 0.000 | 0.000 |
| 8 | horizontal flip | 0.000 | 0.000 |
| 12 | exact copy | 1.000 | 1.000 |
| 12 | PNG→JPEG q=60 | 0.700 | 0.700 |
| 12 | gaussian noise σ=4 | 0.773 | 0.773 |
| 12 | brightness ×1.2 | 0.853 | 0.853 |
| 12 | down/up-scale 50% | 0.687 | 0.687 |
| 12 | centre crop 90% | 0.000 | 0.000 |
| 12 | rotate 8° | 0.000 | 0.000 |
| 12 | horizontal flip | 0.000 | 0.000 |
| 16 | exact copy | 1.000 | 1.000 |
| 16 | PNG→JPEG q=60 | 0.733 | 0.733 |
| 16 | gaussian noise σ=4 | 0.853 | 0.853 |
| 16 | brightness ×1.2 | 0.947 | 0.947 |
| 16 | down/up-scale 50% | 0.760 | 0.760 |
| 16 | centre crop 90% | 0.000 | 0.000 |
| 16 | rotate 8° | 0.000 | 0.000 |
| 16 | horizontal flip | 0.000 | 0.000 |
| 24 | exact copy | 1.000 | 1.000 |
| 24 | PNG→JPEG q=60 | 0.827 | 0.827 |
| 24 | gaussian noise σ=4 | 0.927 | 0.927 |
| 24 | brightness ×1.2 | 0.987 | 0.987 |
| 24 | down/up-scale 50% | 0.820 | 0.820 |
| 24 | centre crop 90% | 0.000 | 0.013 |
| 24 | rotate 8° | 0.000 | 0.000 |
| 24 | horizontal flip | 0.000 | 0.000 |
| 32 | exact copy | 1.000 | 1.000 |
| 32 | PNG→JPEG q=60 | 0.920 | 0.927 |
| 32 | gaussian noise σ=4 | 0.953 | 0.953 |
| 32 | brightness ×1.2 | 0.980 | 0.987 |
| 32 | down/up-scale 50% | 0.900 | 0.907 |
| 32 | centre crop 90% | 0.007 | 0.047 |
| 32 | rotate 8° | 0.000 | 0.000 |
| 32 | horizontal flip | 0.000 | 0.000 |
