# Exp 3 — trigger scanner on trained models, GTSRB (target `1`)

Scanner probe: 5×5 white patch at (row 27, col 27) at 5 positions, target auto-discovered.

| model | trigger | poison_rate | clean_acc | attack_asr_true_trigger | scan_verdict | scan_max_asr | scan_position | scan_target | scan_verdict_n300 | scan_max_asr_n300 | gradcam_patch_flag_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean-seed0 | — | 0.00 | 0.960 | — | LOW | 0.120 | center | 34 | LOW | 0.087 | 0.000 |
| clean-seed1 | — | 0.00 | 0.959 | — | LOW | 0.173 | center | 23 | LOW | 0.172 | 0.010 |
| clean-seed2 | — | 0.00 | 0.960 | — | LOW | 0.099 | center | 10 | LOW | 0.080 | 0.000 |
| backdoor-matching | 4×4 white patch at (row 28, col 28) | 0.05 | 0.954 | 0.950 | MEDIUM | 0.293 | bottom-right | 1 | MEDIUM | 0.342 | 0.920 |
| backdoor-low-poison | 4×4 white patch at (row 28, col 28) | 0.01 | 0.956 | 0.906 | MEDIUM | 0.329 | bottom-right | 1 | MEDIUM | 0.320 | 0.270 |
| backdoor-small | 3×3 white patch at (row 29, col 29) | 0.05 | 0.954 | 0.940 | MEDIUM | 0.358 | bottom-right | 1 | MEDIUM | 0.363 | 0.900 |
| backdoor-off-position | 4×4 white patch at (row 4, col 14) | 0.05 | 0.958 | 0.986 | LOW | 0.159 | center | 34 | LOW | 0.095 | 0.970 |
| backdoor-checkerboard | 4×4 checker patch at (row 28, col 28) | 0.05 | 0.961 | 1.000 | LOW | 0.122 | bottom-right | 1 | LOW | 0.106 | 0.960 |
| backdoor-blend | full-image noise blend, alpha=0.1 | 0.05 | 0.951 | 0.968 | LOW | 0.062 | center | 34 | LOW | 0.058 | — |

Grad-CAM rate for clean models is measured with the `matching` trigger stamped (the false-alarm rate of the Explainability rule).
