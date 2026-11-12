# Exp 3b — scanner ASR vs probe size, GTSRB (white square probe, 5 positions, n=300)

| model | 2px | 3px | 4px | 5px | 6px | 8px | default_verdict | multi_size_asr | best_size | best_target | multi_size_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_s0 | 0.014 | 0.061 | 0.087 | 0.087 | 0.101 | 0.235 | LOW | 0.235 | 8 | 22 | LOW |
| clean_s1 | 0.037 | 0.084 | 0.070 | 0.172 | 0.220 | 0.326 | LOW | 0.326 | 8 | 22 | MEDIUM |
| clean_s2 | 0.054 | 0.072 | 0.058 | 0.080 | 0.207 | 0.214 | LOW | 0.214 | 8 | 13 | LOW |
| bd_matching | 0.036 | 0.251 | 0.964 | 0.342 | 0.385 | 0.462 | MEDIUM | 0.964 | 4 | 1 | HIGH |
| bd_low-poison | 0.022 | 0.407 | 0.920 | 0.320 | 0.389 | 0.237 | MEDIUM | 0.920 | 4 | 1 | HIGH |
| bd_small | 0.037 | 0.952 | 0.590 | 0.363 | 0.154 | 0.238 | MEDIUM | 0.952 | 3 | 1 | HIGH |
| bd_off-position | 0.025 | 0.069 | 0.145 | 0.095 | 0.153 | 0.393 | LOW | 0.393 | 8 | 22 | MEDIUM |
| bd_checkerboard | 0.073 | 0.139 | 0.103 | 0.106 | 0.114 | 0.249 | LOW | 0.249 | 8 | 22 | LOW |
| bd_blend | 0.033 | 0.070 | 0.095 | 0.058 | 0.161 | 0.194 | LOW | 0.194 | 8 | 15 | LOW |
