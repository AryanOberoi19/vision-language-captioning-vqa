# Inference cost per configuration, batch 1, 500 test items per task, 3 repetitions (mean ± sd)

| Configuration | Task | Decoding | J/item above idle | J/item incl. idle | vs FP32 | encoder J | decoding J | Latency median ms | p95 ms | Size MiB | Memory MB | Same output as batch 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | caption | beam3 | 1.803 ± 0.023 | 1.813 ± 0.026 | 1.00 ± 0.00 | 1.002 ± 0.014 | 0.801 ± 0.009 | 168 | 190 | 1761 | 1764 | 100.0% |
| fp32 | vqa | greedy | 1.187 ± 0.014 | 1.191 ± 0.015 | 1.00 ± 0.00 | 1.002 ± 0.014 | 0.185 ± 0.005 | 77 | 87 | 1761 | 1764 | 100.0% |
| enc_fp16 | caption | beam3 | 1.525 ± 0.033 | 1.535 ± 0.036 | 0.85 ± 0.02 | 0.747 ± 0.001 | 0.778 ± 0.033 | 162 | 183 | 1182 | 1185 | 100.0% |
| enc_fp16 | vqa | greedy | 0.934 ± 0.004 | 0.938 ± 0.003 | 0.79 ± 0.01 | 0.747 ± 0.001 | 0.186 ± 0.003 | 71 | 81 | 1182 | 1185 | 100.0% |
| enc_int8 | caption | beam3 | 3.250 ± 0.025 | 3.271 ± 0.032 | 1.80 ± 0.02 | 2.474 ± 0.015 | 0.776 ± 0.040 | 340 | 363 | 894 | 896 | 85.2% |
| enc_int8 | vqa | greedy | 2.664 ± 0.019 | 2.679 ± 0.015 | 2.25 ± 0.04 | 2.474 ± 0.015 | 0.190 ± 0.004 | 252 | 261 | 894 | 896 | 98.0% |
| enc_nf4 | caption | beam3 | 1.761 ± 0.015 | 1.771 ± 0.014 | 0.98 ± 0.02 | 0.970 ± 0.004 | 0.791 ± 0.011 | 172 | 194 | 753 | 756 | 100.0% |
| enc_nf4 | vqa | greedy | 1.152 ± 0.008 | 1.157 ± 0.008 | 0.97 ± 0.01 | 0.970 ± 0.004 | 0.182 ± 0.004 | 83 | 92 | 753 | 756 | 100.0% |
| dec_fp16 | caption | beam3 | 1.613 ± 0.018 | 1.622 ± 0.021 | 0.89 ± 0.01 | 1.002 ± 0.014 | 0.611 ± 0.011 | 152 | 170 | 1460 | 1462 | 93.8% |
| dec_fp16 | vqa | greedy | 1.142 ± 0.013 | 1.147 ± 0.014 | 0.96 ± 0.00 | 1.002 ± 0.014 | 0.140 ± 0.002 | 75 | 82 | 1460 | 1462 | 98.6% |
| dec_int8 | caption | beam3 | 3.275 ± 0.034 | 3.297 ± 0.039 | 1.82 ± 0.03 | 1.002 ± 0.014 | 2.273 ± 0.040 | 368 | 421 | 1349 | 1359 | 76.4% |
| dec_int8 | vqa | greedy | 1.569 ± 0.018 | 1.577 ± 0.021 | 1.32 ± 0.00 | 1.002 ± 0.014 | 0.567 ± 0.010 | 136 | 170 | 1349 | 1359 | 96.8% |
| dec_nf4 | caption | beam3 | 2.526 ± 0.016 | 2.535 ± 0.018 | 1.40 ± 0.01 | 1.002 ± 0.014 | 1.524 ± 0.002 | 162 | 179 | 1294 | 1297 | 95.2% |
| dec_nf4 | vqa | greedy | 1.269 ± 0.006 | 1.274 ± 0.007 | 1.07 ± 0.01 | 1.002 ± 0.014 | 0.267 ± 0.009 | 77 | 86 | 1294 | 1297 | 99.2% |
| both_fp16 | caption | beam3 | 1.365 ± 0.007 | 1.374 ± 0.006 | 0.76 ± 0.01 | 0.747 ± 0.001 | 0.618 ± 0.007 | 148 | 165 | 881 | 882 | 94.6% |
| both_fp16 | vqa | greedy | 0.892 ± 0.007 | 0.896 ± 0.005 | 0.75 ± 0.01 | 0.747 ± 0.001 | 0.145 ± 0.006 | 68 | 76 | 881 | 882 | 99.6% |
| both_int8 | caption | beam3 | 4.750 ± 0.031 | 4.782 ± 0.026 | 2.64 ± 0.05 | 2.474 ± 0.015 | 2.276 ± 0.020 | 540 | 593 | 481 | 490 | 72.4% |
| both_int8 | vqa | greedy | 3.049 ± 0.027 | 3.068 ± 0.021 | 2.58 ± 0.05 | 2.474 ± 0.015 | 0.575 ± 0.012 | 310 | 338 | 481 | 490 | 95.6% |
| both_nf4 | caption | beam3 | 2.475 ± 0.007 | 2.485 ± 0.004 | 1.37 ± 0.02 | 0.970 ± 0.004 | 1.505 ± 0.006 | 166 | 184 | 286 | 288 | 91.6% |
| both_nf4 | vqa | greedy | 1.236 ± 0.005 | 1.241 ± 0.005 | 1.04 ± 0.02 | 0.970 ± 0.004 | 0.267 ± 0.002 | 83 | 92 | 286 | 288 | 99.4% |
| fp32 | caption | greedy | 1.577 ± 0.005 | 1.585 ± 0.008 | 0.87 ± 0.01 | 1.002 ± 0.014 | 0.576 ± 0.012 | 130 | 148 | 1761 | - | 100.0% |
| fp32 | caption | beam5 | 1.926 ± 0.012 | 1.937 ± 0.014 | 1.07 ± 0.01 | 1.002 ± 0.014 | 0.924 ± 0.006 | 180 | 201 | 1761 | - | 100.0% |

## Energy by component (J per item incl. idle, mean over repetitions)

| Configuration | Task | Decoding | CPU | GPU | DRAM | g CO2e per 1,000 items |
|---|---|---|---|---|---|---|
| fp32 | caption | beam3 | 0.305 (17%) | 1.206 (67%) | 0.301 (17%) | 0.37 |
| fp32 | vqa | greedy | 0.111 (9%) | 0.946 (79%) | 0.135 (11%) | 0.24 |
| enc_fp16 | caption | beam3 | 0.273 (18%) | 1.000 (65%) | 0.262 (17%) | 0.31 |
| enc_fp16 | vqa | greedy | 0.106 (11%) | 0.735 (78%) | 0.097 (10%) | 0.19 |
| enc_int8 | caption | beam3 | 0.572 (18%) | 2.329 (71%) | 0.368 (11%) | 0.66 |
| enc_int8 | vqa | greedy | 0.389 (15%) | 2.087 (78%) | 0.203 (8%) | 0.54 |
| enc_nf4 | caption | beam3 | 0.346 (20%) | 1.173 (66%) | 0.253 (14%) | 0.36 |
| enc_nf4 | vqa | greedy | 0.150 (13%) | 0.919 (79%) | 0.088 (8%) | 0.23 |
| dec_fp16 | caption | beam3 | 0.297 (18%) | 1.128 (70%) | 0.198 (12%) | 0.33 |
| dec_fp16 | vqa | greedy | 0.110 (10%) | 0.926 (81%) | 0.110 (10%) | 0.23 |
| dec_int8 | caption | beam3 | 0.835 (25%) | 2.165 (66%) | 0.296 (9%) | 0.67 |
| dec_int8 | vqa | greedy | 0.260 (16%) | 1.180 (75%) | 0.137 (9%) | 0.32 |
| dec_nf4 | caption | beam3 | 0.504 (20%) | 1.859 (73%) | 0.171 (7%) | 0.51 |
| dec_nf4 | vqa | greedy | 0.152 (12%) | 1.023 (80%) | 0.098 (8%) | 0.26 |
| both_fp16 | caption | beam3 | 0.294 (21%) | 0.920 (67%) | 0.160 (12%) | 0.28 |
| both_fp16 | vqa | greedy | 0.109 (12%) | 0.715 (80%) | 0.072 (8%) | 0.18 |
| both_int8 | caption | beam3 | 1.099 (23%) | 3.318 (69%) | 0.364 (8%) | 0.97 |
| both_int8 | vqa | greedy | 0.530 (17%) | 2.331 (76%) | 0.207 (7%) | 0.62 |
| both_nf4 | caption | beam3 | 0.525 (21%) | 1.835 (74%) | 0.124 (5%) | 0.50 |
| both_nf4 | vqa | greedy | 0.194 (16%) | 0.996 (80%) | 0.052 (4%) | 0.25 |
| fp32 | caption | greedy | 0.243 (15%) | 1.055 (67%) | 0.287 (18%) | 0.32 |
| fp32 | caption | beam5 | 0.288 (15%) | 1.329 (69%) | 0.319 (16%) | 0.39 |

Per item = the configuration's encoder window (J per image) + its decoding window (J per item), same repetition. vs FP32: ratio to the FP32 reference of the same repetition and task. Encoder and decoding columns are above idle. Size: parameters and buffers as stored, quantization scales included. Memory: GPU memory held by tensors (weights, caches, activations) with encoder and decoder loaded together, highest value after each of 20 captions and 20 answers, fresh process; the Metal driver's total (allocated in large chunks) is in memory.json. Same output as batch 64: share of first-repetition outputs identical to 05_evaluate.py's (batch 64) for the same configuration. Energy covers the SoC (CPU, GPU) and DRAM; CO2e uses J incl. idle at 727 g/kWh, PUE 1.
