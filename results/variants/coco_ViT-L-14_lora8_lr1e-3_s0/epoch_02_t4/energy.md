# Inference cost per configuration, batch 1, 500 test items per task, 3 repetitions (mean ± sd)

| Configuration | Task | Decoding | J/item above idle | J/item incl. idle | vs FP32 | encoder J | decoding J | Latency median ms | p95 ms | Size MiB | Memory MB | Same output as batch 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | caption | beam3 | 4.301 ± 0.042 | 12.937 ± 0.155 | 1.00 ± 0.00 | 2.480 ± 0.018 | 1.821 ± 0.027 | 254 | 337 | 1761 | 1772 | 100.0% |
| fp32 | vqa | greedy | 2.702 ± 0.009 | 6.209 ± 0.020 | 1.00 ± 0.00 | 2.480 ± 0.018 | 0.222 ± 0.009 | 102 | 122 | 1761 | 1772 | 100.0% |
| enc_fp16 | caption | beam3 | 2.414 ± 0.152 | 9.227 ± 0.037 | 0.71 ± 0.01 | 0.649 ± 0.007 | 1.765 ± 0.157 | 197 | 277 | 1182 | 1194 | 99.2% |
| enc_fp16 | vqa | greedy | 0.871 ± 0.015 | 2.655 ± 0.012 | 0.43 ± 0.00 | 0.649 ± 0.007 | 0.222 ± 0.022 | 50 | 73 | 1182 | 1194 | 100.0% |
| enc_int8 | caption | beam3 | 2.541 ± 0.147 | 12.781 ± 0.011 | 0.99 ± 0.01 | 0.723 ± 0.062 | 1.817 ± 0.095 | 304 | 380 | 894 | 905 | 85.4% |
| enc_int8 | vqa | greedy | 0.944 ± 0.066 | 6.127 ± 0.077 | 0.99 ± 0.01 | 0.723 ± 0.062 | 0.221 ± 0.007 | 150 | 193 | 894 | 905 | 98.0% |
| enc_nf4 | caption | beam3 | 2.640 ± 0.086 | 10.351 ± 0.035 | 0.80 ± 0.01 | 0.883 ± 0.012 | 1.757 ± 0.073 | 226 | 307 | 753 | 766 | 99.4% |
| enc_nf4 | vqa | greedy | 1.099 ± 0.003 | 3.713 ± 0.011 | 0.60 ± 0.00 | 0.883 ± 0.012 | 0.216 ± 0.009 | 77 | 101 | 753 | 766 | 100.0% |
| dec_fp16 | caption | beam3 | 3.367 ± 0.071 | 12.078 ± 0.121 | 0.93 ± 0.01 | 2.480 ± 0.018 | 0.887 ± 0.070 | 257 | 337 | 1460 | 1477 | 91.8% |
| dec_fp16 | vqa | greedy | 2.615 ± 0.039 | 6.172 ± 0.025 | 0.99 ± 0.01 | 2.480 ± 0.018 | 0.135 ± 0.032 | 102 | 126 | 1460 | 1477 | 98.8% |
| dec_int8 | caption | beam3 | 3.204 ± 0.081 | 21.279 ± 0.293 | 1.65 ± 0.04 | 2.480 ± 0.018 | 0.724 ± 0.067 | 534 | 702 | 1349 | 1360 | 76.8% |
| dec_int8 | vqa | greedy | 2.637 ± 0.076 | 7.890 ± 0.037 | 1.27 ± 0.01 | 2.480 ± 0.018 | 0.157 ± 0.064 | 149 | 202 | 1349 | 1360 | 97.0% |
| dec_nf4 | caption | beam3 | 3.401 ± 0.210 | 14.063 ± 0.206 | 1.09 ± 0.01 | 2.480 ± 0.018 | 0.921 ± 0.195 | 316 | 418 | 1294 | 1306 | 90.4% |
| dec_nf4 | vqa | greedy | 2.587 ± 0.032 | 6.425 ± 0.034 | 1.03 ± 0.00 | 2.480 ± 0.018 | 0.107 ± 0.014 | 110 | 140 | 1294 | 1306 | 98.8% |
| both_fp16 | caption | beam3 | 1.508 ± 0.073 | 8.510 ± 0.108 | 0.66 ± 0.01 | 0.649 ± 0.007 | 0.859 ± 0.079 | 204 | 283 | 881 | 897 | 92.4% |
| both_fp16 | vqa | greedy | 0.796 ± 0.023 | 2.600 ± 0.034 | 0.42 ± 0.01 | 0.649 ± 0.007 | 0.147 ± 0.029 | 51 | 73 | 881 | 897 | 98.6% |
| both_int8 | caption | beam3 | 1.543 ± 0.477 | 21.388 ± 0.516 | 1.65 ± 0.03 | 0.723 ± 0.062 | 0.819 ± 0.418 | 588 | 762 | 481 | 492 | 71.6% |
| both_int8 | vqa | greedy | 0.847 ± 0.115 | 7.696 ± 0.115 | 1.24 ± 0.02 | 0.723 ± 0.062 | 0.124 ± 0.055 | 198 | 260 | 481 | 492 | 95.6% |
| both_nf4 | caption | beam3 | 1.767 ± 0.077 | 11.376 ± 0.148 | 0.88 ± 0.02 | 0.883 ± 0.012 | 0.884 ± 0.082 | 283 | 377 | 286 | 298 | 90.0% |
| both_nf4 | vqa | greedy | 0.985 ± 0.032 | 3.905 ± 0.024 | 0.63 ± 0.00 | 0.883 ± 0.012 | 0.102 ± 0.020 | 86 | 112 | 286 | 298 | 99.2% |
| fp32 | caption | greedy | 3.376 ± 0.069 | 11.007 ± 0.073 | 0.85 ± 0.02 | 2.480 ± 0.018 | 0.896 ± 0.055 | 226 | 295 | 1761 | - | 100.0% |
| fp32 | caption | beam5 | 6.187 ± 0.133 | 14.866 ± 0.044 | 1.15 ± 0.02 | 2.480 ± 0.018 | 3.707 ± 0.116 | 256 | 332 | 1761 | - | 100.0% |

Per item = the configuration's encoder window (J per image) + its decoding window (J per item), same repetition. vs FP32: ratio to the FP32 reference of the same repetition and task. Encoder and decoding columns are above idle. Size: parameters and buffers as stored, quantization scales included. Memory: GPU memory held by tensors (weights, caches, activations) with encoder and decoder loaded together, highest value after each of 20 captions and 20 answers, fresh process; the CUDA allocator's reserved total is in memory.json. Same output as batch 64: share of first-repetition outputs identical to 05_evaluate.py's (batch 64) for the same configuration. Energy covers the GPU board only (NVML; no CPU or DRAM counters on this platform); no CO2e is given, since the data centre's grid intensity and PUE are unknown.
