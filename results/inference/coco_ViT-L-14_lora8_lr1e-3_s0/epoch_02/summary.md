# Inference cost, batch 1, 500 test items per pass (mean ± sd over repeats)

| Precision | Task | Decoding | Repeats | J/item above idle | J/item incl. idle | encoder J | decoding J | Latency median ms | p95 ms | Idle W | Peak Metal MB | Size MB | Agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | caption | beam3 | 5 | 1.667 ± 0.018 | 1.702 ± 0.011 | 0.957 ± 0.010 | 0.710 ± 0.018 | 162 | 181 | 0.217 ± 0.084 | 2094 | 1761 | 100.0% |
| fp32 | vqa | greedy | 5 | 1.134 ± 0.006 | 1.149 ± 0.014 | 0.955 ± 0.009 | 0.179 ± 0.004 | 74 | 84 | 0.201 ± 0.160 | 2078 | 1761 | 100.0% |

## Energy by component (J per item incl. idle, mean over repeats; share of the total)

| Precision | Task | Decoding | CPU | GPU | DRAM | g CO2e per 1,000 items |
|---|---|---|---|---|---|---|
| fp32 | caption | beam3 | 0.269 (16%) | 1.134 (67%) | 0.299 (18%) | 0.34 |
| fp32 | vqa | greedy | 0.104 (9%) | 0.913 (79%) | 0.132 (11%) | 0.23 |

GFLOPs (fp32, CPU count over 3 images): encoder 155.5 per image; caption decoding (beam3) 13.60; answer decoding 4.14.

Encoder and decoding columns are per item, above idle. Agreement: share of first-repeat outputs identical to the batched evaluation's (scripts/05_evaluate.py). Energy covers the SoC (CPU, GPU) and DRAM only; CO2e uses J per item incl. idle at 727 g/kWh (CEA, Indian grid), PUE 1.
