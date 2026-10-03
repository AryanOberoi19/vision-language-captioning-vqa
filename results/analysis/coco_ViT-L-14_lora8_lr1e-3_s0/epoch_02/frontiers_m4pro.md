# RQ1: accuracy-cost frontiers, MacBook Pro M4 Pro (coco_ViT-L-14_lora8_lr1e-3_s0 epoch_02.pt, Karpathy test)

Dominance: a configuration dominates another if it is not worse on either axis and better on at least one. Accuracy differences count when the paired 95 % bootstrap interval of the difference excludes zero (1,000 resamples of the test images, shared by all configurations); cost differences when they exceed the pooled sd over the repetitions (energy, latency) or 1 % (size, memory). Energy above idle (methodology §8). Accuracy in points (x 100).


## CIDEr vs energy (captions)

Frontier: Both FP16

| Configuration | CIDEr [95 % CI] | energy (J) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 111.7 [109.7, 113.7] | 1.365 ± 0.007 | yes |  |
| Encoder FP16 | 111.7 [109.7, 113.7] | 1.525 ± 0.033 |  | Both FP16 |
| FP32 greedy | 106.1 [104.2, 108.1] | 1.577 ± 0.005 |  | Encoder FP16, Both FP16 |
| Decoder FP16 | 111.8 [109.8, 113.9] | 1.613 ± 0.018 |  | Encoder FP16, Both FP16 |
| Encoder NF4 | 111.4 [109.3, 113.4] | 1.761 ± 0.015 |  | Encoder FP16, Decoder FP16, Both FP16 |
| FP32 | 111.8 [109.8, 113.7] | 1.803 ± 0.023 |  | Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| FP32 beam 5 | 110.8 [108.8, 112.8] | 1.926 ± 0.012 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Both NF4 | 110.3 [108.4, 112.3] | 2.475 ± 0.007 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Decoder NF4 | 110.2 [108.2, 112.1] | 2.526 ± 0.016 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, Both NF4, FP32 beam 5 |
| Encoder INT8 | 111.7 [109.8, 113.8] | 3.250 ± 0.025 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Decoder INT8 | 111.6 [109.6, 113.8] | 3.275 ± 0.034 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Both INT8 | 112.1 [110.2, 114.2] | 4.750 ± 0.031 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |

## SPICE vs energy (captions)

Frontier: Both FP16

| Configuration | SPICE [95 % CI] | energy (J) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 20.0 [19.7, 20.3] | 1.365 ± 0.007 | yes |  |
| Encoder FP16 | 20.0 [19.7, 20.3] | 1.525 ± 0.033 |  | Both FP16 |
| FP32 greedy | 19.4 [19.1, 19.6] | 1.577 ± 0.005 |  | Encoder FP16, Both FP16 |
| Decoder FP16 | 20.0 [19.7, 20.3] | 1.613 ± 0.018 |  | Encoder FP16, Both FP16 |
| Encoder NF4 | 20.0 [19.7, 20.3] | 1.761 ± 0.015 |  | Encoder FP16, Decoder FP16, Both FP16 |
| FP32 | 20.0 [19.8, 20.3] | 1.803 ± 0.023 |  | Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| FP32 beam 5 | 19.9 [19.6, 20.2] | 1.926 ± 0.012 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Both NF4 | 19.9 [19.6, 20.1] | 2.475 ± 0.007 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Decoder NF4 | 19.9 [19.6, 20.1] | 2.526 ± 0.016 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, Both NF4, FP32 beam 5 |
| Encoder INT8 | 20.0 [19.7, 20.3] | 3.250 ± 0.025 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Decoder NF4, Both FP16, Both NF4, FP32 beam 5 |
| Decoder INT8 | 20.1 [19.8, 20.4] | 3.275 ± 0.034 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Both INT8 | 20.1 [19.8, 20.4] | 4.750 ± 0.031 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |

## CHAIR_i vs energy (captions)

Frontier: Both FP16

| Configuration | CHAIR_i [95 % CI] | energy (J) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 6.5 [5.9, 7.0] | 1.365 ± 0.007 | yes |  |
| Encoder FP16 | 6.5 [5.9, 7.0] | 1.525 ± 0.033 |  | Both FP16 |
| FP32 greedy | 7.5 [6.9, 8.0] | 1.577 ± 0.005 |  | Encoder FP16, Both FP16 |
| Decoder FP16 | 6.4 [5.9, 6.9] | 1.613 ± 0.018 |  | Encoder FP16, Both FP16 |
| Encoder NF4 | 6.4 [5.8, 6.9] | 1.761 ± 0.015 |  | Encoder FP16, Decoder FP16, Both FP16 |
| FP32 | 6.5 [5.9, 7.0] | 1.803 ± 0.023 |  | Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| FP32 beam 5 | 6.0 [5.5, 6.6] | 1.926 ± 0.012 |  | Encoder NF4 |
| Both NF4 | 6.6 [6.0, 7.1] | 2.475 ± 0.007 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Decoder NF4 | 6.8 [6.3, 7.4] | 2.526 ± 0.016 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, Both NF4, FP32 beam 5 |
| Encoder INT8 | 6.3 [5.8, 6.9] | 3.250 ± 0.025 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, Both NF4, FP32 beam 5 |
| Decoder INT8 | 6.4 [5.9, 6.9] | 3.275 ± 0.034 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, Both NF4, FP32 beam 5 |
| Both INT8 | 6.5 [5.9, 7.0] | 4.750 ± 0.031 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Decoder NF4, Both FP16, Both NF4, FP32 beam 5 |

## CIDEr vs latency (captions)

Frontier: FP32 greedy, Both FP16

| Configuration | CIDEr [95 % CI] | latency (ms) | On frontier | Dominated by |
|---|---|---|---|---|
| FP32 greedy | 106.1 [104.2, 108.1] | 130 ± 0.8 | yes |  |
| Both FP16 | 111.7 [109.7, 113.7] | 148 ± 0.5 | yes |  |
| Decoder FP16 | 111.8 [109.8, 113.9] | 152 ± 1.4 |  | Both FP16 |
| Decoder NF4 | 110.2 [108.2, 112.1] | 162 ± 1.2 |  | Encoder FP16, Decoder FP16, Both FP16 |
| Encoder FP16 | 111.7 [109.7, 113.7] | 162 ± 1.0 |  | Decoder FP16, Both FP16 |
| Both NF4 | 110.3 [108.4, 112.3] | 166 ± 1.1 |  | Encoder FP16, Decoder FP16, Decoder NF4, Both FP16 |
| FP32 | 111.8 [109.8, 113.7] | 168 ± 0.9 |  | Encoder FP16, Decoder FP16, Both FP16 |
| Encoder NF4 | 111.4 [109.3, 113.4] | 172 ± 0.5 |  | FP32, Encoder FP16, Decoder FP16, Both FP16 |
| FP32 beam 5 | 110.8 [108.8, 112.8] | 180 ± 2.1 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Decoder NF4, Both FP16, Both NF4 |
| Encoder INT8 | 111.7 [109.8, 113.8] | 340 ± 3.2 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Decoder INT8 | 111.6 [109.6, 113.8] | 368 ± 6.4 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Both INT8 | 112.1 [110.2, 114.2] | 540 ± 7.2 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |

## CIDEr vs size (captions)

Frontier: Both NF4, Both INT8

| Configuration | CIDEr [95 % CI] | size (MiB) | On frontier | Dominated by |
|---|---|---|---|---|
| Both NF4 | 110.3 [108.4, 112.3] | 286 | yes |  |
| Both INT8 | 112.1 [110.2, 114.2] | 481 | yes |  |
| Encoder NF4 | 111.4 [109.3, 113.4] | 753 |  | Both INT8 |
| Both FP16 | 111.7 [109.7, 113.7] | 881 |  | Encoder NF4, Both INT8 |
| Encoder INT8 | 111.7 [109.8, 113.8] | 894 |  | Encoder NF4, Both FP16, Both INT8 |
| Encoder FP16 | 111.7 [109.7, 113.7] | 1182 |  | Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder NF4 | 110.2 [108.2, 112.1] | 1294 |  | Encoder FP16, Encoder INT8, Encoder NF4, Both FP16, Both INT8, Both NF4 |
| Decoder INT8 | 111.6 [109.6, 113.8] | 1349 |  | Encoder FP16, Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder FP16 | 111.8 [109.8, 113.9] | 1460 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder INT8, Both FP16, Both INT8 |
| FP32 | 111.8 [109.8, 113.7] | 1761 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16, Both INT8 |
| FP32 greedy | 106.1 [104.2, 108.1] | 1761 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Decoder NF4, Both FP16, Both INT8, Both NF4, FP32 beam 5 |
| FP32 beam 5 | 110.8 [108.8, 112.8] | 1761 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Decoder NF4, Both FP16, Both INT8, Both NF4 |

## CIDEr vs memory (captions)

Frontier: Both NF4, Both INT8

| Configuration | CIDEr [95 % CI] | memory (MB) | On frontier | Dominated by |
|---|---|---|---|---|
| Both NF4 | 110.3 [108.4, 112.3] | 288 | yes |  |
| Both INT8 | 112.1 [110.2, 114.2] | 490 | yes |  |
| Encoder NF4 | 111.4 [109.3, 113.4] | 756 |  | Both INT8 |
| Both FP16 | 111.7 [109.7, 113.7] | 882 |  | Encoder NF4, Both INT8 |
| Encoder INT8 | 111.7 [109.8, 113.8] | 896 |  | Encoder NF4, Both FP16, Both INT8 |
| Encoder FP16 | 111.7 [109.7, 113.7] | 1185 |  | Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder NF4 | 110.2 [108.2, 112.1] | 1297 |  | Encoder FP16, Encoder INT8, Encoder NF4, Both FP16, Both INT8, Both NF4 |
| Decoder INT8 | 111.6 [109.6, 113.8] | 1359 |  | Encoder FP16, Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder FP16 | 111.8 [109.8, 113.9] | 1462 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder INT8, Both FP16, Both INT8 |
| FP32 | 111.8 [109.8, 113.7] | 1764 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16, Both INT8 |

## VQA vs energy (answers)

Frontier: Both FP16

| Configuration | VQA [95 % CI] | energy (J) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 57.9 [57.2, 58.6] | 0.892 ± 0.007 | yes |  |
| Encoder FP16 | 57.9 [57.2, 58.6] | 0.934 ± 0.004 |  | Both FP16 |
| Decoder FP16 | 57.9 [57.2, 58.6] | 1.142 ± 0.013 |  | Encoder FP16, Both FP16 |
| Encoder NF4 | 57.7 [57.1, 58.4] | 1.152 ± 0.008 |  | Encoder FP16, Both FP16 |
| FP32 | 57.9 [57.2, 58.6] | 1.187 ± 0.014 |  | Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Both NF4 | 57.2 [56.6, 57.9] | 1.236 ± 0.005 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Decoder NF4 | 57.5 [56.9, 58.2] | 1.269 ± 0.006 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Decoder INT8 | 57.8 [57.2, 58.5] | 1.569 ± 0.018 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Encoder INT8 | 57.8 [57.2, 58.5] | 2.664 ± 0.019 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |
| Both INT8 | 57.8 [57.2, 58.5] | 3.049 ± 0.027 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |

## VQA vs latency (answers)

Frontier: Both FP16

| Configuration | VQA [95 % CI] | latency (ms) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 57.9 [57.2, 58.6] | 68 ± 0.4 | yes |  |
| Encoder FP16 | 57.9 [57.2, 58.6] | 71 ± 0.4 |  | Both FP16 |
| Decoder FP16 | 57.9 [57.2, 58.6] | 75 ± 1.2 |  | Encoder FP16, Both FP16 |
| Decoder NF4 | 57.5 [56.9, 58.2] | 77 ± 1.1 |  | FP32, Encoder FP16, Decoder FP16, Both FP16 |
| FP32 | 57.9 [57.2, 58.6] | 77 ± 1.1 |  | Encoder FP16, Decoder FP16, Both FP16 |
| Both NF4 | 57.2 [56.6, 57.9] | 83 ± 0.7 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Decoder NF4, Both FP16 |
| Encoder NF4 | 57.7 [57.1, 58.4] | 83 ± 0.9 |  | FP32, Encoder FP16, Decoder FP16, Decoder NF4, Both FP16 |
| Decoder INT8 | 57.8 [57.2, 58.5] | 136 ± 1.6 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Encoder INT8 | 57.8 [57.2, 58.5] | 252 ± 1.3 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |
| Both INT8 | 57.8 [57.2, 58.5] | 310 ± 0.6 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |

## VQA vs size (answers)

Frontier: Both NF4, Both INT8

| Configuration | VQA [95 % CI] | size (MiB) | On frontier | Dominated by |
|---|---|---|---|---|
| Both NF4 | 57.2 [56.6, 57.9] | 286 | yes |  |
| Both INT8 | 57.8 [57.2, 58.5] | 481 | yes |  |
| Encoder NF4 | 57.7 [57.1, 58.4] | 753 |  | Both INT8 |
| Both FP16 | 57.9 [57.2, 58.6] | 881 |  | Both INT8 |
| Encoder INT8 | 57.8 [57.2, 58.5] | 894 |  | Encoder NF4, Both FP16, Both INT8 |
| Encoder FP16 | 57.9 [57.2, 58.6] | 1182 |  | Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder NF4 | 57.5 [56.9, 58.2] | 1294 |  | Encoder FP16, Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder INT8 | 57.8 [57.2, 58.5] | 1349 |  | Encoder FP16, Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder FP16 | 57.9 [57.2, 58.6] | 1460 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder INT8, Both FP16, Both INT8 |
| FP32 | 57.9 [57.2, 58.6] | 1761 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16, Both INT8 |

## Accuracy per joule (above idle)

| Configuration | CIDEr per J | VQA accuracy per J |
|---|---|---|
| FP32 | 62.0 | 48.8 |
| Encoder FP16 | 73.2 | 62.0 |
| Encoder INT8 | 34.4 | 21.7 |
| Encoder NF4 | 63.2 | 50.1 |
| Decoder FP16 | 69.3 | 50.7 |
| Decoder INT8 | 34.1 | 36.9 |
| Decoder NF4 | 43.6 | 45.3 |
| Both FP16 | 81.8 | 64.9 |
| Both INT8 | 23.6 | 19.0 |
| Both NF4 | 44.6 | 46.3 |
| FP32 greedy | 67.3 | - |
| FP32 beam 5 | 57.5 | - |
