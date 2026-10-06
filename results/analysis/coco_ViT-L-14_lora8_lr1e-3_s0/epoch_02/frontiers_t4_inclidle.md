# RQ1: accuracy-cost frontiers, Colab T4 (coco_ViT-L-14_lora8_lr1e-3_s0 epoch_02.pt, Karpathy test)

Dominance: a configuration dominates another if it is not worse on either axis and better on at least one. Accuracy differences count when the paired 95 % bootstrap interval of the difference excludes zero (1,000 resamples of the test images, shared by all configurations); cost differences when they exceed the pooled sd over the repetitions (energy, latency) or 1 % (size, memory). Energy including idle (the device's idle draw during each item counts; methodology §8 uses above idle, frontiers_<platform>.md). Accuracy in points (x 100).

Cost measured on Colab T4 (results/variants/coco_ViT-L-14_lora8_lr1e-3_s0/epoch_02_t4/energy.json); accuracy is the Mac's full Karpathy-test evaluation at batch 64. batch_check_t4.md scores this platform's batch-1 outputs on the 500-item subset against it.


## CIDEr vs energy (captions)

Frontier: Both FP16

| Configuration | CIDEr [95 % CI] | energy (J) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 111.7 [109.7, 113.7] | 8.510 ± 0.108 | yes |  |
| Encoder FP16 | 111.7 [109.7, 113.7] | 9.227 ± 0.037 |  | Both FP16 |
| Encoder NF4 | 111.4 [109.3, 113.4] | 10.351 ± 0.035 |  | Encoder FP16, Both FP16 |
| FP32 greedy | 106.1 [104.2, 108.1] | 11.007 ± 0.073 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Both NF4 | 110.3 [108.4, 112.3] | 11.376 ± 0.148 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Decoder FP16 | 111.8 [109.8, 113.9] | 12.078 ± 0.121 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Encoder INT8 | 111.7 [109.8, 113.8] | 12.781 ± 0.011 |  | Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| FP32 | 111.8 [109.8, 113.7] | 12.937 ± 0.155 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16 |
| Decoder NF4 | 110.2 [108.2, 112.1] | 14.063 ± 0.206 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, Both NF4 |
| FP32 beam 5 | 110.8 [108.8, 112.8] | 14.866 ± 0.044 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder NF4, Both FP16, Both NF4 |
| Decoder INT8 | 111.6 [109.6, 113.8] | 21.279 ± 0.293 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Both INT8 | 112.1 [110.2, 114.2] | 21.388 ± 0.516 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16 |

## SPICE vs energy (captions)

Frontier: Both FP16

| Configuration | SPICE [95 % CI] | energy (J) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 20.0 [19.7, 20.3] | 8.510 ± 0.108 | yes |  |
| Encoder FP16 | 20.0 [19.7, 20.3] | 9.227 ± 0.037 |  | Both FP16 |
| Encoder NF4 | 20.0 [19.7, 20.3] | 10.351 ± 0.035 |  | Encoder FP16, Both FP16 |
| FP32 greedy | 19.4 [19.1, 19.6] | 11.007 ± 0.073 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Both NF4 | 19.9 [19.6, 20.1] | 11.376 ± 0.148 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Decoder FP16 | 20.0 [19.7, 20.3] | 12.078 ± 0.121 |  | Encoder FP16, Encoder NF4, Both FP16, Both NF4 |
| Encoder INT8 | 20.0 [19.7, 20.3] | 12.781 ± 0.011 |  | Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, Both NF4 |
| FP32 | 20.0 [19.8, 20.3] | 12.937 ± 0.155 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, Both NF4 |
| Decoder NF4 | 19.9 [19.6, 20.1] | 14.063 ± 0.206 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, Both NF4 |
| FP32 beam 5 | 19.9 [19.6, 20.2] | 14.866 ± 0.044 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder NF4, Both FP16, Both NF4 |
| Decoder INT8 | 20.1 [19.8, 20.4] | 21.279 ± 0.293 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16 |
| Both INT8 | 20.1 [19.8, 20.4] | 21.388 ± 0.516 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16 |

## CHAIR_i vs energy (captions)

Frontier: Both FP16

| Configuration | CHAIR_i [95 % CI] | energy (J) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 6.5 [5.9, 7.0] | 8.510 ± 0.108 | yes |  |
| Encoder FP16 | 6.5 [5.9, 7.0] | 9.227 ± 0.037 |  | Both FP16 |
| Encoder NF4 | 6.4 [5.8, 6.9] | 10.351 ± 0.035 |  | Encoder FP16, Both FP16 |
| FP32 greedy | 7.5 [6.9, 8.0] | 11.007 ± 0.073 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Both NF4 | 6.6 [6.0, 7.1] | 11.376 ± 0.148 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Decoder FP16 | 6.4 [5.9, 6.9] | 12.078 ± 0.121 |  | Encoder FP16, Encoder NF4, Both FP16, Both NF4 |
| Encoder INT8 | 6.3 [5.8, 6.9] | 12.781 ± 0.011 |  | Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, Both NF4 |
| FP32 | 6.5 [5.9, 7.0] | 12.937 ± 0.155 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, Both NF4 |
| Decoder NF4 | 6.8 [6.3, 7.4] | 14.063 ± 0.206 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, Both NF4 |
| FP32 beam 5 | 6.0 [5.5, 6.6] | 14.866 ± 0.044 |  | Encoder INT8, Encoder NF4 |
| Decoder INT8 | 6.4 [5.9, 6.9] | 21.279 ± 0.293 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, Both NF4, FP32 beam 5 |
| Both INT8 | 6.5 [5.9, 7.0] | 21.388 ± 0.516 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder NF4, Both FP16, Both NF4, FP32 beam 5 |

## CIDEr vs latency (captions)

Frontier: Encoder FP16

| Configuration | CIDEr [95 % CI] | latency (ms) | On frontier | Dominated by |
|---|---|---|---|---|
| Encoder FP16 | 111.7 [109.7, 113.7] | 197 ± 2.1 | yes |  |
| Both FP16 | 111.7 [109.7, 113.7] | 204 ± 2.1 |  | Encoder FP16 |
| FP32 greedy | 106.1 [104.2, 108.1] | 226 ± 1.3 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Encoder NF4 | 111.4 [109.3, 113.4] | 226 ± 1.9 |  | Encoder FP16, Both FP16 |
| FP32 | 111.8 [109.8, 113.7] | 254 ± 5.6 |  | Encoder FP16, Encoder NF4, Both FP16 |
| FP32 beam 5 | 110.8 [108.8, 112.8] | 256 ± 2.3 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Decoder FP16 | 111.8 [109.8, 113.9] | 257 ± 1.1 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Both NF4 | 110.3 [108.4, 112.3] | 283 ± 2.3 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Encoder INT8 | 111.7 [109.8, 113.8] | 304 ± 3.7 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Decoder NF4 | 110.2 [108.2, 112.1] | 316 ± 1.7 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, Both NF4, FP32 beam 5 |
| Decoder INT8 | 111.6 [109.6, 113.8] | 534 ± 7.3 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, FP32 beam 5 |
| Both INT8 | 112.1 [110.2, 114.2] | 588 ± 10.9 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |

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
| Both NF4 | 110.3 [108.4, 112.3] | 298 | yes |  |
| Both INT8 | 112.1 [110.2, 114.2] | 492 | yes |  |
| Encoder NF4 | 111.4 [109.3, 113.4] | 766 |  | Both INT8 |
| Both FP16 | 111.7 [109.7, 113.7] | 897 |  | Encoder NF4, Both INT8 |
| Encoder INT8 | 111.7 [109.8, 113.8] | 905 |  | Encoder NF4, Both INT8 |
| Encoder FP16 | 111.7 [109.7, 113.7] | 1194 |  | Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder NF4 | 110.2 [108.2, 112.1] | 1306 |  | Encoder FP16, Encoder INT8, Encoder NF4, Both FP16, Both INT8, Both NF4 |
| Decoder INT8 | 111.6 [109.6, 113.8] | 1360 |  | Encoder FP16, Encoder INT8, Encoder NF4, Both FP16, Both INT8 |
| Decoder FP16 | 111.8 [109.8, 113.9] | 1477 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder INT8, Both FP16, Both INT8 |
| FP32 | 111.8 [109.8, 113.7] | 1772 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16, Both INT8 |

## VQA vs energy (answers)

Frontier: Both FP16

| Configuration | VQA [95 % CI] | energy (J) | On frontier | Dominated by |
|---|---|---|---|---|
| Both FP16 | 57.9 [57.2, 58.6] | 2.600 ± 0.034 | yes |  |
| Encoder FP16 | 57.9 [57.2, 58.6] | 2.655 ± 0.012 |  | Both FP16 |
| Encoder NF4 | 57.7 [57.1, 58.4] | 3.713 ± 0.011 |  | Encoder FP16, Both FP16 |
| Both NF4 | 57.2 [56.6, 57.9] | 3.905 ± 0.024 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Encoder INT8 | 57.8 [57.2, 58.5] | 6.127 ± 0.077 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Decoder FP16 | 57.9 [57.2, 58.6] | 6.172 ± 0.025 |  | Encoder FP16, Encoder NF4, Both FP16 |
| FP32 | 57.9 [57.2, 58.6] | 6.209 ± 0.020 |  | Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16 |
| Decoder NF4 | 57.5 [56.9, 58.2] | 6.425 ± 0.034 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16 |
| Both INT8 | 57.8 [57.2, 58.5] | 7.696 ± 0.115 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16 |
| Decoder INT8 | 57.8 [57.2, 58.5] | 7.890 ± 0.037 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Both FP16, Both INT8 |

## VQA vs latency (answers)

Frontier: Encoder FP16, Both FP16

| Configuration | VQA [95 % CI] | latency (ms) | On frontier | Dominated by |
|---|---|---|---|---|
| Encoder FP16 | 57.9 [57.2, 58.6] | 50 ± 0.6 | yes |  |
| Both FP16 | 57.9 [57.2, 58.6] | 51 ± 0.6 | yes |  |
| Encoder NF4 | 57.7 [57.1, 58.4] | 77 ± 4.0 |  | Encoder FP16, Both FP16 |
| Both NF4 | 57.2 [56.6, 57.9] | 86 ± 5.0 |  | Encoder FP16, Encoder NF4, Both FP16 |
| FP32 | 57.9 [57.2, 58.6] | 102 ± 0.1 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Decoder FP16 | 57.9 [57.2, 58.6] | 102 ± 1.0 |  | Encoder FP16, Encoder NF4, Both FP16 |
| Decoder NF4 | 57.5 [56.9, 58.2] | 110 ± 0.3 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Decoder INT8 | 57.8 [57.2, 58.5] | 149 ± 2.8 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Encoder INT8 | 57.8 [57.2, 58.5] | 150 ± 1.7 |  | FP32, Encoder FP16, Encoder NF4, Decoder FP16, Both FP16 |
| Both INT8 | 57.8 [57.2, 58.5] | 198 ± 2.4 |  | FP32, Encoder FP16, Encoder INT8, Encoder NF4, Decoder FP16, Decoder INT8, Both FP16 |

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

## Accuracy per joule (incl. idle)

| Configuration | CIDEr per J | VQA accuracy per J |
|---|---|---|
| FP32 | 8.6 | 9.3 |
| Encoder FP16 | 12.1 | 21.8 |
| Encoder INT8 | 8.7 | 9.4 |
| Encoder NF4 | 10.8 | 15.5 |
| Decoder FP16 | 9.3 | 9.4 |
| Decoder INT8 | 5.2 | 7.3 |
| Decoder NF4 | 7.8 | 9.0 |
| Both FP16 | 13.1 | 22.3 |
| Both INT8 | 5.2 | 7.5 |
| Both NF4 | 9.7 | 14.7 |
| FP32 greedy | 9.6 | - |
| FP32 beam 5 | 7.5 | - |
