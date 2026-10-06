# CNN-LSTM baseline (showtell_resnet50_s0) vs the FP32 reference model (coco_ViT-L-14_lora8_lr1e-3_s0 epoch_02.pt)

Karpathy test, beam3. Baseline: frozen ResNet-50, 13,504,886 trained parameters (LSTM decoder), best epoch 18 of 20 by validation CIDEr (91.4). Reference: CLIP ViT-L/14 + mapping network + GPT-2 with LoRA, 33,336,576 trained parameters.

## Accuracy

| Model | CIDEr [CI] | BLEU-4 | SPICE | CLIPScore | CHAIR_i | CHAIR_s | Words |
|---|---|---|---|---|---|---|---|
| Show and Tell (ResNet-50 + LSTM) | 93.1 [91.2, 94.9] | 28.9 [28.2, 29.5] | 17.9 [17.6, 18.2] | 67.9 [67.5, 68.2] | 13.2 [12.5, 14.0] | 18.9 [17.9, 20.1] | 9.9 |
| FP32 reference | 111.8 [109.8, 113.7] | 33.5 [32.8, 34.2] | 20.0 [19.8, 20.3] | 72.4 [72.1, 72.7] | 6.5 [5.9, 7.0] | 9.8 [8.9, 10.6] | 9.5 |

## Inference cost (batch 1, J per caption, mean ± sd over repetitions, this session)

| Model | J above idle | J incl. idle | vs FP32 (incl. idle) | encoder J | decoding J | Latency median ms | p95 ms | Size MiB | Memory MB |
|---|---|---|---|---|---|---|---|---|---|
| Show and Tell | 0.1907 ± 0.0040 | 0.2314 ± 0.0038 | 0.104 ± 0.003 | 0.0883 ± 0.0022 | 0.1024 ± 0.0018 | 26 | 29 | 141 | 142 |
| FP32 reference | 1.9785 ± 0.0433 | 2.2230 ± 0.0410 | 1 | 1.0685 ± 0.0097 | 0.9100 ± 0.0487 | 160 | 179 | 1761 | 1764 |

Check against step 10 (other session): FP32 reference 1.8027 ± 0.0230 J above idle there, 1.9785 ± 0.0433 J here; 1.8128 ± 0.0255 vs 2.2230 ± 0.0410 J incl. idle.

## Training cost (incl. idle)

| Model | Feature extraction Wh | Training Wh | Training h | Epochs | Wh per epoch |
|---|---|---|---|---|---|
| Show and Tell | 3.7 | 25.4 | 1.27 | 20 | 1.3 |
| FP32 reference | 32.0 | 137.3 | 6.63 | 3 | 45.8 |

## Baseline epochs

| Epoch | lr | Train loss | Val CIDEr | Train min | Wh |
|---|---|---|---|---|---|
| 0 | 5.00e-04 | 3.265 | 75.2 | 3.8 | 1.30 |
| 1 | 5.00e-04 | 2.767 | 82.5 | 4.0 | 1.14 |
| 2 | 5.00e-04 | 2.664 | 84.4 | 3.8 | 1.24 |
| 3 | 4.00e-04 | 2.598 | 84.9 | 3.8 | 1.28 |
| 4 | 4.00e-04 | 2.560 | 87.7 | 3.8 | 1.33 |
| 5 | 4.00e-04 | 2.533 | 86.9 | 3.8 | 1.27 |
| 6 | 3.20e-04 | 2.503 | 87.8 | 3.8 | 1.27 |
| 7 | 3.20e-04 | 2.485 | 88.9 | 3.8 | 1.27 |
| 8 | 3.20e-04 | 2.469 | 88.9 | 3.8 | 1.28 |
| 9 | 2.56e-04 | 2.450 | 89.7 | 3.8 | 1.27 |
| 10 | 2.56e-04 | 2.438 | 89.6 | 3.8 | 1.28 |
| 11 | 2.56e-04 | 2.428 | 89.9 | 3.8 | 1.26 |
| 12 | 2.05e-04 | 2.414 | 90.0 | 3.8 | 1.28 |
| 13 | 2.05e-04 | 2.405 | 89.7 | 3.8 | 1.26 |
| 14 | 2.05e-04 | 2.398 | 89.8 | 3.8 | 1.27 |
| 15 | 1.64e-04 | 2.387 | 91.0 | 3.8 | 1.28 |
| 16 | 1.64e-04 | 2.381 | 91.0 | 3.8 | 1.27 |
| 17 | 1.64e-04 | 2.375 | 91.4 | 3.8 | 1.29 |
| 18 | 1.31e-04 | 2.367 | 91.4 | 3.8 | 1.29 |
| 19 | 1.31e-04 | 2.362 | 91.1 | 3.8 | 1.29 |

Energy covers the SoC (CPU, GPU) and DRAM. CO2e at 727 g/kWh: multiply Wh by 0.727 g. Training Wh: the training steps only (validation decoding excluded), as for the reference model; the reference's feature extraction covers its COCO train, val and test features at FP32.
