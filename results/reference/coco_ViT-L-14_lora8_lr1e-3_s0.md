# coco_ViT-L-14_lora8_lr1e-3_s0 (Karpathy val, beam 3)

| Epochs | CIDEr | VQA | BLEU-4 | SPICE | CHAIR_i | CHAIR_s | Words | Val loss cap / VQA | Train h | Train Wh |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 93.2 | 49.5 | 27.2 | 17.6 | 9.2 | 14.0 | 9.7 | 2.3062 / 1.2720 | 0.58 | 10.5 |
| 1 | 106.1 | 54.7 | 32.4 | 19.1 | 7.9 | 11.8 | 9.2 | 2.1398 / 1.1261 | 2.25 | 44.2 |
| 2 | 109.2 | 57.4 | 32.6 | 19.9 | 6.7 | 9.9 | 9.8 | 2.0811 / 1.0753 | 2.20 | 46.2 |
| 3 | 110.5 | 57.7 | 33.4 | 19.9 | 6.5 | 10.0 | 9.5 | 2.0649 / 1.0703 | 2.18 | 46.9 |

Train h and Wh are per epoch. Epoch 1's include the pilot's 4,428 steps (row 0.25), summed over the sessions that trained them.

Last epoch added CIDEr +1.3, VQA +0.3. Proposed stopping rule (stop when CIDEr < +1.0 and VQA < +0.5): continue.
