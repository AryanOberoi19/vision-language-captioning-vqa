# Encoder check (4,428-step pilots, Karpathy val, beam 3)

Rule: ViT-L/14 if CIDEr +5.0 or VQA +2.0 points, and neither lower by more than 1.0 CIDEr / 0.5 VQA; else ViT-B/32.

| | ViT-B/32 | ViT-L/14 | Difference |
|---|---|---|---|
| CIDEr | 87.0 | 93.2 | +6.2 |
| VQA | 48.1 | 49.5 | +1.4 |
| BLEU-4 | 26.0 | 27.2 | +1.3 |
| SPICE | 16.8 | 17.6 | +0.8 |
| CLIPScore | 68.5 | 68.7 | +0.2 |
| CHAIR_i | 12.3 | 9.2 | -3.1 |
| val_loss_caption | 2.3385 | 2.3062 | -0.0323 |
| val_loss_vqa | 1.2998 | 1.2720 | -0.0278 |
| train_seconds | 1980 | 2083 | +103 |
| train_wh | 12.47 | 10.46 | -2.01 |

Chosen: **ViT-L/14**; the reference run continues coco_ViT-L-14_lora8_lr1e-3_s0.
