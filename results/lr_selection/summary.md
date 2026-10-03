# Learning-rate selection

Updated 2026-09-30T20:36:00.

## Stage A: Flickr8k, LoRA r8, ViT-B/32

Rule: rank by validation loss after 3 epochs, extend the best two to 6, choose the lower minimum.

| Rate | Validation loss per epoch | Unstable |
|---|---|---|
| 3e-5 | 2.6544, 2.5479, 2.4907 | False |
| 1e-4 | 2.5271, 2.4391, 2.3934 | False |
| 3e-4 | 2.4554, 2.3898, 2.3610, 2.3450, 2.3519, 2.3534 | False |
| 1e-3 | 2.4352, 2.3701, 2.3283, 2.3071, 2.3057, 2.3089 | False |

Extended to 6 epochs: 1e-3, 3e-4. Chosen: **1e-3** (lower minimum by 0.0393).

## Stage B: Flickr8k, full fine-tuning

| Rate | Validation loss per epoch |
|---|---|
| 1e-3 | 2.4894, 2.4438, 2.4970 |
| 3e-4 | 2.3531, 2.3037, 2.3532 |

Full fine-tuning did better at the lower rate: decide whether it keeps the shared rate (§5 as written) or gets its own.

## Stage C: COCO + VQA v2 pilot, 4,428 steps

| Rate | Caption validation loss | VQA validation loss |
|---|---|---|
| 1e-3 | 2.3385 | 1.2998 |
| 3e-4 | 2.3714 | 1.3324 |

Final learning rate: **1e-3**. Reference run seed 0 continues from `coco_ViT-B-32_lora8_lr1e-3_s0`.
