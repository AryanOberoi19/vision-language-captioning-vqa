# VizWiz-Captions val, zero-shot (coco_ViT-L-14_lora8_lr1e-3_s0 epoch_02.pt; 7,542 images with references)

Trained on COCO + VQA v2 only. Karpathy test values in brackets for comparison. CIDEr vs FP32: paired 95 % bootstrap interval over the VizWiz images (the step-10 rule).

| Configuration | Why it is here | CIDEr [CI] (COCO) | BLEU-4 (COCO) | SPICE (COCO) | CLIPScore (COCO) | Words | CIDEr vs FP32 [CI] |
|---|---|---|---|---|---|---|---|
| fp32 | reference | 31.1 [30.1, 32.0] (111.8) | 13.4 (33.5) | 7.9 (20.0) | 59.1 (72.4) | 9.7 | - |
| both_fp16 | energy frontier | 31.1 [30.2, 32.1] (111.7) | 13.4 (33.4) | 7.9 (20.0) | 59.1 (72.4) | 9.7 | +0.0 [-0.1, +0.2] |
| fp32_greedy | latency frontier | 30.8 [29.9, 31.7] (106.1) | 12.9 (31.1) | 7.9 (19.4) | 59.8 (72.9) | 9.6 | -0.3 [-0.8, +0.3] |
| both_nf4 | size / memory frontier | 31.1 [30.1, 31.9] (110.3) | 13.4 (33.0) | 7.9 (19.9) | 59.2 (72.3) | 9.7 | -0.0 [-0.5, +0.5] |
| both_int8 | size / memory frontier | 31.0 [30.0, 31.9] (112.1) | 13.4 (33.5) | 7.9 (20.1) | 59.0 (72.3) | 9.7 | -0.1 [-0.5, +0.3] |
