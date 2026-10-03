# RQ2: hallucination (coco_ViT-L-14_lora8_lr1e-3_s0 epoch_02.pt, Karpathy test, 5,000 images)

## CHAIR by caption length

Captions binned by length so that shorter, more generic captions cannot pass for less hallucination (methodology §9). n = captions in the bin.

| Configuration | Words (mean) | <= 8 words: n / CHAIR_s / CHAIR_i | 9-10 words: n / CHAIR_s / CHAIR_i | >= 11 words: n / CHAIR_s / CHAIR_i |
|---|---|---|---|---|
| FP32 | 9.48 | 1173 / 9.5 / 6.5 | 2819 / 8.3 / 5.6 | 1008 / 14.0 / 8.9 |
| Encoder FP16 | 9.47 | 1181 / 9.5 / 6.5 | 2814 / 8.4 / 5.6 | 1005 / 13.8 / 8.8 |
| Encoder INT8 | 9.48 | 1153 / 8.8 / 6.0 | 2849 / 8.2 / 5.5 | 998 / 14.3 / 8.9 |
| Encoder NF4 | 9.49 | 1185 / 8.4 / 6.0 | 2776 / 8.5 / 5.7 | 1039 / 13.6 / 8.5 |
| Decoder FP16 | 9.47 | 1178 / 9.3 / 6.3 | 2812 / 8.5 / 5.7 | 1010 / 13.5 / 8.4 |
| Decoder INT8 | 9.49 | 1165 / 9.5 / 6.6 | 2816 / 8.2 / 5.5 | 1019 / 13.7 / 8.7 |
| Decoder NF4 | 9.52 | 1128 / 8.2 / 5.8 | 2808 / 9.4 / 6.4 | 1064 / 14.3 / 8.9 |
| Both FP16 | 9.46 | 1176 / 9.8 / 6.6 | 2828 / 8.3 / 5.6 | 996 / 13.9 / 8.8 |
| Both INT8 | 9.48 | 1174 / 8.9 / 6.0 | 2821 / 8.9 / 6.0 | 1005 / 13.6 / 8.5 |
| Both NF4 | 9.55 | 1117 / 8.1 / 5.7 | 2775 / 8.8 / 6.0 | 1108 / 14.0 / 8.8 |
| FP32 greedy | 9.47 | 1357 / 8.1 / 5.5 | 2625 / 11.5 / 7.3 | 1018 / 16.4 / 10.0 |
| FP32 beam 5 | 9.56 | 986 / 8.1 / 5.5 | 2953 / 8.1 / 5.4 | 1061 / 12.5 / 8.2 |

## Visual-grounding score of object words

g = log p(word | caption so far, image) - log p(word | caption so far, mean image); natural log, summed over the word's tokens. Each configuration scores its own captions at its own precision. Intervals: 95 % bootstrap over images. A hallucinated-word score closer to zero than the correct-word score means hallucinated objects lean on the language prior rather than the image (H2).

| Configuration | Mentions (hallucinated) | g correct [CI] | g hallucinated [CI] | Difference [CI] | Share g <= 0: correct / hallucinated | g hallucinated vs FP32 [CI] |
|---|---|---|---|---|---|---|
| FP32 | 7,930 (514) | 4.34 [4.26, 4.42] | 3.53 [3.31, 3.74] | -0.81 [-1.05, -0.57] | 6 % / 6 % | - |
| Encoder FP16 | 7,922 (513) | 4.34 [4.26, 4.41] | 3.54 [3.32, 3.74] | -0.80 [-1.05, -0.56] | 6 % / 6 % | +0.01 [-0.01, +0.03] |
| Encoder INT8 | 7,876 (498) | 4.34 [4.26, 4.42] | 3.47 [3.24, 3.67] | -0.87 [-1.11, -0.65] | 6 % / 7 % | -0.06 [-0.21, +0.09] |
| Encoder NF4 | 7,853 (499) | 4.32 [4.24, 4.40] | 3.56 [3.33, 3.78] | -0.75 [-1.00, -0.52] | 6 % / 7 % | +0.04 [-0.14, +0.22] |
| Decoder FP16 | 7,922 (509) | 4.34 [4.26, 4.42] | 3.53 [3.31, 3.75] | -0.80 [-1.05, -0.57] | 6 % / 6 % | +0.01 [-0.04, +0.06] |
| Decoder INT8 | 7,908 (507) | 4.34 [4.26, 4.41] | 3.51 [3.29, 3.74] | -0.82 [-1.06, -0.58] | 6 % / 6 % | -0.02 [-0.13, +0.10] |
| Decoder NF4 | 7,971 (543) | 4.34 [4.26, 4.42] | 3.63 [3.40, 3.85] | -0.71 [-0.98, -0.48] | 5 % / 6 % | +0.10 [-0.07, +0.26] |
| Both FP16 | 7,920 (514) | 4.34 [4.26, 4.42] | 3.54 [3.33, 3.76] | -0.80 [-1.03, -0.56] | 6 % / 6 % | +0.01 [-0.03, +0.07] |
| Both INT8 | 7,880 (512) | 4.33 [4.25, 4.40] | 3.44 [3.21, 3.65] | -0.88 [-1.13, -0.66] | 6 % / 8 % | -0.08 [-0.24, +0.08] |
| Both NF4 | 7,920 (519) | 4.32 [4.24, 4.40] | 3.62 [3.39, 3.86] | -0.70 [-0.95, -0.46] | 5 % / 6 % | +0.09 [-0.10, +0.29] |
| FP32 greedy | 8,348 (622) | 4.50 [4.42, 4.57] | 3.45 [3.25, 3.65] | -1.05 [-1.26, -0.82] | 5 % / 8 % | -0.08 [-0.28, +0.12] |
| FP32 beam 5 | 7,725 (466) | 4.30 [4.21, 4.38] | 3.60 [3.37, 3.83] | -0.69 [-0.95, -0.45] | 7 % / 6 % | +0.08 [-0.09, +0.24] |
