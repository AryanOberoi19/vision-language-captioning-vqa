# RQ3: retention of VQA vs captioning (coco_ViT-L-14_lora8_lr1e-3_s0 epoch_02.pt, Karpathy test)

R_cap = CIDEr / CIDEr(FP32), R_vqa = VQA accuracy / VQA accuracy(FP32), ΔR = R_vqa - R_cap. Only the precision configurations change both heads (methodology §9). Intervals: 95 % bootstrap over test images, captions and questions of an image resampled together.

| Configuration | R_cap [CI] | R_vqa [CI] | ΔR [CI] |
|---|---|---|---|
| Encoder FP16 | 0.9992 [0.9973, 1.0008] | 0.9998 [0.9993, 1.0003] | +0.0006 [-0.0011, +0.0025] |
| Encoder INT8 | 0.9991 [0.9928, 1.0057] | 0.9988 [0.9963, 1.0019] | -0.0003 [-0.0074, +0.0071] |
| Encoder NF4 | 0.9964 [0.9883, 1.0048] | 0.9966 [0.9931, 1.0000] | +0.0003 [-0.0092, +0.0088] |
| Decoder FP16 | 1.0004 [0.9977, 1.0034] | 1.0000 [0.9989, 1.0011] | -0.0004 [-0.0036, +0.0027] |
| Decoder INT8 | 0.9989 [0.9937, 1.0046] | 0.9989 [0.9967, 1.0012] | -0.0000 [-0.0058, +0.0055] |
| Decoder NF4 | 0.9859 [0.9768, 0.9941] | 0.9938 [0.9899, 0.9979] | +0.0080 [-0.0010, +0.0181] |
| Both FP16 | 0.9991 [0.9962, 1.0022] | 1.0001 [0.9990, 1.0012] | +0.0011 [-0.0022, +0.0042] |
| Both INT8 | 1.0031 [0.9959, 1.0098] | 0.9989 [0.9960, 1.0020] | -0.0042 [-0.0118, +0.0036] |
| Both NF4 | 0.9870 [0.9772, 0.9971] | 0.9885 [0.9839, 0.9933] | +0.0016 [-0.0094, +0.0128] |

Mean ΔR over the 9 configurations: +0.0007 [-0.0034, +0.0047]. Wilcoxon signed-rank test (two-sided, n = 9): W = 16.0, p = 0.496.

## Visual-grounding score of answers

g(a) = log p(answer | question, image) - log p(answer | question, mean image), summed over the answer's tokens. Correct: VQA accuracy >= 0.9; wrong: 0. A head that loses visual evidence under compression shows a lower g than FP32.

| Configuration | g all [CI] | vs FP32 [CI] | g correct | g wrong | yes/no | number | other |
|---|---|---|---|---|---|---|---|
| FP32 | 1.57 [1.54, 1.60] | - | 1.39 | 1.80 | 0.34 | 1.01 | 2.66 |
| Encoder FP16 | 1.57 [1.53, 1.60] | +0.001 [-0.001, +0.002] | 1.39 | 1.80 | 0.34 | 1.01 | 2.66 |
| Encoder INT8 | 1.57 [1.54, 1.60] | +0.003 [-0.003, +0.010] | 1.39 | 1.80 | 0.34 | 1.00 | 2.67 |
| Encoder NF4 | 1.56 [1.52, 1.59] | -0.012 [-0.021, -0.003] | 1.37 | 1.79 | 0.34 | 0.98 | 2.64 |
| Decoder FP16 | 1.56 [1.53, 1.60] | -0.003 [-0.005, -0.001] | 1.39 | 1.80 | 0.34 | 0.99 | 2.66 |
| Decoder INT8 | 1.56 [1.52, 1.59] | -0.011 [-0.017, -0.005] | 1.38 | 1.78 | 0.35 | 0.96 | 2.65 |
| Decoder NF4 | 1.54 [1.51, 1.58] | -0.023 [-0.033, -0.013] | 1.37 | 1.75 | 0.32 | 0.88 | 2.67 |
| Both FP16 | 1.56 [1.53, 1.60] | -0.003 [-0.005, -0.000] | 1.38 | 1.80 | 0.34 | 0.99 | 2.66 |
| Both INT8 | 1.55 [1.52, 1.59] | -0.013 [-0.020, -0.006] | 1.39 | 1.77 | 0.34 | 0.95 | 2.65 |
| Both NF4 | 1.54 [1.50, 1.57] | -0.031 [-0.043, -0.020] | 1.37 | 1.75 | 0.31 | 0.87 | 2.65 |
