# Batch 1 (energy runs) vs batch 64 (evaluation), same 500 test images and 500 questions

Accuracy of the first-repetition outputs of 10_measure_variants.py against the evaluation's outputs for the same items. CIDEr's document frequencies come from the 500 images' references, so these values are not comparable with the full-test CIDEr; only the batch-1 vs batch-64 difference is.

| Configuration | Captions identical | CIDEr b64 | CIDEr b1 | Δ | CHAIR_i b64 | CHAIR_i b1 | Answers identical | VQA b64 | VQA b1 | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 | 100.0 % | 114.8 | 114.8 | 0.00 | 6.4 | 6.4 | 100.0 % | 58.9 | 58.9 | 0.00 |
| Encoder FP16 | 100.0 % | 113.8 | 113.8 | 0.00 | 6.5 | 6.5 | 100.0 % | 58.9 | 58.9 | 0.00 |
| Encoder INT8 | 85.2 % | 115.9 | 114.3 | -1.59 | 6.4 | 6.3 | 98.0 % | 58.7 | 58.5 | -0.18 |
| Encoder NF4 | 100.0 % | 116.9 | 116.9 | 0.00 | 6.5 | 6.5 | 100.0 % | 58.6 | 58.6 | 0.00 |
| Decoder FP16 | 93.8 % | 114.8 | 113.8 | -1.01 | 6.8 | 6.6 | 98.6 % | 59.0 | 59.3 | 0.30 |
| Decoder INT8 | 76.4 % | 115.5 | 115.6 | 0.02 | 6.7 | 6.8 | 96.8 % | 58.3 | 58.6 | 0.30 |
| Decoder NF4 | 95.2 % | 112.7 | 113.1 | 0.41 | 7.1 | 7.0 | 99.2 % | 59.3 | 59.2 | -0.08 |
| Both FP16 | 94.6 % | 114.4 | 114.8 | 0.42 | 6.5 | 6.8 | 99.6 % | 58.9 | 58.8 | -0.14 |
| Both INT8 | 72.4 % | 117.3 | 114.5 | -2.72 | 6.5 | 6.6 | 95.6 % | 57.8 | 58.5 | 0.66 |
| Both NF4 | 91.6 % | 111.0 | 111.4 | 0.38 | 7.4 | 7.7 | 99.4 % | 57.9 | 57.9 | 0.00 |
| FP32 greedy | 100.0 % | 109.4 | 109.4 | 0.00 | 7.2 | 7.2 | - | - | - | - |
| FP32 beam 5 | 100.0 % | 113.5 | 113.5 | 0.00 | 7.0 | 7.0 | - | - | - | - |
