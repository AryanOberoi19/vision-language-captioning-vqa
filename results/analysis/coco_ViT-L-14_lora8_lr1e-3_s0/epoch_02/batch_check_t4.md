# Batch 1 (energy runs, Colab T4) vs batch 64 (evaluation, MacBook Pro M4 Pro), same 500 test images and 500 questions

Accuracy of the first-repetition outputs of 10_measure_variants.py against the evaluation's outputs for the same items. CIDEr's document frequencies come from the 500 images' references, so these values are not comparable with the full-test CIDEr; only the batch-1 vs batch-64 difference is.

| Configuration | Captions identical | CIDEr b64 | CIDEr b1 | Δ | CHAIR_i b64 | CHAIR_i b1 | Answers identical | VQA b64 | VQA b1 | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 | 100.0 % | 114.8 | 114.8 | 0.00 | 6.4 | 6.4 | 100.0 % | 58.9 | 58.9 | 0.00 |
| Encoder FP16 | 99.2 % | 113.8 | 114.5 | 0.67 | 6.5 | 6.4 | 100.0 % | 58.9 | 58.9 | 0.00 |
| Encoder INT8 | 85.4 % | 115.9 | 116.1 | 0.22 | 6.4 | 6.6 | 98.0 % | 58.7 | 58.3 | -0.38 |
| Encoder NF4 | 99.4 % | 116.9 | 116.6 | -0.30 | 6.5 | 6.6 | 100.0 % | 58.6 | 58.6 | 0.00 |
| Decoder FP16 | 91.8 % | 114.8 | 114.3 | -0.48 | 6.8 | 6.4 | 98.8 % | 59.0 | 59.1 | 0.10 |
| Decoder INT8 | 76.8 % | 115.5 | 116.5 | 0.92 | 6.7 | 6.9 | 97.0 % | 58.3 | 58.7 | 0.38 |
| Decoder NF4 | 90.4 % | 112.7 | 113.1 | 0.37 | 7.1 | 6.9 | 98.8 % | 59.3 | 59.2 | -0.02 |
| Both FP16 | 92.4 % | 114.4 | 114.6 | 0.18 | 6.5 | 6.3 | 98.6 % | 58.9 | 59.3 | 0.40 |
| Both INT8 | 71.6 % | 117.3 | 116.6 | -0.67 | 6.5 | 5.9 | 95.6 % | 57.8 | 58.2 | 0.40 |
| Both NF4 | 90.0 % | 111.0 | 111.5 | 0.47 | 7.4 | 7.3 | 99.2 % | 57.9 | 57.8 | -0.06 |
| FP32 greedy | 100.0 % | 109.4 | 109.4 | 0.00 | 7.2 | 7.2 | - | - | - | - |
| FP32 beam 5 | 100.0 % | 113.5 | 113.5 | 0.00 | 7.0 | 7.0 | - | - | - | - |
