# Implementation

Code: `Implementation/` in the DL_NLP Project folder on Aryan's MacBook (package `greenvl`, numbered scripts). Started 27 Sep 2026.

## Decisions

- Platform (27 Sep): one MacBook Pro, M4 Pro, PyTorch MPS, for training and all measurement. It is edge-class, not server-class. The edge-vs-server comparison is not required by the course or by RQ1-RQ3 (only H1's last clause depends on it) and cannot answer "offload or run locally" without network and radio energy. Contingency: if NF4/INT8 come out dominated on the Mac, re-measure only the precision configurations on a Colab T4. Code is device-agnostic (`--device`). If the college H100 becomes available, use it for all training.
- Training data (29 Sep): the full dataset as written in the paper: COCO Karpathy train 113,287 images (566,747 caption pairs) and all VQA v2 train2014 questions (443,757). A 20,000-image subset was built on 29 Sep and dropped the same day at Aryan's request; its two files are in Datasets/processed/_to_delete/.
- Learning-rate schedule (29 Sep): constant after a linear warm-up of 0.2 epoch, for every configuration. Runs can then be extended epoch by epoch, and the reference run that sets the epoch budget is itself a valid grid run.
- Learning rates (30 Sep; Aryan left the call to Claude): each adaptation method gets the rate that was best for it under the same fixed selection protocol, instead of one shared rate. LoRA (and the mapping network, trained in every configuration): 1e-3, confirmed by the Stage C COCO pilot; frozen decoder (mapping network only): the LoRA rate; full fine-tuning: 3e-4 (best Flickr8k validation loss 2.304 vs 2.444 at 1e-3, far beyond the 0.02 margin; at 1e-3 it overfit from epoch 3). Reason: a rate tuned for LoRA handicaps full fine-tuning and would bias the adaptation comparison (RQ1) toward LoRA; tuning each method by one protocol is the fair comparison. Stage C tests LoRA only; full fine-tuning's rate comes from Flickr8k.
- Scope (2 Oct, Aryan): only one configuration is trained; the training-side factors (encoder, adaptation, distillation) are dropped as factors. The study's compression levels come from the inference-side factors applied to that one model: precision (FP32, FP16, INT8, NF4 on encoder, decoder, both) and caption decoding (greedy, beam 3, beam 5). RQ1-RQ3 remain answerable from those; lost are the encoder-swap, LoRA-rank / frozen / full fine-tuning and distillation comparisons (3 of the 5 techniques in the literature review), and the training-side cost comparison shrinks to one training-energy figure. The configuration is the one predicted to give the best results at an affordable training cost: GPT-2 small (medium trains 2.6x slower), LoRA rank 8 at 1e-3 (full fine-tuning tied it on Flickr8k, 2.304 vs 2.306, at 1.65x the time, and would merge the two heads), FP32, 3 seeds; encoder decided by the encoder check below.
- Encoder (2 Oct): ViT-L/14, by the encoder check (rule fixed before results; Step 6a below). Training cost is unchanged (features cached; 145 samples/s for both encoders); the cost is at inference (encoder 27 vs 414 images/s; ~11x energy per image when caching features).
- Training proceeds step by step with Aryan; no training run starts without his go-ahead. Flickr8k runs start 29 Sep.

## Differences from the Methodology in Research_Paper.tex (as of 29 Sep)

Changed, text needs updating:
1. Platforms. Paper: server-class GPU plus edge-class device, one frontier each, batch 32 on the server platform, training energy on one designated platform (§8, H1's last clause, Fig. workflow "2 platforms" and "one frontier per platform", §9 RQ1 "for each platform", §10 platform dependence). Now: one MacBook Pro (M4 Pro GPU) for training and measurement, one frontier; T4 re-measurement of the precision configurations only if needed. Open: whether batch 32 is still measured (on the Mac).
2. Energy source (§8). Paper: power telemetry sampled every 100 ms, CodeCarbon where it reads telemetry directly. Now: Apple's IOReport energy counters (zeus-apple-silicon), read at the start and end of each pass; no CodeCarbon. Apple derives the counters from a power model (the same source as powermetrics). Boundary: SoC + DRAM (display, SSD, fans, adapter losses excluded). Idle 60 s, 5 repetitions, randomised order, batch 1, PUE 1, CI 727 g/kWh are unchanged. See "Energy counters" below: CPU and DRAM counters may need powermetrics running, which would bring powermetrics back into §8.
3. Learning rate (§5). Paper: one learning rate for all configurations, chosen on Flickr8k. Now: chosen per adaptation method by the same fixed protocol (scripts/04_select_lr.py): LoRA and frozen 1e-3 (confirmed on COCO + VQA v2), full fine-tuning 3e-4. Changes one sentence of §5.

13. Experimental design (§6, Table 2, Fig. workflow; 2 Oct). Paper: one-factor-at-a-time sweep over encoder, adaptation, distillation, precision and decoding from a ViT-B/32 reference, then a focused grid. Now: one trained configuration (ViT-L/14, LoRA rank 8, GPT-2 small, 3 seeds); the sweep is precision x target and decoding applied to it. The encoder, adaptation and distillation rows of Table 2, the distillation paragraph and the focused grid go; the reference encoder becomes ViT-L/14, with the encoder check reported as the reason.

Unchanged but open:
4. Seeds (§5): three, now for the one trained configuration.
5. Epoch budget (§5): proposed that Flickr8k fixes the learning rate and checks the pipeline, while the epoch budget comes from the reference run on COCO + VQA v2 (validation CIDEr per epoch) and is then held for all configurations. Flickr8k has no VQA questions, and an epoch count found on 30,000 caption pairs does not carry over to 566,747. Changes one sentence of §5.

Not specified in the paper, fixed in code (additions, no contradiction):
6. Mapping network CLIP dim -> 3840 -> 7680 (10 x 768), tanh; CLIP features L2-normalised before it.
7. LoRA on GPT-2's attn.c_attn (query/key/value) and attn.c_proj (output) projections; alpha = 8 for every rank (Hu et al.).
8. VQA input "[prefix] Question: {q} Answer: {a}<eos>", loss on the answer tokens; target is the most common of the ten human answers.
9. NF4 / INT8 through bitsandbytes quantize linear layers only; GPT-2's tied 38.6 M-parameter embedding / LM head stays unquantized, which the model-size results must state.
10. Learning rate constant after warm-up (consistent with §5's "AdamW with linear warm-up"); training precision (fp32 or bf16 autocast) to be fixed on Flickr8k.
11. CLIPScore scorer (§7, "a CLIP model that is not among the encoders in the grid"): OpenCLIP ViT-B/32 trained on LAION-2B (laion/CLIP-ViT-B-32-laion2B-s34B-b79K). Its weights and training data differ from the grid's three OpenAI encoders; its architecture is the same as ViT-B/32, which the paper can state in one clause. w = 2.5, prompt "A photo depicts " (Hessel et al.).
12. Evaluation details: beam 3 is the reference caption decoding (greedy and beam 5 for the decoding factor), answers greedy; length penalty 1.0; captions up to 40 tokens, answers up to 8. Checkpoint = highest validation CIDEr (beam 3) over epochs. Bootstrap: 1,000 resamples of images, percentile intervals; CHAIR_i and VQA accuracy resample images together with all their object mentions or questions.

## Step 4: training code (29 Sep)

- greenvl/data.py: GPT-2 tokenizer (eos as padding), cached features per dataset family (COCO, Flickr8k and VizWiz ids overlap), caption and VQA datasets, right-padding collate, TaskMixer (one task per batch, equal probability; 1 epoch = 2 x caption batches; seeded per epoch so an interrupted epoch replays identically).
- scripts/03_train.py: AdamW, linear warm-up over 0.2 epoch then constant (or --schedule linear), gradient clipping, fp32 or bf16 autocast, validation loss per task each epoch, trained weights saved per epoch (for selection by validation CIDEr in step 5), resumable last.pt, training energy per epoch, idle window at start. --data flickr8k trains the captioning head only. Re-running with a larger --epochs extends a finished constant-schedule run; the result is identical to a run that asked for that many epochs from the start.
- Defaults until fixed on Flickr8k: lr 1e-4, weight decay 0.01, warm-up 0.2 epoch, constant schedule, clip 1.0, batch 64, fp32.
- Offline tests (tiny random GPT-2, fake features): VQA loss only on answer + eos; caption and VQA batches mix ~50/50; with a trainable decoder, loss falls to the data's floor and shuffled features raise validation loss (0.16 -> 1.05), so the prefix is used; kill-and-resume and 2-then-extend-to-4 epochs both give bit-identical weights to an uninterrupted run; a run with changed settings under the same name is refused; a linear-schedule run refuses extension.
- First run on the Mac (29 Sep, 20-step smoke test, Flickr8k, ViT-B/32): runs on MPS, validation loss 3.88 after 20 steps.
- First Flickr8k run f8k_lr1e-4 (29 Sep): epoch 0 train loss 3.09, validation loss 2.527, 229 s of training (~135 samples/s) + 14 s validation. Metal memory reached 29 GB on the 24 GB Mac (PyTorch's default MPS cap is 1.7x the recommended working set) and the run stalled ~230 steps into epoch 1; epoch 0 was saved. Fixes (29 Sep): vocabulary logits computed only at labelled positions (same loss and gradients as GPT-2's built-in loss, verified to float precision; the full logits tensor was ~0.7 GB per batch); MPS allocator capped at the recommended working set (PYTORCH_MPS_HIGH_WATERMARK_RATIO 1.0, LOW 0.8, set in greenvl/device.py unless already set); cache emptied after validation; memory logged every 50 steps. On an error or Ctrl+C the script saves last.pt at the last completed step and writes the error to log.jsonl (tested: interrupt + resume gives bit-identical weights).
- Resume after those fixes (29 Sep): energy counters live (idle 0.089 W: CPU 0.051, DRAM 0.034, GPU 0.004 W); replayed step 500 gave loss 2.5413, identical to the original run, so resume and the labelled-only loss are exact on MPS too. It then stopped 75 steps into epoch 1 with "MPS backend out of memory (MPS allocated: 7.12 GiB, other allocations: 10.47 GiB, max allowed: 17.76 GiB)". Memory outside PyTorch's allocator grows with every new tensor shape on MPS, and the labelled-only logits made the output-layer shape change every batch. Fix: batches padded to a multiple of 16 tokens (at most three lengths per task), training batches always full (last partial batch dropped), output-layer rows padded to a multiple of 512 with ignored dummy rows, MPS cache emptied every 100 steps. Loss unchanged (verified against GPT-2's built-in loss and per-sequence unpadded loss); resume and extension still bit-identical in the offline tests. Flickr8k epoch = 468 steps from epoch 1 on (469 in epoch 0).
- f8k_lr1e-4 completed 29 Sep (3 epochs, lr 1e-4, constant after warm-up): validation loss 2.527 / 2.439 / 2.393, training loss 3.09 / 2.51 / 2.41; still falling, gains shrinking, no overfitting yet. Memory stable at 5.6-10.8 GB during training (1.3 GB after validation), ~140 samples/s, ~3.5 min training + 14 s validation per epoch. Energy with live counters, epoch 2 (468 steps, 29,952 samples, 213 s): 4,858 J = CPU 75 + GPU 4,056 + DRAM 728 J, ~22.8 W, 0.162 J per sample above idle (idle 0.095 W). Epoch 0 energy is GPU only; epoch 1 covers the resumed part only.
- Learning-rate selection driver, scripts/04_select_lr.py (29 Sep; rules fixed before results). Stage A: Flickr8k LoRA r8 at 3e-5 / 1e-4 / 3e-4 for 3 epochs (one more rate beyond the range if the best is at an end), best two by epoch-3 validation loss extended to 6 epochs, chosen = lower minimum validation loss. Stage B: full fine-tuning at the chosen and next lower rate, 3 epochs; reported only (shared rate as in §5 vs its own rate is Aryan's call). Stage C: caches ViT-B/32 COCO features, then a quarter-epoch (4,428-step) caption + VQA pilot at the chosen and next lower rate; switch to the lower rate only if better by > 0.02 on either task or the chosen rate fails. The kept pilot is the first quarter epoch of reference run seed 0 (continuation tested bit-identical to a straight run). One sudo-started powermetrics serves all child runs (GREENVL_KEEPER_PID). Runs and the driver are locked against a second copy; Ctrl+C stops safely and a re-run continues (tested). Development runs keep only their newest weights file (--save-epochs last). Results: results/lr_selection/summary.md. Estimated ~3 h.
- 03_train.py additions (29 Sep): --max-steps stops inside an epoch save step_NNNNNN.pt and a "partial" log entry with validation, and the run can later continue with a larger or no --max-steps; --save-epochs last; run-directory lock.
- Pause and resume (29 Sep, before step 4b started). Ctrl+C, SIGTERM or closing the window now finishes the current step, saves last.pt and exits (code 130); a second Ctrl+C stops at once without saving over the previous last.pt. The selection driver waits for the run to save instead of killing it (before, Python's subprocess.run killed the child 0.25 s after Ctrl+C, so the stop fell back to the last 1,000-step save). Time, energy and training-loss sums of an unfinished epoch are stored in last.pt at every save and added to by the session that finishes the epoch; epoch entries gain "sessions" and, if the Mac slept inside the epoch, "slept_seconds". Energy windows use a clock that stops during sleep (CLOCK_UPTIME_RAW), so a closed lid never counts as run time or idle power. The replay of an interrupted epoch's already-trained batches happens before the measured window opens. A stop during the last validation of a run (or of a --max-steps pilot) is now completed on resume instead of being skipped. Feature caching saves every 8,192 images and resumes from there; the extraction log is written after each set. Tested offline: stops mid-epoch, at an epoch boundary, during validation, during a pilot's validation, twice in one epoch, a forced double Ctrl+C, and SIGHUP all resume to weights bit-identical to an uninterrupted run, with identical epoch losses and energy/time sums matching a fixed-power fake meter; the driver pauses cleanly on Ctrl+C to the whole process group and on a signal to the child alone; interrupted feature caching reproduces the uninterrupted features and order. Steps redone after a hard kill (power loss) are measured once, in the session that keeps them. scripts/progress.py reports progress and time left from the logs; scripts/lr_selection.command starts or resumes step 4b in a Terminal window.
- f8k_lr1e-4's epoch 1 (resumed mid-epoch on 29 Sep, before this change) logged the time and energy of the resumed part only (181.5 s); its losses are unaffected.
- Learning-rate selection, Stages A and B (30 Sep, 14:07-15:52; validation loss per epoch). LoRA r8: 3e-5 2.654 / 2.548 / 2.491; 1e-4 2.527 / 2.439 / 2.393; 3e-4 2.455 / 2.390 / 2.361 / 2.345 / 2.352 / 2.353; 1e-3 (added because 3e-4 was best at the top of the range) 2.435 / 2.370 / 2.328 / 2.307 / 2.306 / 2.309. Chosen 1e-3 (lower minimum by 0.039); it plateaus after epoch 4-5 on Flickr8k. 1e-3 is the top of the grid, so no higher rate was tried. Full fine-tuning: 1e-3 2.489 / 2.444 / 2.497; 3e-4 2.353 / 2.304 / 2.353 (both overfit by epoch 3; training loss 1.72-1.81). Speed and energy with live counters: LoRA ~140 samples/s, ~3.6 min and ~1.4 Wh (~5.1 kJ, ~24 W) per Flickr8k epoch; full fine-tuning ~85 samples/s, ~5.9 min and ~1.9-2.0 Wh per epoch. Stage C: ViT-B/32 COCO feature caching ran at ~300 images/s (73,728 images in ~4 min; paused at 15:56 and resumed).
- Learning-rate selection, Stage C (30 Sep, finished 20:36): 4,428-step COCO + VQA v2 pilot (a quarter epoch, both heads). 1e-3: validation loss caption 2.3385, VQA 1.2998 (33.0 min, 12.47 Wh); 3e-4: caption 2.3714, VQA 1.3324 (32.7 min, 11.98 Wh). The lower rate is worse on both tasks, so 1e-3 stays; the 1e-3 pilot (results/runs/coco_ViT-B-32_lora8_lr1e-3_s0) is the first quarter epoch of reference run seed 0. Caveat: warm-up is 0.2 epoch = 3,542 steps, so only the last ~886 pilot steps ran at the full rate; stability at 1e-3 over a full COCO epoch is checked by the reference run itself. ~145 samples/s on COCO + VQA (~0.45 s/step). ViT-B/32 COCO features: 123,287 images, coco_train at ~320 images/s (0.083 J/image above idle; two sessions), val/test ~75 images/s (DataLoader start-up dominates small sets). Results: results/lr_selection/summary.md.
- Disk (29 Sep): the Mac had 21 GB free of 461 GB. The selection driver writes ~8.5 GB and refuses to start below 12 GB. The full grid needs far more (about 130 MB per LoRA epoch checkpoint, ~630 MB per full-fine-tuning epoch, plus optimiser state in last.pt), so space must be freed before the grid (the Datasets zips are ~25 GB) and unselected epoch checkpoints pruned after step 5.
- Disk: an epoch checkpoint is ~130 MB (mapping network + adapters, fp32), ~630 MB with full fine-tuning; unselected epochs can be pruned after step 5.

## Step 5: evaluation (30 Sep)

- Code:
  - greenvl/decode.py: Hugging Face generate on GPT-2 with the visual prefix as input embeddings (key-value cache on), so decoding runs as it would in deployment. Captions greedy / beam 3 / beam 5; answers greedy. Question prompts are built token for token as in training and batched by prompt length, so no sequence is padded.
  - greenvl/metrics.py: BLEU-4, CIDEr-D and SPICE through pycocoevalcap after its PTB tokenisation. BLEU-4 is recomputed from per-image n-gram statistics so it can be bootstrapped. Also CLIPScore and the bootstrap.
  - greenvl/chair.py: CHAIR.
  - greenvl/vqa_metrics.py: VQA accuracy, with the official evaluator vendored as greenvl/third_party/vqa_eval.py (from LAVIS 1.0.2, BSD-3).
  - scripts/05_evaluate.py: selection and scoring.
- Checks:
  - Generation equals a manual greedy loop token for token.
  - BLEU-4 from per-image statistics equals pycocoevalcap's corpus BLEU-4.
  - Per-question VQA accuracy equals the official evaluate() on the same answers.
  - CHAIR agreed with the images on spot checks: hot dog vs sandwich, traffic light vs sign, frisbee vs baseball, bed vs couch, a fork that is not there.
- CHAIR is a port of Rohrbach et al.'s utils/chair.py, with the same synonym list, double-word and toilet-seat rules, and ground truth (segmentation labels plus reference-caption objects). It differs in two ways:
  - Tokenisation uses a regular expression instead of nltk.word_tokenize.
  - Plurals are singularised with inflect instead of the unmaintained pattern package; words that are already object names are left alone, so "bus" stays "bus".
- SPICE has two issues on the Mac:
  - Its parse cache (LMDB) has no Apple Silicon library, and its serialiser (FST) fails on Java 16+, so the cache is off. The jar and the scores are unchanged.
  - It scores through the Nashorn JavaScript engine, which left the JDK in Java 15. The Mac has Oracle Java 24, and Homebrew's openjdk@11 does not install on macOS 27. Fix: run SpiceScorer on Java 24 with the standalone nashorn-core 15.4 and ASM 9.7 on the class path. These are Maven Central artifacts, SHA-1 checked against Maven Central and pinned by SHA-256 in metrics.py, downloaded once into ~/.greenvl/nashorn.
  - The parse cache matters only for speed: 5,000 images take ~55 s on 12 threads.
- The CLIPScore scorer's image embeddings are cached by 02_extract_features.py under features/CLIPScore-LAION-B-32/ (~75 images/s for a 5,000-image split).
- Time per 5,000-image split on the M4 Pro, batch 64: captions (beam 3) 70 s, 26,729 answers 77 s, SPICE 55 s, CLIPScore and CHAIR a few seconds. These are wall-clock times for evaluation, not the paper's batch-1 latency or energy (step 7).
- First scores, from the quarter-epoch pilot (coco_ViT-B-32_lora8_lr1e-3_s0/step_004428.pt, beam 3, 95 % intervals). Karpathy val:
  - Captions: BLEU-4 26.0 [25.3, 26.6], CIDEr 87.0 [85.2, 88.8], SPICE 16.8 [16.5, 17.1], CLIPScore 68.5 [68.2, 68.9], CHAIR_i 12.3 [11.6, 13.0], CHAIR_s 17.7 [16.7, 18.8], 9.7 words.
  - VQA: 48.1 [47.4, 48.8]; yes/no 67.8, number 32.3, other 37.1.
- Karpathy test, same checkpoint:
  - Captions: BLEU-4 26.7 [26.0, 27.3], CIDEr 88.4 [86.5, 90.1], SPICE 16.9 [16.6, 17.2], CLIPScore 68.5 [68.2, 68.8], CHAIR_i 12.6 [11.9, 13.3], CHAIR_s 18.2 [17.1, 19.2], 9.6 words.
  - VQA (26,280 questions): 47.9 [47.3, 48.6]; yes/no 68.2, number 32.9, other 36.2.
  - VQA-CE (7,801 counterexamples): 16.4 [15.5, 17.2]. It is far below overall accuracy, as expected for questions where the answer priors mislead.
  - The whole test evaluation took 5 min, including the one-off scorer embeddings and CHAIR ground truth.
  - The test split is used here only to exercise the code. Grid results use the checkpoint selected on validation.
  - For scale, ClipCap reports on Karpathy test:
    - MLP mapping network + fine-tuned GPT-2: BLEU-4 32.2, CIDEr 108.4, SPICE 20.1.
    - Frozen GPT-2 behind a transformer mapping network: CIDEr 113.1.
    - A quarter epoch, most of it in warm-up, already reaches 87.
- Grid time re-estimated (30 Sep) with the measured COCO + VQA speed (~145 samples/s, ~0.45 s/step, 17,710 steps per epoch; step 1 assumed 169): ~680 h (~28 days) at 10 epochs × 3 seeds for every configuration, ~14 days at 5 epochs, ~12.8 days at 10 epochs with seeds for the reference only. The epoch budget from the reference run decides which applies. The ~3 h step 4b was the learning-rate selection only.

## Step 6a: encoder check (2 Oct)

- scripts/06_encoder_check.py + encoder_check.command (one password, one powermetrics for every step, Ctrl+C pauses, reopening resumes). Rule written in the script before any ViT-L/14 result: ViT-L/14 if validation CIDEr is higher by >= 5.0 points or VQA accuracy by >= 2.0, and neither is lower by more than 1.0 CIDEr / 0.5 VQA; else ViT-B/32. Thresholds set against L/14's ~3x (captions) to ~7x (answers) inference cost at batch 64 and the val intervals (about +/-1.8 CIDEr, +/-0.7 VQA).
- ViT-L/14 features, COCO Karpathy (11:54-13:23): train 113,287 images in 4,765 s (23.8 img/s, 0.912 J/image above idle), val and test 5,000 each (20.0 img/s, ~1.03 J/image). ViT-B/32 COCO train for comparison: ~320 img/s, 0.083 J/image above idle, so ~11x the energy per image. Feature files 349 MB (train) + 15 MB each (val, test).
- ViT-L/14 pilot coco_ViT-L-14_lora8_lr1e-3_s0: the B/32 pilot's configuration (LoRA r8, lr 1e-3, seed 0, batch 64, fp32, same data order), 4,428 steps; trainable 33.3 M (mapper 32.5 M). 145 samples/s as for B/32; 2,083 s, 37.6 kJ (10.46 Wh) vs B/32's 1,980 s, 44.9 kJ (12.47 Wh). The 16 % lower energy is not an encoder effect (the training computation is the same apart from the mapper's input width): average power was 18.1 vs 22.7 W, so the chip ran in a different power state than on 30 Sep. Run-to-run variation of this size in training energy has to be reported, and training-energy figures compared only within repeated, interleaved measurements.
- Result (Karpathy val, beam 3, 95 % intervals), B/32 -> L/14:
  - CIDEr 87.0 [85.2, 88.8] -> 93.2 [91.3, 94.9] (+6.2)
  - VQA 48.1 [47.4, 48.8] -> 49.5 [48.9, 50.2] (+1.4); yes/no 67.8 -> 68.9, number 32.3 -> 34.7, other 37.1 -> 38.5
  - BLEU-4 26.0 -> 27.2, SPICE 16.8 -> 17.6, CLIPScore 68.5 -> 68.7
  - CHAIR_i 12.3 -> 9.2 [8.6, 9.8], CHAIR_s 17.7 -> 14.0 [13.1, 15.0], at the same mean length (9.7 words): the stronger encoder hallucinates a quarter fewer objects without shorter captions.
  - Validation loss caption 2.3385 -> 2.3062, VQA 1.2998 -> 1.2720.
- Rule outcome: CIDEr +6.2 >= 5.0 and VQA not lower, so ViT-L/14. The reference run continues coco_ViT-L-14_lora8_lr1e-3_s0 (its first quarter epoch). Not started; waiting for Aryan's go-ahead and the stopping rule.
- Still needed for ViT-L/14: VizWiz features (7,750 images, ~6-7 min) before the zero-shot check.

## Step 6b: reference run, seed 0 (2 Oct)

- scripts/07_reference_run.py + reference_run.command: each start trains one more epoch through 03_train.py (or finishes a paused one), then scores the new epoch_NN.pt on Karpathy val with every metric; results/reference/<run>.md tabulates all epochs. One epoch at a time gives the same weights as one long run (constant learning rate after a warm-up counted in steps; optimiser, scheduler, random state and data order restored; tested bit-identical offline and on the pilot continuation).
- Epoch budget (2 Oct, Aryan): 3 epochs, fixed in advance, replacing the proposed stopping rule; the checkpoint is the epoch with the highest validation CIDEr.
- coco_ViT-L-14_lora8_lr1e-3_s0 (continues the encoder-check pilot), Karpathy val, beam 3:

| Epochs | CIDEr | VQA | BLEU-4 | SPICE | CLIPScore | CHAIR_i | CHAIR_s | Words | Val loss cap / VQA | Train h | Train Wh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 93.2 | 49.5 | 27.2 | 17.6 | 68.7 | 9.2 | 14.0 | 9.7 | 2.306 / 1.272 | 0.58 | 10.5 |
| 1 | 106.1 | 54.7 | 32.4 | 19.1 | 70.9 | 7.9 | 11.8 | 9.2 | 2.140 / 1.126 | 2.25 | 44.2 |
| 2 | 109.2 | 57.4 | 32.6 | 19.9 | 72.1 | 6.7 | 9.9 | 9.8 | 2.081 / 1.075 | 2.20 | 46.2 |
| 3 | 110.5 | 57.7 | 33.4 | 19.9 | 72.4 | 6.5 | 10.0 | 9.5 | 2.065 / 1.070 | 2.18 | 46.9 |

  (Epoch 1's time and energy include the pilot's 4,428 steps, two sessions; train h and Wh are per epoch.)
- Selected: epoch_02.pt (epoch 3). Gains were levelling off (+1.3 CIDEr, +0.3 VQA in epoch 3 vs +3.1 / +2.7 in epoch 2), and VQA training loss (1.041) fell below validation loss (1.070) in epoch 3, the first sign of overfitting on that head.
- Seed 0 training: 6.63 h, 137.3 Wh (CPU + GPU + DRAM, incl. idle; ~100 g CO2e at 727 g/kWh). Speed 134-145 samples/s; dips to ~90 when the Mac was in other use.
- Karpathy test, epoch_02.pt, beam 3 (95 % intervals):
  - Captions: BLEU-4 33.5 [32.8, 34.2], CIDEr 111.8 [109.8, 113.7], SPICE 20.0 [19.8, 20.3], CLIPScore 72.4 [72.1, 72.7], CHAIR_i 6.5 [5.9, 7.0], CHAIR_s 9.8 [8.9, 10.6], 9.5 words.
  - VQA (26,280 questions): 57.9 [57.2, 58.6]; yes/no 76.5, number 40.6, other 48.0. VQA-CE (7,801): 28.2 [27.2, 29.2].
  - ClipCap on the same test split (Mokady et al.): MLP + fine-tuned GPT-2 BLEU-4 32.2, CIDEr 108.4, SPICE 20.1; frozen GPT-2 + transformer mapper BLEU-4 33.5, CIDEr 113.1, SPICE 21.1. Not like for like: ClipCap uses ViT-B/32 and trains captioning only.
- Next: seeds 1 and 2 (Aryan's decision), then the inference-side variants and the energy harness.

## Step 1 results

Hardware: Apple M4 Pro, 8P + 4E CPU cores, 16-core GPU, 24 GB unified memory, macOS 27.0, on AC power. Python 3.11.16.

- MPS: fp32, fp16, bf16 and bf16 autocast all run.
- bitsandbytes 0.50.2 NF4 and INT8 run on MPS and CPU, as raw layers and through transformers' BitsAndBytesConfig for GPT-2. On MPS a single 768x3072 layer at batch 64: fp16 0.33 ms, NF4 0.43 ms, INT8 0.84 ms (low-bit is slower than fp16 here).
- GPT-2 footprint through transformers: NF4 191 MB, INT8 232 MB (embedding / LM head unquantized, fp32 here). Load non-quantized parts in fp16 for the precision factor and report size per component.
- Training throughput (batch 64, 24 text tokens + 10 prefix), fp32 / bf16 autocast, samples/s: GPT-2 LoRA r8 169 / 171; frozen 180 / 180; full fine-tuning 127 / 129; GPT-2 medium LoRA r8 65 / 67; GPT-2 medium forward only 154 / 160. Energy per sample from that run (0.116 J fp32, 0.101 J bf16 for LoRA r8) is probably GPU only; see "Energy counters".
- Encoder throughput, fp32 / fp16, images/s: ViT-B/32 414 / 534; ViT-B/16 117 / 144; ViT-L/14 27 / 33. Real extraction of Flickr8k with ViT-B/32 including JPEG decoding: 107 images/s.
- Grid estimate, full data (1 epoch = 566,747 caption pairs + as many VQA samples, 10 epochs as a planning figure): ~564 h (~23.5 days) with 3 seeds for every configuration, ~256 h (~10.7 days) with 3 seeds for the reference only; roughly halves at 5 epochs. Excludes the focused grid, Flickr8k development runs and the CNN-LSTM.

## Energy counters

- 27 Sep check (00b_energy_check.py with sudo powermetrics running alongside): all components reported (CPU, GPU, GPU SRAM, ANE, DRAM); within ~3-4 % of powermetrics under CPU load (6.00 vs 5.75 W) and GPU load (26.06 vs 25.29 W); IOReport's CPU channel ~0.2 W higher in every window; idle ~0.28-0.32 W incl. DRAM; DRAM 2.9 W under GPU load (powermetrics' combined figure excludes DRAM). The first window read 38.8 W where powermetrics read 0.07 W.
- 29 Sep, first real runs (no powermetrics): CPU, DRAM, GPU SRAM and ANE read exactly 0 J over 60-75 s windows; only the GPU counter advanced (15 W training, 6.5 W feature extraction, idle 0.002 W). Most likely the CPU and DRAM counters only advance while powermetrics (or another privileged sampler) is running, and the 38.8 W first window was stale energy released when powermetrics started. The 29 Sep warm-up-window fix addressed the wrong cause.
- Confirmed 29 Sep (00b_energy_check.py without powermetrics): CPU and DRAM 0 J in every window, including a 10 s CPU load; GPU 25.7 W under GPU load, as before.
- Fix (29 Sep): EnergyMeter starts `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500` in the background (output discarded) and discards one settling window. A root shell wrapper stops powermetrics when the Python process exits, even if Python is killed (tested). Needs cached sudo credentials: run `sudo -v` shortly before any script that measures energy; without them the meter warns and records GPU only. Windows where CPU or DRAM does not advance are flagged ("stale_counters"). powermetrics' own CPU use is inside the measured SoC energy; it is present at idle as well, so idle subtraction removes it.
- Energy logged so far (ViT-B/32 Flickr8k extraction, smoke run, f8k_lr1e-4 epoch 0) is GPU only.

## Step status

| Step | Status |
|---|---|
| 1 Environment + benchmark | done 27 Sep |
| 1b Energy counter check | CPU/DRAM counters need powermetrics running (confirmed 29 Sep); EnergyMeter keeps it running; working in the resumed run |
| 2 Splits and eval subsets | done; rerun 29 Sep after dropping the subset; all counts match datasets.md |
| 3 CLIP feature caching (02_extract_features.py) | ViT-B/32 Flickr8k and COCO done (30 Sep); ViT-L/14 COCO done (2 Oct); ViT-L/14 VizWiz to run; ViT-B/16 no longer needed |
| 4 Data pipeline + training script | f8k_lr1e-4 finished 3 epochs 29 Sep (val loss 2.393); memory stable after fixed-shape batches |
| 4b Learning-rate selection (04_select_lr.py) | done 30 Sep: LoRA and frozen 1e-3, full fine-tuning 3e-4 (results/lr_selection/summary.md) |
| 5 Decoding + evaluation (BLEU-4, CIDEr, SPICE, CLIPScore, CHAIR, VQA acc, VQA-CE) | done 30 Sep; each metric checked against its reference implementation; pilot scored on val and test (results/eval/) |
| 6 Reference configuration | encoder check 2 Oct: ViT-L/14; seed 0 done 2 Oct (3 epochs, selected epoch 3: test CIDEr 111.8, VQA 57.9, CHAIR_i 6.5); seeds 1-2 pending |
| 7 Energy measurement harness | |
| 8 Inference-side sweep (precision x target, decoding) on the trained model | training-side factors dropped 2 Oct |
| 9 CNN-LSTM baseline | |
| 10 Analysis (frontiers, grounding score, retention ratios, Grad-CAM) | |
| 11 Gradio demo | |

## Step 2 outputs (Datasets/processed/)

COCO Karpathy 113,287 / 5,000 / 5,000 images, 566,747 training caption pairs, 25,010 reference captions each for val and test. Flickr8k 6,000 / 1,000 / 1,000 (30,000 pairs). VQA v2 train 443,757 questions on 82,783 images, 22,531 distinct answers; eval 26,280 (Karpathy test) and 26,729 (Karpathy val). VQA-CE 7,801 on test. VizWiz val 7,750 images; 5,167 precanned and 964 rejected captions removed; 208 images left without references. Images to encode per encoder: 139,037.

## Libraries

torch 2.14.0, torchvision 0.29.0, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, bitsandbytes 0.50.2, kernels 0.17.1, zeus-apple-silicon 1.1.0, pycocoevalcap 1.2, pycocotools 2.0.11 (Implementation/requirements.lock.txt).
