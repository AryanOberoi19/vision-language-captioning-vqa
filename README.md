# An AI-Based Vision-Language System for Image Captioning and Visual Question Answering

Code for the study in `../Research Paper/Research_Paper.tex`: a unified captioning + VQA model (frozen CLIP encoder, ClipCap mapping network, GPT-2 with one LoRA adapter set per task) and its accuracy-energy trade-offs (Green AI). The Python package is `greenvl` (Green AI, vision-language). Data is read from `../Datasets`; set `GREENVL_DATA` to use another location.

## Layout

```
greenvl/        package: paths, device selection, energy counters, model
scripts/        numbered steps, run in order
results/        reports and measurements (created on first run)
../Datasets/processed/   splits and evaluation subsets built by step 2
../Datasets/features/    cached CLIP features (step 3)
```

## Platform

Training and measurement run on one MacBook Pro (M4 Pro), on its GPU through PyTorch's MPS backend. Code selects the device through `greenvl/device.py`, so the same scripts run on a CUDA GPU (Colab T4, college H100) or CPU with `--device`. Energy comes from the platform's counters (`greenvl/energy.py`): IOReport on Apple Silicon, NVML on NVIDIA.

## Steps

| Step | Script | Status |
|---|---|---|
| 1 | `setup_env.sh`, `scripts/00_check_env.py`: environment, capability checks, throughput and grid-cost estimate | done 27 Sep (`results/env/`) |
| 1b | `scripts/00b_energy_check.py`: energy counters per component, idle vs CPU vs GPU load, optional powermetrics cross-check | done 27 Sep; counters agree with powermetrics within ~3-4 % under load; first window after start-up discarded |
| 2 | `scripts/01_build_splits.py`: Karpathy splits, reference files, VQA / VQA-CE / VizWiz subsets | done 29 Sep; all counts match datasets.md |
| 3 | `scripts/02_extract_features.py`: CLIP embeddings for the three encoders (139,037 images each), with time and energy | ViT-B/32 Flickr8k and COCO done 30 Sep; ViT-L/14 COCO done 2 Oct; ViT-L/14 VizWiz to run |
| 4 | `greenvl/data.py`, `scripts/03_train.py`: tokenisation, one-task-per-batch mixing, training with resumable checkpoints, validation loss and energy per epoch | f8k_lr1e-4 done 29 Sep (3 epochs) |
| 4b | `scripts/04_select_lr.py`: learning-rate selection (Flickr8k LoRA sweep, full fine-tuning check, COCO + VQA pilot), rules fixed in advance | done 30 Sep: LoRA and frozen 1e-3, full fine-tuning 3e-4 (`results/lr_selection/summary.md`) |
| 5 | `greenvl/decode.py`, `greenvl/metrics.py`, `greenvl/chair.py`, `greenvl/vqa_metrics.py`, `scripts/05_evaluate.py`: decoding (greedy, beam 3, beam 5) and evaluation: BLEU-4, CIDEr, SPICE, CLIPScore, CHAIR, VQA accuracy, VQA-CE, each with a 95 % bootstrap interval | done 30 Sep; checked on the 1e-3 pilot (`results/eval/`) |
| 6 | `scripts/06_encoder_check.py`, `scripts/07_reference_run.py`: encoder check (ViT-L/14 vs ViT-B/32, quarter-epoch pilots, rule fixed in advance), then the reference run on COCO + VQA v2 | encoder check done 2 Oct: ViT-L/14 (`results/encoder_check/summary.md`); seed 0 done 2 Oct, 3 epochs (`results/reference/`); seeds 1-2 deferred |
| 7 | `greenvl/inference.py`, `scripts/08_measure_inference.py`: energy, latency, memory, size and FLOPs per caption and per answer at batch 1 (idle subtraction, 5 repetitions, randomised order) | done 3 Oct for the reference model, FP32 (`results/inference/`) |
| 8 | `greenvl/precision.py`, `scripts/09_evaluate_variants.py`, `scripts/10_measure_variants.py`: inference-side sweep on the trained model, precision (FP16, INT8, NF4 on encoder, decoder, both) and caption decoding (greedy, beam 3, beam 5); accuracy on Karpathy test, cost measured component by component. Training-side factors dropped on 2 Oct (one trained configuration) | code ready and smoke-tested 3 Oct; runs pending |
| 9 | CNN-LSTM baseline | |
| 10 | Analysis: frontiers (RQ1), grounding score (RQ2), retention ratios (RQ3), Grad-CAM figures | |
| 11 | Gradio demo with energy per caption | |

## Running steps 1 and 2 (Mac Terminal)

```bash
cd ~/Desktop/Work/STME\ -\ NMIMS/Sem\ 5/DL_NLP\ Project/Implementation
bash setup_env.sh
conda activate greenvl

python scripts/01_build_splits.py        # already run; ~20 s, ends with "All checks passed."
python scripts/00_check_env.py           # ~10-15 min; downloads GPT-2, GPT-2 medium and three CLIP models (~5 GB)
sudo -v && python scripts/00b_energy_check.py --powermetrics   # ~1.5 min; checks the energy counters
```

Run measurements on AC power with Low Power Mode off and other apps closed. `python scripts/00_check_env.py --quick` is a one-minute smoke test.

## Running step 3 (Mac Terminal)

```bash
python scripts/02_extract_features.py --encoder ViT-B/32 --limit 64   # smoke test, ~1 min, writes to features/_smoke
python scripts/02_extract_features.py --encoder all                   # ~2-2.5 h; 60 s idle measurement first
```

Features go to `../Datasets/features/<encoder>/<set>.pt`; time and energy per set to `results/features/`.

## Step 4 (Mac Terminal, when we start training)

```bash
python scripts/02_extract_features.py --encoder ViT-B/32 --sets flickr8k                            # ~1-2 min
python scripts/03_train.py --data flickr8k --run-name smoke --max-steps 20 --idle-seconds 0 --val-batches 5   # ~1 min code check
```

Runs write to `results/runs/<run-name>/` (config, log.jsonl, one `epoch_NN.pt` per epoch, `last.pt`). Re-running an interrupted command resumes where it stopped; a run with different settings under the same name is refused.

To add epochs to a run, re-run it with a larger `--epochs` under the same `--run-name`: it continues from its last epoch, and the result is identical to having asked for that many epochs from the start. This holds for the default constant learning rate after warm-up; a `--schedule linear` run cannot be extended.

Training defaults, to be fixed on Flickr8k: AdamW, learning rate 1e-4, weight decay 0.01, linear warm-up over 0.2 of an epoch then constant, gradient clipping 1.0, batch 64, fp32.

## Step 4b: learning-rate selection (Mac Terminal)

```bash
open scripts/lr_selection.command              # new Terminal window: asks for your password, then runs (~3 h)
python scripts/progress.py                     # what is running, done and left, and the time to finish
python scripts/04_select_lr.py --plan          # the next runs the rules call for
```

`lr_selection.command` runs `sudo -v && python scripts/04_select_lr.py`, keeps the Mac awake while it runs, and copies the output to `results/lr_selection/console.log`. It needs your password once per start (one background powermetrics serves every run) and at least 12 GB free disk. Results in `results/lr_selection/summary.md`. Only one copy of a run, or of this script, can run at a time.

Pausing and resuming: press Ctrl+C in that window (or close it). The run in progress finishes its current step, saves, and everything stops within a few seconds; then the lid can be closed. Open `lr_selection.command` again to continue from the same step. The finished runs are identical to uninterrupted ones (weights, optimiser, learning rate, random state and data order are restored), and an epoch that spans a pause has its time and energy added up over the sessions (its log entry records how many). Feature caching resumes from its last saved block of 8,192 images. Closing the lid without pausing only suspends the run; it continues when the Mac wakes, and the sleep is excluded from the measured time and energy.

## Step 5: evaluation (Mac Terminal)

```bash
python scripts/05_evaluate.py --run coco_ViT-B-32_lora8_lr1e-3_s0 --select      # validation CIDEr of every epoch_NN.pt; keeps the best
python scripts/05_evaluate.py --run coco_ViT-B-32_lora8_lr1e-3_s0               # selected checkpoint, test split, beam 3, all metrics
python scripts/05_evaluate.py --run NAME --checkpoint step_004428.pt --split val --limit 500 --no-spice   # quick look
```

Results go to `results/eval/<run>/`: `<checkpoint>_<split>_<decoding>.json` holds every metric with its 95 % interval, `..._captions.json` each caption with its hallucinated objects, `<checkpoint>_<split>_answers.json` each answer with its accuracy. Scoring the same checkpoint and split again reuses the generated text (`--redecode` decodes afresh). `--decoding greedy | beam3 | beam5` applies to captions; answers are always greedy. VQA-CE is scored on the test split.

On the M4 Pro a 5,000-image split takes about 5 minutes: captions ~70 s (beam 3), 26,000 VQA answers ~80 s, SPICE ~1 min, the rest seconds. The first evaluation on a split also caches the CLIPScore image embeddings and CHAIR's object lists (a few minutes, once).

Needs Java 8 or newer (PTB tokenisation and SPICE are Java). SPICE uses the Nashorn JavaScript engine, which left the JDK in Java 15; on a newer Java it is downloaded once from Maven Central into `~/.greenvl/nashorn` (six jars, 2.5 MB, checked by SHA-256). SPICE's parse cache is off because its database library has no Apple Silicon build, so each SPICE run parses the references again (~20 s).

## Step 6a: encoder check (Mac Terminal)

```bash
open scripts/encoder_check.command     # new Terminal window: asks for your password, then ~2 h; Ctrl+C pauses, reopen to resume
```

Caches ViT-L/14 features for the COCO Karpathy sets, trains the ViT-B/32 pilot's exact configuration on them for the same 4,428 steps, scores both on Karpathy val and applies the rule written in `scripts/06_encoder_check.py`. Result in `results/encoder_check/summary.md`. Run on 2 Oct: ViT-L/14 chosen.

## Step 6b: reference run (Mac Terminal)

```bash
open scripts/reference_run.command     # one more epoch (~2.3 h incl. scoring); password once; Ctrl+C pauses, reopen to resume
```

Each start trains `coco_ViT-L-14_lora8_lr1e-3_s0` to one more full epoch, scores the new checkpoint on Karpathy val and updates `results/reference/<run>.md`. Stopping between epochs gives the same model as one long run. Budget: 3 epochs; the selected checkpoint is the epoch with the highest validation CIDEr. Other seeds: `python scripts/07_reference_run.py --seed 1` (after `sudo -v`).

## Step 7: inference cost (Mac Terminal)

```bash
open scripts/inference.command     # reference model, FP32: password once, ~25 min; Ctrl+C pauses, reopen to resume
python scripts/08_measure_inference.py --summary-only    # rebuild summary.md from the measurements
```

Runs the deployed pipeline one image or question at a time (image file -> CLIP preprocessing -> ViT-L/14 -> mapping network -> GPT-2 generate) on 500 Karpathy test images and 500 test questions. Each pass loads the model, runs 10 unmeasured items, measures 60 s of idle power, then measures encoding and decoding as two separate windows. Five repetitions in a shuffled order. An idle window that reads well above the session's others (macOS background jobs) is measured again, up to three times. Results go to `results/inference/<run>/<checkpoint>/`: `measurements.jsonl` (one record per pass), `flops.json`, `summary.md`. Options: `--precision`, `--decoding greedy beam3 beam5`, `--tasks`, `--n`, `--repeats`, `--label` (a separate results folder, e.g. for a smoke test: `--n 8 --repeats 1 --idle-seconds 5 --label _smoke`).

Close other apps, keep the Mac on AC power, and leave it alone while it runs: other work shows up in the measurement.

## Step 8: inference configurations (Mac Terminal)

```bash
open scripts/variants_accuracy.command   # accuracy of every configuration on Karpathy test: password once, ~1.5-2 h
open scripts/variants_energy.command     # energy, latency, memory: password once, ~2 h for 3 repetitions
python scripts/10_measure_variants.py --summary-only     # rebuild energy.md and summary.md
```

Configurations (`greenvl/precision.py`), all of the trained reference model: FP32 (reference); the encoder, the decoder (mapping network + GPT-2 + both adapter sets) or both at FP16, INT8 or NF4, captions at beam 3; and FP32 captions with greedy and beam 5 decoding.

`variants_accuracy.command` runs `09_evaluate_variants.py`: it caches Karpathy test features with the encoder at FP16, INT8 and NF4 (`02_extract_features.py --precision`), then scores every configuration on the full test split with `05_evaluate.py --precision` (all caption metrics, CHAIR, VQA, VQA-CE). `variants_energy.command` runs `10_measure_variants.py`: each repetition measures one idle window, then 26 windows in a shuffled order, 4 encoder windows (one per encoder precision, on the 1,000 step-7 subset images) and 22 decoding windows (each configuration's decoder, from its own encoder's embeddings), with a 20 s rest before each. A configuration's energy per caption or answer is its encoder window plus its decoding window. Every repetition includes FP32, so each configuration is also given relative to the FP32 of its own repetition. `--repeats 5` later adds two more repetitions without repeating the first three.

Results: `results/variants/<run>/<checkpoint>/`: `accuracy.md`, `energy.md`, `summary.md` (accuracy and cost side by side), with `.json` versions, `energy.jsonl` (every window), `memory.json` (memory per configuration, encoder and decoder loaded together). Ctrl+C pauses either script; reopening resumes. Code check: `python scripts/10_measure_variants.py --n 8 --repeats 1 --pause 1 --idle-seconds 5 --warmup 2 --label _smoke --allow-gpu-only-energy`.

## Fixed implementation choices

- Encoder: CLIP ViT-L/14 (chosen 2 Oct by the encoder check), frozen, features cached.
- Mapping network: ClipCap MLP (CLIP dim -> GPT-2 dim x 10 / 2 -> GPT-2 dim x 10, tanh), 31.5 M parameters for GPT-2 small with ViT-B/32, 32.5 M with ViT-L/14. CLIP features are L2-normalised before the MLP.
- LoRA on `attn.c_attn` and `attn.c_proj` of every GPT-2 block; alpha fixed at 8 (the reference rank) for all ranks, following Hu et al. One adapter set per task (`caption`, `vqa`); only the active task's adapter receives gradients.
- Input formats: captioning `[prefix] caption<eos>`; VQA `[prefix] Question: {q} Answer: {a}<eos>`, loss on answer tokens only.
- VQA training target: the most common of the ten human answers (`multiple_choice_answer`).
- Decoding: Hugging Face `generate` with the visual prefix as input embeddings (key-value cache on), length penalty 1.0, stop at end of text; captions up to 40 tokens, answers up to 8. Beam 3 is the reference caption decoding; question prompts are built exactly as in training and batched by length, so nothing is padded.
- Checkpoint selection: the `epoch_NN.pt` with the highest validation CIDEr (beam 3).
- CLIPScore: 2.5 x max(cos, 0) with the prompt "A photo depicts " (Hessel et al.), scored by OpenCLIP ViT-B/32 trained on LAION-2B (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`), whose weights and training data differ from the grid's three OpenAI encoders.
- CHAIR: Rohrbach et al.'s synonym list and rules; an image's objects are its COCO segmentation labels plus the objects its reference captions mention.
- VQA accuracy: the official evaluator's normalisation and ten leave-one-out subsets (`greenvl/third_party/vqa_eval.py`).
- Intervals: 1,000 bootstrap resamples of images, percentile 95 %; CHAIR_i and VQA accuracy resample each image with all its object mentions or questions.
- Reduced precision (step 8): FP16 casts every weight; INT8 (LLM.int8, outlier threshold 6.0) and NF4 (bitsandbytes defaults) quantize the linear layers and keep everything else in FP16, computing in FP16. GPT-2's output layer shares its weights with the token embedding and stays FP16. The LoRA adapters stay unmerged in FP16, so one decoder serves both tasks. The mapping network goes with the decoder. Model size counts quantization scales.
