#!/usr/bin/env python3
"""Step 11: local demo. Caption an image and answer a question with any precision configuration, and show what that
request cost: latency and energy measured live for this request, next to the averages measured in steps 8-10.

    open scripts/demo.command                       # opens http://127.0.0.1:7860 in the browser
    python scripts/14_demo.py [--port 7860] [--no-browser]

Live energy is one energy-counter reading around this request (CPU, GPU and DRAM with the password given at start,
GPU only without it), incl. idle and above the idle power measured at start-up. A single short request is noisy;
the step-10 averages (500 items x 3 repetitions, batch 1) are the reference values. The first request after
choosing a configuration loads its parts (seconds to a minute); every part is run once on a blank image when
loaded, so the measured request does not include that warm-up. The CNN-LSTM baseline (13_baseline.py) is offered
when its trained weights exist. Runs on this Mac only (no public link).
"""
import argparse
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl.device import get_device, synchronize  # noqa: E402  (first: sets MPS memory limits)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.data import load_tokenizer  # noqa: E402
from greenvl.decode import answer_questions, generate_captions  # noqa: E402
from greenvl.energy import EnergyMeter  # noqa: E402
from greenvl.inference import encode_image  # noqa: E402
from greenvl.measure import CO2_G_PER_KWH, subset  # noqa: E402
from greenvl.precision import CONFIGS, REFERENCE, load_decoder, load_encoder, variant_name  # noqa: E402

RUN, CHECKPOINT = "coco_ViT-L-14_lora8_lr1e-3_s0", "epoch_02.pt"
BASELINE_DIR = paths.RESULTS / "baseline" / "showtell_resnet50_s0"
IDLE_SECONDS = 10
LABELS = {"fp32": "FP32 (reference)", "enc_fp16": "Encoder FP16", "enc_int8": "Encoder INT8", "enc_nf4": "Encoder NF4",
          "dec_fp16": "Decoder FP16", "dec_int8": "Decoder INT8", "dec_nf4": "Decoder NF4", "both_fp16": "Both FP16",
          "both_int8": "Both INT8", "both_nf4": "Both NF4"}
DECODINGS = [("Beam 3 (reference)", "beam3"), ("Greedy", "greedy"), ("Beam 5", "beam5")]


# ---------------------------------------------------------------- models (loaded on first use, kept)

class Models:
    def __init__(self, dev):
        self.dev, self.encoders, self.decoders, self.baseline = dev, {}, {}, None
        self.tok, self.cfg = load_tokenizer("gpt2"), json.loads(
            (paths.RESULTS / "runs" / RUN / "config.json").read_text())
        blank = Path(tempfile.gettempdir()) / "greenvl_demo_blank.png"
        Image.new("RGB", (224, 224), (128, 128, 128)).save(blank)
        self.blank = blank

    def encoder(self, p):
        if p not in self.encoders:
            enc, proc, _ = load_encoder(self.cfg["encoder"], p, self.dev)
            encode_image(enc, proc, self.blank, self.dev, p)  # warm-up
            self.encoders[p] = (enc, proc)
        return self.encoders[p]

    def decoder(self, p):
        if p not in self.decoders:
            model, _, _, _ = load_decoder(RUN, CHECKPOINT, p, self.dev)
            emb = torch.zeros(1, model.mapper.net[0].in_features, device=self.dev)
            generate_captions(model, emb, "greedy", batch_size=1, tokenizer=self.tok)  # warm-up
            answer_questions(model, emb, ["what is this?"], batch_size=1, tokenizer=self.tok)
            self.decoders[p] = model
        return self.decoders[p]

    def showtell(self):
        if self.baseline is None:
            from greenvl.showtell import ENCODER, ShowTell, Vocab, beam_search

            enc, proc, _ = load_encoder(ENCODER, REFERENCE, self.dev)
            vocab = Vocab.load(BASELINE_DIR / "vocab.json")
            model = ShowTell(len(vocab))
            model.load_state_dict(torch.load(BASELINE_DIR / "best.pt", map_location="cpu")["weights"])
            model = model.to(self.dev).eval()
            e = encode_image(enc, proc, self.blank, self.dev, REFERENCE)
            beam_search(model, e[None], 3, vocab)  # warm-up
            self.baseline = (enc, proc, model, vocab, beam_search)
        return self.baseline


# ---------------------------------------------------------------- reference values (steps 8-10)

def reference_values() -> dict:
    """{(variant, task): {"j_above", "j", "lat_ms", "acc"}} from step 10's energy.json and step 8's accuracy.json, and
    the baseline's summary.json if it exists."""
    out = {}
    d = paths.RESULTS / "variants" / RUN / Path(CHECKPOINT).stem
    acc = {}
    if (d / "accuracy.json").exists():
        acc = {r["variant"]: r for r in json.loads((d / "accuracy.json").read_text())["rows"]}
    if (d / "energy.json").exists():
        for r in json.loads((d / "energy.json").read_text())["rows"]:
            v = variant_name(r["config"], r["decoding"]) if r["task"] == "caption" else r["config"]
            a = acc.get(variant_name(r["config"], r["decoding"] if r["task"] == "caption" else "beam3"), {})
            metric = a.get("CIDEr") if r["task"] == "caption" else a.get("VQA")
            out[(v, r["task"])] = {"j_above": r["j_above"][0], "j": r["j"][0], "lat_ms": r["lat_median"][0],
                                   "acc": metric["value"] if metric else None}
    s = BASELINE_DIR / "summary.json"
    if s.exists():
        x = json.loads(s.read_text())
        b = (x.get("cost") or {}).get("baseline")
        if b:
            out[("showtell", "caption")] = {"j_above": b["j_above"][0], "j": b["j"][0], "lat_ms": b["lat_median"][0],
                                            "acc": 100 * x["test"]["CIDEr"]["value"] if x.get("test") else None}
    return out


# ---------------------------------------------------------------- one request

class Demo:
    def __init__(self, dev):
        self.dev = dev
        self.models = Models(dev)
        self.meter = EnergyMeter(dev.type)
        idle = self.meter.measure_idle(IDLE_SECONDS) if self.meter.available else None
        self.idle_w = idle["watts"] if idle else None
        self.ref = reference_values()
        self.lock = threading.Lock()  # one request at a time: energy windows must not overlap

    def energy_note(self) -> str:
        if not self.meter.available:
            return "Energy counters are not available on this device; latency only."
        parts = "CPU, GPU and DRAM" if self.meter.counters_live else "GPU only (start with the password for CPU and DRAM)"
        idle = f", idle {self.idle_w:.2f} W measured at start-up" if self.idle_w is not None else ""
        return f"Live energy: {parts}{idle}."

    def timed(self, fn):
        if self.meter.available:
            self.meter.begin("request")
        t0 = time.perf_counter()
        out = fn()
        synchronize(self.dev)
        seconds = time.perf_counter() - t0
        e = self.meter.end("request") if self.meter.available else None
        j = e.get("total_j") if e else None
        above = j - self.idle_w * e["seconds"] if j is not None and self.idle_w is not None else None
        return out, 1000 * seconds, j, above

    def __call__(self, image_path, config, decoding, question, with_baseline):
        if not image_path:
            return "Upload an image or pick an example.", "", "", ""
        with self.lock:
            enc_p, dec_p = CONFIGS[config]
            enc, proc = self.models.encoder(enc_p)
            model = self.models.decoder(dec_p)
            tok = self.models.tok
            emb, enc_ms, enc_j, enc_above = self.timed(lambda: encode_image(enc, proc, Path(image_path), self.dev,
                                                                            enc_p))
            caption, cap_ms, cap_j, cap_above = self.timed(
                lambda: generate_captions(model, emb[None], decoding, batch_size=1, tokenizer=tok)[0])
            rows = [self.row(f"{LABELS[config]}, caption ({decoding})", enc_ms + cap_ms, enc_j, cap_j, enc_above,
                             cap_above, self.ref.get((variant_name(config, decoding), "caption")), "CIDEr")]
            answer = ""
            if question and question.strip():
                answer, ans_ms, ans_j, ans_above = self.timed(
                    lambda: answer_questions(model, emb[None], [question.strip()], batch_size=1, tokenizer=tok)[0])
                rows.append(self.row(f"{LABELS[config]}, answer", enc_ms + ans_ms, enc_j, ans_j, enc_above, ans_above,
                                     self.ref.get((config, "vqa")), "VQA"))
            base_caption = ""
            if with_baseline and (BASELINE_DIR / "best.pt").exists():
                benc, bproc, bmodel, vocab, beam_search = self.models.showtell()
                bemb, b_enc_ms, b_enc_j, b_enc_above = self.timed(
                    lambda: encode_image(benc, bproc, Path(image_path), self.dev, REFERENCE))
                ids, b_ms, b_j, b_above = self.timed(lambda: beam_search(bmodel, bemb[None], 3, vocab)[0])
                base_caption = vocab.decode(ids)
                rows.append(self.row("Show and Tell (CNN-LSTM), caption (beam3)", b_enc_ms + b_ms, b_enc_j, b_j,
                                     b_enc_above, b_above, self.ref.get(("showtell", "caption")), "CIDEr"))
        return caption, answer, base_caption, TABLE_HEAD + "\n".join(rows)

    @staticmethod
    def row(name, ms, j1, j2, a1, a2, ref, metric):
        """One Markdown table row: this request, the step-10 average, test accuracy."""
        j = j1 + j2 if j1 is not None and j2 is not None else None
        above = a1 + a2 if a1 is not None and a2 is not None else None
        now = f"{ms:.0f} ms"
        if above is not None:
            now += f" · {above:.2f} J above idle ({j:.2f} incl.)"
        elif j is not None:
            now += f" · {j:.2f} J incl. idle"
        ref = ref or {}
        avg = "not measured" if ref.get("j_above") is None else (
            f"{ref['lat_ms']:.0f} ms · {ref['j_above']:.2f} J above idle · "
            f"{ref['j'] * 1000 / 3.6e6 * CO2_G_PER_KWH:.2f} g CO2e per 1,000")
        acc = "-" if ref.get("acc") is None else f"{metric} {ref['acc']:.1f}"
        return f"| {name} | {now} | {avg} | {acc} |"


TABLE_HEAD = ("| Request | This request | Measured average (batch 1) | Accuracy (test) |\n"
              "|---|---|---|---|\n")


def build_ui(demo: Demo, examples: list[str]):
    import gradio as gr

    has_baseline = (BASELINE_DIR / "best.pt").exists()
    with gr.Blocks(title="Green captioning and VQA") as ui:
        gr.Markdown("## Captioning and VQA at different precisions\n"
                    f"CLIP ViT-L/14 + mapping network + GPT-2 with LoRA ({RUN}, {CHECKPOINT}), on this Mac. "
                    + demo.energy_note() + " Measured averages: 500 Karpathy test items x 3 repetitions, batch 1 (step 10; "
                    "the CNN-LSTM baseline from its own run), CPU + GPU + DRAM; test "
                    "accuracy from step 8 (5,000 images).")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="filepath", label="Image")
                config = gr.Dropdown(choices=[(LABELS[c], c) for c in CONFIGS], value=REFERENCE,
                                     label="Precision configuration")
                decoding = gr.Radio(choices=DECODINGS, value="beam3", label="Caption decoding")
                question = gr.Textbox(label="Question (optional)", placeholder="What is the man holding?")
                with_baseline = gr.Checkbox(value=False, label="Also caption with the CNN-LSTM baseline",
                                            visible=has_baseline)
                go = gr.Button("Run", variant="primary")
            with gr.Column():
                caption = gr.Textbox(label="Caption", interactive=False)
                answer = gr.Textbox(label="Answer", interactive=False)
                base_caption = gr.Textbox(label="CNN-LSTM baseline caption", interactive=False, visible=has_baseline)
                table = gr.Markdown()
        if examples:
            gr.Examples(examples=[[e] for e in examples], inputs=[image], label="Karpathy test images")
        go.click(demo, inputs=[image, config, decoding, question, with_baseline],
                 outputs=[caption, answer, base_caption, table], api_name="run")
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = get_device(args.device)
    print(f"device {dev}; measuring idle power for {IDLE_SECONDS} s, then loading FP32", flush=True)
    demo = Demo(dev)
    demo.models.encoder(REFERENCE)
    demo.models.decoder(REFERENCE)
    try:
        examples = [str(paths.COCO_IMAGES / it["file"]) for it in subset("test", 500)["caption"][:6]]
        examples = [e for e in examples if Path(e).exists()]
    except FileNotFoundError:
        examples = []
    ui = build_ui(demo, examples)
    ui.queue(default_concurrency_limit=1)
    try:
        ui.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser)
    finally:
        demo.meter.close()


if __name__ == "__main__":
    main()
