"""End-to-end inference as deployed (methodology §8): image file -> CLIP encoder -> mapping network -> GPT-2, one
image or question at a time. Used to measure energy per caption and per answer, latency, memory, size and FLOPs.

A Pipeline wraps the frozen encoder with its image processor and a trained run (decode.load_run). encode() covers
reading the image, preprocessing, and the encoder; caption() and answer() cover the mapping network and decoding
(decode.generate_captions / answer_questions, so text is produced exactly as in evaluation).
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .decode import DECODING, answer_questions, generate_captions, load_run
from .device import synchronize
from .model import IMAGE_MODELS

PRECISIONS = ("fp32",)  # step 8 adds fp16, int8 and nf4 on the encoder, the decoder or both


@dataclass
class Pipeline:
    encoder: torch.nn.Module
    processor: object
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    cfg: dict
    checkpoint: Path
    precision: str = "fp32"

    @torch.inference_mode()
    def encode(self, image_path: Path) -> torch.Tensor:
        """Projected CLIP embedding of one image (float32, shape [dim]), as cached by 02_extract_features.py."""
        with Image.open(image_path) as im:
            pixels = self.processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"]
        emb = self.encoder(pixel_values=pixels.to(self.device)).image_embeds[0].float()
        synchronize(self.device)
        return emb

    def caption(self, emb: torch.Tensor, decoding: str) -> str:
        out = generate_captions(self.model, emb[None], decoding, batch_size=1, tokenizer=self.tokenizer)[0]
        synchronize(self.device)
        return out

    def answer(self, emb: torch.Tensor, question: str) -> str:
        out = answer_questions(self.model, emb[None], [question], batch_size=1, tokenizer=self.tokenizer)[0]
        synchronize(self.device)
        return out


def load_pipeline(run: str, checkpoint: str, device: torch.device, precision: str = "fp32") -> Pipeline:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from transformers.utils import logging as hf_logging

    from .data import load_tokenizer

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    model, cfg, ckpt = load_run(run, checkpoint, device)
    hf_id = IMAGE_MODELS[cfg["encoder"]][0]
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # the vision-only class skips the text tower; not worth a report
    try:
        encoder = CLIPVisionModelWithProjection.from_pretrained(hf_id).to(device).eval()
    finally:
        hf_logging.set_verbosity(level)
    processor = CLIPImageProcessor.from_pretrained(hf_id)
    return Pipeline(encoder, processor, model, load_tokenizer(cfg["decoder"]), device, cfg, ckpt, precision)


def tensor_bytes(module: torch.nn.Module) -> int:
    """Storage of every parameter and buffer as held in memory (counts quantized storage as stored)."""
    seen, total = set(), 0
    for t in [*module.parameters(), *module.buffers()]:
        if id(t) in seen:  # tied weights (GPT-2's embedding and output layer) count once
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def model_size_mb(p: Pipeline) -> dict:
    """Size per component in MB. The decoder includes both LoRA adapter sets; the mapper is the mapping network."""
    enc = tensor_bytes(p.encoder)
    mapper = tensor_bytes(p.model.mapper)
    dec = tensor_bytes(p.model.decoder)
    return {"encoder": enc / 2**20, "mapper": mapper / 2**20, "decoder": dec / 2**20,
            "total": (enc + mapper + dec) / 2**20}


def count_flops(p: Pipeline, image_paths: list[Path], questions: list[str] | None, decoding: str) -> dict:
    """Mean GFLOPs per image for the encoder, per caption and per answer (mapping network + decoding), counted with
    PyTorch's FlopCounterMode on the CPU. A property of the architecture and the generated lengths, so it is
    computed once, in fp32, for the reference precision."""
    from torch.utils.flop_counter import FlopCounterMode

    cpu = torch.device("cpu")
    pc = Pipeline(p.encoder.to(cpu), p.processor, p.model.to(cpu), p.tokenizer, cpu, p.cfg, p.checkpoint)
    enc, cap, ans = [], [], []
    try:
        for i, path in enumerate(image_paths):
            with FlopCounterMode(display=False) as fc:
                emb = pc.encode(path)
            enc.append(fc.get_total_flops())
            with FlopCounterMode(display=False) as fc:
                pc.caption(emb, decoding)
            cap.append(fc.get_total_flops())
            if questions:
                with FlopCounterMode(display=False) as fc:
                    pc.answer(emb, questions[i])
                ans.append(fc.get_total_flops())
    finally:
        p.encoder.to(p.device)
        p.model.to(p.device)
    mean = lambda xs: sum(xs) / len(xs) / 1e9 if xs else None  # noqa: E731
    return {"encoder_gflops": mean(enc), "caption_decode_gflops": mean(cap), "answer_decode_gflops": mean(ans),
            "images": len(image_paths), "decoding": decoding}


__all__ = ["DECODING", "Pipeline", "PRECISIONS", "count_flops", "load_pipeline", "model_size_mb", "tensor_bytes"]
