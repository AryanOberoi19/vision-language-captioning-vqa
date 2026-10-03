"""End-to-end inference as deployed (methodology §8): image file -> CLIP encoder -> mapping network -> GPT-2, one
image or question at a time. Used to measure energy per caption and per answer, latency, memory, size and FLOPs.

A Pipeline wraps the frozen encoder with its image processor and a trained run (decode.load_run), each at the
precision of a configuration in greenvl/precision.py (FP32 reference; FP16, INT8, NF4 on the encoder, the decoder
or both). encode() covers reading the image, preprocessing, and the encoder; caption() and answer() cover the
mapping network and decoding (decode.generate_captions / answer_questions, so text is produced exactly as in
evaluation).
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .decode import DECODING, answer_questions, generate_captions
from .device import synchronize
from .precision import CONFIGS, REFERENCE, compute_dtype, load_decoder, load_encoder, quant_state_bytes


@dataclass
class Pipeline:
    encoder: torch.nn.Module
    processor: object
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    cfg: dict
    checkpoint: Path
    config: str = REFERENCE

    @property
    def encoder_precision(self) -> str:
        return CONFIGS[self.config][0]

    @property
    def decoder_precision(self) -> str:
        return CONFIGS[self.config][1]

    @torch.inference_mode()
    def encode(self, image_path: Path) -> torch.Tensor:
        """Projected CLIP embedding of one image (float32, shape [dim]), as cached by 02_extract_features.py."""
        return encode_image(self.encoder, self.processor, image_path, self.device, self.encoder_precision)

    def caption(self, emb: torch.Tensor, decoding: str) -> str:
        out = generate_captions(self.model, emb[None], decoding, batch_size=1, tokenizer=self.tokenizer)[0]
        synchronize(self.device)
        return out

    def answer(self, emb: torch.Tensor, question: str) -> str:
        out = answer_questions(self.model, emb[None], [question], batch_size=1, tokenizer=self.tokenizer)[0]
        synchronize(self.device)
        return out


@torch.inference_mode()
def encode_image(encoder, processor, image_path: Path, device: torch.device, precision: str) -> torch.Tensor:
    with Image.open(image_path) as im:
        pixels = processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"]
    emb = encoder(pixel_values=pixels.to(device, compute_dtype(precision))).image_embeds[0].float()
    synchronize(device)
    return emb


def load_pipeline(run: str, checkpoint: str, device: torch.device, config: str = REFERENCE) -> Pipeline:
    from .data import load_tokenizer

    if config not in CONFIGS:
        raise ValueError(f"config must be one of {list(CONFIGS)}")
    enc_p, dec_p = CONFIGS[config]
    model, cfg, ckpt, _ = load_decoder(run, checkpoint, dec_p, device)
    encoder, processor, _ = load_encoder(cfg["encoder"], enc_p, device)
    return Pipeline(encoder, processor, model, load_tokenizer(cfg["decoder"]), device, cfg, ckpt, config)


def tensor_bytes(module: torch.nn.Module) -> int:
    """Storage of every parameter and buffer as held in memory, quantized weights as stored plus their scales.
    Tied weights (GPT-2's embedding and output layer) count once."""
    seen, total = set(), 0
    for t in [*module.parameters(), *module.buffers()]:
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size() + quant_state_bytes(t)
    return total


def model_size_mb(p: Pipeline) -> dict:
    """Size per component in MiB. The decoder includes both LoRA adapter sets; the mapper is the mapping network."""
    return component_sizes(p.encoder, p.model)


def component_sizes(encoder: torch.nn.Module | None, model: torch.nn.Module | None) -> dict:
    out = {}
    if encoder is not None:
        out["encoder"] = tensor_bytes(encoder) / 2**20
    if model is not None:
        out["mapper"] = tensor_bytes(model.mapper) / 2**20
        out["decoder"] = tensor_bytes(model.decoder) / 2**20
    if len(out) == 3:
        out["total"] = sum(out.values())
    return out


def count_flops(p: Pipeline, image_paths: list[Path], questions: list[str] | None, decoding: str) -> dict:
    """Mean GFLOPs per image for the encoder, per caption and per answer (mapping network + decoding), counted with
    PyTorch's FlopCounterMode on the CPU. A property of the architecture and the generated lengths, so it is
    computed once, in fp32, for the reference precision."""
    from torch.utils.flop_counter import FlopCounterMode

    if p.config != REFERENCE:
        raise ValueError("FLOPs are counted on the FP32 reference pipeline")
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


__all__ = ["CONFIGS", "DECODING", "Pipeline", "component_sizes", "count_flops", "encode_image", "load_pipeline",
           "model_size_mb", "tensor_bytes"]
