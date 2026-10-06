"""Reduced numeric precision at inference (methodology §6, Table 2: FP32, FP16, INT8, NF4 on the encoder, the
decoder or both). Applied after training to the trained model; nothing is retrained.

- FP16: every weight and buffer is cast to half precision.
- INT8 (LLM.int8, Dettmers et al. 2022) and NF4 (4-bit NormalFloat, Dettmers et al. 2023), through bitsandbytes:
  the weights of the linear layers are quantized; everything else stays in FP16 (token and position embeddings,
  the encoder's patch convolution, layer norms, and GPT-2's output layer, which shares its weights with the token
  embedding) and computation runs in FP16. INT8 uses LLM.int8's outlier threshold 6.0; NF4 uses bitsandbytes'
  defaults (blocks of 64, no double quantization). bitsandbytes quantizes a layer when it is moved to the device.
- The encoder is the CLIP vision tower with its projection. The decoder is everything after it: the mapping
  network, GPT-2 and both LoRA adapter sets (decided 3 Oct: the mapping network goes with the decoder, which it
  exists to feed). The adapters stay separate (unmerged) and in FP16 on top of the reduced-precision GPT-2, so one
  decoder still serves both tasks; merging them into quantized weights would need one decoder copy per task.
"""
import torch
import torch.nn as nn

from .model import IMAGE_MODELS

PRECISIONS = ("fp32", "fp16", "int8", "nf4")
REFERENCE = "fp32"
# configuration name -> (encoder precision, decoder precision); one factor at a time from the FP32 reference
CONFIGS = {REFERENCE: ("fp32", "fp32"),
           **{f"enc_{p}": (p, "fp32") for p in PRECISIONS[1:]},
           **{f"dec_{p}": ("fp32", p) for p in PRECISIONS[1:]},
           **{f"both_{p}": (p, p) for p in PRECISIONS[1:]}}
# the evaluated set (step 8): the precision factor at beam 3, then the caption-decoding factor at FP32
VARIANTS = [(c, "beam3") for c in CONFIGS] + [(REFERENCE, "greedy"), (REFERENCE, "beam5")]
INT8_THRESHOLD = 6.0


def variant_name(config: str, decoding: str) -> str:
    return config if decoding == "beam3" else f"{config}_{decoding}"


def compute_dtype(precision: str) -> torch.dtype:
    return torch.float32 if precision == "fp32" else torch.float16


def feature_name(encoder: str, precision: str) -> str:
    """Name under which features from this encoder at this precision are cached (Datasets/features/<name>/)."""
    return encoder if precision == "fp32" else f"{encoder}_{precision}"


def config_tag(checkpoint_stem: str, config: str) -> str:
    """File-name stem for a checkpoint evaluated in a configuration; the FP32 reference keeps its original names."""
    return checkpoint_stem if config == REFERENCE else f"{checkpoint_stem}_{config}"


def _conv1d():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except ImportError:  # pragma: no cover
        return ()


def _bnb_layer(layer: nn.Module, precision: str) -> nn.Module:
    """A bitsandbytes INT8 or NF4 layer holding `layer`'s weights (quantized once moved to the device)."""
    import bitsandbytes as bnb

    w = layer.weight.data
    if isinstance(layer, nn.Linear):
        out_f, in_f = w.shape
    else:  # GPT-2's Conv1D stores the weight as (in, out) and computes x @ W + b
        in_f, out_f = w.shape
        w = w.t()
    w = w.to(torch.float16).contiguous()
    has_bias = layer.bias is not None
    if precision == "int8":
        new = bnb.nn.Linear8bitLt(in_f, out_f, bias=has_bias, has_fp16_weights=False, threshold=INT8_THRESHOLD)
        new.weight = bnb.nn.Int8Params(w, requires_grad=False, has_fp16_weights=False)
    elif precision == "nf4":
        new = bnb.nn.Linear4bit(in_f, out_f, bias=has_bias, compute_dtype=torch.float16, quant_type="nf4")
        new.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4")
    else:
        raise ValueError(precision)
    if has_bias:
        new.bias = nn.Parameter(layer.bias.data.to(torch.float16), requires_grad=False)
    return new


def quantize_linears(module: nn.Module, precision: str, skip: tuple = ("lm_head",)) -> int:
    """Replace every nn.Linear and Conv1D in `module` (still on the CPU) with a bitsandbytes layer. Leaves out the
    LoRA adapters' own layers and modules named in `skip`. Returns the number of layers replaced."""
    kinds = (nn.Linear, _conv1d())
    targets = [(name, m) for name, m in module.named_modules()
               if isinstance(m, kinds) and "lora_" not in name and not set(name.split(".")) & set(skip)]
    for name, m in targets:
        parent, _, attr = name.rpartition(".")
        setattr(module.get_submodule(parent) if parent else module, attr, _bnb_layer(m, precision))
    return len(targets)


def convert(module: nn.Module, precision: str) -> int:
    """Bring a CPU fp32 module to `precision` in place (before it is moved to the device). Returns the number of
    quantized layers (0 for fp32 and fp16)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if precision == "fp32":
        return 0
    module.half()
    return quantize_linears(module, precision) if precision in ("int8", "nf4") else 0


def load_encoder(name: str, precision: str, device: torch.device):
    """(CLIP vision encoder with projection at `precision` on `device`, its image processor, quantized layers).
    The CNN-LSTM baseline's ResNet-50 (greenvl/showtell.py) loads here too, at FP32 only."""
    if name == "ResNet-50":
        from .showtell import ResNetEncoder, ResNetProcessor

        if precision != REFERENCE:
            raise ValueError("the ResNet-50 baseline encoder runs at fp32 only")
        return ResNetEncoder().to(device).eval(), ResNetProcessor(), 0
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from transformers.utils import logging as hf_logging

    hf_id = IMAGE_MODELS[name][0]
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # the vision-only class skips the text tower; not worth a report
    try:
        encoder = CLIPVisionModelWithProjection.from_pretrained(hf_id)
    finally:
        hf_logging.set_verbosity(level)
    n = convert(encoder, precision)
    return encoder.to(device).eval(), CLIPImageProcessor.from_pretrained(hf_id), n


def load_decoder(run: str, checkpoint: str, precision: str, device: torch.device):
    """(trained mapping network + GPT-2 + adapters at `precision` on `device`, run config, checkpoint path,
    quantized layers)."""
    from .decode import load_run

    model, cfg, ckpt = load_run(run, checkpoint, torch.device("cpu"))
    n = convert(model, precision)
    model.compute_dtype = compute_dtype(precision)
    return model.to(device).eval(), cfg, ckpt, n


def quant_state_bytes(t: torch.Tensor) -> int:
    """Bytes a bitsandbytes parameter keeps beside its stored weight: INT8 row scales, NF4 block scales."""
    n = 0
    scb = getattr(t, "SCB", None)
    if isinstance(scb, torch.Tensor):
        n += scb.numel() * scb.element_size()
    qs = getattr(t, "quant_state", None)
    while qs is not None:
        for attr in ("absmax", "code"):
            x = getattr(qs, attr, None)
            if isinstance(x, torch.Tensor):
                n += x.numel() * x.element_size()
        qs = getattr(qs, "state2", None)
    return n
