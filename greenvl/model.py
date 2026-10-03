"""Frozen CLIP features -> MLP mapping network -> k prefix vectors -> GPT-2 with one LoRA adapter set per task.

Methodology §3: CLIP and GPT-2 weights stay frozen; the shared mapping network (ClipCap MLP) and the two
adapter sets are the only trained parameters. The adaptation factor (§6, Table 2) swaps the adapters for
nothing ("frozen") or for full fine-tuning of GPT-2 ("full").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

ENCODERS = {  # name -> (Hugging Face id, projected embedding size)
    "ViT-B/32": ("openai/clip-vit-base-patch32", 512),
    "ViT-B/16": ("openai/clip-vit-base-patch16", 512),
    "ViT-L/14": ("openai/clip-vit-large-patch14", 768),
}
# CLIPScore scorer (methodology §7): a CLIP model outside the grid, so captions are not scored by the model that
# produced their visual features. OpenCLIP ViT-B/32 trained on LAION-2B: different weights and training data from
# all three OpenAI encoders above. Its image embeddings are cached like the encoders' (02_extract_features.py).
SCORER = "CLIPScore-LAION-B/32"
IMAGE_MODELS = {**ENCODERS, SCORER: ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 512)}
DECODERS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
}
TASKS = ("caption", "vqa")
ADAPTATIONS = ("frozen", "lora", "full")
LORA_TARGETS = ["attn.c_attn", "attn.c_proj"]  # attention projections only
LOGIT_ROWS = 512  # output-layer rows are padded to a multiple of this (fixed shapes on MPS)
LORA_ALPHA = 8  # Hu et al.: alpha set to the first rank tried (the reference rank 8) and held fixed across ranks


class MappingNetwork(nn.Module):
    """ClipCap MLP: one CLIP embedding -> prefix_len vectors in GPT-2's input space."""

    def __init__(self, clip_dim: int, gpt_dim: int, prefix_len: int = 10):
        super().__init__()
        hidden = gpt_dim * prefix_len // 2
        self.prefix_len, self.gpt_dim = prefix_len, gpt_dim
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, gpt_dim * prefix_len),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).view(feats.shape[0], self.prefix_len, self.gpt_dim)


def build_decoder(
    decoder: str = "gpt2",
    adaptation: str = "lora",
    rank: int = 8,
    alpha: int = LORA_ALPHA,
    dropout: float = 0.0,
    config: GPT2Config | None = None,
):
    """GPT-2 with the requested adaptation. `config` builds a randomly initialised model (offline tests)."""
    if adaptation not in ADAPTATIONS:
        raise ValueError(f"adaptation must be one of {ADAPTATIONS}")
    base = GPT2LMHeadModel(config) if config is not None else GPT2LMHeadModel.from_pretrained(DECODERS[decoder])

    if adaptation == "full":
        base.requires_grad_(True)
        return base
    base.requires_grad_(False)
    if adaptation == "frozen":
        return base

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS,
        fan_in_fan_out=True,  # GPT-2 uses Conv1D (weights stored transposed)
        bias="none",
    )
    model = get_peft_model(base, lora, adapter_name=TASKS[0])
    for task in TASKS[1:]:
        model.add_adapter(task, lora)
    model.set_adapter(TASKS[0])
    return model


class CaptionVQAModel(nn.Module):
    """Prefix-conditioned GPT-2 serving both tasks.

    Captioning:  [prefix] caption <eos>
    VQA:         [prefix] "Question: {q} Answer:" " {a}" <eos>   (loss on the answer tokens only)
    """

    def __init__(self, decoder: nn.Module, clip_dim: int, prefix_len: int = 10, normalize_features: bool = True):
        super().__init__()
        self.decoder = decoder
        self.mapper = MappingNetwork(clip_dim, decoder.config.n_embd, prefix_len)
        self.prefix_len = prefix_len
        self.normalize_features = normalize_features
        self.task = TASKS[0]
        self.compute_dtype = torch.float32  # FP16 when the decoder runs at reduced precision (greenvl/precision.py)

    @property
    def uses_adapters(self) -> bool:
        return hasattr(self.decoder, "peft_config")

    def set_task(self, task: str) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        self.task = task
        if self.uses_adapters:
            self.decoder.set_adapter(task)

    def prefix(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.float()
        if self.normalize_features:
            feats = F.normalize(feats, dim=-1)
        return self.mapper(feats.to(self.compute_dtype))

    def embed(self, feats: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        tokens = self.decoder.get_input_embeddings()(input_ids)
        prefix = self.prefix(feats).to(tokens.dtype)
        embeds = torch.cat([prefix, tokens], dim=1)
        ones = attention_mask.new_ones(attention_mask.shape[0], self.prefix_len)
        return embeds, torch.cat([ones, attention_mask], dim=1)

    def base_lm(self) -> nn.Module:
        """The GPT2LMHeadModel itself (with any adapters injected into its layers)."""
        return self.decoder.get_base_model() if self.uses_adapters else self.decoder

    def forward(self, feats, input_ids, attention_mask, labels):
        """Mean token-level cross-entropy under teacher forcing. labels = -100 where no loss is taken.

        Same value as GPT-2's built-in loss, but vocabulary logits are computed only at the positions that carry
        a label. Prefix, padding and question positions never reach the 50,257-way output layer, which cuts
        memory (the full logits tensor was ~0.7 GB per batch) and compute.
        """
        embeds, mask = self.embed(feats, input_ids, attention_mask)
        base = self.base_lm()
        hidden = base.transformer(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        ignore = labels.new_full((labels.shape[0], self.prefix_len), -100)
        target = torch.cat([ignore, labels], dim=1)[:, 1:].reshape(-1)  # position t predicts token t + 1
        hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        rows = (target != -100).nonzero().squeeze(1)
        # Round the number of rows up to a multiple of LOGIT_ROWS with ignored dummy rows, so the output layer
        # sees a handful of shapes rather than a new one every batch (see data.PAD_MULTIPLE).
        n = rows.numel()
        padded = F.pad(rows, (0, -(-n // LOGIT_ROWS) * LOGIT_ROWS - n))
        rows_target = target[padded].clone()
        rows_target[n:] = -100
        logits = base.lm_head(hidden[padded])
        return F.cross_entropy(logits.float(), rows_target, ignore_index=-100)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Mapping network + every adapter set (inactive adapters included) or the full decoder."""
        return [p for n, p in self.named_parameters() if p.requires_grad or "lora_" in n]

    def parameter_counts(self) -> dict[str, int]:
        counts = {"mapper": sum(p.numel() for p in self.mapper.parameters())}
        for task in TASKS:
            counts[f"lora_{task}"] = sum(p.numel() for n, p in self.decoder.named_parameters() if f".{task}." in n)
        counts["decoder_trainable_full"] = (
            sum(p.numel() for p in self.decoder.parameters()) if not self.uses_adapters and any(
                p.requires_grad for p in self.decoder.parameters()) else 0
        )
        counts["trainable_total"] = sum(p.numel() for p in self.trainable_parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
