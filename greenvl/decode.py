"""Generation from a trained run: captions (greedy, beam 3, beam 5; methodology Table 2) and VQA answers (greedy).

Generation goes through Hugging Face `generate` on GPT-2 with the visual prefix passed as input embeddings, so the
key-value cache is used as in deployment. Prompts are never padded: captions share one prompt length (the prefix),
and questions are batched by prompt length, so every sequence sees the same positions as in training.
"""
import json
from collections import defaultdict
from pathlib import Path

import torch

from . import paths
from .data import MAX_ANSWER_TOKENS, MAX_CAPTION_TOKENS, MAX_QUESTION_TOKENS, load_tokenizer
from .model import CaptionVQAModel, build_decoder

DECODING = {"greedy": 1, "beam3": 3, "beam5": 5}


def weights_files(run_dir: Path) -> list[Path]:
    return sorted([*run_dir.glob("epoch_*.pt"), *run_dir.glob("step_*.pt")])


def load_run(run_name: str, checkpoint: str | None, device: torch.device):
    """(model in eval mode, run config, checkpoint path). checkpoint: a file name in the run folder; default newest."""
    run_dir = paths.RESULTS / "runs" / run_name
    cfg = json.loads((run_dir / "config.json").read_text())
    if checkpoint:
        ckpt = run_dir / checkpoint
    else:
        files = weights_files(run_dir)
        if not files:
            raise FileNotFoundError(f"no epoch_*.pt or step_*.pt in {run_dir}")
        ckpt = max(files, key=lambda p: p.stat().st_mtime)
    state = torch.load(ckpt, map_location="cpu")
    decoder = build_decoder(cfg["decoder"], cfg["adaptation"], cfg["rank"])
    clip_dim = state["mapper.net.0.weight"].shape[1]  # the encoder's embedding size (ENCODERS), as trained
    model = CaptionVQAModel(decoder, clip_dim)
    missing, unexpected = model.load_state_dict(state, strict=False)
    trained = {n for n, p in model.named_parameters() if p.requires_grad or "lora_" in n}
    if unexpected or (trained - set(state)):
        raise ValueError(f"{ckpt.name} does not match the configuration in {run_dir / 'config.json'}")
    return model.to(device).eval(), cfg, ckpt


def _generate(model, embeds, num_beams, max_new_tokens, eos_id):
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    return model.base_lm().generate(
        inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens, num_beams=num_beams,
        do_sample=False, early_stopping=num_beams > 1, length_penalty=1.0, num_return_sequences=1,
        eos_token_id=eos_id, pad_token_id=eos_id)


def _text(tokenizer, seq, eos_id):
    ids = seq.tolist()
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    return tokenizer.decode(ids).strip()


@torch.no_grad()
def generate_captions(model, feats: torch.Tensor, decoding: str = "beam3", batch_size: int = 64,
                      tokenizer=None) -> list[str]:
    """One caption per row of feats (CLIP embeddings, in order)."""
    tokenizer = tokenizer or load_tokenizer()
    eos = tokenizer.eos_token_id
    dev = next(model.parameters()).device
    model.set_task("caption")
    out = []
    for i in range(0, len(feats), batch_size):
        embeds = model.prefix(feats[i: i + batch_size].to(dev))
        seqs = _generate(model, embeds, DECODING[decoding], MAX_CAPTION_TOKENS + 1, eos)
        out += [_text(tokenizer, s, eos) for s in seqs]
    return out


def question_prompt(tokenizer, question: str) -> list[int]:
    """Token ids of 'Question: {q} Answer:' exactly as built for training (greenvl.data.VQADataset)."""
    q = tokenizer(f"Question: {question}", add_special_tokens=False)["input_ids"][:MAX_QUESTION_TOKENS]
    return q + tokenizer(" Answer:", add_special_tokens=False)["input_ids"]


@torch.no_grad()
def answer_questions(model, feats: torch.Tensor, questions: list[str], batch_size: int = 64,
                     tokenizer=None) -> list[str]:
    """Greedy answer per (feats[i], questions[i]). Questions are batched by prompt length, so there is no padding."""
    tokenizer = tokenizer or load_tokenizer()
    eos = tokenizer.eos_token_id
    dev = next(model.parameters()).device
    model.set_task("vqa")
    prompts = [question_prompt(tokenizer, q) for q in questions]
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    answers = [None] * len(questions)
    for length in sorted(by_len):
        idx = by_len[length]
        for j in range(0, len(idx), batch_size):
            chunk = idx[j: j + batch_size]
            ids = torch.tensor([prompts[i] for i in chunk], device=dev)
            embeds, _ = model.embed(feats[chunk].to(dev), ids, torch.ones_like(ids))
            seqs = _generate(model, embeds, 1, MAX_ANSWER_TOKENS + 1, eos)
            for i, s in zip(chunk, seqs):
                answers[i] = _text(tokenizer, s, eos)
    return answers
