"""Building blocks of the analysis (methodology §9): object mentions with their position in the caption, teacher-forced
log-probabilities for the visual-grounding score, the dataset-mean visual prefix, and paired bootstrap resampling.

Grounding score (methodology §9, RQ2), per generated token y_t:
    g(y_t) = log p(y_t | y_<t, I) - log p(y_t | y_<t, Ī)
I is the image's own visual prefix, Ī the mean visual prefix over a fixed sample of training images. A word that
spans several tokens scores the sum over its tokens. A score near zero means the model would have produced the word
about as readily without seeing the image.
"""
import re

import numpy as np
import torch

from .chair import WORD
from .data import MAX_CAPTION_TOKENS

PAD_MULTIPLE = 16  # as in training: few distinct shapes on MPS


# ---------------------------------------------------------------- object mentions with character spans

def object_mentions(chair, caption: str) -> list[tuple[str, str, int, int]]:
    """[(word, category, start, end)] for every MSCOCO object mention in `caption`: exactly CHAIR's caption_objects
    (same singularisation, two-word objects, toilet-seat rule), with the character span of each mention."""
    words = [(chair.singular(m.group()), m.start(), m.end()) for m in WORD.finditer(caption.lower())]
    merged, i = [], 0
    while i < len(words):
        pair = " ".join(w for w, _, _ in words[i: i + 2])
        if pair in chair.double_words:
            merged.append((chair.double_words[pair], words[i][1], words[i + 1][2]))
            i += 2
        else:
            merged.append(words[i])
            i += 1
    if any(w == "toilet" for w, _, _ in merged) and any(w == "seat" for w, _, _ in merged):
        merged = [m for m in merged if m[0] != "seat"]
    return [(w, chair.inverse_synonym[w], s, e) for w, s, e in merged if w in chair.mscoco_objects]


def word_count(caption: str) -> int:
    return len(re.findall(r"[a-z0-9]+", caption.lower()))


# ---------------------------------------------------------------- teacher forcing

def caption_tokens(tokenizer, caption: str):
    """(token ids as in training: caption + eos, character offsets of the caption tokens)."""
    enc = tokenizer(caption, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"][:MAX_CAPTION_TOKENS]
    return ids + [tokenizer.eos_token_id], enc["offset_mapping"][: len(ids)]


@torch.no_grad()
def token_logprobs(model, seqs: list[list[int]], starts: list[int], pad_id: int, feats: torch.Tensor | None = None,
                   prefix: torch.Tensor | None = None, batch_size: int = 64) -> list[np.ndarray]:
    """log p(seq[j] | visual prefix, seq[:j]) for j >= starts[i], for every sequence i, under teacher forcing.
    The prefix is either the model's own from feats[i] (CLIP embeddings, one row per sequence) or one fixed prefix
    [1, k, d] for all (the mean prefix). Sequences are right-padded (causal attention, so padding never affects
    earlier positions) and batched by length."""
    dev = next(model.parameters()).device
    base = model.base_lm()
    wte = model.decoder.get_input_embeddings()
    k = model.prefix_len
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [None] * len(seqs)
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        width = -(-max(len(seqs[i]) for i in idx) // PAD_MULTIPLE) * PAD_MULTIPLE
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), k + width), dtype=torch.long)
        mask[:, :k] = 1
        for r, i in enumerate(idx):
            ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
            mask[r, k: k + len(seqs[i])] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        tok = wte(ids)
        pre = model.prefix(feats[idx].to(dev)) if feats is not None else prefix.expand(len(idx), -1, -1)
        hidden = base.transformer(inputs_embeds=torch.cat([pre.to(dev, tok.dtype), tok], 1),
                                  attention_mask=mask).last_hidden_state
        rows, targets, owners = [], [], []
        for r, i in enumerate(idx):
            for j in range(starts[i], len(seqs[i])):
                rows.append(r * (k + width) + k + j - 1)  # position k + j - 1 predicts token j
                targets.append(seqs[i][j])
                owners.append(i)
        h = hidden.reshape(-1, hidden.shape[-1])[torch.tensor(rows, device=dev)]
        lp = torch.log_softmax(base.lm_head(h).float(), dim=-1)
        vals = lp.gather(1, torch.tensor(targets, device=dev)[:, None])[:, 0].cpu().numpy()
        pos = 0
        for i in idx:
            n = len(seqs[i]) - starts[i]
            out[i] = vals[pos: pos + n]
            pos += n
    return out


@torch.no_grad()
def mean_prefix(model, feats: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """The dataset-mean visual prefix Ī: mean of the model's prefixes over `feats` (CLIP embeddings), [1, k, d]."""
    dev = next(model.parameters()).device
    total = None
    for i in range(0, len(feats), batch_size):
        p = model.prefix(feats[i: i + batch_size].to(dev)).float().sum(0)
        total = p if total is None else total + p
    return (total / len(feats))[None]


# ---------------------------------------------------------------- paired bootstrap over images

def bootstrap_weights(n: int, samples: int = 1000, seed: int = 0) -> np.ndarray:
    """[samples, n] counts: how often each image is drawn in each resample. One matrix is shared by every
    configuration and metric, so differences between configurations are paired."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(samples)]).astype(np.float64)


def boot_mean(per_image: np.ndarray, W: np.ndarray) -> np.ndarray:
    return W @ per_image / W.sum(1)


def boot_ratio(num: np.ndarray, den: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W @ num) / np.maximum(W @ den, 1e-12)


def ci95(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]
