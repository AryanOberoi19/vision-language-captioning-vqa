"""CNN-LSTM baseline (methodology §5): Show and Tell (Vinyals et al. 2015) over cached ResNet-50 features.

A frozen ImageNet ResNet-50 gives one 2048-d vector per image (global average pool, classifier removed). The vector
is projected to the word-embedding size and fed to a one-layer LSTM as its first input; the start token follows, and
each later step predicts the next word (teacher forcing in training). Words come from the evaluation's own PTB
tokenisation of the training captions (lower case, punctuation removed), so generated words are in the form the
references are scored in. Settings are fixed in advance, not tuned (agreed with Aryan, 5 Oct).
"""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ENCODER = "ResNet-50"
WEIGHTS = "IMAGENET1K_V2"  # torchvision's current ResNet-50 weights (80.9 % ImageNet top-1)
FEAT_DIM = 2048
EMBED = HIDDEN = 512
DROPOUT = 0.5
MIN_COUNT = 5  # words seen fewer times in the training captions become <unk>
MAX_WORDS = 16  # longer training captions are cut; generation stops after 16 words
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


# ---------------------------------------------------------------- encoder (same interface as the CLIP encoders)

class ResNetEncoder(nn.Module):
    """ResNet-50 without its classifier; forward(pixel_values=...) returns .image_embeds like the CLIP encoders, so
    02_extract_features.py, greenvl.inference.encode_image and the energy windows use it unchanged."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        net = resnet50(weights=ResNet50_Weights[WEIGHTS])
        net.fc = nn.Identity()
        self.net = net

    def forward(self, pixel_values):
        return SimpleNamespace(image_embeds=self.net(pixel_values))


class ResNetProcessor:
    """torchvision's preprocessing for these weights (resize 232, centre crop 224, ImageNet normalisation), called
    like a Hugging Face image processor. Module level so DataLoader workers can pickle it."""

    def __init__(self):
        from torchvision.models import ResNet50_Weights

        self.transform = ResNet50_Weights[WEIGHTS].transforms()

    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": self.transform(images).unsqueeze(0)}  # a PIL image, resized by PIL as in torchvision


# ---------------------------------------------------------------- vocabulary

class Vocab:
    def __init__(self, words: list[str]):
        self.itos = [PAD, BOS, EOS, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad, self.bos, self.eos, self.unk = (self.stoi[t] for t in (PAD, BOS, EOS, UNK))

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, tokenized: list[str], min_count: int = MIN_COUNT) -> "Vocab":
        counts = Counter(w for c in tokenized for w in c.split())
        return cls(sorted((w for w, n in counts.items() if n >= min_count), key=lambda w: (-counts[w], w)))

    def encode(self, tokenized: str) -> list[int]:
        return [self.stoi.get(w, self.unk) for w in tokenized.split()[:MAX_WORDS]]

    def decode(self, ids) -> str:
        return " ".join(self.itos[i] for i in ids if i not in (self.pad, self.bos, self.eos, self.unk))

    def save(self, path: Path):
        path.write_text(json.dumps({"itos": self.itos[4:], "min_count": MIN_COUNT, "max_words": MAX_WORDS}))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text())["itos"])


def sequences(vocab: Vocab, tokenized: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-length (MAX_WORDS + 1) inputs [<bos>, w1..wn, pad..] and targets [w1..wn, <eos>, pad..]."""
    T = MAX_WORDS + 1
    inp = torch.full((len(tokenized), T), vocab.pad, dtype=torch.int32)
    tgt = torch.full((len(tokenized), T), vocab.pad, dtype=torch.int32)
    for k, c in enumerate(tokenized):
        ids = vocab.encode(c)
        inp[k, : len(ids) + 1] = torch.tensor([vocab.bos] + ids, dtype=torch.int32)
        tgt[k, : len(ids) + 1] = torch.tensor(ids + [vocab.eos], dtype=torch.int32)
    return inp, tgt


# ---------------------------------------------------------------- model

class ShowTell(nn.Module):
    def __init__(self, vocab_size: int, feat_dim: int = FEAT_DIM, embed: int = EMBED, hidden: int = HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.image = nn.Linear(feat_dim, embed)
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, feats: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        """Logits [B, T, V] for the targets of inp ([B, T], starting with <bos>)."""
        x = torch.cat([self.image(feats)[:, None], self.embed(inp)], 1)
        h, _ = self.lstm(self.drop(x))
        return self.out(self.drop(h[:, 1:]))

    def start(self, feats: torch.Tensor):
        """LSTM state after the image step."""
        _, state = self.lstm(self.image(feats)[:, None])
        return state

    def step(self, tokens: torch.Tensor, state):
        h, state = self.lstm(self.embed(tokens)[:, None], state)
        return torch.log_softmax(self.out(h[:, 0]), -1), state


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def beam_search(model: ShowTell, feats: torch.Tensor, beams: int, vocab: Vocab) -> list[list[int]]:
    """Word ids per image. Mirrors the Hugging Face beam search used for GPT-2 (greenvl.decode): the 2k best
    continuations per step, an end token among the first k ends a hypothesis, scored by its summed log-probability
    divided by its length (length_penalty 1); an image is done when k hypotheses have ended (early_stopping); the best
    one is returned. With k = 1 this is greedy decoding. <unk> is never generated."""
    B, k, dev = feats.shape[0], beams, feats.device
    h, c = model.start(feats)
    state = (h.repeat_interleave(k, 1), c.repeat_interleave(k, 1))
    tokens = torch.full((B * k,), vocab.bos, dtype=torch.long, device=dev)
    scores = torch.full((B, k), float("-inf"), device=dev)
    scores[:, 0] = 0.0
    seqs = [[[] for _ in range(k)] for _ in range(B)]
    ended = [[] for _ in range(B)]
    done = [False] * B
    V = len(vocab)
    for t in range(MAX_WORDS + 1):
        logp, state = model.step(tokens, state)
        logp[:, [vocab.pad, vocab.bos, vocab.unk]] = float("-inf")
        if t == MAX_WORDS:  # length limit: every remaining hypothesis ends here
            eos = logp[:, vocab.eos].clone()
            logp.fill_(float("-inf"))
            logp[:, vocab.eos] = eos
        cand = (scores.reshape(-1, 1) + logp).reshape(B, k * V)
        top_s, top_i = cand.topk(2 * k, dim=1)
        top_s, top_i = top_s.tolist(), top_i.tolist()
        new_scores = torch.full((B, k), float("-inf"))
        new_tokens = torch.full((B, k), vocab.eos, dtype=torch.long)
        src = torch.zeros((B, k), dtype=torch.long)
        new_seqs = [[[] for _ in range(k)] for _ in range(B)]
        for b in range(B):
            if done[b]:
                continue
            j = 0
            for rank, (s, i) in enumerate(zip(top_s[b], top_i[b])):
                beam, word = divmod(i, V)
                if s == float("-inf"):
                    break
                if word == vocab.eos:
                    if rank < k:
                        seq = seqs[b][beam]
                        ended[b].append((s / max(len(seq), 1), seq))
                    continue
                if j < k:
                    new_scores[b, j], new_tokens[b, j], src[b, j] = s, word, beam
                    new_seqs[b][j] = seqs[b][beam] + [word]
                    j += 1
                if j == k:
                    break
            if len(ended[b]) >= k or j == 0:
                done[b] = True
        if all(done):
            break
        seqs = new_seqs
        scores = new_scores.to(dev)
        tokens = new_tokens.reshape(-1).to(dev)
        idx = (src + torch.arange(B)[:, None] * k).reshape(-1).to(dev)
        state = (state[0][:, idx], state[1][:, idx])
    return [max(e, key=lambda x: x[0])[1] if e else [] for e in ended]


def generate(model: ShowTell, vocab: Vocab, feats: torch.Tensor, decoding: str, batch_size: int = 64) -> list[str]:
    from .decode import DECODING

    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(feats), batch_size):
        out += [vocab.decode(s) for s in beam_search(model, feats[i: i + batch_size].to(dev), DECODING[decoding],
                                                     vocab)]
    return out
