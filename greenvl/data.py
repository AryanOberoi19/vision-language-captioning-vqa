"""Training data: cached CLIP features plus tokenised captions and VQA questions, batched one task at a time.

Methodology §5: the captioning and VQA losses are summed with equal weight and each batch is drawn from one task
with equal probability. Sequences (the 10 prefix vectors are prepended by the model):
    caption:  caption <eos>                                  loss on every token
    vqa:      Question: {q} Answer: {a} <eos>                loss on " {a} <eos>" only
"""
import json
import random
import zlib

import torch
from torch.utils.data import DataLoader, Dataset

from . import paths
from .model import DECODERS

MAX_CAPTION_TOKENS = 40   # COCO captions average ~13 GPT-2 tokens; longer ones are truncated
MAX_QUESTION_TOKENS = 32  # "Question: {q}" part
MAX_ANSWER_TOKENS = 8
IGNORE = -100
# Pad every batch to a multiple of this many tokens. On Apple's MPS backend each new tensor shape costs memory
# outside PyTorch's allocator (compiled graphs, fragmented heaps); with one shape per batch length that grew until
# the 24 GB Mac ran out (29 Sep 2026). Buckets of 16 leave at most three lengths per task.
PAD_MULTIPLE = 16

# Feature files per dataset family. Image ids are only unique within a family (COCO, Flickr8k and VizWiz ids overlap).
FAMILY_SETS = {"coco": ["coco_train", "coco_val", "coco_test"], "flickr8k": ["flickr8k"], "vizwiz": ["vizwiz_val"]}


def load_tokenizer(decoder: str = "gpt2"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DECODERS[decoder])  # GPT-2 small and medium share one tokenizer
    tok.pad_token = tok.eos_token
    return tok


class FeatureStore:
    """Cached CLIP embeddings of one encoder for one dataset family, looked up by image id."""

    def __init__(self, encoder: str, family: str, sets: list[str] | None = None):
        feats, self.index = [], {}
        for s in sets or FAMILY_SETS[family]:
            path = paths.FEATURES / encoder.replace("/", "-") / f"{s}.pt"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run scripts/02_extract_features.py --encoder {encoder}")
            d = torch.load(path)
            for i, image_id in enumerate(d["ids"].tolist()):
                self.index[image_id] = len(self.index)
            feats.append(d["feats"])
        self.feats = torch.cat(feats)
        self.dim = self.feats.shape[1]

    def get(self, image_id: int) -> torch.Tensor:
        return self.feats[self.index[image_id]]


# ---------------------------------------------------------------- raw rows

def caption_rows(family: str, split: str) -> list[dict]:
    """[{image_id, captions}] for a Karpathy split of COCO or Flickr8k."""
    name = {"coco": "coco_karpathy.json", "flickr8k": "flickr8k_karpathy.json"}[family]
    return json.loads((paths.PROCESSED / name).read_text())[split]


def vqa_rows(split: str) -> list[dict]:
    """[{question_id, image_id, question, answer}]; train = all train2014 questions, val / test = Karpathy images."""
    if split == "train":
        return json.loads((paths.PROCESSED / "vqa_train.json").read_text())
    qs = json.loads((paths.PROCESSED / f"vqa_{split}_questions.json").read_text())["questions"]
    ans = {a["question_id"]: a for a in json.loads((paths.PROCESSED / f"vqa_{split}_annotations.json").read_text())["annotations"]}
    return [{"question_id": q["question_id"], "image_id": q["image_id"], "question": q["question"],
             "answer": ans[q["question_id"]]["multiple_choice_answer"]} for q in qs]


# ---------------------------------------------------------------- datasets

class CaptionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, store: FeatureStore):
        pairs = [(r["image_id"], c) for r in rows for c in r["captions"]]
        self.image_ids = [p[0] for p in pairs]
        enc = tokenizer([p[1] for p in pairs], add_special_tokens=False)["input_ids"]
        eos = tokenizer.eos_token_id
        self.tokens = [t[:MAX_CAPTION_TOKENS] + [eos] for t in enc]
        self.store = store

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        ids = self.tokens[i]
        return self.store.get(self.image_ids[i]), ids, ids


class VQADataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, store: FeatureStore):
        self.image_ids = [r["image_id"] for r in rows]
        eos = tokenizer.eos_token_id
        questions = tokenizer([f"Question: {r['question']}" for r in rows], add_special_tokens=False)["input_ids"]
        answers = tokenizer([f" {r['answer']}" for r in rows], add_special_tokens=False)["input_ids"]
        suffix = tokenizer(" Answer:", add_special_tokens=False)["input_ids"]
        self.inputs, self.labels = [], []
        for q, a in zip(questions, answers):
            prompt = q[:MAX_QUESTION_TOKENS] + suffix
            answer = a[:MAX_ANSWER_TOKENS] + [eos]
            self.inputs.append(prompt + answer)
            self.labels.append([IGNORE] * len(prompt) + answer)
        self.store = store

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.store.get(self.image_ids[i]), self.inputs[i], self.labels[i]


def collate(pad_id: int, multiple: int = PAD_MULTIPLE):
    def fn(batch):
        feats, inputs, labels = zip(*batch)
        width = -(-max(len(x) for x in inputs) // multiple) * multiple
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        lab = torch.full((len(batch), width), IGNORE, dtype=torch.long)
        for j, (x, y) in enumerate(zip(inputs, labels)):
            ids[j, : len(x)] = torch.tensor(x)
            mask[j, : len(x)] = 1
            lab[j, : len(y)] = torch.tensor(y)
        return {"feats": torch.stack(feats), "input_ids": ids, "attention_mask": mask, "labels": lab}

    return fn


def make_loader(dataset: Dataset, batch_size: int, pad_id: int, shuffle: bool, seed: int = 0,
                drop_last: bool = False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, collate_fn=collate(pad_id),
                      num_workers=0, drop_last=drop_last)


class TaskMixer:
    """One epoch of batches, each drawn from one task with the given probability.

    An epoch is one expected pass over the anchor task (captions when present): with equal probabilities that
    is 2 x len(caption loader) steps. A task whose loader runs out mid-epoch restarts with a fresh shuffle.
    Training batches are always full (the last partial batch of a shuffle is dropped), keeping shapes fixed.
    Every random choice is seeded from (seed, epoch), so an interrupted epoch replays identically on resume.
    """

    def __init__(self, datasets: dict, probs: dict, batch_size: int, pad_id: int, seed: int, anchor: str = "caption"):
        self.datasets, self.probs, self.batch_size, self.pad_id, self.seed = datasets, probs, batch_size, pad_id, seed
        self.anchor = anchor if anchor in datasets else next(iter(datasets))
        n_anchor = max(1, len(datasets[self.anchor]) // batch_size)
        self.steps_per_epoch = round(n_anchor / probs[self.anchor])

    def epoch(self, epoch: int, skip: int = 0):
        rng = random.Random(self.seed * 1_000_003 + epoch)
        tasks, weights = list(self.probs), [self.probs[t] for t in self.probs]
        restarts = {t: 0 for t in tasks}

        def fresh(t):
            seed = zlib.crc32(f"{self.seed}-{epoch}-{t}-{restarts[t]}".encode())  # stable across processes
            restarts[t] += 1
            return iter(make_loader(self.datasets[t], self.batch_size, self.pad_id, True, seed, drop_last=True))

        its = {t: fresh(t) for t in tasks}
        for step in range(self.steps_per_epoch):
            task = rng.choices(tasks, weights)[0]
            try:
                batch = next(its[task])
            except StopIteration:
                its[task] = fresh(task)
                batch = next(its[task])
            if step >= skip:
                yield step, task, batch
