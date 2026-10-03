"""Captioning metrics (methodology §7) and bootstrap confidence intervals.

BLEU-4, CIDEr(-D) and SPICE come from pycocoevalcap after its PTB tokenisation (the COCO caption evaluation);
BLEU-4 is recomputed here from per-image n-gram counts, identical to pycocoevalcap's corpus value, so that it can
be bootstrapped. CLIPScore (Hessel et al.) uses a CLIP model outside the grid (model.SCORER). Both PTB tokenisation
and SPICE need Java.
"""
import math
from collections import Counter
from pathlib import Path

import numpy as np

BOOTSTRAP_SAMPLES = 1000
CLIPSCORE_W = 2.5
CLIPSCORE_PROMPT = "A photo depicts "


# ---------------------------------------------------------------- tokenisation

def ptb_tokenize(captions: dict) -> dict:
    """{id: [caption, ...]} -> {id: [tokenised caption, ...]} with the COCO evaluation's PTB tokenizer."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    return PTBTokenizer().tokenize({k: [{"caption": c} for c in v] for k, v in captions.items()})


# ---------------------------------------------------------------- BLEU-4 with per-image statistics

def _ngrams(words, n=4):
    c = Counter()
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            c[tuple(words[i: i + k])] += 1
    return c


def bleu_stats(refs: list[str], hyp: str, n: int = 4) -> np.ndarray:
    """[testlen, reflen(closest), guess_1..n, correct_1..n] for one image, as in pycocoevalcap's BleuScorer."""
    maxref = Counter()
    lens = []
    for r in refs:
        w = r.split()
        lens.append(len(w))
        for g, c in _ngrams(w, n).items():
            maxref[g] = max(maxref[g], c)
    words = hyp.split()
    t = len(words)
    reflen = min((abs(l - t), l) for l in lens)[1]
    guess = [max(0, t - k + 1) for k in range(1, n + 1)]
    correct = [0] * n
    for g, c in _ngrams(words, n).items():
        correct[len(g) - 1] += min(maxref.get(g, 0), c)
    return np.array([t, reflen, *guess, *correct], dtype=np.float64)


def bleu_from_stats(s: np.ndarray, n: int = 4) -> float:
    """Corpus BLEU-n from summed statistics (pycocoevalcap's formula, including its smoothing constants)."""
    tiny, small = 1e-15, 1e-9
    t, r, guess, correct = s[0], s[1], s[2: 2 + n], s[2 + n: 2 + 2 * n]
    bleu = 1.0
    for k in range(n):
        bleu *= (correct[k] + tiny) / (guess[k] + small)
    bleu **= 1.0 / n
    ratio = (t + tiny) / (r + small)
    if ratio < 1:
        bleu *= math.exp(1 - 1 / ratio)
    return float(bleu)


# ---------------------------------------------------------------- CIDEr, SPICE

def cider(gts: dict, res: dict):
    """(corpus CIDEr-D, per-image scores in sorted(gts) order); IDF from the references of the evaluated images."""
    from pycocoevalcap.cider.cider import Cider

    keys = sorted(gts)
    score, scores = Cider().compute_score({k: gts[k] for k in keys}, {k: res[k] for k in keys})
    return float(score), np.asarray(scores, dtype=np.float64)


# SPICE scores scene graphs through the Nashorn JavaScript engine, which left the JDK in Java 15. On a newer Java it
# runs with the standalone Nashorn (and the ASM bytecode library it needs) on the class path; these are the Maven
# Central artifacts, fetched once into GREENVL_NASHORN_DIR (default ~/.greenvl/nashorn) and checked by SHA-256.
MAVEN = "https://repo1.maven.org/maven2/"
NASHORN_JARS = {
    "nashorn-core-15.4.jar": ("org/openjdk/nashorn/nashorn-core/15.4/",
                              "6f816e84dfd63a81d4eaa7829c08337bbaff3ec683ff3bf6bbd90d017a00dc6f"),
    "asm-9.7.jar": ("org/ow2/asm/asm/9.7/", "adf46d5e34940bdf148ecdd26a9ee8eea94496a72034ff7141066b3eea5c4e9d"),
    "asm-commons-9.7.jar": ("org/ow2/asm/asm-commons/9.7/",
                            "389bc247958e049fc9a0408d398c92c6d370c18035120395d4cba1d9d9304b7a"),
    "asm-tree-9.7.jar": ("org/ow2/asm/asm-tree/9.7/", "62f4b3bc436045c1acb5c3ba2d8ec556ec3369093d7f5d06c747eb04b56d52b1"),
    "asm-util-9.7.jar": ("org/ow2/asm/asm-util/9.7/", "37a6414d36641973f1af104937c95d6d921b2ddb4d612c66c5a9f2b13fc14211"),
    "asm-analysis-9.7.jar": ("org/ow2/asm/asm-analysis/9.7/",
                             "7bc6bcbc21379948a0c8c467fb0f864206e5b818f6bc0b546872f5c9f941556f"),
}


def java_major(java: str) -> int:
    import re
    import subprocess

    out = subprocess.run([java, "-version"], capture_output=True, text=True).stderr
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    if not m:
        raise RuntimeError(f"cannot read the version of {java}:\n{out}")
    return int(m.group(2)) if m.group(1) == "1" else int(m.group(1))


def nashorn_dir() -> Path:
    """Folder with NASHORN_JARS, downloading and verifying any that are missing."""
    import hashlib
    import os
    import urllib.request

    d = Path(os.environ.get("GREENVL_NASHORN_DIR", Path.home() / ".greenvl" / "nashorn"))
    d.mkdir(parents=True, exist_ok=True)
    for name, (folder, sha256) in NASHORN_JARS.items():
        f = d / name
        if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == sha256:
            continue
        data = urllib.request.urlopen(MAVEN + folder + name, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError(f"{name} from Maven Central does not match its expected SHA-256")
        tmp = f.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(f)
    return d


def spice_command() -> list[str]:
    """Java command line for SPICE (run from the pycocoevalcap spice folder). GREENVL_SPICE_JAVA selects the java
    executable; default the one on PATH. Java 8-14 run the jar as is; Java 15+ add Nashorn to the class path."""
    import os
    import shutil

    java = os.environ.get("GREENVL_SPICE_JAVA") or shutil.which("java")
    if not java:
        raise RuntimeError("SPICE needs Java (8 or newer) on PATH, or GREENVL_SPICE_JAVA; or pass --no-spice.")
    if java_major(java) <= 14:
        return [java, "-Xmx8G", "-jar", "spice-1.0.jar"]
    cp = os.pathsep.join(["spice-1.0.jar", "lib/*", str(nashorn_dir() / "*")])
    return [java, "-Xmx8G", "-cp", cp, "edu.anu.spice.SpiceScorer"]


def spice(gts: dict, res: dict):
    """(mean SPICE F-score, per-image F in sorted(gts) order), as pycocoevalcap's Spice (same jar, same -subset
    output) but without its parse cache, whose LMDB library has no Apple Silicon build and whose serialiser fails on
    Java 16+. The first call downloads Stanford CoreNLP (pycocoevalcap's get_stanford_models)."""
    import json
    import subprocess
    import tempfile

    import pycocoevalcap.spice as spice_pkg
    from pycocoevalcap.spice.get_stanford_models import get_stanford_models

    spice_dir = Path(spice_pkg.__file__).parent
    get_stanford_models()
    keys = sorted(gts)
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.json", Path(tmp) / "out.json"
        inp.write_text(json.dumps([{"image_id": k, "test": res[k][0], "refs": gts[k]} for k in keys]))
        subprocess.run([*spice_command(), str(inp), "-out", str(out), "-subset", "-silent"], cwd=spice_dir,
                       check=True)
        scores = {r["image_id"]: r["scores"]["All"]["f"] for r in json.loads(out.read_text())}

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    per = np.array([num(scores[k]) for k in keys], dtype=np.float64)
    return float(np.mean(per)), per


# ---------------------------------------------------------------- CLIPScore

class CLIPScorer:
    """CLIP-S(c, v) = w * max(cos(E_text('A photo depicts ' + c), E_image(v)), 0), w = 2.5 (Hessel et al. 2021).
    Image embeddings are cached by scripts/02_extract_features.py --encoder <scorer>."""

    def __init__(self, hf_id: str, device):
        import torch
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        from transformers.utils import logging as hf_logging

        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # the checkpoint's vision weights are unused here; skip that load report
        try:
            self.model = CLIPTextModelWithProjection.from_pretrained(hf_id).to(device).eval()
        finally:
            hf_logging.set_verbosity(level)

    def text_embeddings(self, texts: list[str], batch_size: int = 256):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                t = self.tokenizer([CLIPSCORE_PROMPT + x for x in texts[i: i + batch_size]], padding=True,
                                   truncation=True, max_length=77, return_tensors="pt").to(self.device)
                out.append(torch.nn.functional.normalize(self.model(**t).text_embeds.float(), dim=-1).cpu())
        return torch.cat(out)

    def score(self, image_embeds, texts: list[str]) -> np.ndarray:
        torch = self.torch
        img = torch.nn.functional.normalize(image_embeds.float(), dim=-1)
        cos = (img * self.text_embeddings(texts)).sum(-1)
        return (CLIPSCORE_W * cos.clamp(min=0)).numpy().astype(np.float64)


# ---------------------------------------------------------------- bootstrap over images

def bootstrap_ci(stat, n: int, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0, level: float = 0.95):
    """Percentile interval of stat(idx) over `samples` resamplings of n images with replacement."""
    rng = np.random.default_rng(seed)
    values = np.array([stat(rng.integers(0, n, n)) for _ in range(samples)])
    lo, hi = np.percentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)]


def mean_ci(per_image: np.ndarray, **kw):
    return bootstrap_ci(lambda i: per_image[i].mean(), len(per_image), **kw)


def ratio_ci(num: np.ndarray, den: np.ndarray, **kw):
    """Interval of sum(num)/sum(den) with images resampled (CHAIR_i, VQA accuracy grouped by image)."""
    return bootstrap_ci(lambda i: num[i].sum() / max(den[i].sum(), 1e-12), len(num), **kw)


def caption_scores(gts_raw: dict, res_raw: dict, spice_on: bool = True, clip_scorer=None, image_embeds=None,
                   ci: bool = True) -> dict:
    """All caption metrics for {image_id: [refs]} and {image_id: caption}. image_embeds: scorer embeddings in
    sorted(image_id) order."""
    keys = sorted(gts_raw)
    tok = ptb_tokenize({**{("g", k): v for k, v in gts_raw.items()}, **{("r", k): [res_raw[k]] for k in keys}})
    gts = {k: tok[("g", k)] for k in keys}
    res = {k: tok[("r", k)] for k in keys}

    out = {"images": len(keys)}
    stats = np.stack([bleu_stats(gts[k], res[k][0]) for k in keys])
    out["BLEU-4"] = {"value": bleu_from_stats(stats.sum(0))}
    c, c_per = cider(gts, res)
    out["CIDEr"] = {"value": c}
    if ci:
        out["BLEU-4"]["ci95"] = bootstrap_ci(lambda i: bleu_from_stats(stats[i].sum(0)), len(keys))
        out["CIDEr"]["ci95"] = mean_ci(c_per)
    per_image = {"CIDEr": c_per}
    if spice_on:
        s, s_per = spice(gts, res)
        out["SPICE"] = {"value": s, **({"ci95": mean_ci(s_per)} if ci else {})}
        per_image["SPICE"] = s_per
    if clip_scorer is not None:
        cs = clip_scorer.score(image_embeds, [res_raw[k] for k in keys])
        out["CLIPScore"] = {"value": float(cs.mean()), "scorer": getattr(clip_scorer, "name", None),
                            **({"ci95": mean_ci(cs)} if ci else {})}
        per_image["CLIPScore"] = cs
    lengths = np.array([len(res[k][0].split()) for k in keys], dtype=np.float64)
    out["mean_length_words"] = float(lengths.mean())
    return out, per_image, keys
