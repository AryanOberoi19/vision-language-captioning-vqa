"""CHAIR object hallucination (Rohrbach et al. 2018), ported from the authors' utils/chair.py.

Same synonym list (Datasets/chair/synonyms.txt), double-word rules, toilet/seat rule and ground truth: an image's
objects are its COCO instance-segmentation categories plus the objects its reference captions mention.
Differences, neither of which changes which words count as objects in practice: tokenisation is a regular expression
over lower-cased letters instead of nltk.word_tokenize, and plurals are singularised with `inflect` instead of the
unmaintained `pattern` package (words that are already object names are left as they are, so 'bus' stays 'bus').

  CHAIR_i = hallucinated object mentions / object mentions        (primary, methodology §7)
  CHAIR_s = captions with at least one hallucinated object / captions
"""
import json
import re
from pathlib import Path

COCO_DOUBLE_WORDS = ["motor bike", "motor cycle", "air plane", "traffic light", "street light", "traffic signal",
                     "stop light", "fire hydrant", "stop sign", "parking meter", "suit case", "sports ball",
                     "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
                     "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie", "laptop computer",
                     "stove top oven", "hot dog", "teddy bear", "home plate", "train track"]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]
WORD = re.compile(r"[a-z]+")


class Chair:
    def __init__(self, synonyms_path: Path):
        import inflect

        self._inflect = inflect.engine()
        synonyms = [s.strip().split(", ") for s in Path(synonyms_path).read_text().splitlines() if s.strip()]
        self.mscoco_objects, self.inverse_synonym = set(), {}
        for syn in synonyms:
            self.mscoco_objects.update(syn)
            for s in syn:
                self.inverse_synonym[s] = syn[0]
        self.double_words = {w: w for w in COCO_DOUBLE_WORDS}
        for a in ANIMAL_WORDS:
            self.double_words[f"baby {a}"] = a
            self.double_words[f"adult {a}"] = a
        for v in VEHICLE_WORDS:
            self.double_words[f"passenger {v}"] = v
        self.double_words["bow tie"] = "tie"
        self.double_words["toilet seat"] = "toilet"
        self.double_words["wine glas"] = "wine glass"
        self._singular = {}

    def singular(self, w: str) -> str:
        if w in self.mscoco_objects:
            return w
        if w not in self._singular:
            s = self._inflect.singular_noun(w)
            self._singular[w] = s if s else w
        return self._singular[w]

    def caption_objects(self, caption: str):
        """(words, node_words): MSCOCO object mentions in a caption and the category each maps to."""
        words = [self.singular(w) for w in WORD.findall(caption.lower())]
        merged, i = [], 0
        while i < len(words):
            pair = " ".join(words[i: i + 2])
            if pair in self.double_words:
                merged.append(self.double_words[pair])
                i += 2
            else:
                merged.append(words[i])
                i += 1
        if "toilet" in merged and "seat" in merged:
            merged = [w for w in merged if w != "seat"]
        found = [w for w in merged if w in self.mscoco_objects]
        return found, [self.inverse_synonym[w] for w in found]

    def ground_truth(self, image_ids, instances_json: Path, captions_json: Path) -> dict:
        """{image_id: set of categories} from segmentation labels and reference captions."""
        ids = set(image_ids)
        gt = {i: set() for i in ids}
        inst = json.loads(Path(instances_json).read_text())
        names = {c["id"]: c["name"] for c in inst["categories"]}
        for a in inst["annotations"]:
            if a["image_id"] in ids:
                gt[a["image_id"]].add(self.inverse_synonym[names[a["category_id"]]])
        for a in json.loads(Path(captions_json).read_text())["annotations"]:
            if a["image_id"] in ids:
                gt[a["image_id"]].update(self.caption_objects(a["caption"])[1])
        return gt

    def score(self, captions: dict, gt: dict) -> dict:
        """captions {image_id: caption}; per-caption detail plus totals."""
        per = {}
        for imid, cap in captions.items():
            words, nodes = self.caption_objects(cap)
            halluc = [(w, n) for w, n in zip(words, nodes) if n not in gt[imid]]
            per[imid] = {"mentions": len(nodes), "hallucinated": len(halluc), "hallucinated_words": halluc}
        mentions = sum(p["mentions"] for p in per.values())
        hall = sum(p["hallucinated"] for p in per.values())
        return {"CHAIR_i": hall / max(mentions, 1),
                "CHAIR_s": sum(p["hallucinated"] > 0 for p in per.values()) / max(len(per), 1),
                "object_mentions": mentions, "per_caption": per}
