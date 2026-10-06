#!/usr/bin/env python3
"""Pack what the Colab T4 measurement needs into one zip (run on the Mac; ~330 MB, about a minute).

    python colab/make_bundle.py     # -> DL_NLP Project/colab/greenvl_t4_bundle.zip and a copy of colab_t4.ipynb

Contents: the code (greenvl/, scripts/, requirements); the trained run's config.json and epoch_02.pt; the step-7
subsets (500 captions + 500 questions, and the 8-item smoke-test subset); the 950 distinct COCO images they use; and the
Mac's batch-64 test outputs of every configuration, so the T4 run can report how often its outputs match.
CLIP and GPT-2 are downloaded from the Hugging Face Hub on Colab. Nothing else is needed: 10_measure_variants.py
reads only these files.
"""
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl import paths  # noqa: E402
from greenvl.measure import subset  # noqa: E402

RUN, CHECKPOINT = "coco_ViT-L-14_lora8_lr1e-3_s0", "epoch_02.pt"
OUT = paths.PROJECT_ROOT / "colab" / "greenvl_t4_bundle.zip"


def main():
    impl = paths.IMPL_ROOT
    files = {}  # path inside the zip -> path on disk
    for p in sorted([*impl.glob("greenvl/**/*.py"), *impl.glob("greenvl/third_party/**/*"), *impl.glob("scripts/*.py"),
                     impl / "requirements.txt", impl / "requirements.lock.txt"]):
        if p.is_file() and "__pycache__" not in p.parts:
            files[f"Implementation/{p.relative_to(impl)}"] = p
    run = paths.RESULTS / "runs" / RUN
    for name in ("config.json", CHECKPOINT):
        files[f"Implementation/results/runs/{RUN}/{name}"] = run / name
    images = []
    for n in (500, 8):  # creates subset_test_8.json if missing (the smoke test's items)
        sub = subset("test", n)
        f = paths.RESULTS / "inference" / f"subset_test_{n}.json"
        files[f"Implementation/results/inference/{f.name}"] = f
        images += [it["file"] for task in ("caption", "vqa") for it in sub[task]]
    for rel in sorted(set(images)):
        files[f"Datasets/coco/images/{rel}"] = paths.COCO_IMAGES / rel
    ev = paths.RESULTS / "eval" / RUN
    for p in sorted([*ev.glob(f"{Path(CHECKPOINT).stem}*_test_*_captions.json"),
                     *ev.glob(f"{Path(CHECKPOINT).stem}*_test_answers.json")]):
        files[f"Implementation/results/eval/{RUN}/{p.name}"] = p
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        sys.exit("missing:\n" + "\n".join(missing))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w") as z:
        for arc, p in files.items():
            z.write(p, arc, compress_type=zipfile.ZIP_STORED if p.suffix in (".jpg", ".pt") else zipfile.ZIP_DEFLATED)
    tmp.replace(OUT)
    shutil.copy(Path(__file__).with_name("colab_t4.ipynb"), OUT.parent / "colab_t4.ipynb")  # both files to upload
    n_img = len(set(images))
    print(f"{OUT}: {OUT.stat().st_size / 2**20:.0f} MiB, {len(files)} files ({n_img} images)")


if __name__ == "__main__":
    main()
