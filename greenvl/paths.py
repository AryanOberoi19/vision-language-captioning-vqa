"""Filesystem layout. Data lives next to this folder in ../Datasets unless GREENVL_DATA overrides it."""
import os
from pathlib import Path

IMPL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = IMPL_ROOT.parent

DATA_ROOT = Path(os.environ.get("GREENVL_DATA", PROJECT_ROOT / "Datasets"))
PROCESSED = DATA_ROOT / "processed"
FEATURES = DATA_ROOT / "features"
RESULTS = Path(os.environ.get("GREENVL_RESULTS", IMPL_ROOT / "results"))

COCO_IMAGES = DATA_ROOT / "coco" / "images"
FLICKR8K_IMAGES = DATA_ROOT / "flickr8k" / "Flicker8k_Dataset"
VIZWIZ_IMAGES = DATA_ROOT / "vizwiz" / "images" / "val"
