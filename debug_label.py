import os
import json
from glob import glob

from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
OUT_DIR = os.path.join(BASE_DIR, "debug_labels")

os.makedirs(OUT_DIR, exist_ok=True)

for label_path in glob(os.path.join(LABELS_DIR, "*.json")):
    with open(label_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    img_path = os.path.join(IMAGES_DIR, meta["image_filename"])
    if not os.path.exists(img_path):
        print("Missing image for", label_path)
        continue

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x = meta["tap_x"]
    y = meta["tap_y"]
    r = 10
    draw.ellipse((x - r, y - r, x + r, y + r), fill="red")

    out_name = os.path.basename(label_path).replace(".json", "_debug.png")
    out_path = os.path.join(OUT_DIR, out_name)
    img.save(out_path)
    print("Wrote", out_path)
