import os
import json
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# === CONFIG ===
COCO_PATH = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01_Large_FPCA1/test/annotations/annotations.json"
IMAGE_DIR = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01_Large_FPCA1/test/images"
OUTPUT_DIR = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01_Large_FPCA1/test/annotated_preview"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD COCO JSON ===
with open(COCO_PATH, "r") as f:
    coco = json.load(f)

images_info = {img["id"]: img for img in coco.get("images", [])}
annotations = coco.get("annotations", [])

# === GROUP ANNOTATIONS BY IMAGE ID ===
anns_by_image = defaultdict(list)
for ann in annotations:
    anns_by_image[ann["image_id"]].append(ann)

# Choose a single color (e.g. red) or a palette if you want variety
BOX_COLOR = (255, 0, 0)   # pure red
BOX_WIDTH = 3

# === DRAWING LOOP ===
for img_id, img_info in tqdm(images_info.items(), desc="Annotating images"):
    anns = anns_by_image.get(img_id)
    if not anns:
        continue  # no boxes for this image

    img_path = os.path.join(IMAGE_DIR, os.path.basename(img_info["file_name"]))
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {img_path}") #👀
        continue

    # Load image and convert to RGB for drawing
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw all bounding boxes
    for ann in anns:
        x, y, w, h = ann["bbox"]
        xy = [
            int(round(x)),
            int(round(y)),
            int(round(x + w)),
            int(round(y + h))
        ]
        draw.rectangle(xy, outline=BOX_COLOR, width=BOX_WIDTH)

    # Save annotated image
    out_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0] + "_annotated.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    img.save(out_path)

print(f"\n✅ Done! Annotated images saved to: {OUTPUT_DIR}")