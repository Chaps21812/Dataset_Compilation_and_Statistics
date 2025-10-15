import os
import json
import numpy as np
from tqdm import tqdm
from astropy.io import fits

from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import FourierBasis
from PIL import Image
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA


# -------------------------------
# Utility Functions
# -------------------------------

def load_fits_from_json_entry(entry):
    """Loads a FITS image from 'original_path' in JSON entry."""
    fits_path = entry["original_path"]
    if not os.path.exists(fits_path):
        print(f"⚠️ Missing FITS file: {fits_path}")
        return None
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data


def extract_fpca1(stack, n_basis=5, n_fpca=3):
    """
    Compute first FPCA component image for a temporal stack of images.
    stack shape: (T, H, W)
    returns: (H, W) FPCA component 1 image
    """
    T, H, W = stack.shape
    time_points = np.linspace(0, 1, T)

    # Flatten (H, W, T) -> (N_pixels, T)
    pixel_series = stack.reshape(T, -1).T  # (N, T)

    # Smooth with Fourier basis
    basis = FourierBasis(domain_range=(0, 1), n_basis=n_basis)
    fd_grid = FDataGrid(data_matrix=pixel_series, grid_points=time_points)
    smoother = BasisSmoother(basis=basis)
    smoothed_fd = smoother.fit_transform(fd_grid)

    # FPCA decomposition
    fpca = FPCA(n_components=n_fpca)
    fpca.fit(smoothed_fd)
    fpca_scores = fpca.transform(smoothed_fd)  # shape: (N, n_fpca)

    # Extract first component
    fpca1_image = fpca_scores[:, 0].reshape(H, W)
    return fpca1_image


def normalize_to_png(arr):
    """Normalize float array to 0–255 uint8 for PNG output."""
    arr = np.nan_to_num(arr)
    arr_min, arr_max = np.min(arr), np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min)
    return (norm * 255).astype(np.uint8)


def bbox_overlap(b1, b2):
    """Check if two [x, y, w, h] boxes overlap."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def merge_overlapping_boxes(boxes):
    """
    Merge overlapping boxes into tight bounding boxes.
    boxes: list of [x, y, w, h]
    returns: merged list of [x, y, w, h]
    """
    if not boxes:
        return []

    merged = []
    boxes = [list(b) for b in boxes]

    while boxes:
        base = boxes.pop(0)
        x1, y1, w1, h1 = base
        x2, y2 = x1 + w1, y1 + h1
        merged_any = False
        for other in boxes[:]:
            if bbox_overlap(base, other):
                ox, oy, ow, oh = other
                ox2, oy2 = ox + ow, oy + oh
                # merge to tightest box
                nx1, ny1 = min(x1, ox), min(y1, oy)
                nx2, ny2 = max(x2, ox2), max(y2, oy2)
                base = [nx1, ny1, nx2 - nx1, ny2 - ny1]
                boxes.remove(other)
                merged_any = True
        merged.append(base)
        if merged_any:
            # recheck against all others
            boxes.append(base)
            merged.pop()
    return merged


# -------------------------------
# Main Dataset Creation Function
# -------------------------------

def create_fpca1_png_dataset(
    json_path,
    output_dir="fpca1_png_dataset",
    frames_per_stack=6,
    n_basis=5,
    n_fpca=3,
    n_embeddings=None
):
    """
    Converts COCO dataset with 'original_path' FITS files into FPCA(1) PNG dataset.
    Groups overlapping bounding boxes per temporal stack.
    """
    images_dir = os.path.join(output_dir, "images")
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    with open(json_path, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    # Group images by collect_id
    collects = {}
    for img in images:
        cid = img["collect_id"]
        collects.setdefault(cid, []).append(img)

    print(f"Found {len(collects)} collects total.")

    new_images, new_annotations = [], []
    image_id_counter = 1
    ann_id_counter = 1
    processed = 0

    for collect_id, img_list in tqdm(collects.items(), desc="Processing collects"):
        img_list = sorted(img_list, key=lambda x: x["exp_start_time"])

        # Load FITS frames
        frames = []
        for entry in img_list:
            data = load_fits_from_json_entry(entry)
            if data is not None:
                frames.append(data)

        if len(frames) < frames_per_stack:
            continue

        frames = np.stack(frames, axis=0)
        H, W = frames.shape[1:]

        num_stacks = len(frames) // frames_per_stack
        for s in range(num_stacks):
            stack = frames[s * frames_per_stack:(s + 1) * frames_per_stack]

            # --- FPCA(1) ---
            fpca1_img = extract_fpca1(stack, n_basis=n_basis, n_fpca=n_fpca)
            norm_png = normalize_to_png(fpca1_img)

            # --- Save image ---
            png_name = f"{collect_id}_{s:04d}.png"
            png_path = os.path.join(images_dir, png_name)
            Image.fromarray(norm_png).save(png_path)

            # --- Collect annotations belonging to these frames ---
            frame_ids = [im["id"] for im in img_list[s * frames_per_stack:(s + 1) * frames_per_stack]]
            frame_anns = [a for a in annotations if a["image_id"] in frame_ids]

            # --- Merge overlapping boxes ---
            boxes = [a["bbox"] for a in frame_anns]
            merged_boxes = merge_overlapping_boxes(boxes)

            # --- Add merged annotations ---
            for bbox in merged_boxes:
                new_annotations.append({
                    "id": ann_id_counter,
                    "image_id": image_id_counter,
                    "category_id": frame_anns[0]["category_id"] if frame_anns else 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                ann_id_counter += 1

            # --- Add new image entry ---
            new_images.append({
                "id": image_id_counter,
                "width": W,
                "height": H,
                "file_name": f"images/{png_name}",
                "collect_id": collect_id,
                "num_frames": frames_per_stack,
                "embedding": "FPCA(1)"
            })

            image_id_counter += 1

        processed += 1
        if n_embeddings is not None and processed >= n_embeddings:
            print(f"⚙️ Debug limit reached: {n_embeddings} collects processed.")
            break

    # --- Write new COCO JSON ---
    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }

    json_out = os.path.join(ann_dir, "fpca1_dataset.json")
    with open(json_out, "w") as f:
        json.dump(new_coco, f, indent=4)

    print(f"\n✅ Done! {len(new_images)} FPCA(1) PNGs created.")
    print(f"📁 Images:      {images_dir}")
    print(f"📁 Annotations: {json_out}")


# -------------------------------
# Script Entry Point
# -------------------------------

if __name__ == "__main__":
    create_fpca1_png_dataset(
        json_path="/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01_LargeTrainDemo/val/annotations/annotations.json",   # path to your source COCO JSON
        output_dir="/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/LMNT01_Large_FPCA1/val",          # output directory
        frames_per_stack=6,                      # number of temporal frames per embedding
        n_basis=5,                               # Fourier basis functions
        n_fpca=3,                                # FPCA components (we only use 1st)
        n_embeddings=None                           # debug limit (remove or None for full run)
    )