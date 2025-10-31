import os
from PIL import Image
from pycocotools.coco import COCO
from src.coco_data_loader import Coco16bitGray



# Load COCO annotations
import torch
# Path to your dataset
data_folder = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025-Annotations/2025-10-13"  # must contain images/ and annotations/annotations.json

# Instantiate the dataset
dataset = Coco16bitGray(data_folder)

print(f"Dataset length: {len(dataset)}")

# Iterate over the first few samples
for i in range(min(5, len(dataset))):
    img_tensor, target = dataset[i]

    print(f"\nSample {i}:")
    print("Image tensor shape:", img_tensor.shape)
    print("Image dtype:", img_tensor.dtype)
    print("Min/Max:", img_tensor.min().item(), img_tensor.max().item())
    print("Number of annotations:", len(target))
    
    # Optional: check that the tensor is in [0,1]
    if img_tensor.min() < 0 or img_tensor.max() > 1:
        print("Warning: image tensor values out of range [0,1]")
    
    # Optional: break early for a quick test
    if i == 4:
        break

# Optional: test DataLoader integration
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_idx, (images, targets) in enumerate(loader):
    print(f"\nBatch {batch_idx}:")
    print("Images shape:", images.shape)  # [B, 1, H, W]
    print("Images dtype:", images.dtype)
    print("Number of targets in batch:", len(targets))
    if batch_idx == 1:  # stop after a couple of batches
        break