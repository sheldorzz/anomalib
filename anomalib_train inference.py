#!/usr/bin/env python3
# infer_patchcore.py

import torch
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as T
from anomalib.models import Patchcore

# ─── Configuration (all hard-coded) ────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/patchcore-best.ckpt"
INPUT_IMAGE = "inference/room.jpg"
OUTPUT_MASK = "outputs/room_mask.png"
IMAGE_SIZE = 384
THRESHOLD = 0.5

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1) load the trained model
    model = Patchcore.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()

    # 2) open and preprocess the image
    img = Image.open(INPUT_IMAGE).convert("RGB")
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=model.normalization.mean, std=model.normalization.std),
    ])
    tensor = transform(img).unsqueeze(0)  # shape = (1,3,H,W)

    # 3) forward pass to get per-pixel anomaly map
    with torch.no_grad():
        outputs = model(tensor)
    anomaly_map = outputs["anomaly_map"][0].cpu().numpy()

    # 4) threshold to binary mask
    binary_mask = (anomaly_map >= THRESHOLD).astype(np.uint8) * 255

    # 5) save the mask
    Path(OUTPUT_MASK).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_mask).save(OUTPUT_MASK)
    print(f"\n✅ Binary mask saved to:\n  {OUTPUT_MASK}")


if __name__ == "__main__":
    main()
