#!/usr/bin/env python3
# train_patchcore.py

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from anomalib.data import Folder
from anomalib.models import Patchcore

# ─── Configuration (all hard-coded) ────────────────────────────────────────────
DATA_DIR = "data/train_normals"         # folder of only-good images
OUTPUT_DIR = "checkpoints"             # where to save the best checkpoint
IMAGE_SIZE = 384                       # resize side length for all images
BACKBONE = "wide_resnet50_2"           # frozen, ImageNet-pretrained backbone
LAYERS = ["layer2", "layer3"]          # which feature maps to store
PRETRAINED = True                      # use ImageNet weights
SAMPLING_RATIO = 1.0                   # keep every patch (few images)
N_NEIGHBORS = 3                        # number of nearest neighbors for scoring
MAX_EPOCHS = 1                         # one pass builds the PatchCore memory bank
PRECISION = 16                         # mixed-precision if available

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1) prepare only-normal datamodule
    datamodule = Folder(
        root=DATA_DIR,
        split_ratio={"train": 1.0, "val": 0.0, "test": 0.0},
        image_size=IMAGE_SIZE,
        task="segmentation",
    )

    # 2) build PatchCore model
    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pretrained=PRETRAINED,
        sampling_ratio=SAMPLING_RATIO,
        n_nearest_neighbors=N_NEIGHBORS,
    )

    # 3) checkpointing callback (monitors image-level AUROC on val set)
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="patchcore-best",
        monitor="val_image_auc",
        mode="max",
        save_top_k=1,
    )

    # 4) trainer — one epoch is enough to index all patches
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        precision=PRECISION,
        callbacks=[checkpoint_callback],
    )

    # 5) fit (this builds the memory bank)
    trainer.fit(model, datamodule=datamodule)

    # 6) report
    best_ckpt = checkpoint_callback.best_model_path
    print(f"\n✅ Training complete. Best checkpoint saved at:\n  {best_ckpt}")


if __name__ == "__main__":
    main()
