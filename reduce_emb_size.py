import os
import torch
from pathlib import Path
import shutil
# Script to load VGGT model and extract embeddings from 3 frames
# for building custom architectures on top of VGGT

import os
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import wandb

from stack_dataset import StackDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import gzip

# Path to the directory where the incorrect tensors are saved
import sys
if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = "/Users/maria/ML/data/scenes_part1"  # Path to your dataset

cfg = {
    "batch_size": 4,
    "data_path": data_path,  # Path to your dataset
}

train_dataset = StackDataset(
    data_dir=cfg["data_path"],
    use_cache=False,
    transform=transforms.Compose([
        # transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ]),
    experiment_config={
        "INPUT_TYPE": "GT_DEPTH",
        "PREDICT_TYPE": "GAMMA_WITH_EDGES",
        "DINO_USE_VOL_AS_ADDITIONAL_INPUT": False
        # Add other config parameters as needed
    }
)

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=False)

# Load model
# ==========================================
# Inference and save embeddings
# ==========================================

out_dir = Path(os.path.dirname(cfg["data_path"])) / "vggt_embeddings" / os.path.basename(cfg["data_path"])
out_dir.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    # Save embeddings in batches
    embeddings_batch = []
    for i, batch in enumerate(tqdm(train_loader)):
        folder = batch["folder"]
        image_name = batch["image_name"]
        try: 
            # print(folder)
            tensor = torch.load(out_dir / f"{folder[0]}/{image_name[0]}.pt")

            for i,fold in enumerate(tqdm(folder)): 
                os.makedirs(out_dir / f"{fold}", exist_ok=True)
                # with gzip.open(out_dir / f"{fold}/{image_name[0]}.pt.gz", 'wb') as f:
                torch.save(tensor[i].mean(0), out_dir / f"{fold}/{image_name[0]}_mean.pt")
            # print(tensor.mean(1).shape)

            os.remove(out_dir / f"{folder[0]}/{image_name[0]}.pt")
        except: 
            print("error", folder)