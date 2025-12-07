# Copyright (c) Meta Platforms, Inc. and affiliates.
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

def load_vggt_model(device: str = None, dtype: torch.dtype = None) -> VGGT:
    """
    Load the pretrained VGGT model.
    
    Args:
        device: Device to load model on. Defaults to CUDA if available.
        dtype: Data type for inference. Defaults to bfloat16 on Ampere+ GPUs.
        
    Returns:
        VGGT model loaded with pretrained weights.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if dtype is None and device == "cuda":
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    elif dtype is None:
        dtype = torch.float32
        
    print(f"Loading VGGT model on {device} with dtype {dtype}")
    
    # Load pretrained model from HuggingFace
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    
    return model, device, dtype

class ConvRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        super(ConvRegressor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        ).to(device)
        self.global_pool = nn.AdaptiveAvgPool2d(1).to(device)  # Reduces to (B, 256, 1, 1)
        self.fc = nn.Linear(256, 1).to(device)

        self.print_model_size()

    def forward(self, x):
        # x: (B, C, H, W)
        #assert x.shape[-2] == image_size and x.shape[-1] == image_size 

        x = self.conv_layers(x)
        x = self.global_pool(x).view(x.size(0), -1)  # Flatten to (B, 256)
        return self.fc(x).squeeze()
    
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of weights: {total_params}")


class VGGTWrapper(nn.Module):
    def __init__(self, model: VGGT, device: str, dtype: torch.dtype):
        super().__init__()
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.dtype = dtype

        self.linear = nn.Sequential(
            nn.Linear(2048, 512).to(device),
            nn.ReLU(),
            nn.Linear(512, 1).to(device),
        ).to(device)

    def forward(self, images: torch.Tensor): 
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        
        images = images.to(self.device)
        B, S, C, H, W = images.shape
        
        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        pred_x_patch = self.linear(
            aggregated_tokens_list[-1][:, :, patch_start_idx:, :]).squeeze(-1)  # [B, S, num_patches, 1]

        pred_x_patch = pred_x_patch.sum(dim=-1)  # [B, S, num_patches]
        pred_x_patch = pred_x_patch.mean(dim=1)  # [B, num_patches]
        return pred_x_patch


class VGGTWrapperCNN(nn.Module):
    def __init__(self, model: VGGT, device: str, dtype: torch.dtype):
        super().__init__()
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.dtype = dtype

        # self.cnn = ConvRegressor().to(device)

    def forward(self, images: torch.Tensor): 
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        
        images = images.to(self.device)
        B, S, C, H, W = images.shape
        
        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        
        map_ = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]  # [B, S, num_patches, embed_dim]
        B, S, num_patches, embed_dim = map_.shape
        agg = map_.view(B, S, embed_dim, int((num_patches)**0.5), int((num_patches)**0.5))  # [B*S, embed_dim, H', W']
        agg = agg.sum(dim=1)  # [B, embed_dim, H', W']

        pred = self.cnn(agg)
        return pred
    
    def return_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        
        images = images.to(self.device)
        B, S, C, H, W = images.shape
        
        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        
        map_ = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]  # [B, S, num_patches, embed_dim]

        return map_  # Return aggregated feature map



if __name__ == "__main__":
    
    # ==========================================
    # Configuration
    # ==========================================
    
    # if there is an argument passed, use it as data path
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "/teamspace/studios/this_studio/stackcounting_dataset/train"  # Path to your dataset

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

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=8)

    # Load model
    model, device, dtype = load_vggt_model()
    model = VGGTWrapperCNN(model, device, dtype)
    model.eval()

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
            images = batch["images"].to(device)

            embeddings = model.return_embeddings(images)
            embeddings_batch.append(embeddings.cpu())

            if len(embeddings_batch) >= 16:  # Save when batch is full
                print("saving batch")
                for b in range(len(embeddings_batch)):
                    os.makedirs(out_dir / folder[b], exist_ok=True)
                    torch.save(embeddings_batch[b], out_dir / f"{folder[b]}/{image_name[b]}.pt")
                embeddings_batch = []  # Reset after saving

            if (i+1) % 15 == 0: 
                print("preparing batch")