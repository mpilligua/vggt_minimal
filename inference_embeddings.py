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


def extract_embeddings(
    model: VGGT,
    images: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    extract_all_layers: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Extract embeddings and predictions from VGGT model.
    
    Args:
        model: Loaded VGGT model.
        images: Preprocessed images tensor [S, 3, H, W] or [B, S, 3, H, W].
        device: Device to run inference on.
        dtype: Data type for inference.
        extract_all_layers: If True, return aggregated tokens from all transformer layers.
        
    Returns:
        Dictionary containing:
        - aggregated_tokens_list: List of token tensors from all transformer layers
        - patch_start_idx: Index where patch tokens start (after camera/register tokens)
        - pose_enc: Camera pose encoding [B, S, 9]
        - extrinsic: Extrinsic camera matrices [B, S, 3, 4]
        - intrinsic: Intrinsic camera matrices [B, S, 3, 3]
        - depth: Depth maps [B, S, 1, H, W]
        - depth_conf: Depth confidence [B, S, 1, H, W]
        - world_points: 3D point maps [B, S, 3, H, W]
        - world_points_conf: Point map confidence [B, S, 1, H, W]
        - images: Original input images
    """
    # Add batch dimension if needed
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    
    images = images.to(device)
    B, S, C, H, W = images.shape
    
    print(f"Processing {S} frames with shape {H}x{W}")
    
    results = {}
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else torch.inference_mode():
            # Step 1: Get aggregated tokens from the transformer backbone
            # This is the core feature extraction step
            aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            
            results["aggregated_tokens_list"] = aggregated_tokens_list
            results["patch_start_idx"] = patch_start_idx
            
            # aggregated_tokens_list contains tokens from all 24 transformer layers
            # Each tensor has shape [B, S, num_tokens, embed_dim]
            # where num_tokens = 1 (camera) + 4 (register) + H/14 * W/14 (patches)
            # embed_dim = 2048 (concatenated from two streams)
            
            print(f"Number of transformer layers: {len(aggregated_tokens_list)}")
            print(f"Token tensor shape per layer: {aggregated_tokens_list[0].shape}")
            # print(f"Patch tokens start at index: {patch_start_idx}")
            
            # # Step 2: Extract camera pose encoding
            # if model.camera_head is not None:
            #     pose_enc_list = model.camera_head(aggregated_tokens_list)
            #     pose_enc = pose_enc_list[-1]  # Use last iteration
            #     results["pose_enc"] = pose_enc
            #     results["pose_enc_list"] = pose_enc_list
                
            #     # Convert pose encoding to extrinsic/intrinsic matrices
            #     extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            #     results["extrinsic"] = extrinsic  # [B, S, 3, 4] camera-from-world
            #     results["intrinsic"] = intrinsic  # [B, S, 3, 3]
                
            #     print(f"Camera pose encoding shape: {pose_enc.shape}")
            #     print(f"Extrinsic matrix shape: {extrinsic.shape}")
            #     print(f"Intrinsic matrix shape: {intrinsic.shape}")
            
            # Step 3: Extract depth maps
            # if model.depth_head is not None:
            #     depth, depth_conf = model.depth_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            #     )
            #     results["depth"] = depth
            #     results["depth_conf"] = depth_conf
            #     print(f"Depth map shape: {depth.shape}")
            
            # # Step 4: Extract 3D point maps
            # if model.point_head is not None:
            #     world_points, world_points_conf = model.point_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            #     )
            #     results["world_points"] = world_points
            #     results["world_points_conf"] = world_points_conf
            #     print(f"World points shape: {world_points.shape}")
            
            # Store original images
            results["images"] = images
            
    return results


def get_patch_features(
    aggregated_tokens_list: List[torch.Tensor],
    patch_start_idx: int,
    layer_indices: List[int] = None,
) -> torch.Tensor:
    """
    Extract patch features from specific transformer layers.
    Useful for building custom heads on top of VGGT features.
    
    Args:
        aggregated_tokens_list: List of token tensors from transformer layers.
        patch_start_idx: Index where patch tokens start.
        layer_indices: Which layers to extract features from. 
                      Default uses DPT layers [4, 11, 17, 23].
    
    Returns:
        Tensor of patch features [B, S, num_layers, num_patches, embed_dim]
    """
    if layer_indices is None:
        # Default: use same layers as DPT head
        layer_indices = [4, 11, 17, 23]
    
    features = []
    for idx in layer_indices:
        # Extract only patch tokens (skip camera and register tokens)
        patch_tokens = aggregated_tokens_list[idx][:, :, patch_start_idx:]
        features.append(patch_tokens)
    
    # Stack along new dimension
    features = torch.stack(features, dim=2)  # [B, S, num_layers, num_patches, embed_dim]
    
    return features


def get_camera_tokens(
    aggregated_tokens_list: List[torch.Tensor],
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    Extract camera tokens from a specific layer.
    Camera token is the first token in each frame.
    
    Args:
        aggregated_tokens_list: List of token tensors from transformer layers.
        layer_idx: Which layer to extract from. Default: last layer.
        
    Returns:
        Camera tokens [B, S, embed_dim]
    """
    tokens = aggregated_tokens_list[layer_idx]
    camera_tokens = tokens[:, :, 0]  # First token is camera token
    return camera_tokens

 


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
        "batch_size": 1,
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

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    # Load model
    model, device, dtype = load_vggt_model()
    model = VGGTWrapperCNN(model, device, dtype)
    model.eval()

    # ==========================================
    # Inference and save embeddings
    # ==========================================
    
    out_dir = Path(os.path.dirname(cfg["data_path"])) / "vggt_embeddings" / os.path.basename(cfg["data_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(tqdm(train_loader)):
        folder = batch[2]
        image_name = batch[3]
        images = batch[0].to(device)

        embeddings = model.return_embeddings(images)
        if cfg["batch_size"] > 1:
            for b in range(embeddings.shape[0]):
                os.makedirs(out_dir / folder[b], exist_ok=True)
                torch.save(embeddings[b].cpu(), out_dir / f"{folder[b]}/{image_name[b]}.pt")
        else:
            os.makedirs(out_dir / folder[0], exist_ok=True)
            torch.save(embeddings.cpu(), out_dir / f"{folder[0]}/{image_name[0]}.pt")