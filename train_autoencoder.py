# Copyright (c) Meta Platforms, Inc. and affiliates.
# Script to load VGGT model and extract embeddings from 3 frames
# for building custom architectures on top of VGGT

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None

try:
    from stack_dataset import StackDataset, StackDatasetEmbs
except ImportError:
    StackDataset = None
    StackDatasetEmbs = None

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder

class ViewAggregatorEncoder(nn.Module):
    def __init__(self, d_model=2048, nhead=8, num_layers=6, dim_feedforward=4096):
        super().__init__()
        
        # Learnable CLS token (single vector, will be tiled)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) 
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # src shape: [V, 900, 2048]
        V, S, E = src.shape
        
        # 1. Tile CLS token and prepend to each sequence
        # cls_token_tiled shape: [V, 1, 2048]
        cls_token_tiled = self.cls_token.repeat(V, 1, 1)
        
        # input_with_cls shape: [V, 901, 2048]
        input_with_cls = torch.cat([cls_token_tiled, src], dim=1)
        
        # 2. Process V sequences independently through the encoder
        # Output shape: [V, 901, 2048]
        encoder_output = self.encoder(input_with_cls)
        
        # 3. Extract the CLS token output for each view
        # cls_output_per_view shape: [V, 1, 2048]
        cls_output_per_view = encoder_output[:, 0:1, :]
        
        # 4. Aggregate the V CLS tokens into a single sequence (using mean)
        # aggregated_output shape: [1, 1, 2048]
        aggregated_output = cls_output_per_view.mean(dim=0, keepdim=True)
        
        # 5. Expand the aggregated token to the required sequence length [1, 900, 2048]
        # This step is critical to match the decoder's expected memory shape [1, S, E]
        # The aggregation is now represented by a single token, which must be tiled/expanded 
        # to the sequence length (S=900) for the decoder's cross-attention Key/Value vectors.
        target_seq_len = S 
        final_memory = aggregated_output.repeat(1, target_seq_len, 1)

        # Output shape: [1, 900, 2048]
        return final_memory

from torch.nn import TransformerDecoderLayer, TransformerDecoder

class ViewReconstructorDecoder(nn.Module):
    def __init__(self, d_model=2048, nhead=8, num_layers=6, dim_feedforward=4096):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final linear layer to project back to the output space if needed
        self.linear_out = nn.Linear(d_model, d_model) 

    def forward(self, memory, tgt, V):
        # memory shape: [1, 900, 2048] (Aggregated output from Encoder)
        # tgt shape: [V, 900, 2048] (Initial target sequence / View embeddings)
        
        # 1. Expansion: Repeat the memory V times
        # Expanded memory shape: [V, 900, 2048]
        expanded_memory = memory.repeat(V, 1, 1) 
        
        # 2. Pass expanded memory and target through the decoder
        # The decoder's cross-attention will now operate on V sequences
        # Output shape: [V, 900, 2048]
        decoder_output = self.decoder(tgt, expanded_memory)
        
        # 3. Final projection/reconstruction
        output = self.linear_out(decoder_output)
        
        return output

class ViewTransformer(nn.Module):
    # ... (init function is the same, using the new encoder)
    def __init__(self, d_model=2048, nhead=8, num_layers=6, dim_feedforward=4096):
        super().__init__()
        # Use the CLS-based encoder
        self.encoder = ViewAggregatorEncoder(d_model, nhead, num_layers, dim_feedforward)
        self.decoder = ViewReconstructorDecoder(d_model, nhead, num_layers, dim_feedforward)
        self.initial_target_embedding = nn.Parameter(torch.randn(1, d_model)) 
        
    def forward(self, src):
        V, S, E = src.shape
        
        # 1. Encode, Aggregate via CLS token, and Expand
        # aggregated_memory shape: [1, 900, 2048]
        aggregated_memory = self.encoder(src)
        
        # 2. Prepare Decoder Target Input (tgt)
        # tgt shape: [V, 900, 2048]
        tgt = self.initial_target_embedding.unsqueeze(0).repeat(V, S, 1)
        
        # 3. Decode and Reconstruct
        # output shape: [V, 900, 2048]
        output = self.decoder(aggregated_memory, tgt, V)
        
        return aggregated_memory, output
        

class ConvRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        super(ConvRegressor, self).__init__()
        self.down = nn.Sequential(
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

        x = self.down(x)
        count = self.global_pool(x).view(x.size(0), -1)  # Flatten to (B, 256)

        return self.fc(count).squeeze()
    
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of weights: {total_params}")


if __name__ == "__main__":
    
    # ==========================================
    # Configuration
    # ==========================================
    
    cfg = {
        "batch_size": 2,
        "epochs": 5,
        "lr": 1e-4,
        "data_path": "/Users/maria/ML/data/scenes_part1/",  # Path to your dataset
        "m1": 0.5,
        "m2": 0.1
    }

    train_dataset = StackDatasetEmbs(
        data_dir=cfg["data_path"],
        use_cache=False,
        experiment_config={
            "INPUT_TYPE": "GT_DEPTH",
            "PREDICT_TYPE": "GAMMA_WITH_EDGES",
            "DINO_USE_VOL_AS_ADDITIONAL_INPUT": False
            # Add other config parameters as needed
        }
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # Load model

    model = ViewTransformer()
    model.to(device)    
    model.train()

    loss_fn = nn.L1Loss()
    loss_autoencoder_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ==========================================
    # Training
    # ==========================================
    
    with wandb.init(project="ML_proj2", name="VGGT_finetune"):
        for epoch in range(cfg["epochs"]):
            for i, batch in enumerate(train_loader):
                embeddings = batch["embedding"].to(device)
                gt_count = batch["volume_ratio"].to(device)

                aggregated_memory, out_autoencoder = model(embeddings)

                loss_autoencoder = loss_autoencoder_fn(out_autoencoder, embeddings)

                if i % 1 == 0:
                    print(f"Epoch {epoch}, Step {i}/{len(train_loader)}, Loss autoencoder: {loss_autoencoder.item()}")
                    wandb.log({"loss_autoencoder": loss_autoencoder.item()})
                
                optimizer.zero_grad()
                loss_autoencoder.backward()
                optimizer.step()
    
                if i % 50 == 0:
                    torch.save(model.state_dict(), f"vggt_finetune_epoch.pth")