import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import wandb

from stack_dataset import StackDatasetEmbs


console = Console()

# ============================================================
# Pretty Logging
# ============================================================

def pretty_print_metrics(title: str, metrics: Dict[str, float]):
    table = Table(title=title)
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.6f}")
    console.print(table)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred, eps: float = 1e-8):
    y_true = np.array(y_true.detach().cpu(), dtype=np.float64)
    y_pred = np.array(y_pred.detach().cpu(), dtype=np.float64)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    denom_nae = np.mean(np.abs(y_true)) + eps
    nae = mae / denom_nae

    num = np.sum((y_true - y_pred) ** 2)
    denom_sre = np.sum(y_true ** 2) + eps
    sre = np.sqrt(num / denom_sre)

    smape = np.mean(2 * np.abs(y_pred - y_true) /
                    (np.abs(y_true) + np.abs(y_pred) + eps)) * 100

    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "NAE": nae, "SRE": sre, "SMAPE (%)": smape, "R2": r2}


def compute_metrics_val(model, val_loader, device):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            emb = batch["embedding"].to(device)
            gt = batch["volume_ratio"].to(device)

            pred = model(emb)
            all_preds.append(pred.cpu())
            all_trues.append(gt.cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_trues)

    return compute_metrics(y_true, y_pred)


# ============================================================
# Model
# ============================================================

class ConvRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)
        self.print_model_size()

    def forward(self, x):
        B, E, C = x.shape
        x = x.reshape(B, C, 37, 37)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.global_pool(x)
        x = x.view(B, -1)
        return self.fc(x).squeeze()

    def print_model_size(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Total number of weights: {total}")


# ============================================================
# Training
# ============================================================

# ============================================================
# Training
# ============================================================

if __name__ == "__main__":

    cfg = {
        "batch_size": 100,
        "epochs": 500,
        "lr": 1e-4,
        "emb_path": "/teamspace/studios/this_studio/stackcounting_dataset/vggt_embeddings/train",
        "data_path": "/teamspace/studios/this_studio/stackcounting_dataset/train",
        "patience": 30,         # <- early stopping patience
        "monitor": "MAE"        # <- metric to track
    }

    # Load dataset
    dataset = StackDatasetEmbs(
        data_dir=cfg["data_path"],
        emb_dir=cfg["emb_path"],
        use_cache=False,
        experiment_config={
            "INPUT_TYPE": "GT_DEPTH",
            "PREDICT_TYPE": "COUNT_DIRECTLY",
            "DINO_USE_VOL_AS_ADDITIONAL_INPUT": False,
        }
    )

    total_len = len(dataset)
    val_len = min(100, int(total_len * 0.05))
    train_len = total_len - val_len

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ConvRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.L1Loss()

    # ============================================================
    # Load checkpoint from previous run (if exists)
    # ============================================================

    checkpoint_path = "best_model_mae.pth"
    start_epoch = 0
    load_checkpoint = True

    if load_checkpoint and Path(checkpoint_path).exists():
        console.print(f"[bold green]Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt)

        # OPTIONAL: Restore optimizer state if you also saved it in the future
        # optimizer.load_state_dict(ckpt["optimizer"])

        # Reset early stopping counters properly
        best_metric = float("inf")      # or load from ckpt in future
        patience_counter = 0
        console.print("[bold green]Checkpoint loaded. Training will resume.")
    else:
        console.print("[bold yellow]No checkpoint found. Training from scratch.")


    best_metric = float("inf")
    best_epoch = -1
    patience_counter = 0   # <- early stopping counter

    # ============================================================
    # Main Training Loop
    # ============================================================

    with wandb.init(project="ML_proj2", name="VGGT_finetune"):

        for epoch in range(cfg["epochs"]):

            console.rule(f"[bold green]Epoch {epoch+1}/{cfg['epochs']}")
            model.train()

            pbar = tqdm(train_loader, desc="Training", leave=True)
            for step, batch in enumerate(pbar):
                emb = batch["embedding"].to(device)
                gt = batch["volume_ratio"].to(device)

                pred = model(emb)
                loss = loss_fn(pred, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": float(loss.detach())})
                wandb.log({"train_loss": float(loss.detach())})

                # ---- VALIDATION ----
                if step % 10 == 0 and step > 0:
                    console.print("\n[bold yellow]Running validation...")
                    val_metrics = compute_metrics_val(model, val_loader, device)
                    pretty_print_metrics("Validation Metrics", val_metrics)
                    wandb.log({"metrics_val": val_metrics})

                    current = val_metrics[cfg["monitor"]]

                    # =====================================================
                    #   EARLY STOPPING
                    # =====================================================
                    if current < best_metric:
                        best_metric = current
                        best_epoch = epoch
                        patience_counter = 0

                        torch.save(model.state_dict(), "best_model_mae2.pth")
                        console.print(f"[bold green]New best {cfg['monitor']}: {best_metric:.6f}. Model saved.")
                    else:
                        patience_counter += 1
                        console.print(
                            f"[red]No improvement. Early stopping patience {patience_counter}/{cfg['patience']}"
                        )

                        if patience_counter >= cfg["patience"]:
                            console.print(
                                f"[bold red]Early stopping triggered at epoch {epoch+1}. "
                                f"Best epoch was {best_epoch+1}."
                            )
                            raise SystemExit

    console.print("[bold magenta]Training complete.")
