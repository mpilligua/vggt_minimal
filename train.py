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
from PIL import Image, ImageDraw, ImageFont
import math
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import wandb

# Function to log histograms of the ground truth and predicted values
def log_histograms(gt_counts, pred_counts, step=None):
    """
    Logs histograms of ground truth and predicted counts to WandB.
    :param gt_counts: Ground truth counts (list or numpy array)
    :param pred_counts: Predicted counts (list or numpy array)
    :param step: Global step for logging (optional)
    """
    # Convert to numpy arrays if not already
    gt_counts = np.array(gt_counts.cpu().detach())
    pred_counts = np.array(pred_counts.cpu().detach())
    
    # Create histograms for ground truth and predicted counts
    plt.figure(figsize=(10, 5))

    # Plot for ground truth
    plt.subplot(1, 2, 1)
    plt.hist(gt_counts, bins=50, alpha=0.7, color='blue', label='Ground Truth')
    plt.title('Histogram of Ground Truth Counts')
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot for predictions
    plt.subplot(1, 2, 2)
    plt.hist(pred_counts, bins=50, alpha=0.7, color='orange', label='Predictions')
    plt.title('Histogram of Predicted Counts')
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.legend()

    # Log to WandB
    if wandb.run is not None:
        wandb.log({
            "histograms/gt_counts": wandb.Histogram(gt_counts),
            "histograms/pred_counts": wandb.Histogram(pred_counts),
        }, step=step)

    # Optionally, save the plot as a file if needed
    plt.tight_layout()
    plt.close()

console = Console()

def _safe(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")

def _tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    img_t = img_t.detach().cpu().clamp(0, 1)
    arr = (img_t * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)

def _make_text_header(text: str, width: int, height: int = 60) -> Image.Image:
    header = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(header)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 10), text, fill=(0, 0, 0), font=font)
    return header

def _tile_views(views, max_cols: int = 4, pad: int = 6) -> Image.Image:
    w, h = views[0].size
    cols = min(max_cols, len(views))
    rows = math.ceil(len(views) / cols)
    canvas_w = cols * w + (cols - 1) * pad
    canvas_h = rows * h + (rows - 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
    for idx, im in enumerate(views):
        r = idx // cols
        c = idx % cols
        x = c * (w + pad)
        y = r * (h + pad)
        canvas.paste(im, (x, y))
    return canvas

def _save_panel(images_i: torch.Tensor, gt_i: float, pred_i: float, out_path: Path, max_views: int = 4):
    # images_i: [3,H,W] or [S,3,H,W]
    if images_i.dim() == 3:
        images_i = images_i.unsqueeze(0)

    images_i = images_i.detach().cpu().clamp(0, 1)
    S = images_i.shape[0]
    use_s = min(S, max_views)

    cols = min(4, use_s)
    rows = math.ceil(use_s / cols)

    fig_w = 3.6 * cols
    fig_h = 3.6 * rows + 0.8
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=200)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for ax in axes.flatten():
        ax.axis("off")

    for j in range(use_s):
        r = j // cols
        c = j % cols
        img = images_i[j].permute(1, 2, 0).numpy()
        axes[r, c].imshow(img)

    err = pred_i - gt_i
    fig.suptitle(
        f"GT: {int(gt_i)}   Pred: {int(pred_i)}   Error: {int(err)}",
        y=0.98
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def flatten_metrics(prefix, metrics_dict):
    flat = {}
    for split_name, split_metrics in metrics_dict.items():
        if split_metrics is None:
            continue
        for k, v in split_metrics.items():
            flat[f"{prefix}/{split_name}/{k}"] = float(v)
    return flat


# ============================================================
# Pretty Logging
# ============================================================

def pretty_print_metrics(title: str, metrics: Dict):
    if metrics is None:
        console.print(f"[yellow]{title}: None")
        return

    for split_name, split_metrics in metrics.items():
        if split_metrics is None:
            console.print(f"[yellow]{title} ({split_name}): None (no samples matched)")
            continue

        table = Table(title=f"{title} ({split_name})")
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")
        for k, v in split_metrics.items():
            table.add_row(k, f"{v:.6f}")
        console.print(table)


# ============================================================
# Metrics
# ============================================================

def _compute_metrics_core(gt_counts, pred_counts):
    assert len(gt_counts) == len(pred_counts)
    gt_counts = np.array(gt_counts, dtype=np.float64)
    pred_counts = np.array(pred_counts, dtype=np.float64)

    errors = pred_counts - gt_counts

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    nae = np.sum(np.abs(errors)) / np.sum(gt_counts)
    sre = np.sum(errors**2) / np.sum(gt_counts**2)
    smape = np.mean(
        2 * np.abs(errors) / (np.abs(gt_counts) + np.abs(pred_counts) + 1e-9) * 100
    )

    ss_total = np.sum((gt_counts - np.mean(gt_counts))**2)
    ss_residual = np.sum(errors**2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

    poisson_loss = np.mean(pred_counts - gt_counts * np.log(pred_counts + 1e-9))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "NAE": nae,
        "SRE": sre,
        "SMAPE (%)": smape,
        "R2": r2,
        "Poisson Loss": poisson_loss
    }


def compute_metrics(gt_counts, pred_counts, scene_counts=None, low_count_threshold=2000):
    metrics_all = _compute_metrics_core(gt_counts, pred_counts)

    mask = gt_counts < low_count_threshold
    if torch.any(mask):
        metrics_low = _compute_metrics_core(
            np.array(gt_counts)[mask],
            np.array(pred_counts)[mask]
        )

    return {
        "all": metrics_all,
        f"lt_{low_count_threshold}": metrics_low
    }



def evaluate_and_save_best_worst(
    model,
    loader,
    device,
    split_name: str,
    low_count_threshold=2000,
    count_key="count",
    save_dir: Path = None,
    k_best: int = 10,
    k_worst: int = 10,
    max_views_in_panel: int = 4,
    log_to_wandb: bool = False,
    global_step: int = None,   # <- add this
):

    model.eval()

    all_preds, all_trues, all_counts = [], [], []

    # store only what we need for final saving, not everything
    # each item: (abs_err, folder, image_name, gt, pred, images_tensor_cpu)
    best = []   # keep sorted ascending by abs_err
    worst = []  # keep sorted descending by abs_err

    def _insert_best(item):
        best.append(item)
        best.sort(key=lambda x: x[0])
        if len(best) > k_best:
            best.pop(-1)

    def _insert_worst(item):
        worst.append(item)
        worst.sort(key=lambda x: x[0], reverse=True)
        if len(worst) > k_worst:
            worst.pop(-1)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
            emb = batch["embedding"].to(device)
            gt = batch["volume_ratio"].to(device)

            pred = model(emb)

            pred_cpu = pred.detach().cpu().numpy()
            gt_cpu = gt.detach().cpu().numpy()

            all_preds.append(pred_cpu)
            all_trues.append(gt_cpu)

            if count_key in batch:
                c = batch[count_key]
                if torch.is_tensor(c):
                    c = c.detach().cpu().numpy()
                all_counts.append(c)

            folder = batch.get("folder", None)
            image_name = batch.get("image_name", None)

            B = len(gt_cpu)
            for i in range(B):
                gt_i = float(gt_cpu[i])
                pred_i = float(pred_cpu[i])
                abs_err = abs(pred_i - gt_i)

                f = str(folder[i]) if folder is not None else "nofolder"
                n = str(image_name[i]) if image_name is not None else f"{split_name}_{i:02d}"

                # store CPU tensor for later saving
                item = (int(abs_err), f, n, int(gt_i), int(pred_i), images[i].detach().cpu())

                _insert_best(item)
                _insert_worst(item)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)

    scene_counts = None
    if len(all_counts) > 0:
        scene_counts = np.concatenate(all_counts, axis=0)

    metrics = compute_metrics(
        gt_counts=y_true,
        pred_counts=y_pred,
        scene_counts=scene_counts,
        low_count_threshold=low_count_threshold
    )

    # Save panels
    if save_dir is not None:
        step_folder = f"step_{int(global_step):07d}" if global_step is not None else "step_unknown"
        base = save_dir / split_name / step_folder
        (base / "best").mkdir(parents=True, exist_ok=True)
        (base / "worst").mkdir(parents=True, exist_ok=True)

        for rank, (abs_err, f, n, gt_i, pred_i, img_i) in enumerate(best):
            out_path = base / "best" / (
                f"rank{rank:02d}_err{abs_err}_gt{gt_i}_pred{pred_i}_{_safe(f)}.png"
            )
            _save_panel(img_i, gt_i, pred_i, out_path, max_views=max_views_in_panel)
            if log_to_wandb and wandb.run is not None:
                wandb.log({f"qual/{split_name}/best": wandb.Image(str(out_path))}, step=global_step)

        for rank, (abs_err, f, n, gt_i, pred_i, img_i) in enumerate(worst):
            out_path = base / "worst" / (
                f"rank{rank:02d}_abs{abs_err:.3f}_gt{gt_i:.2f}_pred{pred_i:.2f}_{_safe(f)}_{_safe(n)}.png"
            )
            _save_panel(img_i, gt_i, pred_i, out_path, max_views=max_views_in_panel)
            if log_to_wandb and wandb.run is not None:
                wandb.log({f"qual/{split_name}/worst": wandb.Image(str(out_path))}, step=global_step)


    return metrics


# ============================================================
# Pretty Logging
# ============================================================

# def pretty_print_metrics(title: str, metrics: Dict[str, float]):
#     table = Table(title=title)
#     table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
#     table.add_column("Value", justify="right", style="green")
#     for k, v in metrics.items():
#         table.add_row(k, f"{v:.6f}")
#     console.print(table)


# ============================================================
# Metrics
# ============================================================

# def compute_metrics(y_true, y_pred, eps: float = 1e-8):
#     y_true = np.array(y_true.detach().cpu(), dtype=np.float64)
#     y_pred = np.array(y_pred.detach().cpu(), dtype=np.float64)

#     mae = mean_absolute_error(y_true, y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)

#     denom_nae = np.mean(np.abs(y_true)) + eps
#     nae = mae / denom_nae

#     num = np.sum((y_true - y_pred) ** 2)
#     denom_sre = np.sum(y_true ** 2) + eps
#     sre = np.sqrt(num / denom_sre)

#     smape = np.mean(2 * np.abs(y_pred - y_true) /
#                     (np.abs(y_true) + np.abs(y_pred) + eps)) * 100

#     r2 = r2_score(y_true, y_pred)

#     return {"MAE": mae, "RMSE": rmse, "NAE": nae, "SRE": sre, "SMAPE (%)": smape, "R2": r2}


def compute_metrics_val(model, val_loader, device):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            emb = batch["embedding"].to(device)
            gt = batch["volume_ratio"].to(device)
            folder = batch["folder"]
            print(folder)

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
        x = self.fc(x).squeeze(1)
        return torch.round(x)

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
        "data_path": "/teamspace/studios/this_studio/stackcounting_dataset/",
        "emb_path": "/teamspace/studios/this_studio/stackcounting_dataset/vggt_embeddings/",
        "patience": 30,         # <- early stopping patience
        "monitor": "MAE",        # <- metric to track
        "num_views": 1,
        "force_top_image": False,
        "use_cache": True,
        "val_freq": 10,        # <- validate every N steps
        "k_best_worst": 5,      # <- number of best/worst samples
        "val_size": 10,
    }

    # Load dataset
    train_dataset = StackDatasetEmbs(
        data_dir=cfg["data_path"] + "train",
        emb_dir=cfg["emb_path"] + "train",
        use_cache=cfg["use_cache"],
        experiment_config={
            "INPUT_TYPE": "GT_DEPTH",
            "PREDICT_TYPE": "COUNT_DIRECTLY",
            "DINO_USE_VOL_AS_ADDITIONAL_INPUT": False,
        }
    )

    total_len = len(train_dataset)
    val_len = cfg["val_size"]
    train_len = total_len - val_len

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    test_synt_dataset = StackDatasetEmbs(
        data_dir=cfg["data_path"] + "val",
        emb_dir=cfg["emb_path"] + "val",
        use_cache=cfg["use_cache"],
        experiment_config={
            "INPUT_TYPE": "GT_DEPTH",
            "PREDICT_TYPE": "COUNT_DIRECTLY",
            "DINO_USE_VOL_AS_ADDITIONAL_INPUT": False,
        }
    )

    test_real_dataset = StackDatasetEmbs(
        data_dir=cfg["data_path"] + "test/scenes",
        emb_dir=cfg["emb_path"] + "test",
        use_cache=cfg["use_cache"],
        experiment_config={
            "INPUT_TYPE": "GT_DEPTH",
            "PREDICT_TYPE": "COUNT_DIRECTLY",
            "DINO_USE_VOL_AS_ADDITIONAL_INPUT": False,
        }, 
        test = True
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test_synt_loader = DataLoader(test_synt_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test_real_loader = DataLoader(test_real_dataset, batch_size=cfg["batch_size"], shuffle=False)
    

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test synt samples:", len(test_synt_dataset))
    print("Test real samples:", len(test_real_dataset))

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
        #console.print(f"[bold green]Loading checkpoint from: {checkpoint_path}")
        #ckpt = torch.load(checkpoint_path, map_location=device)
        #model.load_state_dict(ckpt)

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
    global_step = 0
    step = 0

    # ============================================================
    # Main Training Loop
    # ============================================================

    with wandb.init(project="ML_proj2", name="VGGT_finetune-no_ckpt"):

        for epoch in range(cfg["epochs"]):

            console.rule(f"[bold green]Epoch {epoch+1}/{cfg['epochs']}")
            model.train()

            pbar = tqdm(train_loader, desc="Training", leave=True)
            for step, batch in enumerate(pbar):
                emb = batch["embedding"].to(device)
                gt = batch["volume_ratio"].to(device)

                pred = model(emb)
                pred_clamped = torch.clamp(pred, min=1e-6)  
                gt_clamped = torch.clamp(gt, min=1e-6)

                # compute log
                loss_raw = loss_fn(pred_clamped, gt_clamped)
                # loss_raw = loss_fn(torch.log(pred_clamped), torch.log(gt_clamped))

                # zero out small losses
                # loss = torch.where(loss_raw < 0.1, torch.zeros_like(loss_raw), loss_raw)
                loss = loss_raw

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": float(loss.detach())})
                wandb.log({"train_loss": float(loss.detach()), 
                            "train_loss_raw": float(loss_raw.detach()), 
                            "mean_pred": pred_clamped.mean(), 
                            "min_pred": pred_clamped.min(), 
                            "max_pred": pred_clamped.max()})

                # Log histograms of GT and predicted counts
                log_histograms(gt_clamped, pred_clamped)

                # ---- VALIDATION ----
                if step % cfg["val_freq"] == 0 and step > 0:
                    qual_dir = Path("qualitative_panels_best_worst")

                    console.print("\n[bold yellow]Running validation...")
                    val_metrics = compute_metrics_val(model, val_loader, device)
                    pretty_print_metrics("Validation Metrics", val_metrics)
                    wandb.log(flatten_metrics("val", val_metrics))

                    test_synt_metrics = compute_metrics_val(model, test_synt_loader, device)
                    pretty_print_metrics("Test synt Metrics", test_synt_metrics)
                    wandb.log(flatten_metrics("test_synt", test_synt_metrics))

                    test_real_metrics = compute_metrics_val(model, test_real_loader, device)
                    pretty_print_metrics("Test real Metrics", test_real_metrics)
                    wandb.log(flatten_metrics("test_real", test_real_metrics))

                    current = val_metrics[cfg["monitor"]]

                    # =====================================================
                    #   EARLY STOPPING
                    # =====================================================
                    if current < best_metric:
                        best_metric = current
                        best_epoch = epoch
                        patience_counter = 0

                        torch.save(model.state_dict(), "best_model_mae4.pth")
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
