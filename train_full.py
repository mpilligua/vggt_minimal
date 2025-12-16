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

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import wandb

from stack_dataset import StackDataset2
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont
import math
import matplotlib.pyplot as plt


console = Console()

# ============================================================
# Qualitative Examples Saving
# ============================================================

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
    if np.any(mask):
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
            images = batch["images"]          # keep on CPU for saving
            images_dev = images.to(device)

            gt = batch["volume_ratio"].to(device)
            pred = model(images_dev)

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
# Load VGGT Aggregator Only
# ============================================================


def load_vggt_aggregator_only(device: str = None, dtype: torch.dtype = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None and device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    elif dtype is None:
        dtype = torch.float32

    print(f"Loading VGGT on CPU, extracting aggregator only. Target device={device}, dtype={dtype}")

    with torch.no_grad():
        full = VGGT.from_pretrained("facebook/VGGT-1B").to("cpu")
        full.eval()

        aggregator = full.aggregator
        for p in aggregator.parameters():
            p.requires_grad = False

        del full
        torch.cuda.empty_cache()

    aggregator = aggregator.to(device=device, dtype=dtype)
    aggregator.eval()

    return aggregator, device, dtype


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
        return self.fc(x).squeeze(1)  # Only squeeze feature dim, keep batch dim

    def print_model_size(self):
        total = sum(p.numel() for p in self.parameters())
        # pretty print the result in millions
        print(f"Total number of weights CNN: {total / 1e6:.2f}M")



class VGGTWrapperCNN(nn.Module):
    def __init__(self, device: str = None, dtype: torch.dtype = None):
        super().__init__()
        model, device, dtype = load_vggt_aggregator_only(device, dtype)
        self.model = model
        self.device = device
        self.dtype = dtype

        for param in self.model.parameters():
            param.requires_grad = False

        self.cnn = ConvRegressor().to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        
        images = images.to(self.device)
        B, S, C, H, W = images.shape
        
        aggregated_tokens_list, patch_start_idx = self.model(images)
        
        map_ = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]  # [B, S, num_patches, embed_dim]
        out = self.cnn(map_.mean(dim=1))  # Average over S views, then pass through CNN

        rounded_out = torch.round(out)
        return rounded_out
    
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        # pretty print the result only of parameters we are training
        print(f"Total number of weights VGGT + CNN: {total_params / 1e6:.2f}M")

    def print_flops(self, image_size: int = 518):
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"Total FLOPs: {flops.total()/1e9:.2f} GFLOPs")
        except ImportError:
            print("fvcore is not installed. Cannot compute FLOPs.")

def flatten_metrics(prefix, metrics_dict):
    flat = {}
    for split_name, split_metrics in metrics_dict.items():
        if split_metrics is None:
            continue
        for k, v in split_metrics.items():
            flat[f"{prefix}/{split_name}/{k}"] = float(v)
    return flat


# ============================================================
# Training
# ============================================================

if __name__ == "__main__":

    cfg = {
        "batch_size": 2,
        "epochs": 500,
        "lr": 1e-4,
        "data_path": "/Users/maria/ML/data/",
        "patience": 30,         # <- early stopping patience
        "monitor": "MAE",        # <- metric to track
        "num_views": 1,
        "force_top_image": False,
        "use_cache": True,
        "val_freq": 1,        # <- validate every N steps
        "k_best_worst": 5,      # <- number of best/worst samples
        "val_size": 20,
    }

    # Use CPU instead of MPS due to bugs with scaled_dot_product_attention on MPS
    device = torch.device("cpu")

    # Model
    model = VGGTWrapperCNN(device=device)
    model.print_model_size()
    model.print_flops(image_size=518)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.L1Loss()


    # Load dataset
    train_dataset = StackDataset2(
        data_dir=cfg["data_path"] + "train",
        use_cache=cfg["use_cache"],
        num_views=cfg["num_views"],
        force_top_image=cfg["force_top_image"],
        transform=transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
        ]),
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


    test_synt_dataset = StackDataset2(
        data_dir=cfg["data_path"] + "val",
        use_cache=cfg["use_cache"],
        num_views=cfg["num_views"],
        force_top_image=cfg["force_top_image"],
        transform=transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
        ]),
        experiment_config={
            "INPUT_TYPE": "GT_DEPTH",
            "PREDICT_TYPE": "COUNT_DIRECTLY",
            "DINO_USE_VOL_AS_ADDITIONAL_INPUT": False,
        }
    )

    test_real_dataset = StackDataset2(
        data_dir=cfg["data_path"] + "test/scenes",
        use_cache=cfg["use_cache"],
        num_views=cfg["num_views"],
        force_top_image=cfg["force_top_image"],
        transform=transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
        ]),
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
    global_step = 0
    step = 0

    # ============================================================
    # Main Training Loop
    # ============================================================

    with wandb.init(project="ML_proj2", name="VGGT_finetune_v2"):

        for epoch in range(cfg["epochs"]):
            console.rule(f"[bold green]Epoch {epoch+1}/{cfg['epochs']}")
            model.train()

            pbar = tqdm(train_loader, desc="Training", leave=True)
            for i, batch in enumerate(pbar):
                folder = batch["folder"]
                image_name = batch["image_name"]
                images = batch["images"].to(device)
                gt = batch["volume_ratio"].to(device)

                pred = model(images)
                loss = loss_fn(pred, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": float(loss.detach())})
                wandb.log({"train_loss": float(loss.detach())})

                # ---- VALIDATION ----
                if step % cfg["val_freq"] == 0 and step > 0:
                    qual_dir = Path("qualitative_panels_best_worst")

                    val_metrics = evaluate_and_save_best_worst(
                        model, val_loader, device,
                        split_name="val",
                        low_count_threshold=2000,
                        count_key="count",
                        save_dir=qual_dir,
                        k_best=cfg["k_best_worst"],
                        k_worst=cfg["k_best_worst"],
                        max_views_in_panel=4,
                        global_step=global_step,
                    )
                    pretty_print_metrics("Validation Metrics", val_metrics)
                    wandb.log(flatten_metrics("val", val_metrics))

                    test_synt_metrics = evaluate_and_save_best_worst(
                        model, test_synt_loader, device,
                        split_name="test_synt",
                        low_count_threshold=2000,
                        count_key="count",
                        save_dir=qual_dir,
                        k_best=cfg["k_best_worst"],
                        k_worst=cfg["k_best_worst"],
                        max_views_in_panel=4,
                        global_step=global_step,
                    )
                    pretty_print_metrics("Test synt Metrics", test_synt_metrics)
                    wandb.log(flatten_metrics("test_synt", test_synt_metrics))

                    test_real_metrics = evaluate_and_save_best_worst(
                        model, test_real_loader, device,
                        split_name="test_real",
                        low_count_threshold=2000,
                        count_key="count",
                        save_dir=qual_dir,
                        k_best=cfg["k_best_worst"],
                        k_worst=cfg["k_best_worst"],
                        max_views_in_panel=4,
                        global_step=global_step,
                    )
                    pretty_print_metrics("Test real Metrics", test_real_metrics)
                    wandb.log(flatten_metrics("test_real", test_real_metrics))

                    # train: you probably only want best/worst samples, not full metrics
                    # _ = evaluate_and_save_best_worst(
                    #     model, train_loader, device,
                    #     split_name="train",
                    #     low_count_threshold=2000,
                    #     count_key="count",
                    #     save_dir=qual_dir,
                    #     k_best=cfg["k_best_worst"],
                    #     k_worst=cfg["k_best_worst"],
                    #     max_views_in_panel=4,
                    # )

                    current = val_metrics["all"][cfg["monitor"]]

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
                        
                global_step += len(train_loader)
                step += 1

    console.print("[bold magenta]Training complete.")
