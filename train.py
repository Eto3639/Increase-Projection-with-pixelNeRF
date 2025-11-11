import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from pathlib import Path
import yaml
from tqdm import tqdm
import os
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import math
import logging
import csv
from datetime import datetime
import random

# --- Check for and import optional libraries ---
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# --- Local Modules ---
from dataset import DRRDataset
from model import PixelNeRFModel, get_rays

def load_config(path="config.yml"):
    """Load YAML configuration file."""
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        print("Configuration file loaded.")
        return cfg
    except FileNotFoundError:
        print(f"Error: Configuration file '{path}' not found.")
        exit()
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit()

def safe_inverse(matrix_batch):
    """Safely compute the inverse of a batch of 4x4 matrices."""
    inverses = []
    for i in range(matrix_batch.shape[0]):
        try:
            inv = torch.inverse(matrix_batch[i].contiguous().float())
            inverses.append(inv)
        except Exception:
            inverses.append(torch.eye(4, device=matrix_batch.device, dtype=matrix_batch.dtype))
    return torch.stack(inverses, dim=0)

class CsvLogger:
    """A simple CSV logger for training and validation metrics."""
    def __init__(self, log_dir, run_name):
        self.log_path = Path(log_dir) / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = []
        self.file = open(self.log_path, 'w', newline='')
        self.writer = None

    def log(self, data_dict):
        if self.writer is None:
            self.fieldnames = list(data_dict.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        self.writer.writerow(data_dict)
        self.file.flush()

    def close(self):
        self.file.close()

def run_one_epoch(model, loader, optimizer, loss_fns, loss_weights, device, cfg, is_train):
    """Run one epoch of patch-based training or validation."""
    model.train() if is_train else model.eval()
    
    total_losses = {'total': 0.0, 'l1': 0.0, 'lpips': 0.0, 'coarse': 0.0, 'sigma_coarse_mean': 0.0, 'sigma_coarse_max': 0.0, 'sigma_fine_mean': 0.0, 'sigma_fine_max': 0.0}
    l1_fn, lpips_fn = loss_fns['l1'], loss_fns['lpips']
    w_l1, w_lpips = loss_weights['l1'], loss_weights.get('lpips', 0.0)
    processed_batches = 0

    mode = "Train" if is_train else "Validation"
    progress_bar = tqdm(loader, desc=f"Epoch {cfg['epoch_num']}/{cfg['training']['num_epochs']} [{mode}]")

    with torch.set_grad_enabled(is_train):
        for i, batch in enumerate(progress_bar):
            # --- Data Preparation ---
            img_0, img_90, target_img = batch["img_0"].to(device), batch["img_90"].to(device), batch["target_img"].to(device)
            params_0, params_90, target_params = {k: v.to(device) for k,v in batch["params_0"].items()}, {k: v.to(device) for k,v in batch["params_90"].items()}, {k: v.to(device) for k,v in batch["target_params"].items()}
            B, _, H, W = target_img.shape
            K, w2c = target_params["intrinsics"][:, :3, :3], target_params["extrinsics"]
            c2w = safe_inverse(w2c) if cfg['geometry']['invert_extrinsics'] else w2c
            
            if is_train:
                patch_size = cfg['training'].get('patch_size')
                if patch_size is None: raise ValueError("Patch size must be set for training.")
                px = torch.randint(0, W - patch_size + 1, (B,), device=device)
                py = torch.randint(0, H - patch_size + 1, (B,), device=device)
                target_patch = torch.stack([target_img[b, :, py[b]:py[b]+patch_size, px[b]:px[b]+patch_size] for b in range(B)])
                target_pixels = target_patch.permute(0, 2, 3, 1).reshape(B, -1, 1)
                
                rays_o_full, rays_d_full = get_rays(H, W, K, c2w)
                rays_o_full = rays_o_full.reshape(B, H, W, 3)
                rays_d_full = rays_d_full.reshape(B, H, W, 3)

                j_idx, i_idx = torch.meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device), indexing='xy')
                pixel_indices_patch = torch.stack([i_idx, j_idx], dim=-1).view(1, -1, 2)
                batch_offsets = torch.stack([px, py], dim=-1).unsqueeze(1)
                pixel_indices = pixel_indices_patch + batch_offsets

                selected_rays_o = torch.stack([rays_o_full[b, pixel_indices[b, :, 1], pixel_indices[b, :, 0]] for b in range(B)])
                selected_rays_d = torch.stack([rays_d_full[b, pixel_indices[b, :, 1], pixel_indices[b, :, 0]] for b in range(B)])
                
                rays_o, rays_d = selected_rays_o, selected_rays_d
            else:
                rays_o, rays_d = get_rays(H, W, K, c2w)
                rays_o, rays_d = rays_o.reshape(B, -1, 3), rays_d.reshape(B, -1, 3)

            if is_train: optimizer.zero_grad()

            chunk_size = cfg['training']['patch_size']**2 if is_train else cfg['training'].get('visualization_chunk_size', 2048)
            
            all_outputs_coarse, all_outputs_fine = [], []
            batch_is_invalid = False
            for chunk_i in range(0, rays_o.shape[1], chunk_size):
                chunk_rays_o = rays_o[:, chunk_i:chunk_i+chunk_size]
                chunk_rays_d = rays_d[:, chunk_i:chunk_i+chunk_size]
                
                outputs = model(img_0, img_90, params_0, params_90, chunk_rays_o, chunk_rays_d)

                if not torch.all(torch.isfinite(outputs["coarse"])) or ("fine" in outputs and not torch.all(torch.isfinite(outputs["fine"]))):
                    print(f"\n\nERROR in batch {i}: Model output contains NaN/Inf! Skipping batch.")
                    torch.cuda.empty_cache()
                    batch_is_invalid = True
                    break
                
                all_outputs_coarse.append(outputs["coarse"])
                if model.use_fine_model and "fine" in outputs:
                    all_outputs_fine.append(outputs["fine"])

            if batch_is_invalid:
                continue

            pred_pixels_coarse = torch.cat(all_outputs_coarse, dim=1)
            
            if is_train:
                loss_target_pixels = target_pixels
            else:
                loss_target_pixels = target_img.permute(0, 2, 3, 1).reshape(B, -1, 1)

            loss_coarse = l1_fn(pred_pixels_coarse, loss_target_pixels)
            loss = loss_coarse

            if model.use_fine_model and all_outputs_fine:
                pred_pixels_fine = torch.cat(all_outputs_fine, dim=1)
                loss_l1_fine = l1_fn(pred_pixels_fine, loss_target_pixels)
                
                loss_lpips_fine = torch.tensor(0.0, device=device)
                if w_lpips > 0 and lpips_fn is not None:
                    # Reshape for LPIPS
                    pred_img_for_lpips = pred_pixels_fine.reshape(B, H, W, 1).permute(0, 3, 1, 2) if not is_train else pred_pixels_fine.reshape(B, patch_size, patch_size, 1).permute(0, 3, 1, 2)
                    target_for_lpips = target_img if not is_train else target_patch
                    
                    lpips_input_pred = torch.clamp(pred_img_for_lpips * 2 - 1, -1, 1)
                    lpips_input_target = torch.clamp(target_for_lpips * 2 - 1, -1, 1)
                    loss_lpips_fine = lpips_fn(lpips_input_pred, lpips_input_target).mean()

                if not torch.isfinite(loss_l1_fine) or not torch.isfinite(loss_lpips_fine):
                    print(f"\n\nERROR in batch {i}: Loss calculation resulted in NaN/Inf. Skipping batch.")
                    continue

                loss = loss_l1_fine * w_l1 + loss_lpips_fine * w_lpips + loss_coarse
                total_losses['l1'] += loss_l1_fine.item()
                total_losses['lpips'] += loss_lpips_fine.item()

            if is_train:
                if not torch.isfinite(loss):
                    print(f"\n\nERROR in batch {i}: Total loss is {loss}. Skipping backward pass.")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_losses['total'] += loss.item()
            total_losses['coarse'] += loss_coarse.item()
            processed_batches += 1
            progress_bar.set_postfix({"L1_f": f"{total_losses['l1']/processed_batches:.4f}", "LPIPS": f"{total_losses['lpips']/processed_batches:.4f}"})
    
    if processed_batches == 0:
        print("\nWARNING: No batches were processed in this epoch. Returning zero losses.")
        return total_losses # Return zero losses to avoid division by zero

    avg_losses = {k: v / processed_batches for k, v in total_losses.items()}
    return avg_losses


def get_full_image_prediction(model, batch, device, cfg):
    """Render a full image for a given batch (data sample) and return visualization data."""
    model.eval()
    with torch.no_grad():
        img_0 = batch["img_0"].unsqueeze(0).to(device)
        img_90 = batch["img_90"].unsqueeze(0).to(device)
        target_img = batch["target_img"].unsqueeze(0).to(device)
        params_0 = {k: v.unsqueeze(0).to(device) for k, v in batch["params_0"].items()}
        params_90 = {k: v.unsqueeze(0).to(device) for k, v in batch["params_90"].items()}
        target_params = {k: v.unsqueeze(0).to(device) for k, v in batch["target_params"].items()}

        B, _, H, W = target_img.shape
        K = target_params["intrinsics"][:, :3, :3]
        w2c = target_params["extrinsics"]
        c2w = safe_inverse(w2c) if cfg['geometry']['invert_extrinsics'] else w2c
        
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_o, rays_d = rays_o.reshape(B, -1, 3), rays_d.reshape(B, -1, 3)

        chunk_size = cfg['training'].get('visualization_chunk_size', 2048)
        
        all_outputs = []
        for i in range(0, rays_o.shape[1], chunk_size):
            outputs = model(img_0, img_90, params_0, params_90, rays_o[:, i:i+chunk_size, :], rays_d[:, i:i+chunk_size, :])
            all_outputs.append(outputs)
        
        # 特徴マップは最初のチャンクから取得すればよい
        vis_data = {
            "F_0": all_outputs[0].get("F_0"),
            "F_90": all_outputs[0].get("F_90"),
        }

        # 不要になったキーを削除してから、残りを結合する
        keys_to_concat = [k for k in all_outputs[0].keys() if k not in ["F_0", "F_90"]]
        full_outputs = {k: torch.cat([d[k] for d in all_outputs], dim=1) for k in keys_to_concat}

        pred_img_coarse_tensor = full_outputs['coarse'].reshape(B, H, W, 1).permute(0, 3, 1, 2)
        
        pred_img_fine_tensor = None
        if model.use_fine_model and 'fine' in full_outputs:
            pred_img_fine_tensor = full_outputs['fine'].reshape(B, H, W, 1).permute(0, 3, 1, 2)

        vis_data.update({
            "img_0": img_0,
            "img_90": img_90,
            "target_img": target_img,
            "pred_coarse": pred_img_coarse_tensor,
            "pred_fine": pred_img_fine_tensor,
            "z_vals_coarse": full_outputs.get("z_vals_coarse"),
            "weights_coarse": full_outputs.get("weights_coarse"),
            "z_vals_fine": full_outputs.get("z_vals_fine"),
            "weights_fine": full_outputs.get("weights_fine"),
        })

    return vis_data

def visualize_feature_maps(epoch, F_0, F_90):
    """Visualizes the mean of feature maps and returns the figure object."""
    if F_0 is None or F_90 is None:
        return None

    def to_feature_img(tensor):
        # チャンネル次元で平均をとり、[0, 1]に正規化
        mean_feature = tensor.mean(dim=1, keepdim=True)
        mean_feature = (mean_feature - mean_feature.min()) / (mean_feature.max() - mean_feature.min() + 1e-8)
        return (mean_feature.squeeze().cpu().numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Epoch {epoch+1} - Feature Maps', fontsize=16)
    
    axes[0].imshow(to_feature_img(F_0), cmap='viridis')
    axes[0].set_title("Feature Map (0 deg)")
    axes[0].axis('off')

    axes[1].imshow(to_feature_img(F_90), cmap='viridis')
    axes[1].set_title("Feature Map (90 deg)")
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def visualize_angle(epoch, cfg, run_name, angle, vis_data, metrics):
    """Visualize results, conditionally save locally, and log to wandb."""
    print(f"  - Visualizing angle {angle}...")
    
    gt = vis_data["target_img"]
    pred_c = vis_data["pred_coarse"]
    pred_f = vis_data["pred_fine"]
    z_vals_coarse = vis_data["z_vals_coarse"]
    weights_coarse = vis_data["weights_coarse"]
    z_vals_fine = vis_data["z_vals_fine"]
    weights_fine = vis_data["weights_fine"]

    def to_img_np(tensor):
        if tensor is None:
            h, w = cfg['dataset']['img_size']
            return np.zeros((h, w), dtype=np.uint8)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        return (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # --- Main Visualization Figure ---
    main_fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    main_fig.suptitle(f'Epoch {epoch+1} - {run_name} - Angle {angle}', fontsize=16)

    axes[0, 0].imshow(to_img_np(gt), cmap='gray'); axes[0, 0].set_title("Ground Truth")
    axes[0, 1].imshow(to_img_np(pred_c), cmap='gray'); axes[0, 1].set_title("Coarse Prediction")
    axes[0, 2].imshow(to_img_np(pred_f), cmap='gray'); axes[0, 2].set_title("Fine Prediction")

    error_map_c = torch.abs(gt - pred_c) if pred_c is not None else torch.zeros_like(gt)
    im_c = axes[1, 0].imshow(to_img_np(error_map_c), cmap='viridis'); axes[1, 0].set_title("Error Map (Coarse)")
    main_fig.colorbar(im_c, ax=axes[1, 0])

    error_map_f = torch.abs(gt - pred_f) if pred_f is not None else torch.zeros_like(gt)
    im_f = axes[1, 1].imshow(to_img_np(error_map_f), cmap='viridis'); axes[1, 1].set_title("Error Map (Fine)")
    main_fig.colorbar(im_f, ax=axes[1, 1])

    ax_weights = axes[1, 2]
    if z_vals_coarse is not None and weights_coarse is not None:
        _, H, W = gt.shape[-3:]
        center_ray_idx = (H // 2) * W + (W // 2)
        z_c, w_c = z_vals_coarse[0, center_ray_idx].cpu().numpy(), weights_coarse[0, center_ray_idx].cpu().numpy()
        ax_weights.plot(z_c, w_c, 'b-', label='Coarse Weights')

    if z_vals_fine is not None and weights_fine is not None:
        _, H, W = gt.shape[-3:]
        center_ray_idx = (H // 2) * W + (W // 2)
        z_f, w_f = z_vals_fine[0, center_ray_idx].cpu().numpy(), weights_fine[0, center_ray_idx].cpu().numpy()
        ax_weights.plot(z_f, w_f, 'r-', label='Fine Weights')

    ax_weights.set_title(f"Ray Weights\nL1: {metrics['l1']:.4f}, LPIPS: {metrics['lpips']:.4f}")
    ax_weights.set_xlabel("Depth (z)"); ax_weights.set_ylabel("Weight")
    ax_weights.legend(); ax_weights.grid(True)

    for ax in axes.flat: ax.axis('on') if ax != ax_weights else None
    main_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Feature Map Visualization Figure ---
    feature_map_fig = visualize_feature_maps(epoch, vis_data.get("F_0"), vis_data.get("F_90"))

    # --- Local Saving (Conditional) ---
    if cfg['training'].get('save_visualizations_locally', False):
        angle_vis_dir = Path(cfg['training'].get('visualization_dir', 'visualizations')) / run_name / f"angle_{angle}"
        angle_vis_dir.mkdir(parents=True, exist_ok=True)
        
        main_save_path = angle_vis_dir / f"epoch_{(epoch+1):04d}.png"
        main_fig.savefig(main_save_path)
        
        if feature_map_fig:
            feature_map_save_path = angle_vis_dir / f"epoch_{(epoch+1):04d}_features.png"
            feature_map_fig.savefig(feature_map_save_path)
    
    # --- W&B Logging ---
    if WANDB_AVAILABLE and cfg.get('wandb', {}).get('enabled', False) and wandb.run:
        log_payload = {f"visualizations/angle_{angle}": wandb.Image(main_fig)}
        if feature_map_fig:
            log_payload[f"feature_maps/angle_{angle}"] = wandb.Image(feature_map_fig)
        wandb.log(log_payload)
    
    # --- Cleanup ---
    plt.close(main_fig)
    if feature_map_fig:
        plt.close(feature_map_fig)

def visualize_input_data(dataset, num_samples, save_dir, cfg):
    """Saves a few samples from the dataset for visual inspection."""
    if not cfg['training'].get('save_visualizations_locally', False):
        return # ローカル保存が無効なら何もしない

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving {num_samples} data samples to {save_dir}...")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        save_image(sample['img_0'], save_dir / f"sample_{i:02d}_img_0.png")
        save_image(sample['img_90'], save_dir / f"sample_{i:02d}_img_90.png")
        save_image(sample['target_img'], save_dir / f"sample_{i:02d}_target_angle_{sample['info']['target_angle']}.png")

def objective(trial, args):
    cfg = load_config(args.config)
    is_optuna_run = isinstance(trial, optuna.Trial)
    if is_optuna_run and cfg.get('optuna', {}).get('enabled', False):
        hparams = cfg['optuna']['hyperparameters']
        cfg['model']['backbone']['name'] = trial.suggest_categorical('backbone_name', hparams['backbone_name']['choices'])
        # --- ★ Optunaによるハイパーパラメータの提案を追加 ---
        if 'learning_rate' in hparams: 
            cfg['training']['learning_rate'] = trial.suggest_float('learning_rate', hparams['learning_rate']['low'], hparams['learning_rate']['high'], log=hparams['learning_rate'].get('log', False))
        if 'weight_decay' in hparams:
            cfg['training']['weight_decay'] = trial.suggest_float('weight_decay', hparams['weight_decay']['low'], hparams['weight_decay']['high'], log=hparams['weight_decay'].get('log', False))
        if 'd_mlp_width' in hparams:
            cfg['model']['d_mlp_width'] = trial.suggest_categorical('d_mlp_width', hparams['d_mlp_width']['choices'])
        if 'd_feature' in hparams:
            cfg['model']['d_feature'] = trial.suggest_categorical('d_feature', hparams['d_feature']['choices'])
        if 'lpips_net' in hparams:
            cfg['training']['lpips_net'] = trial.suggest_categorical('lpips_net', hparams['lpips_net']['choices'])

    
    # --- Run Name and Directory Setup ---
    base_project_name = cfg['wandb']['project']
    backbone_name = cfg['model']['backbone']['name']
    
    trial_num_str = f"trial_{trial.number}" if is_optuna_run else "default"
    
    # New run_name for logging and directory creation
    run_name_for_ui = f"{trial_num_str}_{backbone_name}"
    # Filesystem-friendly name that includes the project
    fs_run_name = f"{base_project_name}_{run_name_for_ui}"

    # --- Logging Setup ---
    # Print current run info to console
    print("\n" + "="*50)
    print(f"Starting Run: {fs_run_name}")
    print(f"Project: {base_project_name}")
    if is_optuna_run:
        print(f"Trial Number: {trial.number}")
    print(f"Backbone: {backbone_name}")
    print("="*50 + "\n")

    logger = CsvLogger(cfg['training']['log_dir'], fs_run_name)

    # --- W&B Setup ---
    wandb_enabled = cfg.get('wandb', {}).get('enabled', False)
    wandb_entity = cfg.get('wandb', {}).get('entity')
    if wandb_enabled and (not WANDB_AVAILABLE or not wandb_entity or wandb_entity == 'YOUR_WANDB_ENTITY'):
        print("W&B is disabled. Please install wandb and set your entity in config.yml")
        cfg['wandb']['enabled'] = False
    
    if cfg['wandb']['enabled']:
        wandb.init(
            project=base_project_name, 
            entity=wandb_entity, 
            config=cfg, 
            name=run_name_for_ui, 
            reinit=True
        )

    DEVICE = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    
    # --- Dataset ---
    root_dir = Path(cfg['dataset']['root_dir'])
    all_patient_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])

    subset_perc = cfg['dataset'].get('data_subset_percentage', 1.0)
    if subset_perc < 1.0:
        num_patients = int(len(all_patient_dirs) * subset_perc)
        all_patient_dirs = all_patient_dirs[:num_patients]

    random.shuffle(all_patient_dirs)

    train_split_perc = cfg['dataset'].get('train_val_split', 0.8)
    train_size = int(train_split_perc * len(all_patient_dirs))
    train_dirs = all_patient_dirs[:train_size]
    val_dirs = all_patient_dirs[train_size:]

    train_dataset = DRRDataset(cfg=cfg, split='train', patient_dirs=train_dirs)
    val_dataset = DRRDataset(cfg=cfg, split='val', patient_dirs=val_dirs)

    # --- ★デバッグ用: 単一データへの過学習テスト ---
    if cfg['training'].get('overfit_single_image', False):
        print("\n" + "!"*60)
        print("!!! WARNING: OVERFITTING ON A SINGLE IMAGE FOR DEBUGGING !!!")
        print("! This will use only the first data sample for training and validation.")
        print("!"*60 + "\n")
        train_dataset = Subset(train_dataset, [0])
        # 検証も同じデータで行うことで、学習の進捗を純粋に追跡する
        val_dataset = Subset(val_dataset, [0])

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    print(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation images.")

    # --- ★入力データを可視化 ---
    vis_input_dir = Path(cfg['training'].get('visualization_dir', 'visualizations')) / fs_run_name
    visualize_input_data(train_dataset, num_samples=3, save_dir=vis_input_dir / "input_samples", cfg=cfg)

    # --- Model, Optimizer, Losses ---
    model = PixelNeRFModel(cfg=cfg).to(DEVICE)
    if cfg['wandb']['enabled']:
        wandb.watch(model, log="all", log_freq=100) # ★勾配をロギング

    # --- ★ Optimizerにweight_decayを適用 ---
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg['training']['learning_rate'], 
        weight_decay=cfg['training'].get('weight_decay', 0.0)
    )
    lpips_net_type = cfg['training'].get('lpips_net', 'alex')
    loss_fns = {
        'l1': nn.L1Loss(),
        'lpips': lpips.LPIPS(net=lpips_net_type).to(DEVICE) if LPIPS_AVAILABLE and cfg['training']['loss_weights'].get('lpips', 0.0) > 0 else None
    }
    loss_weights = cfg['training']['loss_weights']
    
    scheduler = None
    scheduler_config = cfg['training'].get('scheduler')
    if scheduler_config:
        scheduler_params = scheduler_config.copy()
        scheduler_params.pop('type', None)
        scheduler = CosineAnnealingLR(optimizer, **scheduler_params)

    # --- Validation Samples ---
    val_samples_by_angle = {angle: None for angle in cfg['dataset']['target_angles']}
    if cfg['training'].get('overfit_single_image', False):
        # In overfitting mode, just use the single available sample for visualization
        print("Overfitting mode: Using the single data sample for all visualizations.")
        if len(val_dataset) > 0:
            # Get the single item from the Subset
            single_sample = val_dataset[0] 
            # Assign it to the first available visualization angle slot
            first_angle = cfg['dataset']['target_angles'][0]
            val_samples_by_angle[first_angle] = single_sample
    else:
        # Original logic for finding samples for each angle
        for angle in val_samples_by_angle.keys():
            for i in range(len(val_dataset)):
                # データセットから指定した角度のサンプルを取得しようと試みる
                sample = val_dataset.get_item_for_visualization(i, angle)
                if sample is not None:
                    val_samples_by_angle[angle] = sample
                    break # この角度のサンプルが見つかったので次の角度へ
    print(f"Found validation samples for angles: {[k for k, v in val_samples_by_angle.items() if v is not None]}")

    best_val_loss = float('inf')
    for epoch in range(cfg['training']['num_epochs']):
        cfg['epoch_num'] = epoch + 1
        
        # --- Training ---
        train_losses = run_one_epoch(model, train_loader, optimizer, loss_fns, loss_weights, DEVICE, cfg, is_train=True)
        if scheduler: scheduler.step()

        # --- Validation ---
        val_losses = run_one_epoch(model, val_loader, None, loss_fns, loss_weights, DEVICE, cfg, is_train=False)
        
        # --- ▼▼▼ DIAGNOSTIC BLOCK ▼▼▼ ---
        try:
            print(f"Epoch {epoch+1} | Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")

            # --- Logging ---
            log_data = {
                "epoch": epoch + 1,
                "train_loss_total": train_losses['total'],
                "train_loss_l1": train_losses.get('l1', 0),
                "train_loss_lpips": train_losses.get('lpips', 0),
                "train_loss_coarse": train_losses.get('coarse', 0),
                "val_loss_total": val_losses['total'],
                "val_loss_l1": val_losses.get('l1', 0),
                "val_loss_lpips": val_losses.get('lpips', 0),
                "val_loss_coarse": val_losses.get('coarse', 0),
                "sigma_coarse_mean": train_losses.get('sigma_coarse_mean', 0),
                "sigma_coarse_max": train_losses.get('sigma_coarse_max', 0),
                "sigma_fine_mean": train_losses.get('sigma_fine_mean', 0),
                "sigma_fine_max": train_losses.get('sigma_fine_max', 0),
            }
            if scheduler: log_data['learning_rate'] = scheduler.get_last_lr()[0]
            logger.log(log_data)
            if cfg['wandb']['enabled']: wandb.log(log_data)
        except TypeError as e:
            print("\n\n--- DIAGNOSTIC CATCH ---")
            print(f"Caught a TypeError: {e}")
            print(f"Type of train_losses: {type(train_losses)}")
            print(f"Value of train_losses: {train_losses}")
            print(f"Type of val_losses: {type(val_losses)}")
            print(f"Value of val_losses: {val_losses}")
            print("--- END DIAGNOSTIC CATCH ---\n\n")
            raise e # Re-raise the exception to stop the program

        # --- Visualization ---
        if (epoch + 1) % cfg['training']['val_freq'] == 0:
            print("\nRunning full validation and visualization...")
            for angle, batch in val_samples_by_angle.items():
                if batch is None: continue
                
                vis_data = get_full_image_prediction(model, batch, DEVICE, cfg)
                
                gt_img = vis_data["target_img"]
                pred_fine = vis_data["pred_fine"]
                pred_coarse = vis_data["pred_coarse"]

                if gt_img is not None and pred_coarse is not None:
                    pred_to_eval = pred_fine if model.use_fine_model and pred_fine is not None else pred_coarse
                    
                    vis_metrics = {'l1': 0.0, 'lpips': 0.0}
                    if loss_fns['l1']:
                        vis_metrics['l1'] = loss_fns['l1'](pred_to_eval, gt_img).item()
                    if loss_fns['lpips'] and pred_to_eval is not None:
                        vis_metrics['lpips'] = loss_fns['lpips'](pred_to_eval * 2 - 1, gt_img * 2 - 1).mean().item()
                    
                    log_data_vis = {f'vis/angle_{angle}_l1': vis_metrics['l1'], f'vis/angle_{angle}_lpips': vis_metrics['lpips']}
                    if cfg['wandb']['enabled']: wandb.log(log_data_vis)
                    print(f"  - Angle {angle} | L1: {vis_metrics['l1']:.4f} | LPIPS: {vis_metrics['lpips']:.4f}")

                    visualize_angle(epoch, cfg, fs_run_name, angle, vis_data, vis_metrics)

        # --- Checkpointing, Pruning & Early Stopping ---
        patience = cfg['training'].get('early_stopping_patience', 10)
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            torch.save(model.state_dict(), Path(cfg['training']['checkpoint_dir']) / f"{fs_run_name}_best.pth")
            print(f"  -> New best validation loss: {best_val_loss:.4f}. Checkpoint saved.")
        else:
            patience_counter += 1
            print(f"  -> Validation loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nStopping early. Validation loss has not improved for {patience} epochs.")
            break

        if is_optuna_run:
            trial.report(val_losses['total'] , epoch)
            if trial.should_prune():
                if cfg['wandb']['enabled']: wandb.finish()
                logger.close()
                raise optuna.TrialPruned()

    if cfg['wandb']['enabled']: wandb.finish()
    logger.close()
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="PixelNeRF Training Script with Optuna and WandB")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config file.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if cfg.get('optuna', {}).get('enabled', False):
        if 'backbone_name' not in cfg['optuna']['hyperparameters']:
             cfg['optuna']['hyperparameters']['backbone_name'] = {'choices': [cfg['model']['backbone']['name']]} # Add default backbone if not present

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args), n_trials=cfg['optuna']['n_trials'])
        print(f"\n--- Optuna Study Complete ---\nBest trial: {study.best_trial.number} | Value: {study.best_value:.5f}\nParams: {study.best_params}")
    else:
        class DummyTrial:
            def __init__(self, params):
                self.number = 0
                self.params = params
            def suggest_categorical(self, name, choices, **kwargs):
                if name == 'backbone_name': return self.params.get(name)
                return choices[0] # Default to first choice if not backbone_name
            def report(self, *args): pass
            def should_prune(self): return False

        default_params = {'backbone_name': cfg['model']['backbone']['name']}
        dummy_trial_instance = DummyTrial(default_params)
        print("Optuna is disabled. Running a single training session.")
        objective(dummy_trial_instance, args)

if __name__ == "__main__":
    if not all([WANDB_AVAILABLE, OPTUNA_AVAILABLE, TIMM_AVAILABLE, LPIPS_AVAILABLE]):
        print("Warning: One or more optional libraries (wandb, optuna, timm, lpips) not found.")
        print("Please run 'pip install wandb optuna timm lpips' to use all features.")
    main()
