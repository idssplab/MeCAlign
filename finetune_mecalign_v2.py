"""
Random-search finetuning script for MeCAlign V2.

Goal
----
- use the original EWASCSVBridgeDataset.
- Randomly sample N hyperparameter settings from SEARCH_SPACE.
- For each setting, train MeCAlignV2 and select checkpoints by F1.

Expected project structure in Colab / Drive
-------------------------------------------
Your current code already uses:
    from dataset.ewas_dataset import EWASCSVBridgeDataset
    from models.model_mecalign_v2 import MeCAlignV2

So please make sure these exist before running:
    dataset/ewas_dataset.py
    models/model_mecalign_v2.py

Example
-------
python finetune_mecalign_v2.py \
  --train_csv <path of training set> \
  --valid_csv <path of valid set> \
  --test_csv  <path of test set> \
  --output_dir <output set> \
  --num_trials 30 --epochs 50 --batch_size 16 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from dataset.ewas_dataset import EWASCSVBridgeDataset
from models.model_mecalign_v2 import MeCAlignV2


# ============================================================
# 1) Search space for random search
# ============================================================

    
SEARCH_SPACE = {
    # Model size
    "embed_dim": [128],
    "heads": [1],
    "layers_per_block": [2],
    "dropout": {"type": "uniform", "low": 0.12, "high": 0.22},

    # Basic V1 components
    "probe_fusion_type": ["film"],
    "methy_transform": ["none"],
    "use_value_mlp": [False, True],
    "p_clin_mask": [0.05, 0.10],
    "p_methyl_mask": [0.00, 0.05],
    "use_probe_identity_embedding": [True],
    "clinical_feature_dropout": [0.00, 0.05, 0.10],
    "use_clin_cont_feature_tokens": [True],
    "use_region_tokens": [True],
    "disable_region_film": [False],
    "use_latent_reencoding": [True],
    "num_latent_tokens": [200],
    "use_cpg_gate": [True],
    "gate_tau": [0.7, 1.0, 1.5],
    "use_alignment_tokens": [True],
    "num_alignment_tokens": [4, 8, 12],

    # V2 routed adapters
    "use_routed_adapters": [True],
    "adapter_layers": ["last"],
    "num_adapter_experts": [3],
    "adapter_rank": [8, 12, 16],
    "adapter_scale": [0.075, 0.10, 0.125, 0.15],
    "adapter_scale_learnable": [True],
    "router_tau": [0.7, 1.0],

    # Loss knobs
    "lambda_align": [0.0, 0.001, 0.002],
    "align_loss_type": ["cosine"],

    # Barlow-style anti-collapse cross-modal alignment.
    # Start small. Too large may hurt classification.
    "lambda_barlow": [0.0, 0.0005, 0.001, 0.002, 0.005],
    "barlow_offdiag_weight": [0.002, 0.005, 0.01],

    "lambda_gate_entropy": [0.0],
    "lambda_router_balance": [0.0, 0.0003, 0.0005, 0.001],
    "lambda_adapter_diversity": [0.0, 0.0001, 0.0003],
    "loss_weight_scale": [0.4, 0.5, 0.6, 0.75],

    # Optim
    "lr": {"type": "loguniform", "low": 4.5e-4, "high": 1.2e-4},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 8e-5},
    "grad_clip": [0.5, 1.0],
    "scheduler": ["cosine"],
    "warmup_epochs": [3, 5],
    "eta_min": [1e-6],
}

# ============================================================
# 2) Utilities
# ============================================================

def str2bool_int(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return bool(x)
    x = str(x).strip().lower()
    if x in ["1", "true", "t", "yes", "y"]:
        return True
    if x in ["0", "false", "f", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {x}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def get_cat_cardinalities(ds: EWASCSVBridgeDataset) -> List[int]:
    return [len(ds.cat_maps[col]) for col in ds.cat_cols]


def build_datasets(args: argparse.Namespace):
    train_ds = EWASCSVBridgeDataset(
        csv_file=args.train_csv,
        mode="train",
        id_col=args.id_col,
        label_col=args.label_col,
    )
    valid_ds = EWASCSVBridgeDataset(
        csv_file=args.valid_csv,
        mode="valid",
        scaler=train_ds.scaler,
        cat_maps=train_ds.cat_maps,
        id_col=args.id_col,
        label_col=args.label_col,
        cont_cols=train_ds.cont_cols,
        cat_cols=train_ds.cat_cols,
        cpg_cols=train_ds.probe_ids,
        region_cols=train_ds.region_cols,
    )
    test_ds = None
    if args.test_csv:
        test_ds = EWASCSVBridgeDataset(
            csv_file=args.test_csv,
            mode="test",
            scaler=train_ds.scaler,
            cat_maps=train_ds.cat_maps,
            id_col=args.id_col,
            label_col=args.label_col,
            cont_cols=train_ds.cont_cols,
            cat_cols=train_ds.cat_cols,
            cpg_cols=train_ds.probe_ids,
            region_cols=train_ds.region_cols,
        )
    return train_ds, valid_ds, test_ds


def feature_info_from_dataset(train_ds: EWASCSVBridgeDataset) -> Dict[str, Any]:
    return {
        "probe_ids": list(train_ds.probe_ids),
        "region_cols": list(train_ds.region_cols),
        "cont_cols": list(train_ds.cont_cols),
        "cat_cols": list(train_ds.cat_cols),
        "cat_cardinalities": get_cat_cardinalities(train_ds),
        "num_probes": len(train_ds.probe_ids),
        "num_region_global": len(train_ds.region_cols),
        "num_clin_cont": len(train_ds.cont_cols),
        "num_clin_cat": len(train_ds.cat_cols),
    }


def sample_one_value(spec: Any, rng: random.Random) -> Any:
    if isinstance(spec, list):
        return rng.choice(spec)
    if isinstance(spec, tuple):
        return rng.choice(list(spec))
    if isinstance(spec, dict):
        kind = spec.get("type", "choice")
        if kind == "uniform":
            return rng.uniform(float(spec["low"]), float(spec["high"]))
        if kind == "loguniform":
            low = math.log(float(spec["low"]))
            high = math.log(float(spec["high"]))
            return math.exp(rng.uniform(low, high))
        if kind == "int":
            return rng.randint(int(spec["low"]), int(spec["high"]))
        if kind == "bool":
            return bool(rng.randint(0, 1))
        if kind == "choice":
            return rng.choice(spec["values"])
        raise ValueError(f"Unsupported search space spec: {spec}")
    return spec


def sample_config(search_space: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    cfg = {k: sample_one_value(v, rng) for k, v in search_space.items()}

    # Safety rules for small models.
    if int(cfg.get("layers_per_block", 2)) <= 2 and cfg.get("adapter_layers") == "last2":
        cfg["adapter_layers"] = "last"

    # heads must divide embed_dim.
    if int(cfg["embed_dim"]) % int(cfg["heads"]) != 0:
        cfg["heads"] = 1

    return cfg


def build_model_config(trial_cfg: Dict[str, Any], feature_info: Dict[str, Any]) -> Dict[str, Any]:
    model_keys = [
        "embed_dim", "layers_per_block", "heads", "dropout", "probe_fusion_type",
        "methy_transform", "use_value_mlp", "p_clin_mask", "p_methyl_mask",
        "use_probe_identity_embedding", "clinical_feature_dropout",
        "use_clin_cont_feature_tokens", "use_region_tokens", "disable_region_film",
        "use_latent_reencoding", "num_latent_tokens", "use_cpg_gate", "gate_tau",
        "use_alignment_tokens", "num_alignment_tokens",
        "use_routed_adapters", "adapter_layers", "num_adapter_experts",
        "adapter_rank", "adapter_scale", "adapter_scale_learnable", "router_tau",
    ]
    cfg = {k: trial_cfg[k] for k in model_keys}
    cfg.update({
        "num_probes": feature_info["num_probes"],
        "num_region_global": feature_info["num_region_global"],
        "num_clin_cont": feature_info["num_clin_cont"],
        "num_clin_cat": feature_info["num_clin_cat"],
        "cat_cardinalities": feature_info["cat_cardinalities"],
    })
    return cfg


def build_model(trial_cfg: Dict[str, Any], feature_info: Dict[str, Any]) -> MeCAlignV2:
    model_config = build_model_config(trial_cfg, feature_info)
    return MeCAlignV2(**model_config)


def make_optimizer(trial_cfg: Dict[str, Any], model: nn.Module):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(trial_cfg["lr"]),
        weight_decay=float(trial_cfg["weight_decay"]),
    )


def make_scheduler(trial_cfg: Dict[str, Any], optimizer: torch.optim.Optimizer, epochs: int):
    scheduler = trial_cfg.get("scheduler", "cosine")
    if scheduler == "none":
        return None
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - int(trial_cfg.get("warmup_epochs", 0))),
            eta_min=float(trial_cfg.get("eta_min", 1e-6)),
        )
    if scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(trial_cfg.get("step_size", 10)),
            gamma=float(trial_cfg.get("step_gamma", 0.5)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# ============================================================
# 3) Losses
# ============================================================
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Return the off-diagonal elements of a square matrix.
    """
    n, m = x.shape
    if n != m:
        raise ValueError(f"off_diagonal expects a square matrix, got {x.shape}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    offdiag_weight: float = 0.005,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Barlow Twins-style cross-modal alignment loss.

    z1, z2: [B, D]
      Usually use unnormalized projected embeddings:
        aux["methyl_align_h"], aux["clinical_align_h"]

    Goal:
      1. On-diagonal cross-correlation close to 1
      2. Off-diagonal cross-correlation close to 0
    This avoids simple representation collapse where all samples point to the same direction.
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(f"barlow_twins_loss expects [B,D], got {z1.shape}, {z2.shape}")
    if z1.shape != z2.shape:
        raise ValueError(f"barlow_twins_loss expects same shape, got {z1.shape}, {z2.shape}")

    batch_size, dim = z1.shape

    # Barlow-style statistics are unstable for very small batches.
    # If batch too small, return zero rather than injecting noisy gradients.
    if batch_size < 2:
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

    # Normalize each dimension across batch.
    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0, unbiased=False) + eps)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0, unbiased=False) + eps)

    # Cross-correlation matrix: [D, D]
    c = torch.mm(z1.T, z2) / batch_size

    on_diag = torch.diagonal(c).add(-1.0).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()

    return on_diag + float(offdiag_weight) * off_diag

def alignment_loss_from_aux(aux: Dict[str, Any], loss_type: str = "cosine") -> torch.Tensor:
    device = aux["methyl_pool"].device
    zero = torch.tensor(0.0, device=device)
    if loss_type == "none":
        return zero
    z_m = aux["methyl_align_z"]
    z_c = aux["clinical_align_z"]
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(z_m, z_c, dim=-1)).mean()
    if loss_type == "mse":
        return F.mse_loss(z_m, z_c)
    raise ValueError(f"Unsupported align_loss_type: {loss_type}")


def gate_entropy_loss_from_aux(aux: Dict[str, Any]) -> torch.Tensor:
    gate = aux["cpg_gate"]
    prob = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1).mean()
    return entropy / math.log(gate.size(-1))


def router_balance_loss_from_aux(aux: Dict[str, Any]) -> torch.Tensor:
    # Minimize 1 - normalized entropy of batch-average expert usage.
    # This weakly discourages all samples from selecting one expert.
    layers = aux.get("router_alpha_layers", [])
    if not layers:
        return torch.tensor(0.0, device=aux["methyl_pool"].device)
    losses = []
    for alpha in layers:  # [B, K]
        p = alpha.mean(dim=0)
        k = p.numel()
        ent = -(p * torch.log(p.clamp_min(1e-8))).sum() / math.log(k)
        losses.append(1.0 - ent)
    return torch.stack(losses).mean()


def adapter_diversity_loss_from_aux(aux: Dict[str, Any]) -> torch.Tensor:
    # Output-level diversity: discourage expert outputs from being identical.
    # For each layer, expert_outputs: [B, K, D]. Penalize off-diagonal cosine^2.
    outs = aux.get("adapter_expert_outputs", [])
    if not outs:
        return torch.tensor(0.0, device=aux["methyl_pool"].device)
    losses = []
    for z in outs:
        z = F.normalize(z, dim=-1)  # [B, K, D]
        sim = torch.matmul(z, z.transpose(1, 2))  # [B, K, K]
        k = sim.size(-1)
        if k <= 1:
            continue
        # Boolean indexing does not broadcast a [1,K,K] mask to [B,K,K].
        # Use a [K,K] off-diagonal mask and index the last two dims for every batch.
        off_diag = ~torch.eye(k, dtype=torch.bool, device=sim.device)  # [K, K]
        losses.append((sim[:, off_diag] ** 2).mean())  # [B, K*(K-1)]
    if not losses:
        return torch.tensor(0.0, device=aux["methyl_pool"].device)
    return torch.stack(losses).mean()

def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    aux: Dict[str, Any],
    criterion: nn.Module,
    trial_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = criterion(logits, labels.float())

    lambda_align = float(trial_cfg.get("lambda_align", 0.0))
    lambda_barlow = float(trial_cfg.get("lambda_barlow", 0.0))
    barlow_offdiag_weight = float(trial_cfg.get("barlow_offdiag_weight", 0.005))

    lambda_gate_entropy = float(trial_cfg.get("lambda_gate_entropy", 0.0))
    lambda_router_balance = float(trial_cfg.get("lambda_router_balance", 0.0))
    lambda_adapter_diversity = float(trial_cfg.get("lambda_adapter_diversity", 0.0))

    device = logits.device
    zero = torch.tensor(0.0, device=device)

    align_loss = (
        alignment_loss_from_aux(aux, str(trial_cfg.get("align_loss_type", "cosine")))
        if lambda_align > 0
        else zero
    )

    # Barlow-style anti-collapse alignment loss.
    # Prefer raw projected embeddings. Fall back to normalized z for old model compatibility.
    if lambda_barlow > 0:
        z_m = aux.get("methyl_align_h", aux["methyl_align_z"])
        z_c = aux.get("clinical_align_h", aux["clinical_align_z"])
        barlow_loss = barlow_twins_loss(
            z_m,
            z_c,
            offdiag_weight=barlow_offdiag_weight,
        )
    else:
        barlow_loss = zero

    gate_ent = gate_entropy_loss_from_aux(aux) if lambda_gate_entropy > 0 else zero
    router_bal = router_balance_loss_from_aux(aux) if lambda_router_balance > 0 else zero
    adapter_div = adapter_diversity_loss_from_aux(aux) if lambda_adapter_diversity > 0 else zero

    total = (
        bce
        + lambda_align * align_loss
        + lambda_barlow * barlow_loss
        + lambda_gate_entropy * gate_ent
        + lambda_router_balance * router_bal
        + lambda_adapter_diversity * adapter_div
    )

    logs = {
        "loss": float(total.detach().cpu()),
        "bce_loss": float(bce.detach().cpu()),
        "align_loss": float(align_loss.detach().cpu()),
        "barlow_loss": float(barlow_loss.detach().cpu()),
        "gate_entropy_loss": float(gate_ent.detach().cpu()),
        "router_balance_loss": float(router_bal.detach().cpu()),
        "adapter_diversity_loss": float(adapter_div.detach().cpu()),
    }
    return total, logs

# ============================================================
# 4) Metrics
# ============================================================

def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def find_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    prefix: str,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_specificity": float(specificity),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tp": int(tp),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split: str,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], float]:
    out = {
        f"{split}_auroc": safe_auroc(y_true, y_prob),
        f"{split}_auprc": safe_auprc(y_true, y_prob),
    }
    out.update(binary_metrics_at_threshold(y_true, y_prob, 0.5, f"{split}_thr05"))
    if threshold is None:
        best_thr, _ = find_best_threshold_by_f1(y_true, y_prob)
    else:
        best_thr = float(threshold)
    out.update(binary_metrics_at_threshold(y_true, y_prob, best_thr, f"{split}_f1_threshold"))
    return out, best_thr


# ============================================================
# 5) Train / eval
# ============================================================

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    trial_cfg: Dict[str, Any],
    device: torch.device,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Dict[str, float]:
    model.train()
    total = 0
    '''
    sums = {
        "loss": 0.0,
        "bce_loss": 0.0,
        "align_loss": 0.0,
        "gate_entropy_loss": 0.0,
        "router_balance_loss": 0.0,
        "adapter_diversity_loss": 0.0,
    }
    '''
    sums = {
    "loss": 0.0,
    "bce_loss": 0.0,
    "align_loss": 0.0,
    "barlow_loss": 0.0,
    "gate_entropy_loss": 0.0,
    "router_balance_loss": 0.0,
    "adapter_diversity_loss": 0.0,
    }
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch["label"].float()
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits, aux = model(
                    batch["methy"], batch["region_global"], batch["clin_cont"], batch["clin_cat"],
                    return_aux=True,
                )
                loss, logs = compute_total_loss(logits, labels, aux, criterion, trial_cfg)
            assert scaler is not None
            scaler.scale(loss).backward()
            grad_clip = float(trial_cfg.get("grad_clip", 1.0))
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, aux = model(
                batch["methy"], batch["region_global"], batch["clin_cont"], batch["clin_cat"],
                return_aux=True,
            )
            loss, logs = compute_total_loss(logits, labels, aux, criterion, trial_cfg)
            loss.backward()
            grad_clip = float(trial_cfg.get("grad_clip", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = labels.size(0)
        total += bs
        for k in sums:
            sums[k] += logs[k] * bs

    return {f"train_{k}": v / max(1, total) for k, v in sums.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    trial_cfg: Dict[str, Any],
    device: torch.device,
    split: str,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    total = 0
    '''
    sums = {
        "loss": 0.0,
        "bce_loss": 0.0,
        "align_loss": 0.0,
        "gate_entropy_loss": 0.0,
        "router_balance_loss": 0.0,
        "adapter_diversity_loss": 0.0,
    }
    '''
    sums = {
    "loss": 0.0,
    "bce_loss": 0.0,
    "align_loss": 0.0,
    "barlow_loss": 0.0,
    "gate_entropy_loss": 0.0,
    "router_balance_loss": 0.0,
    "adapter_diversity_loss": 0.0,
    }
    y_true_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []

    for batch in loader:
        batch = to_device(batch, device)
        labels = batch["label"].float()
        logits, aux = model(
            batch["methy"], batch["region_global"], batch["clin_cont"], batch["clin_cat"],
            return_aux=True,
        )
        loss, logs = compute_total_loss(logits, labels, aux, criterion, trial_cfg)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy().astype(int)

        bs = labels.size(0)
        total += bs
        for k in sums:
            sums[k] += logs[k] * bs
        y_true_list.append(y_true)
        y_prob_list.append(probs)

    y_true_all = np.concatenate(y_true_list)
    y_prob_all = np.concatenate(y_prob_list)
    loss_metrics = {f"{split}_{k}": v / max(1, total) for k, v in sums.items()}
    metric_dict, chosen_thr = compute_metrics(y_true_all, y_prob_all, split, threshold=threshold)
    metric_dict.update(loss_metrics)
    return metric_dict, chosen_thr


def make_criterion(args: argparse.Namespace, train_ds: EWASCSVBridgeDataset, trial_cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    y_train = np.asarray(train_ds.labels).astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    base_pos_weight = n_neg / max(1, n_pos)

    if args.use_pos_weight and n_pos > 0:
        scale = float(trial_cfg.get("loss_weight_scale", 1.0))
        pos_weight_value = 1.0 + scale * (base_pos_weight - 1.0)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    else:
        pos_weight_value = 1.0
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    info = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_pos_weight": float(base_pos_weight),
        "effective_pos_weight": float(pos_weight_value),
    }
    return criterion, info


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    trial_id: int,
    trial_cfg: Dict[str, Any],
    model_config: Dict[str, Any],
    feature_info: Dict[str, Any],
    train_ds: EWASCSVBridgeDataset,
    threshold: float,
    metrics: Dict[str, Any],
    selection_note: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "trial_id": int(trial_id),
        "selection_note": selection_note,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "trial_config": copy.deepcopy(trial_cfg),
        "model_config": copy.deepcopy(model_config),
        "feature_info": copy.deepcopy(feature_info),
        "scaler": train_ds.scaler,
        "cat_maps": train_ds.cat_maps,
        "id_col": args.id_col,
        "label_col": args.label_col,
        "best_threshold": float(threshold),
        "metrics": copy.deepcopy(metrics),
    }


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def row_with_config(trial_id: int, trial_cfg: Dict[str, Any], metrics: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    row = {"trial_id": trial_id}
    row.update(extra)
    for k, v in trial_cfg.items():
        row[f"cfg_{k}"] = v
    row.update(metrics)
    return row


def save_result_tables(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)

    # Main table: all settings, sorted by validation AUROC.
    if "valid_auroc" in df.columns:
        df_main = df.sort_values("valid_auroc", ascending=False, na_position="last")
    else:
        df_main = df
    df_main.to_csv(output_dir / "all_settings_performance_sorted_by_valid_auroc.csv", index=False)

    # Top-15 by valid AUROC.
    if "valid_auroc" in df.columns:
        df.sort_values("valid_auroc", ascending=False, na_position="last").head(15).to_csv(
            output_dir / "valid_auroc_top15_settings.csv", index=False
        )

    # Top-15 by test AUROC if test exists.
    if "test_auroc" in df.columns:
        df.sort_values("test_auroc", ascending=False, na_position="last").head(15).to_csv(
            output_dir / "test_auroc_top15_settings.csv", index=False
        )


# ============================================================
# 6) Main random-search loop
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--valid_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--id_col", type=str, default="CaseNo")
    p.add_argument("--label_col", type=str, default="EWAS_label")

    # Random search
    p.add_argument("--num_trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_amp", type=str2bool_int, default=False)
    p.add_argument("--use_pos_weight", type=str2bool_int, default=True)

    # Checkpoint selection: fixed to F1 as requested.
    p.add_argument("--monitor_valid", type=str, default="valid_f1_threshold_f1")
    p.add_argument("--monitor_test", type=str, default="test_f1_threshold_f1")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    output_dir = ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print("Only global best-valid-F1 and best-test-F1 checkpoints will be saved.")

    train_ds, valid_ds, test_ds = build_datasets(args)
    feature_info = feature_info_from_dataset(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    global_best_valid_f1 = -float("inf")
    global_best_test_f1 = -float("inf")
    all_rows: List[Dict[str, Any]] = []
    sampled_keys = set()

    for trial_id in range(1, args.num_trials + 1):
        # Sample unique-ish configs.
        for _ in range(200):
            trial_cfg = sample_config(SEARCH_SPACE, rng)
            key = json.dumps(trial_cfg, sort_keys=True, default=str)
            if key not in sampled_keys:
                sampled_keys.add(key)
                break
        else:
            trial_cfg = sample_config(SEARCH_SPACE, rng)

        trial_seed = args.seed + trial_id * 1009
        set_seed(trial_seed)

        print("\n" + "=" * 90)
        print(f"Trial {trial_id:03d}/{args.num_trials} | seed={trial_seed}")
        print(json.dumps(trial_cfg, ensure_ascii=False, indent=2))

        model_config = build_model_config(trial_cfg, feature_info)
        model = MeCAlignV2(**model_config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        criterion, weight_info = make_criterion(args, train_ds, trial_cfg, device)
        optimizer = make_optimizer(trial_cfg, model)
        scheduler = make_scheduler(trial_cfg, optimizer, args.epochs)
        amp_scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        best_valid_f1_this_trial = -float("inf")
        best_test_f1_this_trial = -float("inf")
        best_row_this_trial: Optional[Dict[str, Any]] = None
        best_valid_epoch_this_trial = -1
        best_test_epoch_this_trial = -1

        for epoch in range(1, args.epochs + 1):
            # Warmup LR.
            warmup_epochs = int(trial_cfg.get("warmup_epochs", 0))
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                warmup_lr = float(trial_cfg["lr"]) * epoch / max(1, warmup_epochs)
                set_lr(optimizer, warmup_lr)

            train_logs = run_one_epoch(
                model, train_loader, optimizer, criterion, trial_cfg, device,
                use_amp=args.use_amp, scaler=amp_scaler,
            )

            if scheduler is not None and epoch > warmup_epochs:
                scheduler.step()

            valid_metrics, valid_best_thr = evaluate(
                model, valid_loader, criterion, trial_cfg, device,
                split="valid", threshold=None,
            )

            row = {
                "epoch": epoch,
                "lr_current": get_lr(optimizer),
                **weight_info,
                **train_logs,
                **valid_metrics,
            }

            if test_loader is not None:
                # Test uses the validation-selected threshold, same style as your V1 code.
                test_metrics, _ = evaluate(
                    model, test_loader, criterion, trial_cfg, device,
                    split="test", threshold=valid_best_thr,
                )
                row.update(test_metrics)

            valid_f1 = float(row.get(args.monitor_valid, float("nan")))
            test_f1 = float(row.get(args.monitor_test, float("nan")))

            # Best valid-F1 within this trial.
            if not math.isnan(valid_f1) and valid_f1 > best_valid_f1_this_trial:
                best_valid_f1_this_trial = valid_f1
                best_valid_epoch_this_trial = epoch
                best_row_this_trial = copy.deepcopy(row)

            # Save global best validation-F1 checkpoint only.
            if not math.isnan(valid_f1) and valid_f1 > global_best_valid_f1:
                global_best_valid_f1 = valid_f1
                payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    trial_id=trial_id,
                    trial_cfg=trial_cfg,
                    model_config=model_config,
                    feature_info=feature_info,
                    train_ds=train_ds,
                    threshold=float(valid_best_thr),
                    metrics=row,
                    selection_note="GLOBAL_BEST_VALID_F1",
                    args=args,
                )
                save_checkpoint(output_dir / "best_valid_f1.pt", payload)

            # Save global best test-F1 checkpoint only. This is analysis only.
            if test_loader is not None and not math.isnan(test_f1) and test_f1 > best_test_f1_this_trial:
                best_test_f1_this_trial = test_f1
                best_test_epoch_this_trial = epoch

            if test_loader is not None and not math.isnan(test_f1) and test_f1 > global_best_test_f1:
                global_best_test_f1 = test_f1
                payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    trial_id=trial_id,
                    trial_cfg=trial_cfg,
                    model_config=model_config,
                    feature_info=feature_info,
                    train_ds=train_ds,
                    threshold=float(valid_best_thr),
                    metrics=row,
                    selection_note="GLOBAL_BEST_TEST_F1_ANALYSIS_ONLY",
                    args=args,
                )
                save_checkpoint(output_dir / "best_test_f1_ANALYSIS_ONLY.pt", payload)

            print(
                f"[T{trial_id:03d} E{epoch:03d}] "
                f"loss={row.get('train_loss', float('nan')):.4f} "
                f"valid_auc={row.get('valid_auroc', float('nan')):.4f} "
                f"valid_auprc={row.get('valid_auprc', float('nan')):.4f} "
                f"valid_f1={row.get('valid_f1_threshold_f1', float('nan')):.4f} "
                f"test_auc={row.get('test_auroc', float('nan')):.4f} "
                f"test_f1={row.get('test_f1_threshold_f1', float('nan')):.4f} "
                f"thr={valid_best_thr:.2f}"
            )

        # Store one summary row per setting: metrics from this trial's best-valid-F1 epoch.
        if best_row_this_trial is None:
            best_row_this_trial = {"epoch": -1}

        extra = {
            "selected_by": "valid_f1_threshold_f1",
            "best_valid_f1_epoch": int(best_valid_epoch_this_trial),
            "best_valid_f1_this_trial": float(best_valid_f1_this_trial),
            "best_test_f1_epoch_this_trial": int(best_test_epoch_this_trial),
            "best_test_f1_this_trial": float(best_test_f1_this_trial) if test_loader is not None else float("nan"),
            "n_params": int(n_params),
            "trial_seed": int(trial_seed),
        }
        all_rows.append(row_with_config(trial_id, trial_cfg, best_row_this_trial, extra))

        # Overwrite only the allowed CSV outputs after each trial.
        save_result_tables(all_rows, output_dir)

        # Free GPU memory between settings.
        del model, optimizer, scheduler, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nRandom search finished.")
    print(f"Global best valid F1: {global_best_valid_f1:.6f}")
    if test_loader is not None:
        print(f"Global best test F1 ANALYSIS_ONLY: {global_best_test_f1:.6f}")
    print("Saved files:")
    for name in [
        "best_valid_f1.pt",
        "best_test_f1_ANALYSIS_ONLY.pt",
        "all_settings_performance_sorted_by_valid_auroc.csv",
        "valid_auroc_top15_settings.csv",
        "test_auroc_top15_settings.csv",
    ]:
        path = output_dir / name
        if path.exists():
            print(f"  - {path}")


if __name__ == "__main__":
    main()
