"""
Bootstrap evaluation for MeCAlign V2 full model.

Purpose
-------
Load a trained MeCAlignV2 checkpoint and run bootstrap confidence intervals on
an input split, usually the test set.

evaluates the full MeCAlignV2 checkpoint.

Expected project structure
--------------------------
Run this from the same project folder where these imports work:
    from dataset.ewas_dataset import EWASCSVBridgeDataset
    from models.model_mecalign_v2 import MeCAlignV2

Example
-------
python bootstrap_mecalign_v2_full_model.py \
  --checkpoint <path of ckpt> \
  --input_csv <path of input data> \
  --output_dir <output path> \
  --n_bootstrap 1000 \
  --batch_size 64 \
  --stratified 1 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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
from torch.utils.data import DataLoader

from dataset.ewas_dataset import EWASCSVBridgeDataset
from models.model_mecalign_v2 import MeCAlignV2


# ============================================================
# Utilities
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


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    # weights_only=False is needed because the checkpoint stores sklearn scalers and Python metadata.
    return torch.load(path, map_location=device, weights_only=False)


def make_input_csv_if_label_missing(input_csv: str, output_dir: Path, label_col: str) -> Tuple[str, bool]:
    df = pd.read_csv(input_csv)
    label_exists = label_col in df.columns
    if label_exists:
        return input_csv, True
    tmp_csv = output_dir / "_input_with_dummy_label.csv"
    df[label_col] = 0.0
    df.to_csv(tmp_csv, index=False)
    return str(tmp_csv), False


def build_dataset_from_checkpoint(
    input_csv: str,
    ckpt: Dict[str, Any],
    output_dir: Path,
) -> Tuple[EWASCSVBridgeDataset, bool]:
    feature_info = ckpt["feature_info"]
    id_col = ckpt.get("id_col", "CaseNo")
    label_col = ckpt.get("label_col", "EWAS_label")
    csv_for_dataset, label_exists = make_input_csv_if_label_missing(input_csv, output_dir, label_col)

    ds = EWASCSVBridgeDataset(
        csv_file=csv_for_dataset,
        mode="test",
        scaler=ckpt["scaler"],
        cat_maps=ckpt["cat_maps"],
        id_col=id_col,
        label_col=label_col,
        cont_cols=feature_info["cont_cols"],
        cat_cols=feature_info["cat_cols"],
        cpg_cols=feature_info["probe_ids"],
        region_cols=feature_info["region_cols"],
    )
    return ds, label_exists


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device, strict: bool = True) -> MeCAlignV2:
    cfg = dict(ckpt["model_config"])

    training_only_keys = [
        "lambda_align",
        "align_loss_type",
        "lambda_barlow",
        "barlow_offdiag_weight",
        "lambda_gate_entropy",
        "lambda_router_balance",
        "lambda_adapter_diversity",
        "loss_weight_scale",
        "lr",
        "weight_decay",
        "grad_clip",
        "scheduler",
        "warmup_epochs",
        "eta_min",
        "use_pos_weight",
    ]
    for k in training_only_keys:
        cfg.pop(k, None)

    for k in list(cfg.keys()):
        if "\n" in str(k) or "# Loss" in str(k):
            cfg.pop(k, None)

    model = MeCAlignV2(**cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    model.to(device)
    model.eval()
    return model

# ============================================================
# Metrics
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


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    return {
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_rate": float(np.mean(y_true)),
    }


def bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
    stratified: bool,
) -> np.ndarray:
    n = len(y_true)
    if not stratified:
        return rng.integers(0, n, size=n)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return rng.integers(0, n, size=n)

    sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(idx)
    return idx


def run_bootstrap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
    stratified: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []

    for b in range(n_bootstrap):
        idx = bootstrap_indices(y_true, rng, stratified=stratified)
        metrics = binary_metrics_at_threshold(y_true[idx], y_prob[idx], threshold=threshold)
        metrics["bootstrap_id"] = b
        rows.append(metrics)

    boot_df = pd.DataFrame(rows)

    point = binary_metrics_at_threshold(y_true, y_prob, threshold=threshold)
    metric_cols = [
        "auroc", "auprc", "accuracy", "balanced_accuracy", "precision", "recall",
        "specificity", "npv", "f1", "positive_rate",
    ]

    summary_rows: List[Dict[str, float]] = []
    for metric in metric_cols:
        vals = boot_df[metric].to_numpy(dtype=float)
        vals_non_nan = vals[~np.isnan(vals)]
        if len(vals_non_nan) == 0:
            summary_rows.append({
                "metric": metric,
                "point": float(point.get(metric, float("nan"))),
                "bootstrap_mean": float("nan"),
                "bootstrap_std": float("nan"),
                "ci_low_2p5": float("nan"),
                "ci_high_97p5": float("nan"),
                "n_valid_bootstrap": 0,
            })
            continue

        summary_rows.append({
            "metric": metric,
            "point": float(point.get(metric, float("nan"))),
            "bootstrap_mean": float(np.mean(vals_non_nan)),
            "bootstrap_std": float(np.std(vals_non_nan, ddof=1)) if len(vals_non_nan) > 1 else 0.0,
            "ci_low_2p5": float(np.percentile(vals_non_nan, 2.5)),
            "ci_high_97p5": float(np.percentile(vals_non_nan, 97.5)),
            "n_valid_bootstrap": int(len(vals_non_nan)),
        })

    summary_df = pd.DataFrame(summary_rows)
    return boot_df, summary_df


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(
    model: MeCAlignV2,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    logits_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    model.eval()
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(
            batch["methy"],
            batch["region_global"],
            batch["clin_cont"],
            batch["clin_cat"],
            return_aux=False,
        )
        probs = torch.sigmoid(logits)
        logits_list.append(logits.detach().cpu().numpy())
        probs_list.append(probs.detach().cpu().numpy())
        labels_list.append(batch["label"].detach().cpu().numpy().astype(int))

    return {
        "logits": np.concatenate(logits_list),
        "probs": np.concatenate(probs_list),
        "labels": np.concatenate(labels_list),
    }


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=None,
                   help="Default: use checkpoint best_threshold. If set, override it.")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratified", type=str2bool_int, default=True,
                   help="If true, resample positives and negatives separately to preserve prevalence.")
    p.add_argument("--save_predictions", type=str2bool_int, default=True)
    p.add_argument("--strict_load", type=str2bool_int, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Output dir: {output_dir}")

    ckpt = load_checkpoint(args.checkpoint, device)
    model = build_model_from_checkpoint(ckpt, device, strict=args.strict_load)
    ds, label_exists = build_dataset_from_checkpoint(args.input_csv, ckpt, output_dir)
    if not label_exists:
        raise ValueError("Bootstrap evaluation needs true labels, but input CSV does not contain the label column.")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    threshold = args.threshold if args.threshold is not None else float(ckpt.get("best_threshold", 0.5))
    print(f"Using threshold = {threshold:.6f}")
    print(f"n_bootstrap = {args.n_bootstrap}, stratified = {bool(args.stratified)}")

    pred = predict(model, loader, device)
    y_true = pred["labels"].astype(int)
    y_prob = pred["probs"].astype(float)
    logits = pred["logits"].astype(float)

    if args.save_predictions:
        pred_df = pd.DataFrame({
            "sample_id": ds.sample_ids,
            "y_true": y_true,
            "logit": logits,
            "prob": y_prob,
            "pred": (y_prob >= threshold).astype(int),
            "threshold": threshold,
        })
        pred_df.to_csv(output_dir / "predictions.csv", index=False)

    boot_df, summary_df = run_bootstrap(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        stratified=bool(args.stratified),
    )

    boot_df.to_csv(output_dir / "bootstrap_metrics_samples.csv", index=False)
    summary_df.to_csv(output_dir / "bootstrap_ci_summary.csv", index=False)

    point_metrics = binary_metrics_at_threshold(y_true, y_prob, threshold=threshold)
    meta = {
        "checkpoint": str(args.checkpoint),
        "input_csv": str(args.input_csv),
        "threshold": float(threshold),
        "n_bootstrap": int(args.n_bootstrap),
        "stratified": bool(args.stratified),
        "n_samples": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "positive_prevalence": float(np.mean(y_true)),
        "point_metrics": point_metrics,
    }
    with open(output_dir / "bootstrap_ci_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "meta": meta,
            "summary": summary_df.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)

    print("\nPoint metrics:")
    print(json.dumps(point_metrics, ensure_ascii=False, indent=2))
    print("\nBootstrap 95% CI summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved bootstrap outputs to: {output_dir}")


if __name__ == "__main__":
    main()
