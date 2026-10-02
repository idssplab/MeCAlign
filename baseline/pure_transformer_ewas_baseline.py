import os
import json
import math
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_comma_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or str(value).strip() == "":
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def excel_col_to_index(col: str) -> int:
    """
    Convert Excel column name to zero-based index.
    Examples:
      A  -> 0
      H  -> 7
      GY -> 206
    """
    col = col.strip().upper()
    idx = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid Excel column letter: {col}")
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_float(x):
    try:
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return x
    except Exception:
        return x


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


# ============================================================
# Metrics and threshold selection
# ============================================================

def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        return float(auc(recall_vals, precision_vals))
    except Exception:
        return float("nan")


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    metrics["threshold"] = float(threshold)
    metrics["auroc"] = safe_auc(y_true, y_prob)
    metrics["auprc"] = safe_auprc(y_true, y_prob)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    try:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
            labels=[0, 1],
        )
        metrics["precision"] = float(prec[1])
        metrics["recall"] = float(rec[1])
        metrics["specificity"] = float(rec[0])
        metrics["f1"] = float(f1[1])
    except Exception:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["specificity"] = 0.0
        metrics["f1"] = 0.0

    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    metrics["num_negative"] = int((y_true == 0).sum())
    metrics["num_positive"] = int((y_true == 1).sum())
    metrics["positive_rate"] = float((y_true == 1).mean())
    return metrics


def find_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 1001) -> Dict[str, float]:
    """
    Select thresholds on validation set only.
    Returns F1-max threshold and Youden's J threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_f1 = -1.0
    best_f1_thresh = 0.5
    best_j = -1.0
    best_j_thresh = 0.5

    thresholds = np.linspace(0.0, 1.0, grid_size)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        try:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average=None,
                zero_division=0,
                labels=[0, 1],
            )
            pos_f1 = float(f1[1])
            sensitivity = float(rec[1])
            specificity = float(rec[0])
            j_score = sensitivity + specificity - 1.0
        except Exception:
            pos_f1 = 0.0
            j_score = -1.0

        if pos_f1 > best_f1:
            best_f1 = pos_f1
            best_f1_thresh = float(t)

        if j_score > best_j:
            best_j = j_score
            best_j_thresh = float(t)

    return {
        "valid_f1_thresh": float(best_f1_thresh),
        "valid_f1_at_thresh": float(best_f1),
        "valid_j_thresh": float(best_j_thresh),
        "valid_j_score": float(best_j),
    }


def print_metrics_block(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Threshold:          {metrics['threshold']:.4f}")
    print(f"AUROC:              {metrics['auroc']:.4f}")
    print(f"AUPRC:              {metrics['auprc']:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall/Sensitivity: {metrics['recall']:.4f}")
    print(f"Specificity:        {metrics['specificity']:.4f}")
    print(f"F1 Score:           {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]))
    print(f"\nClass Distribution: Negative={metrics['num_negative']}, Positive={metrics['num_positive']}")


# ============================================================
# Column resolution
# ============================================================

def resolve_columns(args, train_df: pd.DataFrame) -> Dict[str, List[str]]:
    columns = list(train_df.columns)

    if args.label_col not in train_df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found.")

    # Categorical clinical columns
    if args.cat_cols:
        cat_cols = parse_comma_list(args.cat_cols)
        cat_cols = [c for c in cat_cols if c in train_df.columns]
    else:
        default_cat = ["SEX","smoking","DRK","betel","SPORT","HTN_FAM","edu"]#["SEX", "smoking", "DRK", "betel", "SPORT", "HTN_FAM"]
        cat_cols = [c for c in default_cat if c in train_df.columns]

    # Region-global columns
    if args.region_cols:
        region_cols = parse_comma_list(args.region_cols)
        region_cols = [c for c in region_cols if c in train_df.columns]
    else:
        default_region = [f"ISLAND_{i}" for i in range(1, 6)]
        region_cols = [c for c in default_region if c in train_df.columns]

    # CpG methylation columns
    if args.cpg_cols:
        cpg_cols = parse_comma_list(args.cpg_cols)
        missing = [c for c in cpg_cols if c not in train_df.columns]
        if missing:
            raise ValueError(f"Some cpg_cols are not in train CSV: {missing[:10]}")
    elif args.cpg_start_excel and args.cpg_end_excel:
        start_idx = excel_col_to_index(args.cpg_start_excel)
        end_idx = excel_col_to_index(args.cpg_end_excel)
        if start_idx > end_idx:
            raise ValueError("cpg_start_excel must be before cpg_end_excel.")
        if end_idx >= len(columns):
            raise ValueError(
                f"Excel end column {args.cpg_end_excel} index={end_idx} exceeds CSV width={len(columns)}."
            )
        cpg_cols = columns[start_idx:end_idx + 1]
    else:
        cpg_cols = [c for c in columns if str(c).startswith("cg")]
        if len(cpg_cols) == 0:
            raise ValueError(
                "No CpG columns found. Either use columns starting with 'cg', "
                "or specify --cpg_start_excel H --cpg_end_excel GY, "
                "or provide --cpg_cols col1,col2,..."
            )

    # Continuous clinical columns
    if args.cont_cols:
        cont_cols = parse_comma_list(args.cont_cols)
        cont_cols = [c for c in cont_cols if c in train_df.columns]
    else:
        exclude = set([args.id_col, args.label_col] + cpg_cols + region_cols + cat_cols)
        cont_cols = [
            c for c in columns
            if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])
        ]

    def unique_keep_order(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {
        "cpg_cols": unique_keep_order(cpg_cols),
        "region_cols": unique_keep_order(region_cols),
        "cont_cols": unique_keep_order(cont_cols),
        "cat_cols": unique_keep_order(cat_cols),
    }


# ============================================================
# Preprocessing and Dataset
# ============================================================

class TransformerPreprocessor:
    """
    Fit preprocessing on train set only, then transform train/valid/test.

    Numeric groups:
      - CpG methylation
      - region-global features
      - continuous clinical features

    Categorical groups:
      - clinical categorical features, mapped to integer IDs
    """
    def __init__(
        self,
        cpg_cols: List[str],
        region_cols: List[str],
        cont_cols: List[str],
        cat_cols: List[str],
        scale_numeric: bool = True,
    ):
        self.cpg_cols = list(cpg_cols)
        self.region_cols = list(region_cols)
        self.cont_cols = list(cont_cols)
        self.cat_cols = list(cat_cols)
        self.scale_numeric = scale_numeric

        self.imputers: Dict[str, Optional[SimpleImputer]] = {}
        self.scalers: Dict[str, Optional[StandardScaler]] = {}
        self.cat_maps: Dict[str, Dict[str, int]] = {}

    def _fit_numeric_group(self, df: pd.DataFrame, cols: List[str], group_name: str) -> None:
        if len(cols) == 0:
            self.imputers[group_name] = None
            self.scalers[group_name] = None
            return
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        self.imputers[group_name] = imputer

        if self.scale_numeric:
            scaler = StandardScaler()
            scaler.fit(X_imp)
            self.scalers[group_name] = scaler
        else:
            self.scalers[group_name] = None

    def _transform_numeric_group(self, df: pd.DataFrame, cols: List[str], group_name: str) -> np.ndarray:
        if len(cols) == 0:
            return np.zeros((len(df), 0), dtype=np.float32)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {group_name}: {missing[:10]}")
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        imputer = self.imputers[group_name]
        X_imp = imputer.transform(X)
        scaler = self.scalers[group_name]
        if scaler is not None:
            X_imp = scaler.transform(X_imp)
        return X_imp.astype(np.float32)

    def fit(self, df: pd.DataFrame) -> None:
        self._fit_numeric_group(df, self.cpg_cols, "methylation")
        self._fit_numeric_group(df, self.region_cols, "region")
        self._fit_numeric_group(df, self.cont_cols, "continuous")

        for col in self.cat_cols:
            vals = df[col].fillna("__UNK__").astype(str).values
            uniq = ["__UNK__"] + sorted([v for v in pd.unique(vals) if v != "__UNK__"])
            self.cat_maps[col] = {v: i for i, v in enumerate(uniq)}

    def transform(self, df: pd.DataFrame, label_col: str, id_col: str) -> Dict[str, np.ndarray]:
        methy = self._transform_numeric_group(df, self.cpg_cols, "methylation")
        region = self._transform_numeric_group(df, self.region_cols, "region")
        cont = self._transform_numeric_group(df, self.cont_cols, "continuous")

        cat_arrays = []
        for col in self.cat_cols:
            vals = df[col].fillna("__UNK__").astype(str).values
            mapping = self.cat_maps[col]
            cat_arrays.append(np.array([mapping.get(v, 0) for v in vals], dtype=np.int64))
        if len(cat_arrays) > 0:
            cat = np.vstack(cat_arrays).T.astype(np.int64)
        else:
            cat = np.zeros((len(df), 0), dtype=np.int64)

        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV.")
        labels = df[label_col].astype(np.float32).values

        if id_col in df.columns:
            ids = df[id_col].astype(str).values
        else:
            ids = np.array([str(i) for i in range(len(df))])

        return {
            "methy": methy,
            "region": region,
            "cont": cont,
            "cat": cat,
            "label": labels,
            "ids": ids,
        }

    def get_cat_cardinalities(self) -> List[int]:
        return [len(self.cat_maps[col]) for col in self.cat_cols]

    def to_metadata(self) -> Dict[str, object]:
        return {
            "cpg_cols": self.cpg_cols,
            "region_cols": self.region_cols,
            "cont_cols": self.cont_cols,
            "cat_cols": self.cat_cols,
            "cat_cardinalities": self.get_cat_cardinalities(),
            "scale_numeric": self.scale_numeric,
        }


class EWASTransformerDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.methy = arrays["methy"]
        self.region = arrays["region"]
        self.cont = arrays["cont"]
        self.cat = arrays["cat"]
        self.labels = arrays["label"]
        self.ids = arrays["ids"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "methy": torch.tensor(self.methy[idx], dtype=torch.float32),
            "region": torch.tensor(self.region[idx], dtype=torch.float32),
            "cont": torch.tensor(self.cont[idx], dtype=torch.float32),
            "cat": torch.tensor(self.cat[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ============================================================
# Pure Transformer Model
# ============================================================

class ScalarFeatureTokenizer(nn.Module):
    """
    Convert scalar numeric features to tokens.

    Input:  x [B, N]
    Output: tokens [B, N, D]
    """
    def __init__(self, num_features: int, embed_dim: int, dropout: float):
        super().__init__()
        self.num_features = int(num_features)
        self.value_proj = nn.Linear(1, embed_dim)
        self.feature_embed = nn.Embedding(num_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_features == 0:
            raise RuntimeError("ScalarFeatureTokenizer called with zero features.")
        tokens = self.value_proj(x.unsqueeze(-1))
        feat_ids = torch.arange(self.num_features, device=x.device)
        tokens = tokens + self.feature_embed(feat_ids).unsqueeze(0)
        tokens = self.norm(tokens)
        return self.dropout(tokens)


class CategoricalFeatureTokenizer(nn.Module):
    """
    Convert categorical features to tokens.

    Input:  x_cat [B, K]
    Output: tokens [B, K, D]
    """
    def __init__(self, cardinalities: List[int], embed_dim: int, dropout: float):
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.num_features = len(cardinalities)
        self.embeds = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cardinalities])
        self.feature_embed = nn.Embedding(self.num_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if self.num_features == 0:
            raise RuntimeError("CategoricalFeatureTokenizer called with zero features.")
        tokens = []
        for i, emb in enumerate(self.embeds):
            tokens.append(emb(x_cat[:, i]))
        tokens = torch.stack(tokens, dim=1)
        feat_ids = torch.arange(self.num_features, device=x_cat.device)
        tokens = tokens + self.feature_embed(feat_ids).unsqueeze(0)
        tokens = self.norm(tokens)
        return self.dropout(tokens)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, max(1, embed_dim // 2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(1, embed_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class PureFeatureTransformer(nn.Module):
    """
    Pure token-concat Transformer baseline.

    It does NOT use:
      - FiLM
      - multi-scale encoder
      - cross-fusion block
      - learned token compression
      - separate modality branches

    It simply tokenizes selected feature groups, concatenates tokens,
    applies TransformerEncoder, pools, and classifies.
    """
    VALID_FEATURE_SETS = [
        "clinical_only",
        "methylation_only",
        "methylation_region",
        "clinical_methylation",
        "all",
    ]

    def __init__(
        self,
        feature_set: str,
        num_methy: int,
        num_region: int,
        num_cont: int,
        cat_cardinalities: List[int],
        embed_dim: int = 64,
        num_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.2,
        ffn_mult: int = 4,
        pooling: str = "cls",
    ):
        super().__init__()
        feature_set = feature_set.lower()
        if feature_set not in self.VALID_FEATURE_SETS:
            raise ValueError(f"Invalid feature_set: {feature_set}")
        if pooling not in ["cls", "mean", "attention"]:
            raise ValueError("pooling must be one of: cls, mean, attention")

        self.feature_set = feature_set
        self.embed_dim = embed_dim
        self.pooling = pooling

        self.include_methy = feature_set in ["methylation_only", "methylation_region", "clinical_methylation", "all"]
        self.include_region = feature_set in ["methylation_region", "all"]
        self.include_clin = feature_set in ["clinical_only", "clinical_methylation", "all"]

        self.num_methy = int(num_methy)
        self.num_region = int(num_region)
        self.num_cont = int(num_cont)
        self.num_cat = len(cat_cardinalities)

        if self.include_methy and self.num_methy <= 0:
            raise ValueError("feature_set requires methylation features, but num_methy=0")
        if self.include_region and self.num_region <= 0:
            raise ValueError("feature_set requires region features, but num_region=0")
        if self.include_clin and (self.num_cont + self.num_cat) <= 0:
            raise ValueError("feature_set requires clinical features, but no clinical features were found")

        self.methy_tok = ScalarFeatureTokenizer(num_methy, embed_dim, dropout) if num_methy > 0 else None
        self.region_tok = ScalarFeatureTokenizer(num_region, embed_dim, dropout) if num_region > 0 else None
        self.cont_tok = ScalarFeatureTokenizer(num_cont, embed_dim, dropout) if num_cont > 0 else None
        self.cat_tok = CategoricalFeatureTokenizer(cat_cardinalities, embed_dim, dropout) if len(cat_cardinalities) > 0 else None

        # Type embeddings: 0=methylation, 1=region, 2=clinical continuous, 3=clinical categorical
        self.type_embed = nn.Embedding(4, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

        if pooling == "attention":
            self.attn_pool = AttentionPooling(embed_dim, dropout)
        else:
            self.attn_pool = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _add_type(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        type_emb = self.type_embed(torch.tensor(type_id, device=tokens.device))
        return tokens + type_emb.view(1, 1, -1)

    def forward(self, methy: torch.Tensor, region: torch.Tensor, cont: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        token_list = []

        if self.include_methy:
            t = self.methy_tok(methy)
            token_list.append(self._add_type(t, 0))

        if self.include_region:
            t = self.region_tok(region)
            token_list.append(self._add_type(t, 1))

        if self.include_clin:
            if self.num_cont > 0:
                t = self.cont_tok(cont)
                token_list.append(self._add_type(t, 2))
            if self.num_cat > 0:
                t = self.cat_tok(cat)
                token_list.append(self._add_type(t, 3))

        if len(token_list) == 0:
            raise RuntimeError("No tokens were created. Check feature_set and column configuration.")

        x = torch.cat(token_list, dim=1)
        batch_size = x.size(0)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.input_dropout(x)

        x = self.encoder(x)
        x = self.out_norm(x)

        if self.pooling == "cls":
            pooled = x[:, 0]
        elif self.pooling == "mean":
            pooled = x[:, 1:].mean(dim=1)
        else:
            pooled = self.attn_pool(x[:, 1:])

        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def _init_weights(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=max(0.02, self.embed_dim ** -0.5))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=max(0.02, self.embed_dim ** -0.5))
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ============================================================
# Losses
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, gamma: float = 1.5, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal_weight * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class ComboLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def make_criterion(args, train_labels: np.ndarray, device: torch.device, using_sampler: bool) -> nn.Module:
    train_labels = np.asarray(train_labels).astype(int)
    n_pos = max(1, int((train_labels == 1).sum()))
    n_neg = max(1, int((train_labels == 0).sum()))
    raw_pos_weight = n_neg / n_pos

    if args.loss_weight_scale == 0.0:
        pos_weight_tensor = None
    elif using_sampler and not args.allow_double_weighting:
        pos_weight_tensor = None
        print("Using WeightedRandomSampler -> disabling pos_weight to avoid double-weighting.")
    else:
        scaled_pos_weight = 1.0 + (raw_pos_weight - 1.0) * args.loss_weight_scale
        pos_weight_tensor = torch.tensor([scaled_pos_weight], dtype=torch.float32, device=device)
        print(f"Using pos_weight={scaled_pos_weight:.4f} (raw={raw_pos_weight:.4f}, scale={args.loss_weight_scale})")

    if args.loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if args.loss_type == "combo":
        return ComboLoss(bce_weight=args.bce_weight, pos_weight=pos_weight_tensor)
    if args.loss_type == "focal":
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, pos_weight=pos_weight_tensor)
    raise ValueError(f"Unknown loss_type: {args.loss_type}")


# ============================================================
# Optimizer / scheduler / train / eval
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


def get_optimizer_scheduler(model: nn.Module, args, steps_per_epoch: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def warmup_lambda(current_step):
        if warmup_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    if args.scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=args.eta_min,
        )
    elif args.scheduler == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    else:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True,
        )
    return optimizer, warmup_scheduler, main_scheduler


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def forward_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        methy=batch["methy"],
        region=batch["region"],
        cont=batch["cont"],
        cat=batch["cat"],
    )


def train_one_epoch(model, loader, criterion, optimizer, warmup_scheduler, device, epoch, args) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
    for batch in pbar:
        batch = batch_to_device(batch, device)
        labels = batch["label"]

        optimizer.zero_grad()
        logits = forward_batch(model, batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()

        total_loss += float(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels).astype(int)
    metrics = calc_metrics(all_labels, all_probs, threshold=0.5)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


@torch.no_grad()
def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = forward_batch(model, batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        all_probs.extend(np.atleast_1d(probs).tolist())
        all_labels.extend(np.atleast_1d(labels).tolist())
    return np.asarray(all_probs, dtype=float), np.asarray(all_labels, dtype=int)


@torch.no_grad()
def evaluate(model, loader, criterion, device, args) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        labels = batch["label"]
        logits = forward_batch(model, batch)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(np.atleast_1d(probs).tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_probs = np.asarray(all_probs, dtype=float)
    all_labels = np.asarray(all_labels, dtype=int)

    metrics_05 = calc_metrics(all_labels, all_probs, threshold=0.5)
    thresholds = find_thresholds(all_labels, all_probs, grid_size=args.threshold_grid_size)
    metrics_j = calc_metrics(all_labels, all_probs, threshold=thresholds["valid_j_thresh"])
    metrics_f1 = calc_metrics(all_labels, all_probs, threshold=thresholds["valid_f1_thresh"])

    return {
        "loss": total_loss / max(1, len(loader)),
        "probs": all_probs,
        "labels": all_labels,
        "metrics_0.5": metrics_05,
        "metrics_valid_j": metrics_j,
        "metrics_valid_f1": metrics_f1,
        "thresholds": thresholds,
    }


def choose_monitor_value(eval_result: Dict[str, object], monitor_metric: str) -> float:
    if monitor_metric == "loss":
        return -float(eval_result["loss"])
    if monitor_metric == "auc" or monitor_metric == "auroc":
        return float(eval_result["metrics_0.5"]["auroc"])
    if monitor_metric == "auprc":
        return float(eval_result["metrics_0.5"]["auprc"])
    if monitor_metric == "balanced_acc":
        return float(eval_result["metrics_valid_j"]["balanced_accuracy"])
    if monitor_metric == "f1":
        return float(eval_result["metrics_valid_f1"]["f1"])
    if monitor_metric == "recall":
        return float(eval_result["metrics_valid_j"]["recall"])
    raise ValueError(f"Unknown monitor_metric: {monitor_metric}")


# ============================================================
# Save helpers
# ============================================================

def save_checkpoint(path: str, model: nn.Module, optimizer, epoch: int, best_monitor: float, args, metadata: Dict[str, object]):
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_monitor": float(best_monitor),
        "args": vars(args),
        "metadata": metadata,
    }
    torch.save(checkpoint, path)


def save_predictions(path: str, ids: np.ndarray, labels: np.ndarray, probs: np.ndarray, thresholds: Dict[str, float]) -> None:
    df = pd.DataFrame({
        "CaseNo": ids,
        "true_label": labels.astype(int),
        "predicted_prob": probs.astype(float),
        "predicted_class_0.5": (probs >= 0.5).astype(int),
        "predicted_class_valid_j": (probs >= thresholds["valid_j_thresh"]).astype(int),
        "predicted_class_valid_f1": (probs >= thresholds["valid_f1_thresh"]).astype(int),
    })
    df.to_csv(path, index=False)


# ============================================================
# One feature-set experiment
# ============================================================

def make_loader(dataset: Dataset, args, shuffle: bool, sampler=None, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels_int = labels.astype(int)
    class_counts = np.bincount(labels_int, minlength=2)
    class_weights = np.zeros_like(class_counts, dtype=np.float32)
    for cls, count in enumerate(class_counts):
        class_weights[cls] = 1.0 / count if count > 0 else 0.0
    sample_weights = np.array([class_weights[y] for y in labels_int], dtype=np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_model_for_feature_set(args, feature_set: str, preprocessor: TransformerPreprocessor) -> PureFeatureTransformer:
    return PureFeatureTransformer(
        feature_set=feature_set,
        num_methy=len(preprocessor.cpg_cols),
        num_region=len(preprocessor.region_cols),
        num_cont=len(preprocessor.cont_cols),
        cat_cardinalities=preprocessor.get_cat_cardinalities(),
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        ffn_mult=args.ffn_mult,
        pooling=args.pooling,
    )


def run_feature_set_experiment(
    args,
    feature_set: str,
    preprocessor: TransformerPreprocessor,
    train_arrays: Dict[str, np.ndarray],
    valid_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    device: torch.device,
) -> List[Dict[str, object]]:
    print("\n" + "#" * 100)
    print(f"Running Pure Transformer baseline: {feature_set}")
    print("#" * 100)

    out_dir = os.path.join(args.output_dir, feature_set)
    ensure_dir(out_dir)

    train_ds = EWASTransformerDataset(train_arrays)
    valid_ds = EWASTransformerDataset(valid_arrays)
    test_ds = EWASTransformerDataset(test_arrays)

    train_labels = train_arrays["label"].astype(int)
    sampler = make_weighted_sampler(train_labels) if args.use_weighted_sampler else None

    train_loader = make_loader(train_ds, args, shuffle=True, sampler=sampler, drop_last=True)
    valid_loader = make_loader(valid_ds, args, shuffle=False, sampler=None, drop_last=False)
    test_loader = make_loader(test_ds, args, shuffle=False, sampler=None, drop_last=False)

    model = build_model_for_feature_set(args, feature_set, preprocessor).to(device)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = make_criterion(args, train_labels, device, using_sampler=args.use_weighted_sampler)
    optimizer, warmup_scheduler, main_scheduler = get_optimizer_scheduler(model, args, steps_per_epoch=len(train_loader))
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode="max")

    metadata = {
        "feature_set": feature_set,
        "preprocessor": preprocessor.to_metadata(),
        "model_type": "PureFeatureTransformer",
    }
    save_json(make_jsonable(metadata), os.path.join(out_dir, "metadata.json"))

    best_monitor = -float("inf")
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, warmup_scheduler, device, epoch, args)
        valid_result = evaluate(model, valid_loader, criterion, device, args)
        monitor_value = choose_monitor_value(valid_result, args.monitor_metric)

        if epoch >= args.warmup_epochs:
            if args.scheduler == "plateau":
                main_scheduler.step(monitor_value)
            else:
                main_scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "train_acc_0.5": train_metrics["accuracy"],
            "valid_loss": valid_result["loss"],
            "valid_auroc": valid_result["metrics_0.5"]["auroc"],
            "valid_auprc": valid_result["metrics_0.5"]["auprc"],
            "valid_acc_0.5": valid_result["metrics_0.5"]["accuracy"],
            "valid_bacc_0.5": valid_result["metrics_0.5"]["balanced_accuracy"],
            "valid_bacc_j": valid_result["metrics_valid_j"]["balanced_accuracy"],
            "valid_f1_j": valid_result["metrics_valid_j"]["f1"],
            "valid_f1_best": valid_result["metrics_valid_f1"]["f1"],
            "valid_recall_j": valid_result["metrics_valid_j"]["recall"],
            "valid_spec_j": valid_result["metrics_valid_j"]["specificity"],
            "valid_j_thresh": valid_result["thresholds"]["valid_j_thresh"],
            "valid_f1_thresh": valid_result["thresholds"]["valid_f1_thresh"],
            "monitor_value": monitor_value,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"valid_loss={row['valid_loss']:.4f} | "
            f"valid_auc={row['valid_auroc']:.4f} | "
            f"valid_auprc={row['valid_auprc']:.4f} | "
            f"valid_bacc_j={row['valid_bacc_j']:.4f} | "
            f"valid_f1_best={row['valid_f1_best']:.4f} | "
            f"j_th={row['valid_j_thresh']:.3f} | "
            f"lr={row['lr']:.2e}"
        )

        if monitor_value > best_monitor:
            best_monitor = monitor_value
            best_epoch = epoch
            save_checkpoint(
                os.path.join(out_dir, "best_model.pth"),
                model,
                optimizer,
                epoch,
                best_monitor,
                args,
                metadata,
            )

            best_threshold_info = {
                "epoch": int(epoch),
                "monitor_metric": args.monitor_metric,
                "best_monitor": float(best_monitor),
                "valid_loss": float(valid_result["loss"]),
                "valid_auroc": float(valid_result["metrics_0.5"]["auroc"]),
                "valid_auprc": float(valid_result["metrics_0.5"]["auprc"]),
                "valid_metrics_0.5": valid_result["metrics_0.5"],
                "valid_metrics_j": valid_result["metrics_valid_j"],
                "valid_metrics_f1": valid_result["metrics_valid_f1"],
                **valid_result["thresholds"],
            }
            save_json(make_jsonable(best_threshold_info), os.path.join(out_dir, "best_thresholds.json"))
            print(f"  ✓ Saved best model at epoch {epoch+1} ({args.monitor_metric}={best_monitor:.4f})")

        save_checkpoint(
            os.path.join(out_dir, "last_model.pth"),
            model,
            optimizer,
            epoch,
            best_monitor,
            args,
            metadata,
        )

        if early_stopping(monitor_value):
            print(f"Early stopping at epoch {epoch+1}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # Load best checkpoint for final valid/test evaluation
    checkpoint = torch.load(os.path.join(out_dir, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    thresholds = json.load(open(os.path.join(out_dir, "best_thresholds.json"), "r"))
    threshold_map = {
        "0.5": 0.5,
        "valid_j": float(thresholds["valid_j_thresh"]),
        "valid_f1": float(thresholds["valid_f1_thresh"]),
    }

    rows = []
    for split_name, loader, arrays in [
        ("valid", valid_loader, valid_arrays),
        ("test", test_loader, test_arrays),
    ]:
        probs, labels = predict_probs(model, loader, device)
        for threshold_name, threshold_value in threshold_map.items():
            metrics = calc_metrics(labels, probs, threshold_value)
            metrics["feature_set"] = feature_set
            metrics["split"] = split_name
            metrics["threshold_name"] = threshold_name
            metrics["best_epoch"] = int(best_epoch + 1)
            metrics["monitor_metric"] = args.monitor_metric
            metrics["best_monitor"] = float(best_monitor)
            metrics["model_type"] = "PureFeatureTransformer"
            rows.append(metrics)

            if split_name == "test":
                print_metrics_block(
                    f"{feature_set} | TEST | threshold={threshold_name}",
                    metrics,
                )

        if split_name == "test":
            save_predictions(
                os.path.join(out_dir, f"predictions_{feature_set}_test.csv"),
                ids=arrays["ids"],
                labels=labels,
                probs=probs,
                thresholds={
                    "valid_j_thresh": threshold_map["valid_j"],
                    "valid_f1_thresh": threshold_map["valid_f1"],
                },
            )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "final_metrics.csv"), index=False)
    return rows


# ============================================================
# Args and main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Pure Transformer baselines for EWAS CSV")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--id_col", type=str, default="CaseNo")
    parser.add_argument("--label_col", type=str, default="EWAS_label")

    # Column selection
    parser.add_argument("--cpg_start_excel", type=str, default=None, help="Excel-style start column for CpG features, e.g. H")
    parser.add_argument("--cpg_end_excel", type=str, default=None, help="Excel-style end column for CpG features, e.g. GY")
    parser.add_argument("--cpg_cols", type=str, default=None, help="Comma-separated CpG columns. Overrides auto detection and Excel range.")
    parser.add_argument("--region_cols", type=str, default=None, help="Comma-separated region-global columns. Default: ISLAND_1~ISLAND_5 if present.")
    parser.add_argument("--cont_cols", type=str, default=None, help="Comma-separated continuous clinical columns. Default: numeric columns excluding ID/label/CpG/region/cat.")
    parser.add_argument("--cat_cols", type=str, default=None, help="Comma-separated categorical clinical columns. Default: SEX,smoking,DRK,betel,SPORT,HTN_FAM if present.")
    parser.add_argument("--no_scale_numeric", action="store_true", help="Disable StandardScaler for numeric features.")

    parser.add_argument(
        "--feature_sets",
        type=str,
        default="clinical_only,methylation_only,methylation_region,clinical_methylation,all",
        help="Comma-separated feature sets: clinical_only,methylation_only,methylation_region,clinical_methylation,all",
    )

    # Model hyperparameters
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "attention"])

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "plateau"])
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--step_gamma", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Imbalance and loss
    parser.add_argument("--loss_type", type=str, default="combo", choices=["bce", "combo", "focal"])
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--focal_alpha", type=float, default=0.7)
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--loss_weight_scale", type=float, default=0.5)
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--allow_double_weighting", action="store_true")

    # Model selection and misc
    parser.add_argument("--monitor_metric", type=str, default="balanced_acc", choices=["loss", "auc", "auroc", "auprc", "balanced_acc", "f1", "recall"])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--threshold_grid_size", type=int, default=1001)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    save_json(vars(args), os.path.join(args.output_dir, "args.json"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CSV files...")
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)
    test_df = pd.read_csv(args.test_csv)
    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape : {test_df.shape}")

    col_info = resolve_columns(args, train_df)
    save_json(col_info, os.path.join(args.output_dir, "feature_columns.json"))

    print("\nResolved feature columns:")
    print(f"  CpG columns:        {len(col_info['cpg_cols'])}")
    print(f"  Region columns:     {len(col_info['region_cols'])} -> {col_info['region_cols']}")
    print(f"  Clinical cont cols: {len(col_info['cont_cols'])} -> {col_info['cont_cols']}")
    print(f"  Clinical cat cols:  {len(col_info['cat_cols'])} -> {col_info['cat_cols']}")

    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        labels = df[args.label_col].astype(int).values
        print(
            f"{split_name}: negative={(labels == 0).sum()}, "
            f"positive={(labels == 1).sum()}, positive_rate={(labels == 1).mean():.4f}"
        )

    preprocessor = TransformerPreprocessor(
        cpg_cols=col_info["cpg_cols"],
        region_cols=col_info["region_cols"],
        cont_cols=col_info["cont_cols"],
        cat_cols=col_info["cat_cols"],
        scale_numeric=not args.no_scale_numeric,
    )
    preprocessor.fit(train_df)
    save_json(make_jsonable(preprocessor.to_metadata()), os.path.join(args.output_dir, "preprocessor_metadata.json"))

    train_arrays = preprocessor.transform(train_df, label_col=args.label_col, id_col=args.id_col)
    valid_arrays = preprocessor.transform(valid_df, label_col=args.label_col, id_col=args.id_col)
    test_arrays = preprocessor.transform(test_df, label_col=args.label_col, id_col=args.id_col)

    feature_sets = parse_comma_list(args.feature_sets)
    all_rows = []
    for feature_set in feature_sets:
        rows = run_feature_set_experiment(
            args=args,
            feature_set=feature_set,
            preprocessor=preprocessor,
            train_arrays=train_arrays,
            valid_arrays=valid_arrays,
            test_arrays=test_arrays,
            device=device,
        )
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)
    preferred_cols = [
        "model_type", "feature_set", "split", "threshold_name", "threshold",
        "auroc", "auprc", "accuracy", "balanced_accuracy",
        "precision", "recall", "specificity", "f1",
        "tn", "fp", "fn", "tp",
        "num_negative", "num_positive", "positive_rate",
        "best_epoch", "monitor_metric", "best_monitor",
    ]
    other_cols = [c for c in summary_df.columns if c not in preferred_cols]
    summary_df = summary_df[preferred_cols + other_cols]

    summary_path = os.path.join(args.output_dir, "pure_transformer_summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to: {summary_path}")

    test_valid_j = summary_df[(summary_df["split"] == "test") & (summary_df["threshold_name"] == "valid_j")]
    test_valid_j = test_valid_j.sort_values(["balanced_accuracy", "f1", "auprc"], ascending=False)
    ranking_path = os.path.join(args.output_dir, "pure_transformer_test_ranking_valid_j.csv")
    test_valid_j.to_csv(ranking_path, index=False)

    print("\n" + "=" * 100)
    print("Pure Transformer ranking by TEST metrics using validation-selected Youden's J threshold")
    print("=" * 100)
    show_cols = [
        "feature_set", "auroc", "auprc", "accuracy", "balanced_accuracy",
        "precision", "recall", "specificity", "f1", "threshold",
    ]
    if len(test_valid_j) > 0:
        print(test_valid_j[show_cols].to_string(index=False))
    print(f"\nSaved ranking to: {ranking_path}")
    print("Done.")


if __name__ == "__main__":
    main()
