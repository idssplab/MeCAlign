import os
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
    auc,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError(
        "xgboost is not installed. In Colab, run: !pip install -q xgboost"
    ) from e


# ============================================================
# Utility functions
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


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
      A -> 0
      H -> 7
      GY -> 206
    """
    col = col.strip().upper()
    idx = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid Excel column letter: {col}")
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


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


def compute_scale_pos_weight(y_train: np.ndarray, strategy: str) -> float:
    y_train = np.asarray(y_train).astype(int)
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    ratio = n_neg / n_pos

    strategy = str(strategy).lower()
    if strategy == "none" or strategy == "1" or strategy == "false":
        return 1.0
    if strategy == "auto" or strategy == "ratio":
        return float(ratio)
    if strategy == "sqrt":
        return float(np.sqrt(ratio))

    try:
        return float(strategy)
    except Exception:
        raise ValueError(
            "scale_pos_weight must be one of: none, auto, ratio, sqrt, or a numeric value."
        )


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
# Feature builder
# ============================================================

class XGBFeatureBuilder:
    def __init__(
        self,
        id_col: str,
        label_col: str,
        cpg_cols: List[str],
        region_cols: List[str],
        cont_cols: List[str],
        cat_cols: List[str],
    ):
        self.id_col = id_col
        self.label_col = label_col
        self.cpg_cols = list(cpg_cols)
        self.region_cols = list(region_cols)
        self.cont_cols = list(cont_cols)
        self.cat_cols = list(cat_cols)

        self.numeric_imputers: Dict[str, SimpleImputer] = {}
        self.cat_encoder: Optional[OneHotEncoder] = None
        self.cat_feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        for group_name, cols in [
            ("methylation", self.cpg_cols),
            ("region", self.region_cols),
            ("continuous", self.cont_cols),
        ]:
            if len(cols) == 0:
                self.numeric_imputers[group_name] = None
                continue
            X = df[cols].apply(pd.to_numeric, errors="coerce")
            imputer = SimpleImputer(strategy="median")
            imputer.fit(X)
            self.numeric_imputers[group_name] = imputer

        if len(self.cat_cols) > 0:
            X_cat = df[self.cat_cols].fillna("__MISSING__").astype(str)
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            encoder.fit(X_cat)
            self.cat_encoder = encoder
            self.cat_feature_names = list(encoder.get_feature_names_out(self.cat_cols))
        else:
            self.cat_encoder = None
            self.cat_feature_names = []

    def _transform_numeric(self, df: pd.DataFrame, cols: List[str], group_name: str) -> Tuple[np.ndarray, List[str]]:
        if len(cols) == 0:
            return np.zeros((len(df), 0), dtype=np.float32), []
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input CSV for {group_name}: {missing_cols[:10]}")
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        imputer = self.numeric_imputers[group_name]
        X_imp = imputer.transform(X).astype(np.float32)
        return X_imp, list(cols)

    def _transform_cat(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        if len(self.cat_cols) == 0 or self.cat_encoder is None:
            return np.zeros((len(df), 0), dtype=np.float32), []
        missing_cols = [c for c in self.cat_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing categorical columns in input CSV: {missing_cols}")
        X_cat = df[self.cat_cols].fillna("__MISSING__").astype(str)
        X_oh = self.cat_encoder.transform(X_cat).astype(np.float32)
        return X_oh, self.cat_feature_names

    def transform_group(self, df: pd.DataFrame, feature_set: str) -> Tuple[np.ndarray, List[str]]:
        feature_set = feature_set.lower()

        X_methy, names_methy = self._transform_numeric(df, self.cpg_cols, "methylation")
        X_region, names_region = self._transform_numeric(df, self.region_cols, "region")
        X_cont, names_cont = self._transform_numeric(df, self.cont_cols, "continuous")
        X_cat, names_cat = self._transform_cat(df)

        X_clin = np.concatenate([X_cont, X_cat], axis=1)
        names_clin = names_cont + names_cat

        if feature_set == "clinical_only":
            return X_clin, names_clin
        if feature_set == "methylation_only":
            return X_methy, names_methy
        if feature_set == "methylation_region":
            return np.concatenate([X_methy, X_region], axis=1), names_methy + names_region
        if feature_set == "clinical_methylation":
            return np.concatenate([X_clin, X_methy], axis=1), names_clin + names_methy
        if feature_set == "all":
            return np.concatenate([X_clin, X_methy, X_region], axis=1), names_clin + names_methy + names_region

        raise ValueError(f"Unknown feature_set: {feature_set}")


# ============================================================
# Column resolution
# ============================================================

def resolve_columns(args, train_df: pd.DataFrame) -> Dict[str, List[str]]:
    columns = list(train_df.columns)

    if args.label_col not in train_df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found.")

    # Categorical columns
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

    # CpG columns
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

    # Remove accidental duplicates while preserving order
    def unique_keep_order(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    cpg_cols = unique_keep_order(cpg_cols)
    region_cols = unique_keep_order(region_cols)
    cont_cols = unique_keep_order(cont_cols)
    cat_cols = unique_keep_order(cat_cols)

    return {
        "cpg_cols": cpg_cols,
        "region_cols": region_cols,
        "cont_cols": cont_cols,
        "cat_cols": cat_cols,
    }


# ============================================================
# Training / evaluation
# ============================================================

def build_xgb(args, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        objective="binary:logistic",
        eval_metric=args.eval_metric,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        tree_method=args.tree_method,
    )


def save_feature_importance(model: XGBClassifier, feature_names: List[str], output_path: str, top_k: int = 100) -> None:
    try:
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        df_imp.head(top_k).to_csv(output_path, index=False)
    except Exception as e:
        print(f"[Warning] Failed to save feature importance: {e}")


def save_predictions(
    output_path: str,
    ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float],
) -> None:
    df = pd.DataFrame({
        "CaseNo": ids,
        "true_label": y_true.astype(int),
        "predicted_prob": y_prob.astype(float),
        "predicted_class_0.5": (y_prob >= 0.5).astype(int),
        "predicted_class_valid_j": (y_prob >= thresholds["valid_j_thresh"]).astype(int),
        "predicted_class_valid_f1": (y_prob >= thresholds["valid_f1_thresh"]).astype(int),
    })
    df.to_csv(output_path, index=False)


def run_one_feature_set(
    args,
    feature_set: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    builder: XGBFeatureBuilder,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    test_ids: np.ndarray,
) -> List[Dict[str, object]]:
    print("\n" + "#" * 90)
    print(f"Running XGBoost baseline: {feature_set}")
    print("#" * 90)

    X_train, feature_names = builder.transform_group(train_df, feature_set)
    X_valid, _ = builder.transform_group(valid_df, feature_set)
    X_test, _ = builder.transform_group(test_df, feature_set)

    print(f"Feature set: {feature_set}")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_valid: {X_valid.shape}")
    print(f"  X_test : {X_test.shape}")

    scale_pos_weight = compute_scale_pos_weight(y_train, args.scale_pos_weight)
    print(f"  scale_pos_weight = {scale_pos_weight:.4f}")

    model = build_xgb(args, scale_pos_weight=scale_pos_weight)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    thresholds = find_thresholds(y_valid, valid_prob, grid_size=args.threshold_grid_size)
    print("Validation-selected thresholds:")
    print(json.dumps(thresholds, indent=2))

    thresholds_path = os.path.join(args.output_dir, f"thresholds_{feature_set}.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    rows = []
    threshold_map = {
        "0.5": 0.5,
        "valid_j": thresholds["valid_j_thresh"],
        "valid_f1": thresholds["valid_f1_thresh"],
    }

    for split_name, y_true, y_prob in [
        ("valid", y_valid, valid_prob),
        ("test", y_test, test_prob),
    ]:
        for threshold_name, threshold_value in threshold_map.items():
            metrics = calc_metrics(y_true, y_prob, threshold_value)
            metrics["feature_set"] = feature_set
            metrics["split"] = split_name
            metrics["threshold_name"] = threshold_name
            metrics["num_features"] = int(X_train.shape[1])
            metrics["scale_pos_weight"] = float(scale_pos_weight)
            rows.append(metrics)

            if split_name == "test":
                print_metrics_block(
                    title=f"{feature_set} | TEST | threshold={threshold_name}",
                    metrics=metrics,
                )

    pred_path = os.path.join(args.output_dir, f"predictions_{feature_set}_test.csv")
    save_predictions(pred_path, test_ids, y_test, test_prob, thresholds)

    imp_path = os.path.join(args.output_dir, f"feature_importance_{feature_set}.csv")
    save_feature_importance(model, feature_names, imp_path, top_k=args.top_k_importance)

    return rows


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost baselines for EWAS CSV")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--id_col", type=str, default="CaseNo")
    parser.add_argument("--label_col", type=str, default="EWAS_label")

    # Column selection
    parser.add_argument("--cpg_start_excel", type=str, default=None,
                        help="Excel-style start column for CpG features, e.g. H")
    parser.add_argument("--cpg_end_excel", type=str, default=None,
                        help="Excel-style end column for CpG features, e.g. GY")
    parser.add_argument("--cpg_cols", type=str, default=None,
                        help="Comma-separated CpG columns. Overrides auto detection and Excel range.")
    parser.add_argument("--region_cols", type=str, default=None,
                        help="Comma-separated region-global columns. Default: ISLAND_1~ISLAND_5 if present.")
    parser.add_argument("--cont_cols", type=str, default=None,
                        help="Comma-separated continuous clinical columns. Default: numeric columns excluding ID/label/CpG/region/cat.")
    parser.add_argument("--cat_cols", type=str, default=None,
                        help="Comma-separated categorical clinical columns. Default: SEX,smoking,DRK,betel,SPORT,HTN_FAM if present.")

    parser.add_argument(
        "--feature_sets",
        type=str,
        default="clinical_only,methylation_only,methylation_region,clinical_methylation,all",
        help="Comma-separated feature sets to run. Options: clinical_only,methylation_only,methylation_region,clinical_methylation,all",
    )

    # XGBoost hyperparameters
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=float, default=3.0)
    parser.add_argument("--reg_lambda", type=float, default=5.0)
    parser.add_argument("--reg_alpha", type=float, default=0.5)
    parser.add_argument("--eval_metric", type=str, default="aucpr", choices=["auc", "aucpr", "logloss"])
    parser.add_argument("--scale_pos_weight", type=str, default="sqrt",
                        help="none, auto/ratio, sqrt, or numeric value. Recommended first try: sqrt.")
    parser.add_argument("--tree_method", type=str, default="hist")
    parser.add_argument("--n_jobs", type=int, default=-1)

    parser.add_argument("--threshold_grid_size", type=int, default=1001)
    parser.add_argument("--top_k_importance", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading CSV files...")
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)
    test_df = pd.read_csv(args.test_csv)

    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape : {test_df.shape}")

    col_info = resolve_columns(args, train_df)
    with open(os.path.join(args.output_dir, "feature_columns.json"), "w") as f:
        json.dump(col_info, f, indent=2)

    print("\nResolved feature columns:")
    print(f"  CpG columns:        {len(col_info['cpg_cols'])}")
    print(f"  Region columns:     {len(col_info['region_cols'])} -> {col_info['region_cols']}")
    print(f"  Clinical cont cols: {len(col_info['cont_cols'])} -> {col_info['cont_cols']}")
    print(f"  Clinical cat cols:  {len(col_info['cat_cols'])} -> {col_info['cat_cols']}")

    y_train = train_df[args.label_col].astype(int).values
    y_valid = valid_df[args.label_col].astype(int).values
    y_test = test_df[args.label_col].astype(int).values

    if args.id_col in test_df.columns:
        test_ids = test_df[args.id_col].astype(str).values
    else:
        test_ids = np.array([str(i) for i in range(len(test_df))])

    print("\nClass distribution:")
    for name, y in [("train", y_train), ("valid", y_valid), ("test", y_test)]:
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        print(f"  {name}: negative={n_neg}, positive={n_pos}, positive_rate={n_pos / len(y):.4f}")

    builder = XGBFeatureBuilder(
        id_col=args.id_col,
        label_col=args.label_col,
        cpg_cols=col_info["cpg_cols"],
        region_cols=col_info["region_cols"],
        cont_cols=col_info["cont_cols"],
        cat_cols=col_info["cat_cols"],
    )
    builder.fit(train_df)

    feature_sets = parse_comma_list(args.feature_sets)
    all_rows = []
    for feature_set in feature_sets:
        rows = run_one_feature_set(
            args=args,
            feature_set=feature_set,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            builder=builder,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            test_ids=test_ids,
        )
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)

    preferred_cols = [
        "feature_set", "split", "threshold_name", "threshold",
        "auroc", "auprc", "accuracy", "balanced_accuracy",
        "precision", "recall", "specificity", "f1",
        "tn", "fp", "fn", "tp",
        "num_negative", "num_positive", "positive_rate",
        "num_features", "scale_pos_weight",
    ]
    other_cols = [c for c in summary_df.columns if c not in preferred_cols]
    summary_df = summary_df[preferred_cols + other_cols]

    summary_path = os.path.join(args.output_dir, "xgb_baseline_summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to: {summary_path}")

    test_valid_j = summary_df[(summary_df["split"] == "test") & (summary_df["threshold_name"] == "valid_j")]
    test_valid_j = test_valid_j.sort_values(
        ["balanced_accuracy", "f1", "auprc"],
        ascending=False,
    )

    print("\n" + "=" * 90)
    print("Ranking by TEST metrics using validation-selected Youden's J threshold")
    print("=" * 90)
    show_cols = [
        "feature_set", "auroc", "auprc", "accuracy", "balanced_accuracy",
        "precision", "recall", "specificity", "f1", "threshold",
    ]
    print(test_valid_j[show_cols].to_string(index=False))

    ranking_path = os.path.join(args.output_dir, "xgb_test_ranking_valid_j.csv")
    test_valid_j.to_csv(ranking_path, index=False)
    print(f"\nSaved ranking to: {ranking_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
