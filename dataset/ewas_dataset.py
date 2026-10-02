import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class EWASCSVBridgeDataset(Dataset):
    """
    CSV dataset for EWAS tabular files.

    Expected columns:
      - ID column, default: CaseNo
      - Label column, default: EWAS_label
      - CpG columns starting with 'cg'
      - Region-global columns, default: ISLAND_1 ~ ISLAND_5
      - Clinical categorical columns (optional)
      - Remaining numeric columns are treated as continuous clinical features

    Output API:
      {
        "methy": ...,
        "region_global": ...,
        "clin_cont": ...,
        "clin_cat": ...,
        "label": ...,
      }
    """

    def __init__(
        self,
        csv_file,
        probe_meta_file=None,  
        mode="train",
        scaler=None,
        cat_maps=None,
        id_col="CaseNo",
        label_col="EWAS_label",
        cont_cols=None,
        cat_cols=None,
        cpg_cols=None,
        region_cols=None,
    ):
        self.df = pd.read_csv(csv_file).copy()

        # -------------------------
        # IDs / labels
        # -------------------------
        if id_col in self.df.columns:
            self.sample_ids = self.df[id_col].astype(str).str.strip().values
        else:
            self.sample_ids = np.array([str(i) for i in range(len(self.df))])

        if label_col not in self.df.columns:
            raise ValueError(f"Label column '{label_col}' not found in {csv_file}")
        self.labels = self.df[label_col].astype(np.float32).values

        # -------------------------
        # CpG columns
        # -------------------------
        if cpg_cols is None:
            cpg_cols = [c for c in self.df.columns if str(c).startswith("cg")]
        if len(cpg_cols) == 0:
            raise ValueError("No CpG columns found. Expected columns starting with 'cg'.")

        self.probe_ids = [str(c).strip() for c in cpg_cols]
        self.num_probes = len(self.probe_ids)
        self.methy_data = self.df[self.probe_ids].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

        probe_medians = np.nanmedian(self.methy_data, axis=0)
        for i in range(self.methy_data.shape[1]):
            mask = np.isnan(self.methy_data[:, i])
            self.methy_data[mask, i] = probe_medians[i] if not np.isnan(probe_medians[i]) else 0.0

        # -------------------------
        # Region-global columns
        # -------------------------
        if region_cols is None:
            default_region = [f"ISLAND_{i}" for i in range(1, 6)]
            region_cols = [c for c in default_region if c in self.df.columns]
        self.region_cols = list(region_cols)

        if len(self.region_cols) > 0:
            region_df = self.df[self.region_cols].apply(pd.to_numeric, errors="coerce")
            X_region = region_df.fillna(region_df.median()).fillna(0.0).values.astype(np.float32)
        else:
            X_region = np.zeros((len(self.df), 0), dtype=np.float32)

        self.num_region_global = X_region.shape[1]

        # -------------------------
        # Clinical categorical
        # -------------------------
        CLINICAL_CAT_COLS = [
            "SEX",
            "smoking",
            "DRK",
            "betel",
            "SPORT",
            "HTN_FAM",
            "edu",   
        ]

        CLINICAL_CONT_COLS = [
            "T_CHO_b",
            "LDL_b",
            "TG_b",
            "HDL_b",
            "WHR_b",
            "BMI_b",
            "uric_acid_b",
            "creatinine_b",
            "egfr_b",
            "BUN_b",
            "age_b",
            "microalbumin_b",
            "albumin_b",
            "T_BILIRUBIN_b",
            "DBP_b",
            "SBP_b",
            "SGPT_b",
            "SGOT_b",
            "GAMMA_GT_b",
            "HBA1C_b",
            "FBG_b",
            "HR_b",
        ]

        if cat_cols is None:
          default_cat = [
              "SEX",
              "smoking",
              "DRK",
              "betel",
              "SPORT",
              "HTN_FAM",
              "edu",
          ]
          cat_cols = [c for c in default_cat if c in self.df.columns]

        self.cat_cols = list(cat_cols)

        for col in self.cat_cols:
            if col not in self.df.columns:
                self.df[col] = 0

        self.cat_maps = {}
        cat_arrays = []

        if mode == "train":
            for col in self.cat_cols:
                vals = self.df[col].fillna("__UNK__").astype(str).values
                uniq = ["__UNK__"] + sorted([v for v in pd.unique(vals) if v != "__UNK__"])
                mapping = {v: i for i, v in enumerate(uniq)}
                self.cat_maps[col] = mapping
                mapped = np.array([mapping.get(v, 0) for v in vals], dtype=np.int64)
                cat_arrays.append(mapped)
        else:
            if cat_maps is None:
                raise ValueError("cat_maps must be provided for valid/test mode.")
            self.cat_maps = cat_maps
            for col in self.cat_cols:
                vals = self.df[col].fillna("__UNK__").astype(str).values
                mapping = self.cat_maps[col]
                mapped = np.array([mapping.get(v, 0) for v in vals], dtype=np.int64)
                cat_arrays.append(mapped)

        self.cat_data = np.vstack(cat_arrays).T if len(cat_arrays) > 0 else np.zeros((len(self.df), 0), dtype=np.int64)

        # -------------------------
        # Clinical continuous (exclude region_global)
        # -------------------------
        '''
        if cont_cols is None:
            exclude = set([id_col, label_col] + self.probe_ids + self.cat_cols + self.region_cols)
            cont_cols = [
                c for c in self.df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(self.df[c])
            ]
        self.cont_cols = list(cont_cols)
        '''
        if cont_cols is None:
          default_cont = [
              "T_CHO_b",
              "LDL_b",
              "TG_b",
              "HDL_b",
              "WHR_b",
              "BMI_b",
              "uric_acid_b",
              "creatinine_b",
              "egfr_b",
              "BUN_b",
              "age_b",
              "microalbumin_b",
              "albumin_b",
              "T_BILIRUBIN_b",
              "DBP_b",
              "SBP_b",
              "SGPT_b",
              "SGOT_b",
              "GAMMA_GT_b",
              "HBA1C_b",
              "FBG_b",
              "HR_b",
          ]
          cont_cols = [c for c in default_cont if c in self.df.columns]

        self.cont_cols = list(cont_cols)

        if len(self.cont_cols) > 0:
            cont_df = self.df[self.cont_cols].apply(pd.to_numeric, errors="coerce")
            X_cont = cont_df.fillna(cont_df.median()).fillna(0.0).values.astype(np.float32)
        else:
            X_cont = np.zeros((len(self.df), 0), dtype=np.float32)

        if mode == "train":
            cont_scaler = StandardScaler() if X_cont.shape[1] > 0 else None
            region_scaler = StandardScaler() if X_region.shape[1] > 0 else None
            self.cont_data = cont_scaler.fit_transform(X_cont) if cont_scaler is not None else X_cont
            self.region_data = region_scaler.fit_transform(X_region) if region_scaler is not None else X_region
            self.scaler = {
                "cont": cont_scaler,
                "region": region_scaler,
            }
        else:
            if scaler is None:
                raise ValueError("scaler must be provided for valid/test mode.")
            self.scaler = scaler
            cont_scaler = scaler.get("cont", None) if isinstance(scaler, dict) else None
            region_scaler = scaler.get("region", None) if isinstance(scaler, dict) else None
            self.cont_data = cont_scaler.transform(X_cont) if (cont_scaler is not None and X_cont.shape[1] > 0) else X_cont
            self.region_data = region_scaler.transform(X_region) if (region_scaler is not None and X_region.shape[1] > 0) else X_region

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "methy": torch.tensor(self.methy_data[idx], dtype=torch.float32),
            "region_global": torch.tensor(self.region_data[idx], dtype=torch.float32),
            "clin_cont": torch.tensor(self.cont_data[idx], dtype=torch.float32),
            "clin_cat": torch.tensor(self.cat_data[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
