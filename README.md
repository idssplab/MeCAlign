# MecAlign

**MecAlign: Methylation–Clinical Alignment Network with Clinical-Routed Low-Rank Adapters**

This repository provides the implementation of MecAlign, a deep learning framework for hypertension prediction using both clinical variables and DNA methylation features. The model is designed to jointly model baseline clinical information and CpG methylation M-values, while allowing the clinical representation to condition the methylation pathway through CpG gating and clinical-routed low-rank adapters.

This repository is prepared for anonymous review. All commands below use placeholder paths and should be modified according to the local environment.

---

## Model Overview

MecAlign contains two main branches:

1. **Clinical Branch**
   Numerical and categorical baseline variables are transformed into clinical tokens and encoded by a Clinical Transformer Encoder. An attention-pooled clinical context vector is then generated and used as a conditioning signal for the methylation pathway.

2. **Methylation Branch**
   CpG M-values are projected into CpG token embeddings. Global island-derived region features are incorporated through Static Region FiLM, which modulates CpG token representations. The pooled clinical context is also used to generate a CpG-wise gate that adaptively reweights the original CpG tokens. Region tokens are appended before Latent Query Re-encoding compresses the methylation sequence into latent methylation tokens.

3. **Clinical-Routed Transformer**
   The latent methylation tokens are processed by a Clinical-Routed Transformer. In selected Transformer layers, clinical-routed low-rank adapter modules are inserted after the self-attention and feed-forward network updates. A Clinical Router uses the pooled clinical context to compute mixture weights over multiple lightweight low-rank experts.

4. **Alignment Fusion and Prediction**
   Shared alignment tokens attend to the concatenated methylation and clinical token representations. The methylation, clinical, and alignment representations are separately pooled, concatenated, and passed to the final classification head.

---

## Repository Structure

```text
.
├── baseline/
│   ├── baseline_model.ipynb
│   ├── baseline_boostrap.ipynb
│   ├── xgboost_ewas_baseline.py
│   ├── pure_transformer_ewas_baseline.py
│   ├── finetune_setting/
│   │   ├── catboost_finetune_search_space.yaml
│   │   ├── clinical_only_finetune_search_space.yaml
│   │   ├── ft_transformer_finetune_search_space.yaml
│   │   ├── pure_transformer_finetune_search_space.yaml
│   │   ├── tabm_finetune_search_space.yaml
│   │   └── xgboost_finetune_search_space.yaml
│   └── best_setting/
│       ├── *.csv
│       └── *.pt
│
├── checkpoint/
│   └── mecalign_best_valid.pt
│
├── dataset/
│   └── ewas_dataset.py
│
├── models/
│   └── model_mecalign_v2.py
│
├── bootstrap_mecalign_v2_full_model.py
├── finetune_mecalign_v2.py
├── mecalign_v2_finetune_search_space.yaml
└── README.md
```

### Folder Description

* `baseline/`
  Contains the implementation and evaluation scripts for all baseline models, including XGBoost, CatBoost, pure Transformer, FT-Transformer, TabM, and TabPFN.

* `baseline/baseline_model.ipynb`
  Notebook for running baseline models.

* `baseline/baseline_boostrap.ipynb`
  Notebook for running bootstrap evaluation for baseline models.

* `baseline/xgboost_ewas_baseline.py`
  XGBoost baseline implementation.

* `baseline/pure_transformer_ewas_baseline.py`
  Pure Transformer baseline implementation.

* `baseline/finetune_setting/`
  Contains the YAML files that document the fine-tuning search spaces and tuning protocols for baseline models. These files are provided for reproducibility and configuration tracking.

* `baseline/best_setting/`
  Stores the best hyperparameter settings, CSV files, or checkpoints for baseline models.

* `checkpoint/`
  Contains the best validation checkpoint of the proposed MecAlign model.

* `dataset/`
  Contains the dataset loading and preprocessing code.

* `dataset/ewas_dataset.py`
  Dataset class used by MecAlign.

* `models/`
  Contains the proposed model implementation.

* `models/model_mecalign_v2.py`
  Full implementation of MecAlign.

* `bootstrap_mecalign_v2_full_model.py`
  Script for bootstrap evaluation of the proposed MecAlign model.

* `finetune_mecalign_v2.py`
  Script for fine-tuning MecAlign.

* `mecalign_v2_finetune_search_space.yaml`
  YAML file describing the fine-tuning search space and tuning protocol for the proposed MecAlign model.

---

## Fine-tuning Configuration Files

For reproducibility, the fine-tuning search spaces are documented as YAML configuration files.

The proposed MecAlign model uses the configuration file located at the repository root:

```text
mecalign_v2_finetune_search_space.yaml
```

Baseline fine-tuning configuration files are stored under:

```text
baseline/finetune_setting/
```

The baseline configuration files include:

```text
baseline/finetune_setting/
├── catboost_finetune_search_space.yaml
├── clinical_only_finetune_search_space.yaml
├── ft_transformer_finetune_search_space.yaml
├── pure_transformer_finetune_search_space.yaml
├── tabm_finetune_search_space.yaml
└── xgboost_finetune_search_space.yaml
```

These files summarize the search space, tuning strategy, selection metric, evaluation threshold policy, and saved outputs for each model. They are intended to make the hyperparameter tuning process transparent without requiring reviewers to inspect every tuning script.

---

## Requirements

The code was implemented using PyTorch and evaluated in a Google Colab environment with an NVIDIA Tesla T4 GPU.

Typical dependencies include:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn==1.5.2
pip install xgboost catboost
pip install tqdm
```

Additional dependencies may be required for specific baseline models such as FT-Transformer, TabM, or TabPFN.

```bash
# FT-Transformer
pip install pytorch_tabular

# TabM
pip install tabm
pip install git+https://github.com/yandex-research/tabm.git
pip install rtdl_num_embeddings

# TabPFN 3.0 Ecosystem
pip install tabpfn
pip install tabpfn-client

# TabPFN Extensions
git clone https://github.com/PriorLabs/tabpfn-extensions
pip install -e tabpfn-extensions[all]
```

To run TabPFN 3.0 via cloud inference (`tabpfn-client`), an API token must be acquired from **PriorLabs**:

* **API Key Portal:** [PriorLabs User Portal](https://ux.priorlabs.ai/)

Before executing the model, set the token as an environment variable:

```bash
export TABPFN_TOKEN="your_api_token_here"
```

---

## Data Format

The input data should be provided as CSV files with the same schema used by the dataset loader in:

```text
dataset/ewas_dataset.py
```

The expected inputs include:

* CpG methylation features
* Global island-derived region features
* Continuous clinical variables
* Categorical clinical variables
* Binary hypertension label

Example file organization:

```text
<data_root>/
├── train.csv
├── valid.csv
└── test.csv
```

Please replace the placeholder paths in the commands below with the actual local paths.

---

## Training MecAlign

To fine-tune MecAlign, run:

```bash
python finetune_mecalign_v2.py \
  --train_csv <path_to_train_csv> \
  --valid_csv <path_to_valid_csv> \
  --test_csv <path_to_test_csv> \
  --output_dir <path_to_output_dir> \
  --num_trials 30 \
  --epochs 50 \
  --batch_size 16 \
  --seed 42 \
  --use_amp 0
```

Example:

```bash
python finetune_mecalign_v2.py \
  --train_csv ./data/train.csv \
  --valid_csv ./data/valid.csv \
  --test_csv ./data/test.csv \
  --output_dir ./outputs/mecalign_v2 \
  --num_trials 30 \
  --epochs 50 \
  --batch_size 16 \
  --seed 42 \
  --use_amp 0
```

The corresponding MecAlign fine-tuning search space is documented in:

```text
mecalign_finetune_search_space.yaml
```

---

## Bootstrap Evaluation

To evaluate the trained MecAlign model using bootstrap resampling, run:

```bash
python bootstrap_mecalign_v2_full_model.py \
  --checkpoint <path_to_checkpoint> \
  --input_csv <path_to_test_csv> \
  --output_dir <path_to_bootstrap_output_dir> \
  --n_bootstrap 1000 \
  --batch_size 64 \
  --stratified 1 \
  --seed 42
```

Example:

```bash
python bootstrap_mecalign_v2_full_model.py \
  --checkpoint ./checkpoint/best_valid_f1.pt \
  --input_csv ./data/test.csv \
  --output_dir ./outputs/bootstrap_test \
  --n_bootstrap 1000 \
  --batch_size 64 \
  --stratified 1 \
  --seed 42
```

The script performs 1,000 bootstrap resampling iterations on the test set. The `--stratified 1` option enables stratified bootstrap sampling.

---

## Baseline Models

The baseline models are provided under the `baseline/` directory. The included baselines are:

* XGBoost
* CatBoost
* Pure Transformer
* FT-Transformer
* TabM
* TabPFN
* Clinical-only Transformer

The main baseline notebook is:

```text
baseline/baseline_model.ipynb
```

The bootstrap evaluation notebook for baseline models is:

```text
baseline/baseline_boostrap.ipynb
```

Best hyperparameter settings and model checkpoints are stored in:

```text
baseline/best_setting/
```

Fine-tuning search spaces and tuning protocols for baseline models are documented in:

```text
baseline/finetune_setting/
```

---

## Checkpoints

The best validation checkpoint of the proposed MecAlign model is stored under:

```text
checkpoint/
```

For example:

```text
checkpoint/mecalign_best_valid.pt
```

This checkpoint can be used directly for bootstrap evaluation.

---

## Reproducibility

The main experiments use the following default settings:

* Number of MecAlign random-search trials: `30`
* MecAlign maximum training epochs: `50`
* Batch size for MecAlign training: `16`
* Batch size for bootstrap evaluation: `64`
* Random seed: `42`
* Number of bootstrap iterations: `1000`
* Stratified bootstrap: enabled

The fine-tuning search spaces for the proposed model and baseline models are documented in YAML files:

* `mecalign_v2_finetune_search_space.yaml`
* `baseline/finetune_setting/*.yaml`

All file paths in this README are anonymized placeholders. Please replace them with local paths before running the scripts.
