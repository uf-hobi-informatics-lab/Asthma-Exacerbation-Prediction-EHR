# Multimodal EHR-Based Prediction of Pediatric Asthma Exacerbations

This repository contains code for multimodal electronic health record (EHR) modeling and SHAP-based model interpretation for pediatric asthma exacerbation prediction.

Associated manuscript:

- Preprint: [Multimodal EHR-Based Prediction of Pediatric Asthma Exacerbations](https://www.medrxiv.org/content/10.64898/2026.02.25.26347091v1)
- Conference status: Accepted to the AMIA 2026 Amplify Informatics Conference

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── generate_shap.py
│   └── train_models.py
└── src/
    └── asthma_exacerbation/
        ├── data_utils.py
        ├── metrics.py
        ├── modeling.py
        └── shap_utils.py
```

## What the code does

- Loads a feature table and a label file without assuming any private filesystem paths
- Builds binary labels from either a patient list or an explicit label column
- Trains multiple baseline machine learning models with 5-fold stratified cross-validation
- Uses Bayesian hyperparameter search on fold 1 and reuses the selected settings on later folds
- Evaluates baseline, undersampling, and oversampling strategies
- Saves fold-level metrics, summary metrics, fitted models, and best hyperparameters
- Generates SHAP summary plots and ranked feature-importance outputs for trained models

## Data expectations

The repository does not include patient-level data.

Expected inputs:

1. A feature CSV containing one row per patient and an identifier column such as `deid_pat_id`
2. A label CSV containing either:
   - a patient identifier column listing positive cases, or
   - a patient identifier column plus an explicit binary label column

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you already have a long-lived Python environment, creating a fresh virtual environment is recommended to avoid `numpy` or `scipy` version conflicts.

## Example commands

Train the full modeling pipeline:

```bash
python scripts/train_models.py \
  --features-csv data/features.csv \
  --labels-csv data/labels.csv \
  --output-dir outputs/training_run \
  --id-column deid_pat_id \
  --bayes-iterations 100 \
  --scale-note-features
```

Train with an explicit label column and selected models only:

```bash
python scripts/train_models.py \
  --features-csv data/features.csv \
  --labels-csv data/labels_with_targets.csv \
  --label-column label \
  --output-dir outputs/selected_models \
  --models xgboost random_forest logistic_regression \
  --sampling-strategies baseline oversampling
```

Generate SHAP outputs for a trained model:

```bash
python scripts/generate_shap.py \
  --model-path outputs/training_run/fold_1/xgboost_baseline.joblib \
  --features-csv data/features.csv \
  --labels-csv data/labels.csv \
  --output-dir outputs/shap_xgboost \
  --id-column deid_pat_id \
  --top-k 15 \
  --max-samples 1000
```

## Outputs

The training pipeline writes:

- `fold_<n>/model_metrics.csv`
- `fold_<n>/*.joblib`
- `all_fold_metrics.csv`
- `cv_summary_metrics.csv`
- `best_params.json`
- `run_config.json`

The SHAP pipeline writes:

- `shap_summary.png`
- `shap_feature_importance.csv`
- `shap_run_config.json`

## Notes for public release

- No site-specific storage paths are required
- No patient-level data are bundled in this repository
- Dataset-specific exclusions should be passed through command-line arguments instead of hard-coded into source files
