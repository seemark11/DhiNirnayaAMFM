# Classification Model Training and Evaluation

This repository contains a Python script for training, tuning, and evaluating classification models for diagnostic prediction using precomputed feature CSV files.

## Overview

The pipeline supports:

- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Hyperparameter optimization using Optuna
- Standardized preprocessing
- Performance reporting on train and test sets
- SHAP based feature explainability

## Models Trained

- Support Vector Classifier (SVC)
- Decision Tree Classifier

Hyperparameters are optimized using Optuna with 5 fold stratified cross validation.

---

## Input Data Format

The script expects CSV files with the following structure:

- Feature columns
- Target column: `dx`
- Auxiliary column: `mmse` (excluded from training)

### File Naming Convention

```text
dct_num_<DCT>_num_R_form_<R>_<feat_type>.csv
```
Where:
* feat_type ∈ {variance, ddct, combined}
* dct_num ∈ {2,3,4,5,6}
* num_R_form ∈ {4,5,6,7,8}

Train and test files must be stored in separate directories.

## Directory Structure
```txt
base_data/
├── train_csv_<suffix>/
└── test_csv_<suffix>/

base_results/
└── <suffix>/
    ├── classification_svc_combined_results.csv
    ├── classification_dt_combined_results.csv
    └── classification/
        └── num_R_form_*/
            └── dct_num_*/
                └── feat_type_*/
```


`suffix` is automatically constructed as:
```text
specstrides_<value>_specwindowsecs_<value>
```

## Command-Line Arguments

The script accepts the following arguments:
```txt
python train_classification.py --specwindowsecs 5 --specstrides 200
```

The training script supports the following command line arguments:

| Argument | Type | Description | Default |
|--------|------|-------------|---------|
| `--specwindowsecs` | int | Spectrogram window size in seconds used during feature extraction | 5 |
| `--specstrides` | int | Stride (hop length) for spectrogram computation | 200 |

## Preprocessing
*   Features are standardized using `StandardScaler`
*   Labels (`dx`) are encoded using LabelEncoder
*   The scaler is saved for each configuration

## Evaluation Metrics

The following metrics are computed for both train and test sets:
*   Accuracy
*   Precision
*   Recall
*   F1-score
*   Confusion matrix components (TN, FP, FN, TP)

Results are appended to CSV files:
*   `classification_svc_combined_results.csv`
*   `classification_dt_combined_results.csv`

## Model and Artifact Saving

For each configuration, the following are saved:
*   Best trained model (`.pkl`)
*   Feature scaler (`.pkl`)
*   SHAP values (`.pkl`)
*   SHAP summary plots:
    *   Bar plot
    *   Beeswarm plot

## Explainability
SHAP is used to interpret model predictions:
*   `TreeExplainer` for decision trees
*   `KernelExplainer` for SVC

Feature importance plots are automatically generated and stored per experiment.

## Notes
*   Class labels are assumed to be binary.
*   The script performs exhaustive training across all feature and parameter combinations.
*   Execution time can be significant due to Optuna optimization.

---

