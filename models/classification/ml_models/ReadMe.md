# Classification: ML and Statistical Pipelines

This repository implements classification models for diagnostic prediction from precomputed feature CSV files.
It includes **two pipelines**:

1. **ML Classification Pipeline** – purely machine learning with hyperparameter tuning.  
2. **Statistical + ML Classification Pipeline** – incorporates classical statistical tests and effect size measures for feature analysis and selection prior to ML modeling.

The pipeline is designed for binary classification problems with high-dimensional features, where both predictive performance and interpretability are important.

---
## Features

### 1. ML Classification Pipeline
- Models:
  - Support Vector Classifier (SVC)
  - Decision Tree Classifier (DT)
- Hyperparameter tuning with **Optuna** (5-fold cross-validation)
- Feature scaling using **StandardScaler**
- Label encoding using **LabelEncoder**
- Performance metrics: 
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix components (TN, FP, FN, TP)
- Model explainability using **SHAP**
- Results saved to CSV

### 2. Statistical + ML Classification Pipeline
- Statistical feature selection:
  - **VIF** (Variance Inflation Factor) for multicollinearity
  - **Point-biserial correlation**
  - **Spearman correlation**
  - **Mann–Whitney U test**
  - **Mutual information**
  - **Effect size via Cliff’s delta**
- VIF-based pruning applied before ML modeling
- Selected features passed to the same ML classifiers (SVC, DT)
- SHAP-based explainability and result saving identical to ML-only pipeline

---

## Directory Structure

```

base_data/
├── train_csv_specstrides_<value>_specwindowsecs_<value>/
└── test_csv_specstrides_<value>_specwindowsecs_<value>/

base_results/
└── specstrides_<value>_specwindowsecs_<value>/
├── classification_svr_combined_results.csv
├── classification_dt_combined_results.csv
└── classification/
└── num_R_form_*/
└── dct_num_*/
└── feat_type_*/
├── best_SVR_model.pkl
├── best_DT_model.pkl
├── scaler_SVR.pkl
├── best_SVR_shap_values.pkl
├── best_DT_shap_values.pkl
├── best_SVR_shap_bar.png
├── best_SVR_shap_beeswarm.png
├── best_DT_shap_bar.png
└── best_DT_shap_beeswarm.png

````

- `suffix` is automatically generated as:  
`specstrides_<value>_specwindowsecs_<value>`

---

## Input Data Format

- CSV files with:
  - Feature columns
  - Target column: `dx`
  - Auxiliary column: `mmse` (excluded from classification features)
- File naming convention:

```text
dct_num_<DCT>_num_R_form_<R>_<feat_type>.csv
````

* `feat_type ∈ {variance, ddct, combined}`
* `dct_num ∈ {2,3,4,5,6}`
* `num_R_form ∈ {4,5,6,7,8}`

---

## Command-Line Interface (CLI)

```bash
python train_classification.py --specwindowsecs 5 --specstrides 200
```

| Argument           | Description                        | Default |
| ------------------ | ---------------------------------- | ------- |
| `--specwindowsecs` | Spectrogram window size in seconds | 5       |
| `--specstrides`    | Spectrogram stride                 | 200     |

---

## Preprocessing

* Features are standardized using **StandardScaler**
* Labels (`dx`) are encoded numerically using *LabelEncoder*
* Feature scaling is fit only on training data
* Scalers are saved for each experimental configuration

---

## Evaluation Metrics

Computed for **both train and test sets**:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix components:
    * True Negatives (TN)
    * False Positives (FP)
    * False Negatives (FN)
    * True Positives (TP)

Results appended to:

* `classification_svr_combined_results.csv`
* `classification_dt_combined_results.csv`

---

## Model Saving & Explainability

* Trained models saved as `.pkl`
* Feature scaler saved as `.pkl`
* SHAP values saved as `.pkl`
* SHAP summary plots automatically generated:

  * Bar plot (global importance)
  * Beeswarm plot (distribution of feature effects)

---

## Explainability

SHAP is used to interpret classification model predictions:
- `TreeExplainer` for Decision Tree classifiers
- `KernelExplainer` for SVC models

Explainability artifacts are stored alongside model outputs for full reproducibility.

---

## Notes

* The pipeline assumes binary classification.
* All feature sets, DCT configurations, and parameter combinations are evaluated exhaustively.
* Execution time can be significant due to Optuna-based hyperparameter optimization.
* Statistical analysis is performed prior to ML training in the Statistical + ML pipeline.

---

## References

* [SHAP documentation](https://shap.readthedocs.io/en/latest/)
* [Optuna documentation](https://optuna.org/)

---


