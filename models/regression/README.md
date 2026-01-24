# Regression Project: ML and Statistical Pipelines

This repository implements regression models for predicting continuous outcomes (e.g., MMSE) from precomputed feature CSV files.  
It includes **two pipelines**:

1. **ML Regression Pipeline** – purely machine learning with hyperparameter tuning.  
2. **Statistical + ML Regression Pipeline** – incorporates feature selection using classical statistics and effect size measures before ML modeling.

---

## Features

### 1. ML Regression Pipeline
- Models:
  - Support Vector Regressor (SVR)
  - Decision Tree Regressor (DT)
- Hyperparameter tuning with **Optuna** (5-fold cross-validation)
- Feature scaling using **StandardScaler**
- Performance metrics: RMSE, MAE, R², Pearson correlation
- Model explainability using **SHAP**
- Results saved to CSV

### 2. Statistical + ML Regression Pipeline
- Statistical feature selection:
  - **VIF** (Variance Inflation Factor)
  - **OLS p-values**
  - **Pearson & Spearman correlation**
  - **Mutual information**
  - **Standardized effect size**
  - **Cohen's f²**
- Selected features fed into the same ML models (SVR, DT)
- SHAP explainability and results saved similarly

---

## Directory Structure

```

base_data/
├── train_csv_specstrides_<value>*specwindowsecs*<value>/
└── test_csv_specstrides_<value>*specwindowsecs*<value>/

base_results/
└── specstrides_<value>*specwindowsecs*<value>/
├── regression_svr_combined_results.csv
├── regression_dt_combined_results.csv
└── regression/
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
  - Auxiliary column: `mmse` (target for regression, excluded from features)
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
python train_regression.py --specwindowsecs 5 --specstrides 200
```

| Argument           | Description                        | Default |
| ------------------ | ---------------------------------- | ------- |
| `--specwindowsecs` | Spectrogram window size in seconds | 5       |
| `--specstrides`    | Spectrogram stride                 | 200     |

---

## Preprocessing

* Features are standardized using **StandardScaler**
* Labels (`mmse`) remain unscaled for regression
* Scalers saved for each configuration

---

## Evaluation Metrics

Computed for **both train and test sets**:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² score
* Pearson correlation coefficient

Results appended to:

* `regression_svr_combined_results.csv`
* `regression_dt_combined_results.csv`

---

## Model Saving & Explainability

* Trained models saved as `.pkl`
* SHAP values saved as `.pkl`
* SHAP summary plots automatically generated:

  * Bar plot
  * Beeswarm plot

---

## Notes

* Pipelines perform **exhaustive training** across all feature sets and parameters.
* Execution can be **time-consuming** due to Optuna optimization.
* Statistical + ML pipeline applies **feature selection** before ML modeling.

---

## References

* [SHAP documentation](https://shap.readthedocs.io/en/latest/)
* [Optuna documentation](https://optuna.org/)

---


