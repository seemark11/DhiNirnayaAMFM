# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Classification pipeline with statistical analysis and machine learning.

This script implements an end-to-end classification framework including
feature-level statistical analysis, model training, evaluation, and
explainability.

Main components:
- Statistical feature analysis using VIF, correlation, non-parametric tests,
  mutual information, and effect sizes
- Feature selection based on multicollinearity (VIF)
- Classification models: Support Vector Classifier (SVC) and Decision Tree
- Hyperparameter optimization using Optuna with stratified cross-validation
- Comprehensive evaluation on train and test sets using accuracy, precision,
  recall, F1-score, and confusion matrix components
- Model explainability using SHAP values and summary visualizations
- Result and artifact saving for reproducibility

Intended use:
- Binary classification problems with high-dimensional feature spaces
- Scenarios requiring both predictive performance and statistical interpretability

Author: Seema
"""

# ===============================
# Imports
# ===============================

import os 
import time 
import optuna
from sklearn.pipeline import Pipeline
import numpy as np
import pickle  
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pointbiserialr, spearmanr, mannwhitneyu
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif


# -------------------------
# Helper function: Evaluate model on train and test
# -------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, label="Model", params=None, average='binary', pos_label=1):
    """
    Compute classification performance metrics for train and test sets.

    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion matrix components (TN, FP, FN, TP)

    Supports binary and multi-class settings via averaging strategy.
    """

        
    results = {"model": label}
    
    # Store hyperparameters (if provided)
    if params:
        for k, v in params.items():
            results[f"param_{k}"] = v

    # -------- Training metrics --------
    y_train_pred = model.predict(X_train)
    results["train_Accuracy"] = accuracy_score(y_train, y_train_pred) 
    results["train_Precision"] = precision_score(y_train, y_train_pred, average=average, pos_label=pos_label, zero_division=0) 
    results["train_Recall"] = recall_score(y_train, y_train_pred, average=average, pos_label=pos_label, zero_division=0) 
    results["train_F1-score"] = f1_score(y_train, y_train_pred, average=average, pos_label=pos_label, zero_division=0)
    
    # Confusion matrix for training
    cm = confusion_matrix(y_train, y_train_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    # tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    results["train_TN"] = tn
    results["train_FP"] = fp
    results["train_FN"] = fn
    results["train_TP"] = tp

    # -------- Test metrics --------
    y_test_pred = model.predict(X_test)
    results["test_Accuracy"] = accuracy_score(y_test, y_test_pred)
    results["test_Precision"] = precision_score(y_test, y_test_pred, average=average, pos_label=pos_label, zero_division=0)
    results["test_Recall"] = recall_score(y_test, y_test_pred, average=average, pos_label=pos_label, zero_division=0)
    results["test_F1-score"] = f1_score(y_test, y_test_pred, average=average, pos_label=pos_label, zero_division=0)    
    # results["test_Confusion_Matrix"] = confusion_matrix(y_test, y_test_pred).tolist()  # Convert to list
    # Confusion matrix for test
    cm = confusion_matrix(y_test, y_test_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    # tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    results["test_TN"] = tn
    results["test_FP"] = fp
    results["test_FN"] = fn
    results["test_TP"] = tp

    return results

# --------------------------------------------------
# Optuna objective: Support Vector Classifier
# --------------------------------------------------
def objective_svc(trial):
    """
    Hyperparameter optimization objective for SVC.
    Uses stratified cross-validation accuracy.
    """
        
    C = trial.suggest_loguniform("C", 1e-2, 1e2)
    gamma = trial.suggest_loguniform("gamma", 1e-4, 1e0)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])

    # Define model inside pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42))
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

    scores = cross_val_score(model, X_train, y_train, 
                             cv=cv, scoring="accuracy").mean()
    return scores

# --------------------------------------------------
# Optuna objective: Decision Tree Classifier
# --------------------------------------------------
def objective_tree(trial):
    """
    Hyperparameter optimization objective for Decision Tree.
    """
        
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])


    # Define model inside pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42))
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

    scores = cross_val_score(model, X_train, y_train, 
                             cv=cv, scoring="accuracy").mean()
    return scores


# --------------------------------------------------
# Save trained model and compute SHAP explanations
# --------------------------------------------------
def save_model_and_shap(model, X_train, X_test, model_name="model", shap_dir="shap_outputs"):
    """
    Saves a trained model and computes SHAP values for feature explainability.
    SHAP values and plots are saved automatically.

    Parameters:
        model : trained scikit-learn model (any regressor or classifier)
        X_train : pd.DataFrame or np.array for training features
        X_test : pd.DataFrame or np.array for test features
        model_name : str, prefix for saved files
        shap_dir : str, folder to save SHAP outputs
    """

    
    # ----- Choose appropriate SHAP explainer -----
    model_str = str(type(model))
    if 'Tree' in model_str:
        explainer = shap.TreeExplainer(model)
    elif 'Linear' in model_str:
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)
    
    # ----- Compute SHAP values -----
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_test)
    
    # ----- Save SHAP values -----
    shap_values_path = os.path.join(shap_dir, f"{model_name}_shap_values.pkl")
    with open(shap_values_path, "wb") as f:
        pickle.dump(shap_values, f)
    print(f"SHAP values saved: {shap_values_path}")
    
    # ----- Save SHAP summary plots -----
    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    bar_plot_path = os.path.join(shap_dir, f"{model_name}_shap_bar.png")
    plt.tight_layout()
    plt.savefig(bar_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar plot saved: {bar_plot_path}")

    # Beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    beeswarm_plot_path = os.path.join(shap_dir, f"{model_name}_shap_beeswarm.png")
    plt.tight_layout()
    plt.savefig(beeswarm_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP beeswarm plot saved: {beeswarm_plot_path}")
    
    return None

# ===============================
# Statistical Feature Selection
# ===============================

#-----------------------------------------------------------
# 1. Multicollinearity analysis via Variance Inflation Factor
#-----------------------------------------------------------
def calculate_vif(X):
    X_const = sm.add_constant(X)
    return pd.DataFrame({
        "feature": X.columns,
        "VIF": [
            variance_inflation_factor(X_const.values, i + 1)
            for i in range(len(X.columns))
        ]
    })

def vif_feature_selection(X, thresh=5.0):
    X_sel = X.copy()

    while True:
        vif_df = calculate_vif(X_sel)
        max_vif = vif_df["VIF"].max()

        if max_vif <= thresh:
            break

        drop_feat = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
        print(f"Dropping {drop_feat} (VIF={max_vif:.2f})")
        X_sel = X_sel.drop(columns=[drop_feat])

    return X_sel, vif_df

#-----------------------------------------------------------
# 2. Effect size computation (Cliff's delta)
#-----------------------------------------------------------
def cliffs_delta(x1, x2):
    """
    Effect size for two independent samples.
    """
    n1, n2 = len(x1), len(x2)
    gt = sum(i > j for i in x1 for j in x2)
    lt = sum(i < j for i in x1 for j in x2)
    return (gt - lt) / (n1 * n2)

#-----------------------------------------------------------
# 3. Feature-label statistical association analysis
#-----------------------------------------------------------
def compute_classification_feature_stats(X, y):
    """
    Computes classification-relevant statistics per feature.
    Assumes binary y encoded as {0,1}.
    """

    mi = mutual_info_classif(X, y, random_state=21)

    rows = []
    for i, col in enumerate(X.columns):

        x0 = X.loc[y == 0, col]
        x1 = X.loc[y == 1, col]

        # Point-biserial correlation
        pb_r, pb_p = pointbiserialr(y, X[col])

        # Spearman correlation
        sp_r, sp_p = spearmanr(X[col], y)

        # Mann–Whitney U test
        mw_stat, mw_p = mannwhitneyu(x0, x1, alternative="two-sided")

        # Effect size
        delta = cliffs_delta(x0.values, x1.values)

        rows.append({
            "feature": col,
            "pointbiserial_r": pb_r,
            "pointbiserial_p": pb_p,
            "spearman_r": sp_r,
            "spearman_p": sp_p,
            "mannwhitney_p": mw_p,
            "cliffs_delta": delta,
            "mutual_info": mi[i]
        })

    return pd.DataFrame(rows)

#-----------------------------------------------------------
# 4. Combined statistical feature summary
#-----------------------------------------------------------
def compute_all_classification_feature_stats(X, y, vif_thresh=5.0):
    """
    Full statistical feature analysis for classification.
    """

    # VIF pruning
    X_sel, vif_df = vif_feature_selection(X, thresh=vif_thresh)

    # Feature-label statistics
    stats_df = compute_classification_feature_stats(X_sel, y)

    # Merge VIF
    stats_df = stats_df.merge(vif_df, on="feature", how="left")

    return X_sel, stats_df


#%% 
# --------------------------------------------------
# Parse command-line arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Classification training")
parser.add_argument("--specwindowsecs", type=int, default=5)
parser.add_argument("--specstrides", type=int, default=200)
args = parser.parse_args()

print("specwindowsecs:", args.specwindowsecs)
print("specstrides:", args.specstrides)

specwindowsecs = args.specwindowsecs
specstrides = args.specstrides

# --------------------------------------------------
# Dataset paths and output directories
# --------------------------------------------------
base_data = Path(<path/to/dataset>)
base_results = Path(<path/to/results/folder>)

suffix = f"specstrides_{specstrides}_specwindowsecs_{specwindowsecs}"
train_path = base_data / f"train_csv_{suffix}"
test_path = base_data / f"test_csv_{suffix}"

working_dir = base_results / suffix
# Create the directory if it doesn't exist
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# Write results for each set
res_df_path_svc = working_dir / "classification_svc_combined_results.csv"   
res_df_svc = pd.DataFrame(["feat_set", "dct_num", "num_R_form", 
                           "model", "param_C", "param_gamma", "param_kernel",                         
                           "train_Accuracy", "train_Precision", "train_Recall", "train_F1-score", 
                           "train_TN", "train_FP", "train_FN", "train_TP",                            
                           "test_Accuracy", "test_Precision", "test_Recall", "test_F1-score", 
                           "test_TN", "test_FP", "test_FN", "test_TP"]).transpose()
res_df_svc.to_csv(res_df_path_svc, header=False, index=False)
 
res_df_path_dt = working_dir / "classification_dt_combined_results.csv"   
res_df_dt = pd.DataFrame(["feat_set", "dct_num", "num_R_form", 
                           "model", "param_max_depth", "param_min_samples_split", "param_min_samples_leaf", "param_max_features", 
                           "train_Accuracy", "train_Precision", "train_Recall", "train_F1-score", 
                           "train_TN", "train_FP", "train_FN", "train_TP",  
                           "test_Accuracy", "test_Precision", "test_Recall", "test_F1-score", 
                           "test_TN", "test_FP", "test_FN", "test_TP"]).transpose()
res_df_dt.to_csv(res_df_path_dt, header=False, index=False)
    
     
#%%
# ===============================
# Experimental configurations
# ===============================
dct_nums = [2, 3, 4, 5, 6] 
num_R_forms = [4, 5, 6, 7, 8]
feat_set = ["variance", "ddct", "combined"]
vif_thresh = 5

# Loop over each file
for feat_type in feat_set:
        
    for dct_num in dct_nums:
        
        for num_R_form in num_R_forms:
            
            # Path to train/test feature files
            ip_file_path_train = Path(train_path / f"dct_num_{dct_num}_num_R_form_{num_R_form}_{feat_type}.csv")   
            ip_file_path_test = Path(test_path / f"dct_num_{dct_num}_num_R_form_{num_R_form}_{feat_type}.csv")  
                   
            # Output paths 
            stat_summary_csv_path = "classification_feature_stats_combined.csv"
            model_path_svc = "best_SVC_model.pkl"
            model_path_dt = "best_DT_model.pkl"
            scaler_path = "scaler_SVC.pkl"
            
            #%% Load train/test feature files
            # Train data
            train_df = pd.read_csv(ip_file_path_train)
            # Shuffle dataset
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            # Separate features and labels
            X_train = train_df.drop(columns=['mmse', 'dx'])
            X_train.isna().sum()
            y_train = train_df['dx']     
            y_train.isna().sum()
            
            # Test data
            test_df = pd.read_csv(ip_file_path_test)
            # Separate features and labels
            X_test = test_df.drop(columns=['mmse', 'dx'])
            X_test.isna().sum()
            y_test = test_df['dx']     
            y_test.isna().sum()
            
            # Create a LabelEncoder object
            label_encoder = LabelEncoder()
            # Convert labels to numeric values
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            #%% Feature selection
            X_train, classification_stats_df = compute_all_classification_feature_stats(X_train, y_train, vif_thresh=vif_thresh)
            
            # Pick the same features from test set
            X_test = X_test[X_train.columns]

            #%% Scale features
            # Initialize the scaler
            scaler = StandardScaler()        
    
            # Fit on training data and transform, keeping columns and index
            X_train_scaler = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            
            # Transform test data using the same scaler, keeping columns and index
            X_test_scaler = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)

            # Output directory for this configuration
            result_dir = f"classification/num_R_form_{num_R_form}/dct_num_{dct_num}/feat_type_{feat_type}"
            # Convert to absolute path 
            result_dir = os.path.abspath(result_dir)
            os.makedirs(result_dir, exist_ok=True)
            os.chdir(result_dir)
            
            #%%
            # Save feature stats
            classification_stats_df.to_csv(stat_summary_csv_path, index=False)

            # Save the scaler
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        
            del ip_file_path_train, ip_file_path_test
            del train_df, test_df
            
            #%%           
            # Start timer
            start_time = time.time()
            
            # -------------------------
            # Train and evaluate SVC
            # -------------------------
            print("Tuning SVC \n")
            study_svc = optuna.create_study(direction="maximize")
            study_svc.optimize(objective_svc, n_trials=150)

            best_svc = SVC(**study_svc.best_params, probability=True, random_state=21)
            best_svc.fit(X_train_scaler, y_train)
            
            # End timer
            end_time = time.time()
            
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Grid Search completed in {elapsed_time:.2f} seconds.")
            
            # Save the model using pickle
            with open(model_path_svc, 'wb') as model_file:
                pickle.dump(best_svc, model_file)
            print(f"Model saved to {model_path_svc}.")  
            
            # Get results for train and test set
            svc_results = evaluate_model(best_svc, X_train_scaler, y_train, X_test_scaler, y_test,
                                         label="SVC", params=study_svc.best_params)    
                              
            # -------------------------
            # Save results to CSV
            # -------------------------            
            df_scalars = pd.DataFrame({
                'feat_type': [feat_type],
                'dct_num': [dct_num],
                'num_R_form': [num_R_form]})
            
            df_svc = pd.DataFrame([svc_results])
            
            res_df = pd.concat([df_scalars, df_svc], axis=1)            
            res_df.to_csv(res_df_path_svc, header=False, index=False, mode='a')
            
            del res_df
            del svc_results

            # SHAP values
            save_model_and_shap(model=best_svc, X_train=X_train_scaler, X_test=X_test_scaler, model_name="best_SVC", shap_dir=result_dir)
            
            #%%                            
            # Start timer
            start_time = time.time()
            
            # -------------------------
            # Train and evaluate Decision Tree
            # -------------------------
            print("Tuning Decision Tree \n")
            study_tree = optuna.create_study(direction="maximize")
            study_tree.optimize(objective_tree, n_trials=150) 
            
            best_tree = DecisionTreeClassifier(**study_tree.best_params, random_state=21)
            best_tree.fit(X_train_scaler, y_train)
            
            # End timer
            end_time = time.time()
            
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Grid Search completed in {elapsed_time:.2f} seconds.")
            
            # Save the model using pickle
            with open(model_path_dt, 'wb') as model_file:
                pickle.dump(best_tree, model_file)
            print(f"Model saved to {model_path_dt}.")  
            
            # Get results for train and test set
            tree_results = evaluate_model(best_tree, X_train_scaler, y_train, X_test_scaler, y_test,
                                          label="DT", params=study_tree.best_params)
            
            # -------------------------
            # Save results to CSV
            # -------------------------            
            df_scalars = pd.DataFrame({
                'feat_type': [feat_type],
                'dct_num': [dct_num],
                'num_R_form': [num_R_form]})
            
            df_dt = pd.DataFrame([tree_results])
            
            res_df = pd.concat([df_scalars, df_dt], axis=1)            
            res_df.to_csv(res_df_path_dt, header=False, index=False, mode='a')           
            
            del res_df
            del tree_results
            
            # SHAP values
            save_model_and_shap(model=best_tree, X_train=X_train_scaler, X_test=X_test_scaler, model_name="best_DT", shap_dir=result_dir)
            
            os.chdir(working_dir)
