# -*- coding: utf-8 -*-
"""
Regression pipeline with statistical and ML modeling.

Features:
- Feature selection via VIF, correlation, mutual information, p-values
- Statistical effect sizes (standardized β, Cohen's f²)
- Regression models: SVR, Decision Tree
- Hyperparameter tuning with Optuna
- SHAP explainability
- Result saving for reproducibility

@author: seema
"""

# ===============================
# Imports
# =============================== 
import os 
import time
import optuna
import argparse
import numpy as np
import pickle  
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

#%%
# -------------------------
# Helper function: Model evaluation
# -------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, label="Model", params=None):
    """
    Compute regression metrics for train and test sets.
    Metrics: RMSE, MAE, R², Pearson correlation
    """
        
    results = {"model": label}
    
    # Add best hyperparameters if available
    if params:
        for k, v in params.items():
            results[f"param_{k}"] = v

    # Train set evaluation
    y_train_pred = model.predict(X_train)
    results["train_rmse"] = np.sqrt(mean_squared_error(y_train, y_train_pred))
    results["train_mae"] = mean_absolute_error(y_train, y_train_pred)
    results["train_r2"] = r2_score(y_train, y_train_pred)
    results["train_pearson"], _ = pearsonr(y_train, y_train_pred)

    # Test set evaluation
    y_test_pred = model.predict(X_test)
    results["test_rmse"] = np.sqrt(mean_squared_error(y_test, y_test_pred))
    results["test_mae"] = mean_absolute_error(y_test, y_test_pred)
    results["test_r2"] = r2_score(y_test, y_test_pred)
    results["test_pearson"], _ = pearsonr(y_test, y_test_pred)

    return results

#%%
# -------------------------
# Optuna objective for SVR
# -------------------------
def objective_svr(trial):
    C = trial.suggest_loguniform("C", 1e-2, 1e2)
    epsilon = trial.suggest_loguniform("epsilon", 1e-3, 1)
    gamma = trial.suggest_loguniform("gamma", 1e-4, 1e0)

    # Define model inside pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
    ])
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=21)

    scores = cross_val_score(model, X_train, y_train, 
                             cv=cv, scoring='neg_mean_squared_error').mean()
        
    return scores

# -------------------------
# Optuna objective for Decision Tree
# -------------------------
def objective_tree(trial):
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])

    # Define model inside pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=21))
    ])

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=21)

    scores = cross_val_score(model, X_train, y_train, 
                             cv=cv, scoring='neg_mean_squared_error').mean()
     
    return scores

#%%
# ===============================
# SHAP explainability
# ===============================
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
# 1. VIF computation and feature selection
#-----------------------------------------------------------
def calculate_vif(df):
    """
    Compute VIF for each feature
    """

    df_const = sm.add_constant(df)
    vif = pd.DataFrame()
    vif["feature"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df_const.values, i + 1)
                  for i in range(len(df.columns))]
    return vif

def vif_feature_selection(X, thresh=5):
    """
    Remove features iteratively with VIF > thresh
    """
        
    X_copy = X.copy()
    while True:
        vif_df = calculate_vif(X_copy)
        max_vif = vif_df["VIF"].max()
        if max_vif <= thresh:
            break
        remove_feature = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
        print(f"Removing {remove_feature} with VIF={max_vif:.2f}")
        X_copy = X_copy.drop(columns=[remove_feature])
    return X_copy, vif_df

#-----------------------------------------------------------
# 2. Compute Pearson, Spearman, Mutual Information
#-----------------------------------------------------------
def compute_correlations_and_mi(X, y):
    stats_list = []

    # Mutual information
    mi = mutual_info_regression(X, y, random_state=21)
    mi_dict = dict(zip(X.columns, mi))

    for col in X.columns:
        try:
            p_corr, p_val = pearsonr(X[col], y)
        except:
            p_corr, p_val = np.nan, np.nan

        try:
            s_corr, s_val = spearmanr(X[col], y)
        except:
            s_corr, s_val = np.nan, np.nan

        stats_list.append({
            "feature": col,
            "pearson_r": p_corr,
            "pearson_p": p_val,
            "spearman_r": s_corr,
            "spearman_p": s_val,
            "mutual_info": mi_dict[col]
        })

    return pd.DataFrame(stats_list)

#-----------------------------------------------------------
# 3. Compute OLS p-values
#-----------------------------------------------------------
def compute_ols_pvalues(X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model.pvalues.drop("const")

#-----------------------------------------------------------
# 4. Compute standardized effect size (β_std)
#-----------------------------------------------------------
def compute_standardized_effect_size(X, y):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_std = scaler_X.fit_transform(X)
    y_std = scaler_y.fit_transform(y.values.reshape(-1,1)).ravel()

    X_std_const = sm.add_constant(X_std)
    model = sm.OLS(y_std, X_std_const).fit()
    effect_size = pd.Series(model.params[1:], index=X.columns)
    return effect_size

#-----------------------------------------------------------
# 5. Compute Cohen's f²
#-----------------------------------------------------------
def cohens_f2(X, y):
    model_full = LinearRegression()
    model_full.fit(X, y)
    r2_full = r2_score(y, model_full.predict(X))
    
    f2_dict = {}
    for col in X.columns:
        X_reduced = X.drop(columns=[col])
        model_reduced = LinearRegression()
        model_reduced.fit(X_reduced, y)
        r2_reduced = r2_score(y, model_reduced.predict(X_reduced))
        f2 = (r2_full - r2_reduced) / (1 - r2_full) if r2_full < 1 else 0
        f2_dict[col] = f2
    
    return pd.Series(f2_dict)

#-----------------------------------------------------------
# 6. Combine all stats into a single DataFrame
#-----------------------------------------------------------
def compute_all_feature_stats(X, y, vif_thresh=5):
    # VIF-based selection
    X_selected, vif_df = vif_feature_selection(X, thresh=vif_thresh)

    # Correlations + MI
    corr_mi_df = compute_correlations_and_mi(X_selected, y)

    # OLS p-values
    pvals = compute_ols_pvalues(X_selected, y)

    # Standardized effect size
    effect_size = compute_standardized_effect_size(X_selected, y)

    # Cohen's f²
    f2_values = cohens_f2(X_selected, y)

    # Merge all stats
    stats_df = corr_mi_df.merge(vif_df, on="feature", how="left")
    stats_df["ols_pvalue"] = stats_df["feature"].map(pvals)
    stats_df["effect_size_std"] = stats_df["feature"].map(effect_size)
    stats_df["cohens_f2"] = stats_df["feature"].map(f2_values)

    return X_selected, stats_df

# ===============================
# CLI arguments
# ===============================
parser = argparse.ArgumentParser(description="Regression with statistical feature analysis")
parser.add_argument("--specwindowsecs", type=int, default=5)
parser.add_argument("--specstrides", type=int, default=200)
args = parser.parse_args()

print("specwindowsecs:", args.specwindowsecs)
print("specstrides:", args.specstrides)
specwindowsecs = args.specwindowsecs
specstrides = args.specstrides

# ===============================
# Paths
# ===============================
base_data = Path(<path/to/dataset>)
base_results = Path(<path/to/results/folder>)

suffix = f"specstrides_{specstrides}_specwindowsecs_{specwindowsecs}"
train_path = base_data / f"train_csv_{suffix}"
test_path = base_data / f"test_csv_{suffix}"
working_dir = base_results / suffix
# Create the directory if it doesn't exist
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# ===============================
# Result CSV initialization
# ===============================
res_df_path_svr = working_dir / "regression_svr_combined_results.csv"   
res_df_svr = pd.DataFrame(["feat_set", "dct_num", "num_R_form", 
                           "model", "param_C", "param_epsilon", "param_gamma",                         
                           "train_rmse", "train_mae", "train_r2", "train_pearson", 
                           "test_rmse", "test_mae", "test_r2", "test_pearson"]).transpose()
res_df_svr.to_csv(res_df_path_svr, header=False, index=False)
 
res_df_path_dt = working_dir / "regression_dt_combined_results.csv"   
res_df_dt = pd.DataFrame(["feat_set", "dct_num", "num_R_form", 
                           "model", "param_max_depth", "param_min_samples_split", "param_min_samples_leaf", "param_max_features", 
                           "train_rmse", "train_mae", "train_r2", "train_pearson", 
                           "test_rmse", "test_mae", "test_r2", "test_pearson"]).transpose()
res_df_dt.to_csv(res_df_path_dt, header=False, index=False)
    
#%%
# ===============================
# Experiment grid
# ===============================
dct_nums = [2, 3, 4, 5, 6] 
num_R_forms = [4, 5, 6, 7, 8]
feat_set = ["variance", "ddct", "combined"]
vif_thresh = 5

# Loop over each file
for feat_type in feat_set:
        
    for dct_num in dct_nums:
        
        for num_R_form in num_R_forms:
            
            # Obtain train/test csv path
            ip_file_path_train = Path(train_path / f"dct_num_{dct_num}_num_R_form_{num_R_form}_{feat_type}.csv")  # Update with actual file path pattern   
            ip_file_path_test = Path(test_path / f"dct_num_{dct_num}_num_R_form_{num_R_form}_{feat_type}.csv")  # Update with actual file path pattern   
                      
            # Output paths 
            stat_summary_csv_path = "regression_feature_stats_combined.csv"
            model_path_svr = "best_SVR_model.pkl"
            model_path_dt = "best_DT_model.pkl"
            scaler_path = "scaler_SVR.pkl"
            
            #%% # Load train/test data
            # Train data
            train_df = pd.read_csv(ip_file_path_train)
            # Shuffle dataset
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            # train_df = train_df.iloc[0:50, :]
            X_train = train_df.drop(columns=['mmse', 'dx'])
            X_train.isna().sum()
            y_train = train_df['mmse']     
            y_train.isna().sum()
            
            # Test data
            test_df = pd.read_csv(ip_file_path_test)
            X_test = test_df.drop(columns=['mmse', 'dx'])
            X_test.isna().sum()
            y_test = test_df['mmse']     
            y_test.isna().sum()
            
            #%% Feature selection
            X_train, regression_stats_df = compute_all_feature_stats(X_train, y_train, vif_thresh=vif_thresh)
            
            # Pick the same features from test set
            X_test = X_test[X_train.columns]

            #%% Standardization
            # Initialize the scaler
            scaler = StandardScaler()
    
            # Fit on training data and transform, keeping columns and index
            X_train_scaler = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index).copy()
            
            # Transform test data using the same scaler, keeping columns and index
            X_test_scaler = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index).copy()

            # Generate path to save results 
            result_dir = f"regression/num_R_form_{num_R_form}/dct_num_{dct_num}/feat_type_{feat_type}"
            # Convert to absolute path 
            result_dir = os.path.abspath(result_dir)
            os.makedirs(result_dir, exist_ok=True)
            os.chdir(result_dir)
            
            #%%
            # Save feature stats
            regression_stats_df.to_csv(stat_summary_csv_path, index=False)

            # Save the scaler
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        
            del ip_file_path_train, ip_file_path_test
            del train_df, test_df
            
            #%%           
            # Start timer
            start_time = time.time()
            
            # -------------------------
            # SVR Training & SHAP
            # -------------------------
            print("Tuning SVR \n")
            study_svr = optuna.create_study(direction="maximize")
            study_svr.optimize(objective_svr, n_trials=150)

            best_svr = SVR(**study_svr.best_params, kernel="rbf")
            best_svr.fit(X_train_scaler, y_train)
            
            # End timer
            end_time = time.time()
            
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Grid Search completed in {elapsed_time:.2f} seconds.")
            
            # Save the model using pickle
            with open(model_path_svr, 'wb') as model_file:
                pickle.dump(best_svr, model_file)
            print(f"Model saved to {model_path_svr}.")  
            
            # Get results for train and test set
            svr_results = evaluate_model(best_svr, X_train_scaler, y_train, X_test_scaler, y_test,
                                         label="SVR", params=study_svr.best_params)    
                              
            # -------------------------
            # Save results to CSV
            # -------------------------            
            df_scalars = pd.DataFrame({
                'feat_type': [feat_type],
                'dct_num': [dct_num],
                'num_R_form': [num_R_form]})
            
            df_svr = pd.DataFrame([svr_results])
            
            res_df = pd.concat([df_scalars, df_svr], axis=1)            
            res_df.to_csv(res_df_path_svr, header=False, index=False, mode='a')
            
            
            del res_df
            del svr_results

            # SHAP values
            save_model_and_shap(model=best_svr, X_train=X_train_scaler, X_test=X_test_scaler, model_name="best_SVR", shap_dir=result_dir)
                    
            #%%                            
            # Start timer
            start_time = time.time()
            
            # -------------------------
            # Decision Tree Training & SHAP
            # -------------------------
            print("Tuning Decision Tree \n")
            study_tree = optuna.create_study(direction="maximize")
            study_tree.optimize(objective_tree, n_trials=150)
            
            best_tree = DecisionTreeRegressor(**study_tree.best_params, random_state=42)
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
