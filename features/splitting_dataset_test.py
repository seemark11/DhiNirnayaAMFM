# -*- coding: utf-8 -*-
"""
Split extracted feature datasets into different feature groups.

For each (dct_num, num_R_form) configuration, this script:
1. Loads the combined feature CSV
2. Separates variance-based AM/FM features
3. Separates DCT-based AM/FM features
4. Creates three datasets:
   - variance features only
   - DCT features only
   - combined variance + DCT features
5. Retains only selected label/metadata columns

Author: Seema
"""

import os
import pandas as pd
from pathlib import Path
import argparse

#%% Parse command-line arguments
parser = argparse.ArgumentParser(description="Split feature datasets by feature type")
parser.add_argument("--specwindowsecs", type=int, default=5,
                    help="Spectrogram window length in seconds")
parser.add_argument("--specstrides", type=int, default=200,
                    help="Spectrogram stride length")

args = parser.parse_args()

print("specwindowsecs:", args.specwindowsecs)
print("specstrides:", args.specstrides)
specwindowsecs = args.specwindowsecs
specstrides = args.specstrides

#%%
wrking_dr = Path(<path/to/processed/test_csv_specstrides_" + str(specstrides) + "_specwindowsecs_" + str(specwindowsecs) + "/")      
os.chdir(wrking_dr)

#%%
# Parameter combinations used during feature extraction
dct_nums = [2, 3, 4, 5, 6] 
num_R_forms = [4, 5, 6, 7, 8]

# Select only 'mmse' and 'dx' as additional columns
additional_columns = ["mmse", "dx"]

# Loop through all feature configuration files
for dct_num in dct_nums:
    for num_R_form in num_R_forms:
        
        file_path = f"dct_num_{dct_num}_num_R_form_{num_R_form}_combined.csv"
    
        # Load combined feature dataset
        features_df = pd.read_csv(file_path)
    
        # Identify variance-based AM/FM features
        cols_var_am_fm = [c for c in features_df.columns if c.startswith(("var_am_", "var_fm_"))]
        # Identify DCT-based AM/FM features
        cols_dct_am_fm = [c for c in features_df.columns if c.startswith(("dct_am_", "dct_fm_"))]
        
        # Create feature subsets
        df_group1 = features_df[additional_columns + cols_var_am_fm]
        df_group2 = features_df[additional_columns + cols_dct_am_fm]
        df_full = features_df[additional_columns + cols_var_am_fm + cols_dct_am_fm]

        # Save split datasets
        df_group1.to_csv(f"dct_num_{dct_num}_num_R_form_{num_R_form}_variance.csv", index=False)
        df_group2.to_csv(f"dct_num_{dct_num}_num_R_form_{num_R_form}_ddct.csv", index=False)
        df_full.to_csv(f"dct_num_{dct_num}_num_R_form_{num_R_form}_combined.csv", index=False)
    
        print(f"Processed and saved files for dct_num_{dct_num}_num_R_form_{num_R_form}")
