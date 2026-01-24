#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for feature extraction and dataset construction.

This script:
1. Reads preprocessed audio file paths
2. Associates them with labels
3. Extracts AM-FM spectrogram-based features
4. Iterates over multiple parameter configurations
5. Saves merged feature-label CSVs for downstream modeling

Author: Seema
"""

print(f"\n==== Entered code ====")

import os 
import numpy as np
import pandas as pd
import argparse
from utils import *
from get_am_fm_spectrogram_feats import *	# Low frequency spectrogram functions
from rfa_single_conf import *

print(f"\n==== Libraries loaded ====")

#%% Parse command-line arguments
parser = argparse.ArgumentParser(description="AM-FM spectrogram feature extraction")
parser.add_argument("--specwindowsecs", type=int, default=5,
                    help="Spectrogram window length in seconds")
parser.add_argument("--specstrides", type=int, default=200,
                    help="Spectrogram stride length")

args = parser.parse_args()

print("specwindowsecs:", args.specwindowsecs)
print("specstrides:", args.specstrides)

# Assign CLI arguments to variables
specwindowsecs = args.specwindowsecs
specstrides = args.specstrides

# Parameter sweeps
dct_nums = [2, 3, 4, 5, 6] 
num_R_forms = [4, 5, 6, 7, 8]
 
#%%
# Text file containing paths to processed (combined) audio files
processed_files_txt = <path/to/processed/test/audio/txt/file>
# CSV containing labels for each sample
classification_labels_file_csv = <path/to/processed/train/classification/label/file>
regression_labels_file_csv = <path/to/processed/train/regression/label/file>
# Output directory for merged feature CSVs
merged_csv = Path(<path/to/processed/test_csv_specstrides_" + str(specstrides) + "_specwindowsecs_" + str(specwindowsecs) + "/")     
# Create output directory if it does not exist
merged_csv.mkdir(parents=True, exist_ok=True)
print(f"Folder '{merged_csv}' is ready")

#%%
# Read processed file list
processed_files = []
if os.path.exists(processed_files_txt):
    with open(processed_files_txt, "r") as f:
        processed_files = [line.strip() for line in f.readlines()]

# Extract sample names by removing suffixes
# Assumes filenames follow: <sample>_combined.wav
processed_files_clean = [os.path.basename(f).replace("_combined", "").replace(".wav", "") for f in processed_files]

# Create DataFrame linking sample names to audio paths
df_processed = pd.DataFrame({"file_name": processed_files_clean, "processed_path": processed_files})

#%%
# Read labels CSV
# Classification labels
classification_labels = pd.read_csv(classification_labels_file_csv)
# process classification labels 
classification_labels["dx"] = classification_labels["Dx"].map({"Control": "cn", "ProbableAD": "ad"})
classification_labels.drop(columns = ["Dx"], inplace=True)

# Regression labels
regression_labels = pd.read_csv(regression_labels_file_csv)
# Rename the column from 'MMSE' to 'mmse'
regression_labels.rename(columns={"MMSE": "mmse"}, inplace=True)

# Merge classification and regression labels
df_labels = pd.merge(classification_labels, regression_labels, on="ID")
df_labels.rename(columns={"ID": "adressfname"}, inplace=True)

#%%
# Extract features for each wav file
# Outer loop: dct_num
for dct_num in dct_nums:
    print(f"\n==== Running for dct_num={dct_num} ====")

    # Inner loop: num_R_form
    for num_R_form in num_R_forms:
        print(f"   -> Processing with num_R_form={num_R_form}")
        
        features_list = []
        
        for idx, row in df_processed.iterrows():
            
            sample_name = row["file_name"]
            wav_path = row["processed_path"]
        
            features = extract_features_spectrogram(wav_path = wav_path, 
                                         specwindowsecs = specwindowsecs, 
                                         specstrides = specstrides, 
                                         dct_num = dct_num, 
                                         num_R_form = num_R_form)
        
            # Add metadata (sample_name, wav_path) into the Series
            meta = pd.Series({"sample_name": sample_name, "wav_path": wav_path})
        
            # Concatenate metadata + features (metadata first)
            processed_data = pd.concat([meta, features])
            
            features_list.append(processed_data)   
        
        #%%
        # Concatenate the arrays
        features_df = pd.DataFrame(features_list)
        #%%     
        # Merging df_labels and df_processed on different columns
        df_merged = pd.merge(df_labels, features_df, left_on="adressfname", right_on="sample_name", how="inner")

        #%% Save merged CSV
        output_file_path = os.path.normpath(os.path.join(merged_csv, f"dct_num_{dct_num}_num_R_form_{num_R_form}_combined.csv"))
                    
        df_merged.to_csv(output_file_path, index=False)
        
        print(f"Merged file saved to: {merged_csv}\n")
        print("====================================\n")
#%%