# -*- coding: utf-8 -*-
"""
Generate a list of processed training audio files.

Traverses the processed training audio directory and collects
paths of all combined .wav files (from 'ad' and 'cn' subfolders).
The resulting list is saved to a text file for downstream use.

Author: Seema
"""

import os

# Base output directory containing processed audio
output_base_path = <path/to/train/audio/files>
# Path to the text file that will store the list of audio files
output_list_file = <path/to/processed/train/audio/txt/file>

#%%
# Ensure the output directory exists before proceeding
if not os.path.exists(output_base_path):
    print("Output directory does not exist. Exiting.")
    exit()
#%%
# List to store full paths of processed audio files
processed_files = []

# Traverse the output directory and collect .wav files
for subfolder in ["ad", "cn"]:
    subfolder_path = os.path.join(output_base_path, subfolder)
    
    if os.path.exists(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.endswith(".wav"):
                file_path = os.path.normpath(os.path.join(subfolder_path, file))
                processed_files.append(file_path)
#%%
# Write collected file paths to a text file (one path per line)
with open(output_list_file, "w") as f:
    for file_path in processed_files:
        f.write(file_path + "\n")

print(f"List of processed files saved to: {output_list_file}")
#%%