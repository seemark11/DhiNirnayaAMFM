# -*- coding: utf-8 -*-
"""
Pre-process test audio files by extracting and concatenating
segments spoken by the participant ('PAR').

For each .wav file:
- Reads corresponding segmentation CSV
- Extracts all PAR segments
- Concatenates them into a single audio file
- Saves the combined audio in a mirrored directory structure

Author: Seema
"""

from pydub import AudioSegment
import pandas as pd
import os

# Base paths
# audio_base_path: root directory containing original .wav files
# csv_base_path: root directory containing segmentation CSVs
# output_base_path: root directory to store processed audio
audio_base_path = <path/to/test/audio/files>
csv_base_path = <path/to/test/audio/segmentation/details>
output_base_path = <path/to/output/test/audio/files>

# Ensure the output directory exists
os.makedirs(output_base_path, exist_ok=True)

total_files = 0
processed_files = 0

#%%   
# Iterate over all WAV files in the audio directory
for audio_file in os.listdir(audio_base_path):
    if audio_file.endswith(".wav"):
        # Corresponding CSV file
        file_id = os.path.splitext(audio_file)[0]  # Get the filename without extension
        csv_file = f"{file_id}.csv"
        audio_file_path = os.path.normpath(os.path.join(audio_base_path, audio_file))
        csv_file_path = os.path.normpath(os.path.join(csv_base_path, csv_file))
        
        # Skip files without corresponding CSV
        if os.path.exists(csv_file_path):
            print(f"Processing {audio_file} with {csv_file}")
            total_files += 1
            
            # Load segmentation and audio
            df = pd.read_csv(csv_file_path)
            audio = AudioSegment.from_wav(audio_file_path)
            
            # Select participant speech segments only
            par_segments = df[df['speaker'] == 'PAR']
            
            # Check if the filtered DataFrame is empty
            if par_segments.empty:
                print(f"No 'PAR' segments found in {audio_file}. Skipping to next file.")
                
            else:
                # Process the non-empty par_segments
                print(f"Processing {audio_file} with {len(par_segments)} segments.")

                # Initialize an empty AudioSegment for the combined result
                combined_audio = AudioSegment.empty()
                
                # Extract and concatenate all PAR segments
                for _, row in par_segments.iterrows():
                    # Assumes begin and end are in milliseconds
                    start_time = row['begin']  
                    end_time = row['end']      
                    
                    # Extract the segment and append it to the combined audio
                    segment = audio[start_time:end_time]
                    combined_audio += segment
                
                # Save concatenated audio
                output_file_path = os.path.normpath(os.path.join(output_base_path, f"{file_id}_combined.wav"))
                combined_audio.export(output_file_path, format="wav")
                print(f"Combined audio saved to: {output_file_path}")
                processed_files += 1
        else:
            print(f"CSV file not found for {audio_file}")
                
#%%
print(f'\n Total # of files {total_files}.')
print(f'# files processed {processed_files}.')
