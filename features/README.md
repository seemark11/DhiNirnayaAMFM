## Setup

1. **Clone the repository**

```bash
git clone https://github.com/seemark11/DhiNirnayaAMFM
cd DhiNirnayaAMFM
```

2. **Create and activate the environment**

```bash 
conda env create -f features/environment.yaml
conda activate rfa_env   
```

---

## Workflow

### 1. Pre-process audio files (`pre_processing_audio_files_train.py` and `pre_processing_audio_files_test.py`)

**Purpose**: Extracts and concatenates segments spoken by the participant ("PAR") from raw audio files.

* **Input**: 
  * Directory containing `.wav` audio files.
  * Corresponding segmentation `.csv` files

* **Output**: Processed audio files named `<sample>_combined.wav`


### 2. Generate list of usable audio files (`list_audio_files_train.py` and `list_audio_files_test.py`)

**Purpose**: Creates a text file containing paths to all processed audio files to be used for feature extraction.

* **Input**: Directory containing processed `_combined.wav` files
* **Output**: `processed_files.txt` containing one audio file path per line


### 3. Extract features (`main.py` and `main_test.py`)

**Purpose**: Extracts AM–FM spectrogram-based features from processed audio files and merges them with labels.
* **Inputs**:

  * `.txt` file containing paths to processed `.wav` files
  * `.csv` file containing labels
  * Output directory for saving extracted features

* **Command-line arguments**:

  * --specwindowsecs: Spectrogram window length in seconds
  * --specstrides: Spectrogram stride length

* **Output**: `.csv` files containing extracted features for different parameter configurations
(varying dct_num and num_R_form)


### 4. Split the features (`splitting_dataset_train.py` and `splitting_dataset_test.py`)

* **Purpose**: Splits the extracted feature datasets into:
  * Variance-based AM/FM features
  * DCT-based AM/FM features
  * Combined variance + DCT features

* **Inputs**:
  `.csv` files containing extracted features

* **Output**: Separate `.csv` files for:
  * Variance features
  * DCT features
  * Combined features

---

