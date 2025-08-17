

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

### 1. Pre-process audio files (`pre_processing_audio_files.py`)

* **Purpose**: Extracts `"PAR"` segments from input `.wav` files.
* **Input**: Path to folder with `.wav` file.
* **Output**: Processed file `combined.wav`.

### 2. Generate list of usable audio files (`list_audio_files.py`)

* **Purpose**: Creates a `.txt` file containing paths of all usable `_combine.wav` files.
* **Input**: Folder containing processed `_combine.wav` files.
* **Output**: `processed_files.txt`.


### 3. Extract features (`main.py`)

* **Purpose**: Extracts features from audio and aligns them with labels.
* **Inputs**:

  * Path to `.txt` file with links to usable `.wav` files.
  * Path to `.csv` file with labels.
  * Output path for saving the features.
* **Output**: `.csv` file containing extracted features.

---

