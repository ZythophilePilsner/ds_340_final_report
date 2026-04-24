# DS340 Final Report: GTZAN And Suno Music Genre Classification

This repository contains the final portable project deliverable for comparing music genre classification on human-recorded GTZAN audio and AI-generated Suno audio.

The main analysis file is `final_submission_v2.ipynb`.

## Final Repository Files

The repo is intentionally small. The important files are:

- `final_submission_v2.ipynb`: final notebook that rebuilds the full analysis and visual evidence from raw audio folders
- `prepare_ai_music.py`: script that downloads and prepares the `ai_music/` evaluation set from Hugging Face
- `requirements.txt`: Python packages needed for the script and notebook
- `DATASET_MANIFEST.md`: short dataset layout reference

## Dataset Sources

- AI music source: [humair025/suno-audio on Hugging Face](https://huggingface.co/datasets/humair025/suno-audio)
- Human music source: [GTZAN Dataset - Music Genre Classification on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## Step-By-Step Run Guide

Follow these steps in order on a fresh computer.

### 1. Clone the repository

```bash
git clone https://github.com/ZythophilePilsner/ds_340_final_report.git
cd ds_340_final_report
```

### 2. Create and activate a virtual environment

Python 3.11 or 3.12 is recommended.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the required Python packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Download the GTZAN dataset and place it in the repo root

Open the Kaggle page:

[GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

Download and unzip the dataset. The notebook only needs the `genres_original` audio folder.

Rename `genres_original` to `human_music` and place it in the repository root.

If Kaggle extracts into a nested path such as `Data/genres_original`, move that folder and rename it:

```bash
mv /path/to/extracted/Data/genres_original human_music
```

After this step, your repository should look like:

```text
ds_340_final_report/
  final_submission_v2.ipynb
  prepare_ai_music.py
  requirements.txt
  DATASET_MANIFEST.md
  human_music/
    blues/
    classical/
    country/
    disco/
    hiphop/
    jazz/
    metal/
    pop/
    reggae/
    rock/
```

### 5. Build the AI music evaluation folder

Run:

```bash
python prepare_ai_music.py
```

What the script does:

- downloads the public Suno dataset from Hugging Face
- selects a deterministic balanced subset
- writes 30 processed clips per genre
- trims or pads every clip to 30 seconds
- creates `ai_music/` with the ten genre folders
- writes `ai_music_manifest.csv`

After this finishes, your repository should also contain:

```text
ai_music/
  blues/
  classical/
  country/
  disco/
  hiphop/
  jazz/
  metal/
  pop/
  reggae/
  rock/
```

### 6. Start Jupyter and open the final notebook

```bash
jupyter notebook final_submission_v2.ipynb
```

### 7. Run the notebook from top to bottom

Inside Jupyter:

- open `final_submission_v2.ipynb`
- if needed, leave the dataset input cell at its default values because it already expects `human_music/` and `ai_music/` in the repo root
- click `Kernel -> Restart & Run All`

The notebook includes a runtime dependency check cell near the top. On a clean machine, that cell can install any missing notebook packages into the active kernel before the main imports run.

## What `final_submission_v2.ipynb` Does

The notebook rebuilds the full workflow directly from `human_music/` and `ai_music/`:

1. loads raw audio from both dataset folders
2. extracts MFCC segment features
3. trains the parent-style CNN on human music
4. evaluates the CNN on held-out human music
5. applies the same CNN to AI music
6. builds MFCC-statistics features and trains classical ML models
7. compares the final methods on human and AI evaluations
8. generates visual evidence between the major steps
9. saves final tables, metrics, and confusion matrices

## Output Files

When the notebook finishes, it writes outputs under:

```text
artifacts/final_submission_v2/
```

That folder includes:

- `final_method_comparison.csv`
- `mfcc_stats_method_comparison.csv`
- `mfcc_stats_track_results.csv`
- `cnn_ai_track_results.csv`
- `final_summary.json`
- `final_ai_track_confusion_matrix.png`
- `cnn_training_history.csv`
- visual checkpoint plots produced during the notebook run

## Portability Notes

- The notebook does not require precomputed JSON caches, saved models, or hidden local files.
- The repository alone is not enough to run the project; you must also add `human_music/` from Kaggle and generate `ai_music/` with `prepare_ai_music.py`.
- Once those two dataset folders exist, `final_submission_v2.ipynb` is designed to run on another computer directly from the cloned repository.
