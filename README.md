# DS340 Midterm Progress

This repository shares the core code for a DS340 midterm project on music genre classification and evaluation on AI-generated music.

## Included Files

- `midterm_progress.py`: standalone script for preprocessing, CNN training, and evaluation
- `load_data.ipynb`: notebook for downloading the Hugging Face Suno dataset in manageable batches
- `requirements.txt`: Python dependencies needed for the script and notebook

## Project Summary

The workflow follows the parent paper's general structure:
- extract MFCC-based features from local audio
- train a CNN on the GTZAN dataset
- evaluate the trained model on AI-generated music from the Suno dataset

The main question is whether a model trained on human-curated music genres generalizes well to AI-generated songs.

## Expected Local Data

Large datasets are not included in this repository.

The code expects these local folders:
- `data/genres_original`
- `data/suno-audio`

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

GTZAN-only run:

```bash
python midterm_progress.py --skip-ai-eval
```

Full GTZAN + AI evaluation:

```bash
python midterm_progress.py --rebuild-gtzan-json --rebuild-ai-features --retrain-model
```

Artifacts are written to:

```bash
artifacts/midterm_progress
```

## Note

The Suno dataset uses free-form tags, so AI labels are mapped heuristically into the GTZAN genre set.
