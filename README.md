# DS340 Midterm Progress

This repository contains the code for a DS340 project based on a music genre classification paper replication and an extension to AI-generated music.

## Project Goal

The project follows the parent paper's general workflow:
- extract MFCC-based audio features
- train a CNN genre classifier on GTZAN
- evaluate the same model on AI-generated music from the Suno dataset

The main question is whether a genre classifier trained on human-curated music still performs well on AI-generated songs.

## Included Files

- `midterm_progress.py`: standalone script for preprocessing, model training, and evaluation
- `demo_1.ipynb`: notebook version of the full project workflow
- `demo_2.ipynb`: notebook version with a cleaner local-environment setup
- `trial_1.ipynb`: local analysis notebook for Suno tag counts
- `load_data.ipynb`: notebook for downloading the Hugging Face dataset in batches

## Local Data

Large datasets are not stored in this repository.

Expected local folders:
- `data/genres_original`
- `data/suno-audio`

## Environment Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Main Script

GTZAN-only run:

```bash
python midterm_progress.py --skip-ai-eval
```

Full GTZAN + AI evaluation:

```bash
python midterm_progress.py --rebuild-gtzan-json --rebuild-ai-features --retrain-model
```

Outputs are written under:

```bash
artifacts/midterm_progress
```

## Notes

- The Suno dataset uses free-form tags, so AI genre labels are mapped heuristically into the GTZAN label set.
- Large raw datasets, trained models, and generated artifacts are ignored by Git to keep the repository lightweight.

