# DS340 Midterm Project

This repository contains the code for a DS340 midterm project on music genre classification. The main workflow trains a paper-style CNN on GTZAN MFCC features and optionally evaluates the same model on locally stored AI-generated music. The latest notebook also adds a research-backed MFCC-statistics plus classical ML comparison for AI-generated music.

## Repository Contents

- `demo_3.ipynb`: end-to-end notebook covering preprocessing, MFCC extraction, CNN training, GTZAN evaluation, and optional AI music evaluation
- `demo_6.ipynb`: latest extended notebook with the parent CNN workflow, AI-music inference comparisons, and MFCC-statistics classifiers such as linear SVM, RBF SVM, logistic regression, and Extra Trees
- `midterm_progress.py`: script version of the core GTZAN plus AI-evaluation workflow
- `load_data.ipynb`: notebook for loading or organizing the local Suno dataset batches
- `demo_1.ipynb`, `demo_2.ipynb`, `trial_1.ipynb`: earlier exploratory notebooks
- `requirements.txt`: Python dependencies for the notebooks and scripts

## Public Upload Note

`demo_3.ipynb` and `demo_6.ipynb` are included in a public-safe form:

- stored without cell outputs or execution counts
- cleaned of hardcoded personal filesystem paths
- configured to use repository-local `data/` and `artifacts/` folders
- trimmed to avoid keeping unnecessary AI metadata fields such as prompts and track titles in notebook-generated artifacts

## Project Workflow

1. Load one GTZAN example track and walk through waveform, rectification, smoothing, FFT, STFT, spectrogram, and MFCC views.
2. Build MFCC segment datasets from the local GTZAN audio folders.
3. Train the CNN used in the paper replication.
4. Evaluate the trained model on a GTZAN test split.
5. Optionally map local Suno tags into the GTZAN label space and evaluate the same model on AI-generated tracks.
6. In `demo_6.ipynb`, compare additional AI-music decision rules and research-backed MFCC-statistics classifiers.

## Demo 6 Extension

`demo_6.ipynb` keeps the previous parent-paper CNN implementation and adds two new comparison blocks:

- AI-music inference rules, including section-aware/core-window voting and probability-weighted voting.
- MFCC-statistics classifiers, where each segment is represented by summary statistics such as mean, standard deviation, quartiles, min/max, and delta-MFCC statistics before training classical ML models.

In the local cached run, the strongest AI-music method was the MFCC-statistics linear SVM with track-level majority voting:

```text
Parent CNN majority vote:          accuracy 0.2167 | macro F1 0.1943
Core-window CNN vote:              accuracy 0.2567 | macro F1 0.2350
MFCC-statistics linear SVM vote:   accuracy 0.3533 | macro F1 0.3245
```

This result is useful for the final paper's Novelty or Contributions section because it shows a domain-shift improvement on AI-generated music, not only another GTZAN-only model.

## Expected Local Data

Large datasets are not included in the repository.

The code expects:

- `data/genres_original`
- `data/suno-audio` for the optional AI-evaluation section

Generated outputs are written to `artifacts/demo_3` or `artifacts/demo_6` and are ignored by git.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Working With The Notebooks

Launch Jupyter and open the notebook you want to run:

```bash
jupyter notebook demo_3.ipynb
jupyter notebook demo_6.ipynb
```

The notebooks assume the local dataset folders above already exist. If `data/suno-audio` is missing, the AI-evaluation sections are skipped automatically.

## Running The Script

GTZAN-only run:

```bash
python midterm_progress.py --skip-ai-eval
```

Full GTZAN plus AI evaluation:

```bash
python midterm_progress.py --rebuild-gtzan-json --rebuild-ai-features --retrain-model
```

## Notes

- The GTZAN model uses MFCC segments as inputs to a CNN that follows the paper replication structure.
- The AI-evaluation extension uses heuristic tag mapping because the Suno dataset does not ship with GTZAN-style labels.
- The `demo_6` extension compares multiple methods and writes result tables under `artifacts/demo_6`, including `research_method_comparison.csv` when run locally.
