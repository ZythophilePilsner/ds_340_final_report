# Dataset Manifest

`final_submission_v2.ipynb` uses two dataset folders in the repository root.

## 1. `human_music/`

Source: [GTZAN Dataset - Music Genre Classification on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

Purpose:

- training the parent-style CNN baseline
- held-out human-music evaluation
- training the MFCC-statistics classical ML models

Expected structure:

```text
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

In the Kaggle download, this folder is usually named `genres_original`. Rename it to `human_music` and place it in the repository root.

## 2. `ai_music/`

Source: [humair025/suno-audio on Hugging Face](https://huggingface.co/datasets/humair025/suno-audio)

Purpose:

- cross-domain evaluation on AI-generated music

Expected structure:

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

Create this folder by running:

```bash
python prepare_ai_music.py
```

The script downloads the public Suno dataset, builds a deterministic balanced subset, exports 30-second clips, and writes `ai_music_manifest.csv`.

## Reproducibility Note

`final_submission_v2.ipynb` rebuilds the pipeline directly from `human_music/` and `ai_music/`. It does not depend on hidden local files, cached JSON feature exports, or older demo notebooks.
