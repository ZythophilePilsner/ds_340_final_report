#!/usr/bin/env python3
"""Midterm progress pipeline for local GTZAN and Suno evaluation.

This script packages the project work into one shareable file that:
1. Extracts MFCC features from the local GTZAN dataset.
2. Trains the paper-style CNN used in the replication workflow.
3. Evaluates the model on GTZAN test data.
4. Maps local Suno AI tracks into GTZAN genres with tag heuristics.
5. Evaluates the same model on AI-generated music.

Example:
    python midterm_progress.py --rebuild-gtzan-json --retrain-model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Audio, load_from_disk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

AUDIO_SUFFIXES = {".wav", ".mp3", ".au", ".flac", ".ogg", ".m4a"}
GENRE_ALIASES = {
    "blues": {"blues"},
    "classical": {"classical"},
    "country": {"country"},
    "disco": {"disco"},
    "hiphop": {"hip hop", "hiphop"},
    "jazz": {"jazz"},
    "metal": {"metal"},
    "pop": {"pop"},
    "reggae": {"reggae"},
    "rock": {"rock"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the paper-style GTZAN CNN and evaluate it on AI-generated Suno music."
    )
    parser.add_argument(
        "--gtzan-dir",
        default="data/genres_original",
        help="Local GTZAN root folder with 10 genre subfolders.",
    )
    parser.add_argument(
        "--ai-dir",
        default="data/suno-audio",
        help="Local Suno dataset root containing batch_* folders.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/midterm_progress",
        help="Directory where JSON, model, reports, plots, and CSV outputs will be saved.",
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--track-duration", type=int, default=30)
    parser.add_argument("--num-mfcc", type=int, default=13)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-ai-tracks-per-genre",
        type=int,
        default=30,
        help="Maximum AI tracks kept per mapped GTZAN genre. Use -1 for no limit.",
    )
    parser.add_argument(
        "--rebuild-gtzan-json",
        action="store_true",
        help="Recompute GTZAN MFCC features from raw audio.",
    )
    parser.add_argument(
        "--rebuild-ai-features",
        action="store_true",
        help="Recompute MFCC features for the mapped AI music subset.",
    )
    parser.add_argument(
        "--retrain-model",
        action="store_true",
        help="Train a new CNN even if a saved model already exists.",
    )
    parser.add_argument(
        "--allow-multi-label",
        action="store_true",
        help="Allow AI tracks with multiple matched genre tags and keep the first match.",
    )
    parser.add_argument(
        "--skip-ai-eval",
        action="store_true",
        help="Run only the GTZAN training and test pipeline.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(data, fp, indent=2)


def extract_mfcc_segments_from_file(
    file_path: Path,
    sample_rate: int,
    track_duration: int,
    num_mfcc: int,
    n_fft: int,
    hop_length: int,
    num_segments: int,
) -> list[np.ndarray]:
    signal, sr = librosa.load(file_path, sr=sample_rate, duration=track_duration)
    samples_per_track = sample_rate * track_duration
    samples_per_segment = int(samples_per_track / num_segments)
    expected_vectors = math.ceil(samples_per_segment / hop_length)

    segments: list[np.ndarray] = []
    for segment_idx in range(num_segments):
        start_sample = samples_per_segment * segment_idx
        finish_sample = start_sample + samples_per_segment

        mfcc = librosa.feature.mfcc(
            y=signal[start_sample:finish_sample],
            sr=sr,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        ).T

        if len(mfcc) == expected_vectors:
            segments.append(mfcc)

    return segments


def save_mfcc_from_folder(
    dataset_path: Path,
    json_path: Path,
    sample_rate: int,
    track_duration: int,
    num_mfcc: int,
    n_fft: int,
    hop_length: int,
    num_segments: int,
) -> dict:
    data = {"mapping": [], "labels": [], "mfcc": []}

    for genre_idx, genre_dir in enumerate(sorted(p for p in dataset_path.iterdir() if p.is_dir())):
        data["mapping"].append(genre_dir.name)
        print(f"Processing GTZAN genre: {genre_dir.name}")

        for audio_file in sorted(genre_dir.iterdir()):
            if audio_file.suffix.lower() not in AUDIO_SUFFIXES:
                continue

            try:
                segments = extract_mfcc_segments_from_file(
                    audio_file,
                    sample_rate=sample_rate,
                    track_duration=track_duration,
                    num_mfcc=num_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    num_segments=num_segments,
                )
            except Exception as exc:
                print(f"SKIP GTZAN file: {audio_file} | {type(exc).__name__}: {exc}")
                continue

            for mfcc in segments:
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(genre_idx)

    save_json(json_path, data)
    return data


def load_mfcc_json(json_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    with json_path.open("r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"], dtype=np.float32)
    y = np.array(data["labels"], dtype=np.int64)
    return data, X, y


def prepare_datasets_from_json(
    json_path: Path,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data, X, y = load_mfcc_json(json_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
    )

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return data["mapping"], X_train, X_val, X_test, y_train, y_val, y_test


def build_paper_cnn(input_shape: tuple[int, ...], num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (2, 2), activation="relu"),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_history_plot(history_df: pd.DataFrame, output_path: Path) -> None:
    if history_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if "accuracy" in history_df:
        ax.plot(history_df["accuracy"], label="train_accuracy")
    if "val_accuracy" in history_df:
        ax.plot(history_df["val_accuracy"], label="val_accuracy")
    if "loss" in history_df:
        ax.plot(history_df["loss"], label="train_loss")
    if "val_loss" in history_df:
        ax.plot(history_df["val_loss"], label="val_loss")
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def normalize_tags(raw_tags: str) -> set[str]:
    return {
        " ".join(part.strip().lower().split())
        for part in str(raw_tags).split(",")
        if part and part.strip()
    }


def map_tags_to_gtzan_genre(raw_tags: str, strict_single_label: bool) -> str | None:
    tags = normalize_tags(raw_tags)
    matched: list[str] = []
    for genre, aliases in GENRE_ALIASES.items():
        if any(alias in tags for alias in aliases):
            matched.append(genre)
    matched = sorted(set(matched))
    if len(matched) == 1:
        return matched[0]
    if strict_single_label:
        return None
    return matched[0] if matched else None


def collect_ai_eval_metadata(
    ai_root: Path,
    max_ai_tracks_per_genre: int | None,
    strict_single_label: bool,
) -> pd.DataFrame:
    records: list[dict] = []

    for batch_dir in sorted(p for p in ai_root.iterdir() if p.is_dir() and p.name.startswith("batch_")):
        try:
            ds = load_from_disk(str(batch_dir))
        except Exception as exc:
            print(f"SKIP Suno batch: {batch_dir} | {type(exc).__name__}: {exc}")
            continue

        wanted_columns = [
            column
            for column in ["id", "title", "tags", "prompt", "duration", "model_name", "status"]
            if column in ds.column_names
        ]
        meta = ds.select_columns(wanted_columns)

        for row_index in range(len(meta)):
            row = meta[row_index]
            mapped_genre = map_tags_to_gtzan_genre(
                row.get("tags", ""),
                strict_single_label=strict_single_label,
            )
            if mapped_genre is None:
                continue
            records.append(
                {
                    "batch": batch_dir.name,
                    "row_index": row_index,
                    "track_id": str(row.get("id", "")),
                    "title": str(row.get("title", "")),
                    "tags": str(row.get("tags", "")),
                    "prompt": str(row.get("prompt", "")),
                    "duration": row.get("duration"),
                    "model_name": str(row.get("model_name", "")),
                    "status": str(row.get("status", "")),
                    "mapped_genre": mapped_genre,
                }
            )

    metadata_df = pd.DataFrame(records)
    if metadata_df.empty:
        return metadata_df

    metadata_df = metadata_df.sort_values(["mapped_genre", "batch", "row_index"]).reset_index(drop=True)
    if max_ai_tracks_per_genre is not None:
        metadata_df = (
            metadata_df.groupby("mapped_genre", group_keys=False)
            .head(max_ai_tracks_per_genre)
            .reset_index(drop=True)
        )
    return metadata_df


def extract_mfcc_segments_from_audio_bytes(
    audio_bytes: bytes,
    suffix: str,
    sample_rate: int,
    track_duration: int,
    num_mfcc: int,
    n_fft: int,
    hop_length: int,
    num_segments: int,
) -> list[np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=suffix or ".mp3", delete=True) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        return extract_mfcc_segments_from_file(
            Path(tmp_file.name),
            sample_rate=sample_rate,
            track_duration=track_duration,
            num_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            num_segments=num_segments,
        )


def build_ai_eval_json(
    metadata_df: pd.DataFrame,
    ai_root: Path,
    output_json_path: Path,
    label_names: list[str],
    genre_to_id: dict[str, int],
    sample_rate: int,
    track_duration: int,
    num_mfcc: int,
    n_fft: int,
    hop_length: int,
    num_segments: int,
) -> dict:
    data = {
        "mapping": label_names,
        "labels": [],
        "mfcc": [],
        "track_ids": [],
        "track_titles": [],
        "track_genres": [],
    }

    for batch_name, batch_rows in metadata_df.groupby("batch"):
        batch_ds = load_from_disk(str(ai_root / batch_name)).cast_column("audio", Audio(decode=False))
        for record in batch_rows.to_dict("records"):
            row = batch_ds[int(record["row_index"])]
            audio_info = row["audio"]
            audio_bytes = audio_info.get("bytes")
            if audio_bytes is None:
                print(f"SKIP missing audio bytes for {record['track_id']}")
                continue

            suffix = Path(audio_info.get("path", "track.mp3")).suffix or ".mp3"
            try:
                segments = extract_mfcc_segments_from_audio_bytes(
                    audio_bytes,
                    suffix=suffix,
                    sample_rate=sample_rate,
                    track_duration=track_duration,
                    num_mfcc=num_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    num_segments=num_segments,
                )
            except Exception as exc:
                print(f"SKIP AI track: {record['track_id']} | {type(exc).__name__}: {exc}")
                continue

            label_id = genre_to_id[record["mapped_genre"]]
            for mfcc in segments:
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(int(label_id))
                data["track_ids"].append(str(record["track_id"]))
                data["track_titles"].append(str(record["title"]))
                data["track_genres"].append(str(record["mapped_genre"]))

    save_json(output_json_path, data)
    return data


def load_ai_eval_json(json_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    with json_path.open("r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"], dtype=np.float32)[..., np.newaxis]
    y = np.array(data["labels"], dtype=np.int64)
    return data, X, y


def evaluate_split(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    split_name: str,
    artifact_dir: Path,
) -> dict:
    loss, accuracy = model.evaluate(X, y, verbose=0)
    y_pred = model.predict(X, verbose=0).argmax(axis=1)
    macro_f1 = f1_score(y, y_pred, average="macro")
    report = classification_report(
        y,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )

    write_text(artifact_dir / f"{split_name}_report.txt", report)
    save_confusion_matrix(
        y_true=y,
        y_pred=y_pred,
        label_names=label_names,
        title=f"{split_name.replace('_', ' ').title()} Confusion Matrix",
        output_path=artifact_dir / f"{split_name}_confusion_matrix.png",
    )

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "y_pred": y_pred,
        "report": report,
    }


def evaluate_ai_track_level(
    segment_results_df: pd.DataFrame,
    label_names: list[str],
    artifact_dir: Path,
) -> dict:
    track_results_df = (
        segment_results_df.groupby(["track_id", "title", "true_label"], as_index=False)
        .agg(
            true_label_id=("true_label_id", "first"),
            pred_label_id=("pred_label_id", lambda series: Counter(series).most_common(1)[0][0]),
            num_segments=("pred_label_id", "size"),
        )
    )
    track_results_df["pred_label"] = track_results_df["pred_label_id"].map(lambda idx: label_names[idx])
    track_results_df.to_csv(artifact_dir / "ai_track_results.csv", index=False)

    y_true = track_results_df["true_label_id"].to_numpy(dtype=np.int64)
    y_pred = track_results_df["pred_label_id"].to_numpy(dtype=np.int64)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    write_text(artifact_dir / "ai_track_report.txt", report)
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
        title="AI Music Track-Level Confusion Matrix",
        output_path=artifact_dir / "ai_track_confusion_matrix.png",
    )

    return {
        "track_results_df": track_results_df,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "report": report,
    }


def main() -> None:
    args = parse_args()

    gtzan_dir = resolve_path(args.gtzan_dir)
    ai_dir = resolve_path(args.ai_dir)
    artifact_dir = resolve_path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not gtzan_dir.exists():
        raise FileNotFoundError(f"Could not find GTZAN folder: {gtzan_dir}")
    if not args.skip_ai_eval and not ai_dir.exists():
        raise FileNotFoundError(f"Could not find Suno folder: {ai_dir}")

    gtzan_json_path = artifact_dir / "gtzan_data.json"
    model_path = artifact_dir / "paper_cnn.keras"
    history_csv_path = artifact_dir / "gtzan_training_history.csv"
    history_plot_path = artifact_dir / "gtzan_training_history.png"
    ai_json_path = artifact_dir / "suno_ai_eval_mfcc.json"
    ai_metadata_csv_path = artifact_dir / "suno_ai_eval_metadata.csv"
    ai_segment_results_csv_path = artifact_dir / "ai_segment_results.csv"
    summary_json_path = artifact_dir / "run_summary.json"

    keras.utils.set_random_seed(args.seed)

    if args.rebuild_gtzan_json or not gtzan_json_path.exists():
        print("Building GTZAN MFCC JSON from local audio...")
        save_mfcc_from_folder(
            dataset_path=gtzan_dir,
            json_path=gtzan_json_path,
            sample_rate=args.sample_rate,
            track_duration=args.track_duration,
            num_mfcc=args.num_mfcc,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            num_segments=args.num_segments,
        )

    label_names, X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets_from_json(
        json_path=gtzan_json_path,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.seed,
    )
    genre_to_id = {genre: idx for idx, genre in enumerate(label_names)}

    print(f"Label order: {label_names}")
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    if model_path.exists() and not args.retrain_model:
        print(f"Loading saved model from {model_path}")
        model = keras.models.load_model(model_path)
        history_df = pd.read_csv(history_csv_path) if history_csv_path.exists() else pd.DataFrame()
    else:
        print("Training CNN...")
        model = build_paper_cnn(X_train.shape[1:], len(label_names))
        optimiser = keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(
            optimizer=optimiser,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
        )
        model.save(model_path)
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(history_csv_path, index=False)
        save_history_plot(history_df, history_plot_path)

    gtzan_metrics = evaluate_split(
        model=model,
        X=X_test,
        y=y_test,
        label_names=label_names,
        split_name="gtzan_test",
        artifact_dir=artifact_dir,
    )

    print(f"GTZAN test accuracy: {gtzan_metrics['accuracy']:.4f}")
    print(f"GTZAN test macro F1: {gtzan_metrics['macro_f1']:.4f}")
    print(gtzan_metrics["report"])

    summary: dict[str, object] = {
        "project_root": str(PROJECT_ROOT),
        "gtzan_dir": str(gtzan_dir),
        "artifact_dir": str(artifact_dir),
        "label_names": label_names,
        "gtzan_test_accuracy": gtzan_metrics["accuracy"],
        "gtzan_test_macro_f1": gtzan_metrics["macro_f1"],
    }

    if args.skip_ai_eval:
        save_json(summary_json_path, summary)
        print(f"Saved outputs to {artifact_dir}")
        return

    strict_single_label = not args.allow_multi_label
    max_ai_tracks_per_genre = None if args.max_ai_tracks_per_genre < 0 else args.max_ai_tracks_per_genre
    ai_metadata_df = collect_ai_eval_metadata(
        ai_root=ai_dir,
        max_ai_tracks_per_genre=max_ai_tracks_per_genre,
        strict_single_label=strict_single_label,
    )
    ai_metadata_df.to_csv(ai_metadata_csv_path, index=False)

    if ai_metadata_df.empty:
        summary["ai_eval"] = "No AI tracks could be mapped into the GTZAN label set."
        save_json(summary_json_path, summary)
        print("No AI tracks could be mapped into the GTZAN genre labels.")
        print(f"Saved outputs to {artifact_dir}")
        return

    print("Mapped AI tracks kept by genre:")
    print(ai_metadata_df["mapped_genre"].value_counts().sort_index())

    if args.rebuild_ai_features or not ai_json_path.exists():
        print("Building AI MFCC JSON from mapped Suno tracks...")
        build_ai_eval_json(
            metadata_df=ai_metadata_df,
            ai_root=ai_dir,
            output_json_path=ai_json_path,
            label_names=label_names,
            genre_to_id=genre_to_id,
            sample_rate=args.sample_rate,
            track_duration=args.track_duration,
            num_mfcc=args.num_mfcc,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            num_segments=args.num_segments,
        )

    ai_eval_data, X_ai, y_ai = load_ai_eval_json(ai_json_path)
    if len(X_ai) == 0:
        summary["ai_eval"] = "Mapped AI tracks produced zero usable MFCC segments."
        save_json(summary_json_path, summary)
        print("Mapped AI tracks produced zero usable MFCC segments.")
        print(f"Saved outputs to {artifact_dir}")
        return

    ai_segment_metrics = evaluate_split(
        model=model,
        X=X_ai,
        y=y_ai,
        label_names=label_names,
        split_name="ai_segment",
        artifact_dir=artifact_dir,
    )

    ai_pred = ai_segment_metrics["y_pred"]
    segment_results_df = pd.DataFrame(
        {
            "track_id": ai_eval_data["track_ids"],
            "title": ai_eval_data["track_titles"],
            "true_label_id": y_ai,
            "pred_label_id": ai_pred,
            "true_label": [label_names[idx] for idx in y_ai],
            "pred_label": [label_names[idx] for idx in ai_pred],
        }
    )
    segment_results_df.to_csv(ai_segment_results_csv_path, index=False)

    ai_track_metrics = evaluate_ai_track_level(
        segment_results_df=segment_results_df,
        label_names=label_names,
        artifact_dir=artifact_dir,
    )

    print(f"AI segment-level accuracy: {ai_segment_metrics['accuracy']:.4f}")
    print(f"AI segment-level macro F1: {ai_segment_metrics['macro_f1']:.4f}")
    print(ai_segment_metrics["report"])
    print(f"AI track-level accuracy: {ai_track_metrics['accuracy']:.4f}")
    print(f"AI track-level macro F1: {ai_track_metrics['macro_f1']:.4f}")
    print(ai_track_metrics["report"])

    summary.update(
        {
            "ai_dir": str(ai_dir),
            "strict_single_label": strict_single_label,
            "max_ai_tracks_per_genre": max_ai_tracks_per_genre,
            "mapped_ai_tracks": int(len(ai_metadata_df)),
            "ai_segment_accuracy": ai_segment_metrics["accuracy"],
            "ai_segment_macro_f1": ai_segment_metrics["macro_f1"],
            "ai_track_accuracy": ai_track_metrics["accuracy"],
            "ai_track_macro_f1": ai_track_metrics["macro_f1"],
        }
    )
    save_json(summary_json_path, summary)
    print(f"Saved outputs to {artifact_dir}")


if __name__ == "__main__":
    main()
