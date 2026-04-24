from __future__ import annotations

import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset


DATASET_IDS = [
    "humair025/suno-audio",
    "Humair332/suno-audio",
]
OUTPUT_ROOT = Path("ai_music")
OUTPUT_METADATA_CSV = Path("ai_music_manifest.csv")

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

GENRE_ALIASES = {
    "blues": {"blues", "blues rock", "bluesrock"},
    "classical": {"classical", "classical piano", "classical music"},
    "country": {"country", "country pop", "country rock"},
    "disco": {"disco"},
    "hiphop": {"hip hop", "hiphop", "hip-hop"},
    "jazz": {"jazz", "acid jazz", "smooth jazz"},
    "metal": {"metal", "heavy metal", "death metal", "black metal"},
    "pop": {"pop", "synthpop", "synth pop"},
    "reggae": {"reggae", "dub reggae"},
    "rock": {"rock", "hard rock", "soft rock", "alt rock", "alternative rock"},
}

TARGET_PER_GENRE = 30
CLIP_SECONDS = 30
SAMPLE_RATE = 22050


def safe_name(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text))
    return text.strip("_")[:80]


def normalize_tag(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_tags(raw_tags: str) -> list[str]:
    return [normalize_tag(part) for part in str(raw_tags).split(",") if part and part.strip()]


def tag_matches_alias(tag: str, alias: str) -> bool:
    return re.search(rf"\b{re.escape(alias)}\b", tag) is not None


def map_tags_to_genre(raw_tags: str) -> str | None:
    tags = normalize_tags(raw_tags)
    matched = set()

    for genre, aliases in GENRE_ALIASES.items():
        if any(tag_matches_alias(tag, alias) for tag in tags for alias in aliases):
            matched.add(genre)

    if len(matched) == 1:
        return next(iter(matched))
    return None


def standardize_track_length(signal: np.ndarray) -> np.ndarray:
    target_samples = SAMPLE_RATE * CLIP_SECONDS
    if len(signal) < target_samples:
        signal = np.pad(signal, (0, target_samples - len(signal)))
    else:
        signal = signal[:target_samples]
    return signal


def export_clip_from_audio_info(audio_info: dict, output_path: Path) -> None:
    audio_bytes = audio_info.get("bytes")
    audio_path = audio_info.get("path")

    if audio_bytes is not None:
        suffix = Path(audio_path or "track.mp3").suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            signal, _ = librosa.load(temp_audio.name, sr=SAMPLE_RATE, mono=True)
    elif audio_path:
        signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    else:
        raise ValueError("Audio row did not include bytes or a usable path.")

    signal = standardize_track_length(signal)
    sf.write(output_path, signal, SAMPLE_RATE)


def iter_stream_rows(dataset_id: str):
    dataset = load_dataset(dataset_id, split="train", streaming=True)
    try:
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    yield from dataset


def iter_batch_rows(dataset_id: str):
    for batch_index in range(50):
        batch_name = f"batch_{batch_index}"
        dataset = load_dataset(dataset_id, data_dir=batch_name, split="train")
        try:
            dataset = dataset.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
        for row in dataset:
            row["_batch_name"] = batch_name
            yield row


def select_dataset_rows():
    errors = []
    for dataset_id in DATASET_IDS:
        for loader_name, row_iter_factory in [("streaming", iter_stream_rows), ("batch", iter_batch_rows)]:
            try:
                print(f"Trying {dataset_id} with {loader_name} mode...")
                row_iter = row_iter_factory(dataset_id)
                first_row = next(row_iter)

                def chained_rows():
                    yield first_row
                    yield from row_iter

                return dataset_id, chained_rows()
            except Exception as exc:
                errors.append(f"{dataset_id} ({loader_name}): {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not access the Hugging Face dataset.\n" + "\n".join(errors))


def reset_output_folder():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for genre in GENRES:
        (OUTPUT_ROOT / genre).mkdir(parents=True, exist_ok=True)


def main():
    reset_output_folder()

    selected_counts = defaultdict(int)
    selected_track_ids = {genre: set() for genre in GENRES}
    metadata_rows = []

    dataset_id, row_iter = select_dataset_rows()
    print(f"Using dataset source: {dataset_id}")

    for row in row_iter:
        if all(selected_counts[genre] >= TARGET_PER_GENRE for genre in GENRES):
            break

        if str(row.get("status", "")).lower() != "complete":
            continue

        genre = map_tags_to_genre(row.get("tags", ""))
        if genre is None or selected_counts[genre] >= TARGET_PER_GENRE:
            continue

        track_id = str(row.get("id", "")).strip()
        if not track_id or track_id in selected_track_ids[genre]:
            continue

        output_index = selected_counts[genre] + 1
        output_path = OUTPUT_ROOT / genre / f"{genre}_{output_index:02d}_{safe_name(track_id)}.wav"

        try:
            export_clip_from_audio_info(row["audio"], output_path)
        except Exception as exc:
            print(f"Skipping {track_id}: {type(exc).__name__}")
            continue

        selected_track_ids[genre].add(track_id)
        selected_counts[genre] += 1
        metadata_rows.append(
            {
                "genre": genre,
                "index_within_genre": output_index,
                "track_id": track_id,
                "title": row.get("title", ""),
                "tags": row.get("tags", ""),
                "duration": row.get("duration", ""),
                "model_name": row.get("model_name", ""),
                "status": row.get("status", ""),
                "output_file": str(output_path),
            }
        )
        print(f"{genre}: {selected_counts[genre]:02d}/{TARGET_PER_GENRE} -> {output_path.name}")

    metadata_df = pd.DataFrame(metadata_rows).sort_values(["genre", "index_within_genre"]).reset_index(drop=True)
    metadata_df.to_csv(OUTPUT_METADATA_CSV, index=False)

    print("\nFinal counts:")
    for genre in GENRES:
        print(f"{genre}: {selected_counts[genre]}")

    incomplete = [genre for genre in GENRES if selected_counts[genre] < TARGET_PER_GENRE]
    if incomplete:
        raise RuntimeError(
            "Could not build a complete balanced AI subset. Missing tracks for: "
            + ", ".join(incomplete)
        )

    print(f"\nCreated {OUTPUT_ROOT}/ with 30 clips per genre.")
    print(f"Saved metadata to {OUTPUT_METADATA_CSV}")


if __name__ == "__main__":
    main()
