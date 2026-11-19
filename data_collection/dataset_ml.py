import os
import json
import re


def generate_ml_dataset_from_covers(
    covers_dir="./data/covers",
    genres_json_path="./data/bands_genres_cleaned.json",
    output_json_path="./data/dataset_ml.json"
):
    if not os.path.exists(genres_json_path):
        print(f"Erreur : Fichier des genres introuvable : {genres_json_path}")
        return

    if not os.path.exists(covers_dir):
        print(f"Erreur : Dossier des covers introuvable : {covers_dir}")
        return

    print(f"Chargement des genres depuis {genres_json_path}...")
    with open(genres_json_path, "r", encoding="utf-8") as f:
        bands_genres = json.load(f)
        normalized_genres = {
            re.sub(r'[<>:"/\\|?*]', "_", band): (band, genres)
            for band, genres in bands_genres.items()
        }

    ml_dataset = []
    skipped_covers = 0
    files = [f for f in os.listdir(covers_dir) if f.lower().endswith(".jpg")]

    print(f"Analyse de {len(files)} fichiers dans {covers_dir}...")

    for filename in files:
        name_without_ext = os.path.splitext(filename)[0]
        artist_name, album_name, genres = None, "Unknown", None

        if " - " in name_without_ext:
            tokens = name_without_ext.split(" - ")
            for i in range(1, len(tokens)):
                candidate = " - ".join(tokens[:i]).strip()
                if candidate in normalized_genres:
                    artist_name, genres = normalized_genres[candidate]
                    album_name = " - ".join(tokens[i:]).strip()
                    break
        elif name_without_ext in normalized_genres:
            artist_name, genres = normalized_genres[name_without_ext]

        if not genres:
            skipped_covers += 1
            continue

        cover_path = os.path.join(covers_dir, filename)

        ml_dataset.append({
            "band": artist_name,
            "album": album_name,
            "cover_path": cover_path,
            "genres": genres
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ml_dataset, f, indent=4, ensure_ascii=False)

    print(f"\nDataset ML généré avec succès : {output_json_path}")
    print(f"  Total covers analysées : {len(files)}")
    print(f"  Covers conservées      : {len(ml_dataset)}")
    print(f"  Covers ignorées (null) : {skipped_covers}")


if __name__ == "__main__":
    generate_ml_dataset_from_covers()