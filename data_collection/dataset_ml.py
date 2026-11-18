import os
import json

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

    ml_dataset = []
    files = [f for f in os.listdir(covers_dir) if f.lower().endswith(".jpg")]
    
    print(f"Analyse de {len(files)} fichiers dans {covers_dir}...")

    for filename in files:
        name_without_ext = os.path.splitext(filename)[0]
        
        # Le format est "Artiste - Album.jpg" : on sépare sur le premier " - "
        if " - " in name_without_ext:
            parts = name_without_ext.split(" - ", 1)
            artist_name = parts[0].strip()
            album_name = parts[1].strip()
        else:
            artist_name = name_without_ext
            album_name = "Unknown"

        cover_path = os.path.join(covers_dir, filename)
        
        # Récupération du genre brut depuis bands_genres.json
        genre = bands_genres.get(artist_name, None)

        ml_dataset.append({
            "band": artist_name,
            "album": album_name,
            "cover_path": cover_path,
            "genre": genre
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ml_dataset, f, indent=4, ensure_ascii=False)

    print(f"Dataset ML généré avec succès : {output_json_path} ({len(ml_dataset)} entrées)")

if __name__ == "__main__":
    generate_ml_dataset_from_covers()