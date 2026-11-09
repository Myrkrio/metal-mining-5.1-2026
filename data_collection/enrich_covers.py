import json
import os
from pathlib import Path
from scrap_cover import init_selenium_driver, get_metal_archives_cover

# Chemin de ton JSON existant et du dossier de destination des images
JSON_PATH = "./data/dataset.json" 
COVERS_DIR = "./data/covers"

os.makedirs(COVERS_DIR, exist_ok=True)

def enrich_dataset_with_covers():
    if not os.path.exists(JSON_PATH):
        print(f"Erreur : Fichier JSON introuvable à l'emplacement {JSON_PATH}")
        return

    # 1. Charger le JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Récupération de la racine du dataset (selon ton format avec la clé "dataset")
    dataset = data.get("dataset", data)

    print("Lancement du navigateur Firefox...")
    driver = init_selenium_driver(headless=True)

    try:
        total_artists = len(dataset)
        
        START_INDEX = 0  # <--- Tu modifies juste ça pour choisir où reprendre
        artist_count = 0

        for i, (artist_key, artist_data) in enumerate(dataset.items(), start=1):
            # Si on n'est pas encore arrivé à l'artiste voulu, on passe au suivant
            if i < START_INDEX:
                continue

            artist_count += 1
            artist_name = artist_data.get("name", artist_key)
            albums = artist_data.get("albums", {})

            print(f"\n[{artist_count}/{total_artists}] Traitement de l'artiste : {artist_name}")

            for album_key, album_data in albums.items():
                album_name = album_data.get("name", album_key)

                # Si la cover est déjà téléchargée et existe sur le disque, on passe
                if album_data.get("cover_path") and os.path.exists(album_data["cover_path"]):
                    print(f"Cover déjà présente pour : {album_name}")
                    continue

                print(f"Recherche de la cover pour : {album_name}...")

                # Appel de la fonction de scraping avec Firefox
                cover_data = get_metal_archives_cover(driver, artist_name, album_name)

                if cover_data:
                    filename, mime_type, binary_data = cover_data
                    
                    # Nettoyage sécurisé du nom de fichier pour éviter les erreurs de caractères spéciaux
                    safe_artist = artist_name.replace("/", "-").replace("\\", "-")
                    safe_album = album_name.replace("/", "-").replace("\\", "-")
                    output_filename = f"{safe_artist} - {safe_album}.jpg"
                    output_path = os.path.join(COVERS_DIR, output_filename)

                    with open(output_path, "wb") as f:
                        f.write(binary_data)

                    # Ajout du chemin dans l'album du JSON
                    album_data["cover_path"] = output_path
                    print(f" Cover sauvegardée : {output_path}")
                else:
                    album_data["cover_path"] = None
                    print(f" Aucune cover trouvée pour : {album_name}")

        # Sauvegarde intermédiaire/finale du JSON mis à jour
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("\n JSON mis à jour avec succès avec tous les chemins de covers !")

    finally:
        print("Fermeture du navigateur.")
        driver.quit()

if __name__ == "__main__":
    enrich_dataset_with_covers()
