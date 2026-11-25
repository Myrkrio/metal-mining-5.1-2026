import os
import torch
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from models import get_model
import numpy as np

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")

    print("Chargement des données de test...")
    _, _, test_ds, all_genres = get_dataloaders(json_path="data/dataset_ml.json", batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    print(f"{test_loader.dataset.__len__()} échantillons de test chargés.")
    
    model = get_model(architecture="convnext-base", num_classes=len(all_genres)).to(device)
    
    model_path = "./results/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Erreur : Poids non trouvés à {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\nAnalyse des prédictions (avec seuil de confiance à 0.3)...")
    
    # Suivi pour quelques genres clés
    target_keywords = ["death metal", "black metal", "heavy metal"]
    keyword_stats = {kw: {"true_positives": 0, "total_real": 0, "total_pred": 0} for kw in target_keywords}

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                 logits = model(images)
            probs = torch.sigmoid(logits)  # Probabilités entre 0 et 1 pour chaque genre
            
            # Seuil de décision multi-label (ex: > 0.3)
            preds = (probs >= 0.3).cpu().numpy()
            targets = targets.numpy()

            for i in range(len(targets)):
                true_labels = set(np.where(targets[i] == 1.0)[0])
                pred_labels = set(np.where(preds[i] == True)[0])

                for kw in target_keywords:
                    # Trouver tous les indices de genres correspondant au mot-clé
                    matching_idxs = [idx for idx, g in enumerate(all_genres) if kw in g.lower()]
                    
                    for idx in matching_idxs:
                        is_real = idx in true_labels
                        is_pred = idx in pred_labels
                        
                        if is_real:
                            keyword_stats[kw]["total_real"] += 1
                        if is_pred:
                            keyword_stats[kw]["total_pred"] += 1
                        if is_real and is_pred:
                            keyword_stats[kw]["true_positives"] += 1

    print("\n" + "=" * 50)
    print("STATISTIQUES RÉELLES PAR GENRE (Multi-Label)")
    print("=" * 50)
    for kw, stats in keyword_stats.items():
        tp = stats["total_positives"] if "total_positives" in stats else stats["true_positives"]
        real = stats["total_real"]
        pred = stats["total_pred"]
        print(f"• Genre contenant '{kw.upper()}':")
        print(f"  - Présents dans la vérité terrain : {real}")
        print(f"  - Prédits par le modèle : {pred}")
        print(f"  - Bien prédits (Intersection) : {tp}")
        print("-" * 40)

if __name__ == "__main__":
    evaluate_model()