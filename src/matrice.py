import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from models import get_model
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Metrics(ABC):
    """Interface commune pour les métriques d'évaluation."""

    @abstractmethod
    def confusion_matrix(self, y_true, y_pred, labels=None):
        """Retourne la matrice de confusion."""

    @abstractmethod
    def classification_report(self, y_true, y_pred, target_names=None):
        """Retourne les mesures precision, recall et F1."""

    @abstractmethod
    def roc_auc(self, y_true, y_score):
        """Calcule l'AUC ROC."""


class SklearnMetrics(Metrics):
    """Implémentation des métriques via scikit-learn."""

    def confusion_matrix(self, y_true, y_pred, labels=None):
        return confusion_matrix(y_true, y_pred, labels=labels)

    def classification_report(self, y_true, y_pred, target_names=None):
        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
        )

    def roc_auc(self, y_true, y_score):
        try:
            return roc_auc_score(y_true, y_score, multi_class="ovr")
        except (ValueError, TypeError):
            return roc_auc_score(y_true, y_score)

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")

    print("Chargement des données de test...")
    _, _, test_ds, all_genres = get_dataloaders(json_path="data/dataset_ml.json", batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f"Nombre total de classes : {len(all_genres)}")

    # Initialisation du modèle et chargement des poids
    model = get_model(architecture="convnext-base", num_classes=len(all_genres)).to(device)
    
    model_path = "./results/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Erreur : Aucun poids sauvegardé trouvé à {model_path}. Lance d'abord l'entraînement.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_trues = []
    all_scores = []

    print("\nÉvaluation sur le set of Test...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            probs = torch.sigmoid(logits)
            _, predicted_indices = torch.max(probs, dim=1)

            all_scores.append(probs.float().detach().cpu().numpy())
            
            for i, pred_idx in enumerate(predicted_indices):
                true_vector = targets[i]
                true_indices = (true_vector == 1.0).nonzero(as_tuple=True)[0]
                true_idx = true_indices[0].item() if len(true_indices) > 0 else 0
                
                all_preds.append(pred_idx.item())
                all_trues.append(true_idx)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    all_scores = np.concatenate(all_scores, axis=0)

    y_true_binary = np.eye(len(all_genres))[all_trues]
    auc_score = roc_auc_score(
        y_true_binary,
        all_scores,
        multi_class="ovr",
        average="macro",
    )

    # Courbe ROC micro-moyenne pour afficher la métrique sur Matplotlib
    y_true_flat = y_true_binary.ravel()
    scores_flat = all_scores.ravel()
    fpr, tpr, _ = roc_curve(y_true_flat, scores_flat)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"Courbe ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--",
             label="Classifier aléatoire")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC micro-moyenne")
    plt.text(
        0.60,
        0.12,
        f"AUC macro = {auc_score:.3f}\nAUC ROC = {roc_auc:.3f}",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/roc_auc_metric.png", dpi=300)
    print(f"\nMétrique ROC/AUC affichée et sauvegardée sous : roc_auc_metric.png")
    plt.close()

    # 1. Analyse ciblée pour les genres clés (Death, Black, Heavy)
    target_keywords = ["death", "black", "heavy"]
    print("\n" + "=" * 60)
    print("PERFORMANCES SUR LES GENRES CLÉS (DIAGONALE)")
    print("=" * 60)

    for kw in target_keywords:
        # Trouver les indices qui contiennent le mot-clé (insensible à la casse)
        matching_indices = [i for i, g in enumerate(all_genres) if kw in g.lower()]
        if not matching_indices:
            print(f"Aucun genre trouvé pour le mot-clé : '{kw}'")
            continue
            
        print(f"\n--- Résultats pour les genres contenant '{kw.upper()}' ---")
        for idx in matching_indices[:5]:  # Limiter aux 5 premiers pour la lisibilité si y'en a trop
            genre_name = all_genres[idx]
            # Masque pour ce genre spécifique dans la vérité terrain
            mask = (all_trues == idx)
            total_real = mask.sum()
            if total_real > 0:
                correct = (all_preds[mask] == idx).sum()
                accuracy_genre = (correct / total_real) * 100
                print(f"  • {genre_name} : {correct}/{total_real} bien prédits ({accuracy_genre:.1f}%)")
            else:
                print(f"  • {genre_name} : 0 exemple dans le set de test")

    # 2. Génération d'une matrice filtrée pour les top genres les plus fréquents dans le test
    # On sélectionne par exemple les 15 genres les plus représentés dans all_trues
    counts = np.bincount(all_trues, minlength=len(all_genres))
    top_indices = np.argsort(counts)[-15:][::-1] # Top 15
    top_genres = [all_genres[i] for i in top_indices]

    # Filtrer les prédictions et vérités pour ne garder que ces top 15
    filter_mask = np.isin(all_trues, top_indices)
    if filter_mask.sum() > 0:
        cm_filtered = confusion_matrix(all_trues[filter_mask], all_preds[filter_mask], labels=top_indices)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=top_genres, yticklabels=top_genres)
        plt.xlabel('Prédictions')
        plt.ylabel('Vérité Terrain')
        plt.title('Matrice de Confusion (Top 15 Genres Principaux)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_filtered = "./results/confusion_matrix_top15.png"
        plt.savefig(output_filtered, dpi=300)
        print(f"\nMatrice de confusion ciblée (Top 15) sauvegardée sous : {output_filtered}")
        plt.close()

if __name__ == "__main__":
    evaluate_model()