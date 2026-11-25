import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from models import get_model, unfreeze_layers
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


def build_optimizer(model, lr):
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def train_model():
    # Détection du GPU ou repli sur le CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")

    batch_size = 32
    epochs = 15
    patience = 3  # Nombre d'époques sans amélioration avant l'Early Stopping
    architecture = "convnext-tiny"
    unfreeze_epoch = 3  # Début de l'époque 4 (0-indexed) pour passer au level 2

    print("Chargement des données (Split 80/10/10)...")
    train_ds, val_ds, test_ds, all_genres = get_dataloaders(json_path="data/dataset_ml.json", batch_size=batch_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Nombre de classes cibles (genres uniques) : {len(all_genres)}")

    # Initialisation du modèle (ConvNeXt-Tiny par défaut, modifiable pour le benchmark ResNet50)
    model = get_model(architecture=architecture, num_classes=len(all_genres)).to(device)

    # Fonction de perte adaptée au multi-label (Binary Cross Entropy avec Logits)
    criterion = nn.BCEWithLogitsLoss()

    # Optimiseur : on optimise uniquement les paramètres dont requires_grad=True (Level 1 par défaut)
    optimizer = build_optimizer(model, lr=1e-4)

    # Modification pour l'Early Stopping basé sur l'AUC-ROC (on cherche à maximiser)
    best_val_auc = 0.0
    patience_counter = 0

    print("\nDémarrage de l'entraînement multi-label...")
    for epoch in range(epochs):
        # Dégel progressif vers le Level 2 (Tête + derniers blocs profonds)
        if epoch == unfreeze_epoch:
            unfreeze_layers(model, architecture=architecture, level=2)
            optimizer = build_optimizer(model, lr=1e-5)

        model.train()
        train_loss = 0.0

        # Boucle d'entraînement
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Gestion de la structure de sortie (Hugging Face renvoie un objet avec .logits, PyTorch standard direct)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_ds)

        # Phase de validation (Calcul Loss + Collecte pour l'AUC-ROC multi-label)
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)

                # Application de la sigmoïde pour obtenir des probabilités entre 0 et 1 (multi-label)
                preds = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(targets.cpu().numpy())

        epoch_val_loss = val_loss / len(val_ds)

        # Concaténation de tous les batches pour le calcul global de l'AUC-ROC
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)

        try:
            # average='macro' ou 'samples' selon la préférence en multi-label
            epoch_val_auc = roc_auc_score(val_targets, val_preds, average='macro')
        except ValueError:
            epoch_val_auc = 0.0

        print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC-ROC: {epoch_val_auc:.4f}")

        # Logique Early Stopping & Sauvegarde du meilleur modèle basées sur l'AUC-ROC (maximisation)
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Sauvegarde du meilleur modèle (best_model.pth) suite à une amélioration de l'AUC-ROC.")
        else:
            patience_counter += 1
            print(f"Pas d'amélioration de l'AUC-ROC de validation ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping déclenché : arrêt anticipé de l'entraînement.")
                break

    print("\nEntraînement terminé avec succès !")


if __name__ == "__main__":
    train_model()