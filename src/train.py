import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from models import get_model, unfreeze_layers
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


def compute_pos_weights(train_ds, device, max_weight=10.0):
    """Calcule un pos_weight adouci par classe a partir du train set uniquement."""
    targets_list = []
    for _, target in train_ds:
        t = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
        targets_list.append(t)

    targets_arr = np.vstack(targets_list).astype(np.float32)
    n_samples = targets_arr.shape[0]
    pos_counts = targets_arr.sum(axis=0)
    neg_counts = n_samples - pos_counts

    safe_pos = np.maximum(pos_counts, 1.0)
    raw_weights = neg_counts / safe_pos

    smoothed_weights = np.clip(np.sqrt(raw_weights), 1.0, max_weight)
    smoothed_weights[pos_counts == 0] = 1.0

    return torch.tensor(smoothed_weights, dtype=torch.float32, device=device)


def build_optimizer(model, lr, weight_decay=1e-2):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du peripherique : {device}")

    batch_size = 32
    epochs = 15
    patience = 3
    architecture = "convnext-base"
    unfreeze_epoch = 3  # Passage au level 2 au debut de l'epoque 4 (0-indexed)
    use_pos_weight = True  # Mets False si tu veux tester sans pondération

    print("Chargement des donnees (Split 80/10/10)...")
    train_ds, val_ds, test_ds, all_genres = get_dataloaders(
        json_path="data/dataset_ml.json", batch_size=batch_size
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(all_genres)
    print(f"Nombre de classes cibles (genres uniques) : {num_classes}")

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    if use_pos_weight:
        pos_weight = compute_pos_weights(train_ds, device=device, max_weight=10.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Level 1 : seule la tete personnalisee est entrainable
    optimizer = build_optimizer(model, lr=1e-4)

    best_val_auc = 0.0
    patience_counter = 0

    print("\nDemarrage de l'entrainement multi-label...")
    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            unfreeze_layers(model, architecture=architecture, level=2)
            # Level 2 : On peut ajuster le taux d'apprentissage pour le degel partiel
            optimizer = build_optimizer(model, lr=1e-5)

        model.train()
        train_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, targets = images.to(device), targets.to(device).float()

            optimizer.zero_grad()
            logits = model(images)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_ds)

        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).float()
                logits = model(images)

                loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(targets.cpu().numpy())

        epoch_val_loss = val_loss / len(val_ds)

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)

        try:
            epoch_val_auc = roc_auc_score(val_targets, val_preds, average="macro")
        except ValueError:
            epoch_val_auc = 0.0

        print(
            f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val AUC-ROC: {epoch_val_auc:.4f}"
        )

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Sauvegarde du meilleur modele (best_model.pth) suite a une amelioration de l'AUC-ROC.")
        else:
            patience_counter += 1
            print(f"Pas d'amelioration de l'AUC-ROC de validation ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping declenche : arret anticipe de l'entrainement.")
                break

    print("\nEntrainement termine avec succes !")


if __name__ == "__main__":
    train_model()