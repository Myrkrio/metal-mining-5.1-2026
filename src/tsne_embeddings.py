import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from models import get_model


def extract_features(model, loader, device, max_samples=2500):
    model.eval()
    embeddings_list = []
    targets_list = []
    seen = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = model.extract_embeddings(images)
            embeddings_list.append(emb.float().cpu().numpy())
            targets_list.append(targets.numpy())

            seen += images.size(0)
            if max_samples and seen >= max_samples:
                break

    embeddings = np.vstack(embeddings_list)
    targets = np.vstack(targets_list)
    if max_samples:
        embeddings = embeddings[:max_samples]
        targets = targets[:max_samples]
    return embeddings, targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = "convnext-base"
    batch_size = 32

    _, _, test_ds, all_genres = get_dataloaders(
        json_path="data/dataset_ml.json", batch_size=batch_size
    )

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = get_model(
        architecture=architecture,
        num_classes=len(all_genres),
        dropout=0.3,
    ).to(device)
    model.load_state_dict(torch.load("./results/best_model.pth", map_location=device))

    # 1. Extraction des embeddings (ex: dimension 1024 pour convnext-base)
    embeddings, targets = extract_features(model, test_loader, device, max_samples=10000)
    print(f"Embeddings extraits : {embeddings.shape}")

    # 2. Projection 2D avec t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    coords_2d = tsne.fit_transform(embeddings)

    # 3. Coloration simple : on prend le premier genre actif (ou "Other" si aucun)
    # En multi-label, une cover peut avoir plusieurs genres : c'est une visualisation indicative
    genre_indices = np.argmax(targets, axis=1)
    has_any_label = targets.sum(axis=1) > 0

    plt.figure(figsize=(11, 8))

    # Affiche les points sans label en gris clair
    if np.any(~has_any_label):
        plt.scatter(
            coords_2d[~has_any_label, 0],
            coords_2d[~has_any_label, 1],
            c="lightgray",
            s=10,
            alpha=0.4,
            label="Aucun genre",
        )

    # Colorie les genres les plus frequents pour garder une legende lisible
    counts_in_subset = targets.sum(axis=0)
    top_genre_idx = np.argsort(counts_in_subset)[::-1]

    for idx in top_genre_idx:
        if counts_in_subset[idx] == 0:
            continue
        mask = has_any_label & (genre_indices == idx)
        if np.any(mask):
            plt.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                s=14,
                alpha=0.75,
                label=f"{all_genres[idx]} ({int(counts_in_subset[idx])})",
            )

    plt.title(f"t-SNE des embeddings ({architecture})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig("tsne_covers.png", dpi=200)
    plt.show()
    print("Graphique sauvegarde dans tsne_covers.png")


if __name__ == "__main__":
    main()