import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False


class SquarePad:
    """Ajoute un padding noir pour garder le format carre sans deformer la cover."""
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')


def extract_genres_list(item):
    """
    Recupere les genres en supposant qu'ils sont deja bien formates
    (soit une liste, soit une chaine propre type 'Doom, Death Metal').
    """
    raw = item.get("genres") or item.get("genre") or item.get("raw_genre")

    if isinstance(raw, list):
        return [str(g).strip().title() for g in raw if str(g).strip()]
    elif isinstance(raw, str):
        return [g.strip().title() for g in raw.split(",") if g.strip()]

    return ["Unknown"]


def build_label_matrix(items, genre_to_idx):
    """Construit la matrice binaire (N, num_classes) pour la stratification multi-label."""
    matrix = np.zeros((len(items), len(genre_to_idx)), dtype=np.int32)
    for i, item in enumerate(items):
        for g in extract_genres_list(item):
            if g in genre_to_idx:
                matrix[i, genre_to_idx[g]] = 1
    return matrix


def stratified_split_items(valid_items, genre_to_idx, random_state=42):
    """Repartit valid_items en ~80% train / 10% val / 10% test en conservant les genres."""
    if HAS_ITERSTRAT and len(valid_items) >= 10:
        indices = np.arange(len(valid_items))
        label_matrix = build_label_matrix(valid_items, genre_to_idx)

        # 10% test
        msss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=random_state)
        train_val_idx, test_idx = next(msss_test.split(indices, label_matrix))

        # 10% du total pour val sur les 90% restants (1/9 = 0.1111...)
        msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1.0 / 9.0, random_state=random_state)
        train_rel_idx, val_rel_idx = next(msss_val.split(train_val_idx, label_matrix[train_val_idx]))

        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]

        train_items = [valid_items[i] for i in train_idx]
        val_items = [valid_items[i] for i in val_idx]
        test_items = [valid_items[i] for i in test_idx]
        return train_items, val_items, test_items

    # Repli si iterative-stratification n'est pas installe : stratification sur le genre le plus rare de chaque item
    genre_counts = np.sum(build_label_matrix(valid_items, genre_to_idx), axis=0)
    rarest_label = []
    for item in valid_items:
        genres = [g for g in extract_genres_list(item) if g in genre_to_idx]
        if not genres:
            rarest_label.append("Unknown")
        else:
            rarest_g = min(genres, key=lambda g: genre_counts[genre_to_idx[g]])
            # Si le genre rare a < 3 occurrences, on le regroupe pour eviter une erreur de stratify
            rarest_label.append(rarest_g if genre_counts[genre_to_idx[rarest_g]] >= 3 else "Rare_Group")

    try:
        train_val_items, test_items = train_test_split(
            valid_items, test_size=0.10, random_state=random_state, stratify=rarest_label
        )
        rarest_train_val = [
            rarest_label[i] for i, it in enumerate(valid_items) if it in set(id(x) for x in train_val_items)
        ]
        # Recalcule proprement les labels associes a train_val_items
        item_to_rare = {id(valid_items[i]): rarest_label[i] for i in range(len(valid_items))}
        strat_tv = [item_to_rare[id(x)] for x in train_val_items]
        train_items, val_items = train_test_split(
            train_val_items, test_size=1.0 / 9.0, random_state=random_state, stratify=strat_tv
        )
    except ValueError:
        train_val_items, test_items = train_test_split(valid_items, test_size=0.10, random_state=random_state)
        train_items, val_items = train_test_split(train_val_items, test_size=1.0 / 9.0, random_state=random_state)

    return train_items, val_items, test_items


class MetalCoverDataset(Dataset):
    def __init__(self, items, all_genres, transform=None):
        self.items = items
        self.all_genres = all_genres
        self.genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item["cover_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        target_vector = torch.zeros(len(self.all_genres), dtype=torch.float32)
        item_genres = extract_genres_list(item)
        for g in item_genres:
            if g in self.genre_to_idx:
                target_vector[self.genre_to_idx[g]] = 1.0

        return image, target_vector


def get_dataloaders(json_path="data/dataset_ml.json", batch_size=32):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Le fichier {json_path} est introuvable.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_items = []
    all_genres_set = set()

    for item in data:
        cover_path = item.get("cover_path")
        if cover_path and os.path.exists(cover_path):
            genres = extract_genres_list(item)
            all_genres_set.update(genres)
            valid_items.append(item)

    all_genres = sorted(list(all_genres_set))
    genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}

    train_items, val_items, test_items = stratified_split_items(valid_items, genre_to_idx, random_state=42)

    transform_pipeline = transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = MetalCoverDataset(train_items, all_genres, transform_pipeline)
    val_ds = MetalCoverDataset(val_items, all_genres, transform_pipeline)
    test_ds = MetalCoverDataset(test_items, all_genres, transform_pipeline)

    return train_ds, val_ds, test_ds, all_genres