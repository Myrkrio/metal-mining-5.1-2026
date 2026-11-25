import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ConvNextModel


class CoverEmbeddingClassifier(nn.Module):
    """
    Extrait les embeddings via un backbone (ConvNeXt-Base par defaut)
    puis applique une tete de classification personnalisee.
    """

    def __init__(self, architecture="convnext-base", num_classes=10, dropout=0.3, pretrained=True):
        super().__init__()
        self.architecture = architecture.lower()

        if self.architecture in ("convnext", "convnext-base", "convnext_base"):
            model_id = "facebook/convnext-base-224"
            print(f"Chargement de l'architecture {model_id} depuis Hugging Face...")
            self.backbone = ConvNextModel.from_pretrained(model_id)
            embed_dim = self.backbone.config.hidden_sizes[-1]  # 1024 pour convnext-base
            print(f"Dimension des embeddings : {embed_dim}")
            self.backbone_kind = "convnext"

        elif self.architecture in ("convnext-tiny", "convnext_tiny"):
            model_id = "facebook/convnext-tiny-224"
            print(f"Chargement de l'architecture {model_id} depuis Hugging Face...")
            self.backbone = ConvNextModel.from_pretrained(model_id)
            embed_dim = self.backbone.config.hidden_sizes[-1]  # 768 pour convnext-tiny
            print(f"Dimension des embeddings : {embed_dim}")
            self.backbone_kind = "convnext"

        elif self.architecture == "resnet50":
            print("Chargement de l'architecture ResNet-50 (Benchmark)...")
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            embed_dim = resnet.fc.in_features  # 2048 pour resnet50
            resnet.fc = nn.Identity()
            self.backbone = resnet
            self.backbone_kind = "resnet50"

        else:
            raise ValueError(
                f"Architecture non supportee : {architecture}. "
                "Choisis 'convnext-base', 'convnext-tiny' ou 'resnet50'."
            )

        # Tete de classification personnalisee (remplace l'ancienne tete ImageNet)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Gel initial : Linear Probing (seule la tete personnalisee est entrainable)
        unfreeze_layers(self, architecture=self.architecture, level=1)

    def extract_embeddings(self, x):
        """Retourne les embeddings (vecteurs de features) avant la tete de classification."""
        if self.backbone_kind == "convnext":
            outputs = self.backbone(pixel_values=x)
            return outputs.pooler_output
        return self.backbone(x)

    def forward(self, x):
        embeddings = self.extract_embeddings(x)
        logits = self.classifier(embeddings)
        return logits


def get_model(architecture="convnext-base", num_classes=10, dropout=0.3, pretrained=True):
    return CoverEmbeddingClassifier(
        architecture=architecture,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )


def unfreeze_layers(model, architecture="convnext-base", level=1):
    """
    Gere le degel progressif des couches :
    - Level 1 : Tete personnalisee uniquement (Linear Probing)
    - Level 2 : Tete + dernier bloc profond du backbone (Degel partiel)
    - Level 3 : Tout le reseau (Full Fine-Tuning)
    """
    arch_lower = architecture.lower()

    if level == 1:
        print("Niveau 1 : Seule la tete de classification personnalisee est entrainable.")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

    elif level == 2:
        print("Niveau 2 : Degel partiel (Tete + derniers blocs profonds).")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

        if "convnext" in arch_lower:
            for p in model.backbone.encoder.stages[-1].parameters():
                p.requires_grad = True
        elif "resnet50" in arch_lower:
            for p in model.backbone.layer4.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Architecture non supportee pour level=2 : {architecture}")

    elif level == 3:
        print("Niveau 3 : Degel complet (Full Fine-Tuning de tout le reseau).")
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError("Niveau de degel inconnu (choisis 1, 2 ou 3).")