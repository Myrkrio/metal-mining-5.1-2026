import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification
import torchvision.models as models

def get_model(architecture="convnext-tiny", num_classes=10, pretrained=True):
    """
    Initialise le modèle (ConvNeXt ou ResNet50) et gèle le backbone par défaut
    pour respecter la stratégie de Linear Probing initiale.
    """
    arch_lower = architecture.lower()
    
    if "convnext" in arch_lower:
        # Chargement de ConvNeXt depuis Hugging Face (ex: facebook/convnext-tiny-224)
        model_id = f"facebook/{arch_lower}-224"
        print(f"Chargement de l'architecture {model_id} depuis Hugging Face...")
        
        model = AutoModelForImageClassification.from_pretrained(
            model_id, 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )
        
        # ÉTAPE 1 : Gel total du backbone (Linear Probing)
        for param in model.parameters():
            param.requires_grad = False
            
        # Dégel uniquement de la tête de classification finale
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif "resnet50" in arch_lower:
        print("Chargement de l'architecture ResNet-50 (Benchmark)...")
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        
        # Adaptation de la couche linéaire finale pour notre nombre de classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Gel total du backbone
        for param in model.parameters():
            param.requires_grad = False
            
        # Dégel uniquement de la couche fc
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Architecture non supportée : {architecture}. Choisis 'convnext-tiny' ou 'resnet50'.")

    return model

def unfreeze_layers(model, architecture="convnext-tiny", level=1):
    """
    Gère le dégel progressif des couches selon les résultats de l'entraînement :
    - Level 1 : Tête de classification uniquement (Linear Probing)
    - Level 2 : Tête + derniers blocs de features (Dégel partiel)
    - Level 3 : Tout le réseau (Full Fine-Tuning)
    """
    arch_lower = architecture.lower()
    
    if level == 1:
        print("Niveau 1 : Seule la tête de classification est entraînable.")
        # Le gel initial s'en charge déjà, mais on s'assure que seul le classifier/fc est actif
        if "convnext" in arch_lower:
            for p in model.parameters(): p.requires_grad = False
            for p in model.classifier.parameters(): p.requires_grad = True
        else:
            for p in model.parameters(): p.requires_grad = False
            for p in model.fc.parameters(): p.requires_grad = True
            
    elif level == 2:
        print("Niveau 2 : Dégel partiel (Tête + derniers blocs profonds).")
        if "convnext" in arch_lower:
            for p in model.parameters(): p.requires_grad = False
            for p in model.classifier.parameters(): p.requires_grad = True
            # Dégel du dernier stage de ConvNeXt
            for p in model.convnext.encoder.stages[-1].parameters():
                p.requires_grad = True
        else:
            for p in model.parameters(): p.requires_grad = False
            for p in model.fc.parameters(): p.requires_grad = True
            for p in model.layer4.parameters():
                p.requires_grad = True
                
    elif level == 3:
        print("Niveau 3 : Dégel complet (Full Fine-Tuning de tout le réseau).")
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("Niveau de dégel inconnu (choisis 1, 2 ou 3).")