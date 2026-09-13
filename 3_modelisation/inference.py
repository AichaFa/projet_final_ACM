"""
Cœur d'inférence de l'Auditeur de Cohérence Médicale.

Reçoit une radiographie thoracique et son compte rendu, et renvoie
un verdict (cohérent ou incohérent) accompagné d'un score.

Inférence autonome : chargement de BioViL-T (public, licence MIT),
reconstruction du classifieur entraîné et chargement de ses poids
portables (model.safetensors), puis application du seuil de décision
de 0,5.
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

# Encodeur d'image de BioViL-T, fourni par le paquet health_multimodal
# (installation : hi-ml-multimodal).
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "microsoft/BiomedVLP-BioViL-T"   # BioViL-T, public, licence MIT
WEIGHTS_PATH = "model.safetensors"           # poids portables du classifieur
SEUIL = 0.5                                  # >= 0,5 : cohérent ; sinon : incohérent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Architecture du classifieur (identique au notebook de modélisation)
# ---------------------------------------------------------------------------
class VisualProjectionLayer(nn.Module):
    """Projette les patches d'image (dimension 128) vers la dimension du texte (768)."""

    def __init__(self, img_dim=128, text_dim=768):
        super().__init__()
        self.projector = nn.Linear(img_dim, text_dim)

    def forward(self, img_patches):
        x = img_patches.permute(0, 2, 3, 1)   # [batch, 14, 14, 128]
        x = x.flatten(1, 2)                    # [batch, 196, 128]
        return self.projector(x)               # [batch, 196, 768]


class CrossAttentionClassifierBiovil(nn.Module):
    """Bloc d'attention croisée : le texte interroge l'image, puis classification."""

    def __init__(self, embed_dim=768, num_heads=8, dropout=0.3):
        super().__init__()
        self.projection_layer = VisualProjectionLayer()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, img_patches, text_tokens):
        img_patches_proj = self.projection_layer(img_patches)
        norm_text = self.layer_norm1(text_tokens)
        attn_output, _ = self.cross_attention(
            query=norm_text, key=img_patches_proj, value=img_patches_proj
        )
        x = attn_output + text_tokens
        x = self.layer_norm2(self.ffn(x)) + x
        x_pooled, _ = torch.max(x, dim=1)
        return self.classifier(x_pooled)       # score brut (logit), forme [batch, 1]


# ---------------------------------------------------------------------------
# Chargement (une seule fois) de BioViL-T et du classifieur
# ---------------------------------------------------------------------------
_ressources = {}

def charger_modeles():
    if _ressources:
        return _ressources

    # Côté texte : tokenizer et modèle CXR-BERT de BioViL-T
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

    # Côté image : encodeur image de BioViL-T
    image_model = get_biovil_t_image_encoder().to(DEVICE).eval()
    image_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)

    # Classifieur entraîné : architecture reconstruite, puis chargement des poids portables
    classifieur = CrossAttentionClassifierBiovil()
    classifieur.load_state_dict(load_file(WEIGHTS_PATH))
    classifieur = classifieur.to(DEVICE).eval()

    _ressources.update(
        tokenizer=tokenizer,
        text_model=text_model,
        image_model=image_model,
        image_transform=image_transform,
        classifieur=classifieur,
    )
    return _ressources


# ---------------------------------------------------------------------------
# Préparation des entrées
# ---------------------------------------------------------------------------
def encoder_texte(compte_rendu, r):
    """Transforme le compte rendu en une séquence de vecteurs, forme [1, 512, 768]."""
    inputs = r["tokenizer"](
        compte_rendu,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        sortie = r["text_model"](
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True,
        )
    return sortie.last_hidden_state


def encoder_image(image, r):
    """Transforme la radiographie en patches, forme [1, 128, 14, 14]."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("L")
    tenseur = r["image_transform"](image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sortie = r["image_model"](tenseur)
    return sortie.projected_patch_embeddings


# ---------------------------------------------------------------------------
# Prédiction complète : image + texte -> verdict
# ---------------------------------------------------------------------------
def predire(image, compte_rendu):
    """
    Renvoie un dictionnaire :
      - verdict     : "COHÉRENT" ou "INCOHÉRENT"
      - prediction  : 1 (cohérent) ou 0 (incohérent)
      - probabilite : P(cohérent), entre 0 et 1
      - confiance   : proximité au verdict rendu, entre 0,5 et 1
    """
    r = charger_modeles()

    patches_image = encoder_image(image, r)
    sequence_texte = encoder_texte(compte_rendu, r)

    with torch.no_grad():
        logit = r["classifieur"](patches_image, sequence_texte)
        probabilite = torch.sigmoid(logit).item()

    prediction = 1 if probabilite >= SEUIL else 0
    verdict = "COHÉRENT" if prediction == 1 else "INCOHÉRENT"
    confiance = probabilite if prediction == 1 else 1.0 - probabilite

    return {
        "verdict": verdict,
        "prediction": prediction,
        "probabilite": round(probabilite, 4),
        "confiance": round(confiance, 4),
    }


# ---------------------------------------------------------------------------
# Test manuel
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exemple_image = "exemple_radio.png"
    exemple_texte = "No acute cardiopulmonary process."
    print(predire(exemple_image, exemple_texte))
