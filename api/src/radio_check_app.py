"""
API de l'Auditeur de Cohérence Médicale (FastAPI).

Reçoit une radiographie (image) et un compte rendu (texte), les passe dans
BioViL-T et le classifieur à attention croisée, puis renvoie un score de
cohérence. C'est le moteur appelé par l'application.

Chargement autonome : BioViL-T (public, licence MIT) et le classifieur
reconstruit puis chargé depuis les poids portables locaux (model.safetensors).
Aucune dépendance à un serveur MLflow ni à un stockage S3.

Les grosses bibliothèques (torch, transformers, BioViL-T) sont importées à
l'intérieur des fonctions qui les utilisent : un simple test qui ne fait que
lire le fichier et appeler /health n'a donc pas besoin de les installer. Le
comportement de l'API, lui, est identique quand elle tourne réellement.
"""

import os  # Accès au système de fichiers (chemin des poids)
import io  # Lire le fichier image reçu directement en mémoire
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)  # Briques de l'API web
from PIL import Image  # Pillow : ouvre l'image reçue

# Chemin des poids portables, résolu par rapport à ce fichier (donc valable
# quel que soit le dossier de lancement).
POIDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.safetensors")

# Création de l'application API et de son titre
app = FastAPI(title="BioVil Cross-Attention+MLP Inference API")

# Variables globales : elles contiendront les modèles et outils, chargés une
# seule fois au démarrage puis réutilisés à chaque requête. Elles valent None au départ.
device = None  # "cuda" (carte graphique) ou "cpu"
tokenizer = None  # Découpe le texte en jetons pour le modèle
text_model = None  # Encodeur de texte
image_model = None  # Encodeur d'image
image_transform = None  # Prétraitement de l'image (redimensionnement, recadrage)
cross_att_classifier = None  # Classifieur final (attention croisée)


def _construire_classifieur(device):
    # Reconstruit l'architecture du classifieur (identique au notebook de
    # modélisation) puis y charge les poids portables. Aucune exécution de
    # code externe : le format safetensors ne contient que des chiffres.
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    class VisualProjectionLayer(nn.Module):
        def __init__(self, img_dim=128, text_dim=768):
            super().__init__()
            self.projector = nn.Linear(img_dim, text_dim)

        def forward(self, img_patches):
            x = img_patches.permute(0, 2, 3, 1)
            x = x.flatten(1, 2)
            return self.projector(x)

    class CrossAttentionClassifierBiovil(nn.Module):
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
            return self.classifier(x_pooled)

    modele = CrossAttentionClassifierBiovil()
    modele.load_state_dict(load_file(POIDS))
    return modele.to(device).eval()


# COMPOSANT DE DÉMARRAGE
# Cette fonction s'exécute UNE fois, au démarrage de l'API : elle charge les modèles.
@app.on_event("startup")
def load_all_models_and_assets():
    global device, tokenizer, text_model, image_model, image_transform, cross_att_classifier
    try:
        # Les grosses bibliothèques sont importées ici, au moment où on en a besoin
        import torch
        from transformers import AutoTokenizer, AutoModel
        from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
        from health_multimodal.image.data.transforms import (
            create_chest_xray_transform_for_inference,
        )

        # Choix du matériel : carte graphique si disponible, sinon processeur
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Dépôt BioViL-T (côté texte)
        model_id = "microsoft/BiomedVLP-BioViL-T"

        # Tokenizer et modèle de texte spécialisés radiologie (CXR-BERT)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        text_model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(
            device
        )
        text_model.eval()  # Mode évaluation (on utilise le modèle, on ne l'entraîne pas)

        # Encodeur d'image BioViL-T + prétraitement associé
        image_model = get_biovil_t_image_encoder().to(device)
        image_transform = create_chest_xray_transform_for_inference(
            resize=512, center_crop_size=448
        )
        image_model.eval()

        # Classifieur à attention croisée, reconstruit et chargé depuis les
        # poids portables locaux (plus de MLflow ni de S3)
        cross_att_classifier = _construire_classifieur(device)

        print("All 3 models and processors loaded into memory successfully!")
    except Exception as e:
        # En cas d'erreur au chargement, on l'affiche et on interrompt le démarrage
        print(f"Erreur au démarrage : {str(e)}")
        raise e


# HEALTH CHECK ENDPOINT (petite route "es-tu en vie ?")
# Elle ne touche pas aux modèles : elle répond donc instantanément. Elle sert aux
# tests et au suivi (l'API est-elle démarrée ? les modèles sont-ils chargés ?).
@app.get("/health")
def health():
    models_ready = None not in (cross_att_classifier, text_model, image_model)
    return {"status": "ok", "models_loaded": models_ready}


# CHAÎNES DE PRÉTRAITEMENT
def get_text_embeddings(report_text):
    import torch  # utilisé seulement au calcul (déjà chargé une fois l'API démarrée)

    # Transforme le compte rendu en jetons, puis en représentation numérique (embeddings)
    inputs = tokenizer(
        report_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with (
        torch.no_grad()
    ):  # Pas de calcul de gradient : on ne fait qu'utiliser le modèle
        outputs = text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True,
        )
    return outputs.last_hidden_state


def get_image_embeddings_from_pil(pil_image):
    import torch

    # Reçoit directement l'objet image (en mémoire) et le prépare pour le modèle
    raw_image = pil_image.convert("L")  # Conversion en niveaux de gris
    processed_tensor = image_transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_outputs = image_model(processed_tensor)
    return image_outputs.projected_patch_embeddings


# LA ROUTE DE PRÉDICTION
@app.post("/predict")
async def predict(
    text_input: str = Form(...),  # Le compte rendu, envoyé via le formulaire
    image_file: UploadFile = File(...),  # La radiographie, envoyée en tant que fichier
):
    # Garde-fou : si une requête arrive avant que les modèles soient chargés, on refuse poliment
    if None in (cross_att_classifier, text_model, image_model):
        raise HTTPException(
            status_code=503, detail="Models are initializing. Try again shortly."
        )

    import torch  # nécessaire pour le calcul ci-dessous (l'API est démarrée à ce stade)

    try:
        # Lecture du fichier reçu, directement en mémoire, sous forme d'image PIL
        image_bytes = await image_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Prétraitements : on obtient les représentations du texte et de l'image
        sequence_outputs = get_text_embeddings(text_input)
        patch_img_emb = get_image_embeddings_from_pil(pil_image)

        # Passage dans le classifieur à attention croisée -> score de cohérence
        with torch.no_grad():
            outputs = cross_att_classifier(patch_img_emb, sequence_outputs).squeeze(1)
            probability = torch.sigmoid(outputs).item()  # Score entre 0 et 1
            prediction = int(probability >= 0.5)  # 1 = cohérent, 0 = incohérent

        # Réponse renvoyée à l'application (au format JSON)
        return {
            "status": "success",
            "prediction": prediction,
            "probability": round(probability, 4),
        }

    except Exception as e:
        # En cas d'erreur pendant l'inférence, on renvoie une erreur claire
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
