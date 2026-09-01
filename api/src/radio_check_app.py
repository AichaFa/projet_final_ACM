# ============================================================================
# Model API (FastAPI) - Auditeur de Cohérence Médicale
# ----------------------------------------------------------------------------
# Rôle : recevoir une radiographie (image) et un compte rendu (texte), les
# passer dans le modèle BioViL-T + le classifieur à attention croisée, puis
# renvoyer un score de cohérence. C'est le "moteur" appelé par l'application.
#
# Ce qui a changé par rapport à ta version d'origine :
#   - commentaires explicatifs en français ;
#   - ajout de la route /health ;
#   - les grosses bibliothèques (torch, transformers, BioViL-T, MLflow) sont
#     désormais importées À L'INTÉRIEUR des fonctions qui les utilisent, et non
#     plus tout en haut. Pourquoi ? Ces paquets ne servent que lorsque les
#     modèles se chargent ou qu'une prédiction est calculée. En les important
#     là, un simple test (qui ne fait que "lire" le fichier et appeler /health)
#     n'a plus besoin de les installer. Le comportement de l'API, lui, est
#     identique quand elle tourne réellement.
# ============================================================================

import os  # Accès aux variables d'environnement (ex : APP_URI)
import io  # Lire le fichier image reçu directement en mémoire
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)  # Briques de l'API web
from PIL import Image  # Pillow : ouvre l'image reçue

# Création de l'application API et de son titre
app = FastAPI(title="BioVil Cross-Attention+MLP Inference API")

# Variables globales : elles contiendront les modèles et outils, chargés une
# seule fois au démarrage puis réutilisés à chaque requête. Elles valent None au départ.
device = None  # "cuda" (carte graphique) ou "cpu"
tokenizer = None  # Découpe le texte en jetons pour le modèle
text_model = None  # Encodeur de texte
image_model = None  # Encodeur d'image
image_transform = None  # Prétraitement de l'image (redimensionnement, recadrage)
cross_att_classifier = (
    None  # Classifieur final (attention croisée), chargé depuis MLflow
)


# COMPOSANT DE DÉMARRAGE
# Cette fonction s'exécute UNE fois, au démarrage de l'API : elle charge les modèles.
@app.on_event("startup")
def load_all_models_and_assets():
    global device, tokenizer, text_model, image_model, image_transform, cross_att_classifier
    try:
        # Les grosses bibliothèques sont importées ici, au moment où on en a besoin
        import torch
        import mlflow.pytorch
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

        # Connexion au MLflow hébergé (adresse lue dans la variable d'environnement APP_URI),
        # puis chargement du classifieur à attention croisée (dernière version enregistrée)
        mlflow.set_tracking_uri(os.environ.get("APP_URI"))
        model_uri = "models:/biovil_cross_attention_mlp/latest"

        cross_att_classifier = mlflow.pytorch.load_model(
            model_uri, map_location=torch.device("cpu")
        )
        cross_att_classifier.to(device).eval()

        print("All 3 models and processors loaded into memory successfully!")
    except Exception as e:
        # En cas d'erreur au chargement, on l'affiche et on interrompt le démarrage
        print(f"❌ Startup Error: {str(e)}")
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
            outputs = cross_att_classifier(
                patch_img_emb, sequence_outputs[:, :256, :]
            ).squeeze(1)
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
