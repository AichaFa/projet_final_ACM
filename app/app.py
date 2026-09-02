# ============================================================================
# Auditeur de Cohérence Médicale - Application AUTONOME (Gradio)
# ----------------------------------------------------------------------------
# Elle charge les encodeurs BioViL-T PUBLICS (image + texte) et calcule
# elle-même la cohérence entre la radiographie et le compte rendu, par
# similarité. Aucune dépendance à une API externe (fini les erreurs 503).
# ============================================================================

import math
from pathlib import Path

import gradio as gr
import torch
import spaces
from health_multimodal.image.inference_engine import ImageInferenceEngine
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from transformers import AutoTokenizer, AutoModel

SEUIL = 0.5

# --- Chargement des encodeurs publics (une seule fois, au démarrage) ---
image_encoder = get_biovil_t_image_encoder()
transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)
image_engine = ImageInferenceEngine(image_encoder, transform)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T", trust_remote_code=True
)
text_model = AutoModel.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T", trust_remote_code=True
)
text_model.eval()


@spaces.GPU
def verifier(image_path, texte):
    if image_path is None or not texte or not texte.strip():
        return "Merci de fournir une radiographie ET un compte rendu."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with torch.no_grad():
            # Embedding de l'image
            image_embedding = image_engine.get_projected_global_embedding(
                Path(image_path)
            )
            if not isinstance(image_embedding, torch.Tensor):
                image_embedding = torch.tensor(image_embedding)
            image_embedding = image_embedding.to(device)
            if image_embedding.ndim == 1:
                image_embedding = image_embedding.unsqueeze(0)
            image_embedding = image_embedding / image_embedding.norm(
                dim=-1, keepdim=True
            )

            # Embedding du texte
            inputs = tokenizer(
                texte,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            ).to(device)
            text_model.to(device)
            if hasattr(text_model, "get_projected_text_embeddings"):
                text_embedding = text_model.get_projected_text_embeddings(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            else:
                outputs = text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                text_embedding = (
                    outputs.pooler_output
                    if hasattr(outputs, "pooler_output")
                    else outputs[0][:, 0, :]
                )
            if text_embedding.ndim == 1:
                text_embedding = text_embedding.unsqueeze(0)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            # Similarité -> probabilité
            similarite = torch.mm(image_embedding, text_embedding.t()).item()
            proba = 1 / (1 + math.exp(-similarite * 4))
    except Exception as e:
        return f"Erreur lors de l'analyse : {e}"

    coherent = proba >= SEUIL
    verdict = "COHÉRENT" if coherent else "INCOHÉRENT"
    return f"Résultat : {verdict}  -  score de confiance : {proba:.3f}"


with gr.Blocks(title="Auditeur de Cohérence Médicale") as demo:
    gr.Markdown(
        "## Auditeur de Cohérence Médicale\n"
        "Vérifiez si un compte rendu radiologique correspond bien à sa "
        "radiographie thoracique."
    )
    with gr.Row():
        image_in = gr.Image(label="Radiographie thoracique", type="filepath")
        texte_in = gr.Textbox(label="Compte rendu radiologique", lines=12)
    bouton = gr.Button("Lancer la vérification", variant="primary")
    resultat = gr.Textbox(label="Résultat", interactive=False)
    bouton.click(verifier, inputs=[image_in, texte_in], outputs=resultat)


if __name__ == "__main__":
    demo.launch()
