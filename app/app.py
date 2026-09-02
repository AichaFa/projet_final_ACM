# ============================================================================
# Auditeur de Cohérence Médicale - Application AUTONOME (Gradio)
# Thème : Médical Clair (Fond blanc/gris clair, accents Bleu Navy, cartes pures)
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

# --- Chargement des encodeurs publics ---
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

# --- Icônes de résultat ---
ICONE_OK = (
    "<svg width='28' height='28' viewBox='0 0 24 24' fill='none'>"
    "<circle cx='12' cy='12' r='11' fill='#10b981'/>"
    "<path d='M7 12.5l3 3 7-7' stroke='#ffffff' stroke-width='2.5' "
    "fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>"
)
ICONE_NON = (
    "<svg width='28' height='28' viewBox='0 0 24 24' fill='none'>"
    "<circle cx='12' cy='12' r='11' fill='#ef4444'/>"
    "<path d='M8 8l8 8M16 8l-8 8' stroke='#ffffff' stroke-width='2.5' "
    "stroke-linecap='round'/></svg>"
)


# Function modifiée pour générer des cartes claires
def _carte(accent, icone, titre, detail, fond_clair):
    return (
        f"<div style='background:{fond_clair}; border-left:6px solid {accent}; "
        f"border-radius:12px; padding:20px; display:flex; align-items:center; "
        f"gap:16px; margin-top:12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);'>{icone}"
        f"<div><div style='font-size:20px; font-weight:700; color:{accent};'>{titre}</div>"
        f"<div style='color:#334155; margin-top:4px; font-size:15px;'>{detail}</div></div></div>"
    )


@spaces.GPU
def verifier(image_path, texte):
    if image_path is None or not texte or not texte.strip():
        return _carte(
            "#d97706",
            "",
            "Informations manquantes",
            "Merci de fournir une radiographie ET un compte rendu.",
            "#fffbe8",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with torch.no_grad():
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

            similarite = torch.mm(image_embedding, text_embedding.t()).item()
            proba = 1 / (1 + math.exp(-similarite * 4))
    except Exception as e:
        return _carte("#dc2626", "", "Erreur lors de l'analyse", str(e), "#fef2f2")

    if proba >= SEUIL:
        return _carte(
            "#059669",
            ICONE_OK,
            "COHÉRENT",
            f"Le compte rendu correspond à la radiographie. "
            f"Score de confiance : <b>{proba:.3f}</b>",
            "#ecfdf5",
        )
    return _carte(
        "#dc2626",
        ICONE_NON,
        "INCOHÉRENT",
        f"Le compte rendu ne correspond pas à la radiographie. "
        f"Score de confiance : <b>{proba:.3f}</b>",
        "#fef2f2",
    )


def effacer():
    return None, "", ""


# --- CSS Thème Médical / Navy ---
CSS = """
/* Fond général de la page */
html, body, .gradio-container, .app { background: #f8fafc !important; }

.gradio-container {
  max-width: 1400px !important;
  width: 95% !important;
  margin: 0 auto !important;
  color: #1e293b !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Textes et étiquettes */
.gradio-container label, .gradio-container .prose, .gradio-container .prose *,
.gradio-container span, .gradio-container p,
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  color: #0f172a !important;
}

/* En-tête Médical Navy */
#entete {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
  padding: 24px 30px; 
  border-radius: 16px; 
  margin-bottom: 20px;
  box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25); 
  text-align: center;
}
#entete h1 { 
  margin: 0; 
  font-size: 28px; 
  font-weight: 700;
  color: #ffffff !important; 
  letter-spacing: -0.5px;
}
#entete p { 
  margin: 8px 0 0 0; 
  color: #93c5fd !important; 
  font-size: 15px;
}

#carte_blanche { background: transparent !important; border: none !important; padding: 0 !important; }

/* Conteneurs d'image et de texte */
.col-image, .col-texte {
  background: #ffffff !important; 
  border: 1px solid #e2e8f0 !important;
  border-radius: 14px; 
  padding: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

#ligne_entrees { flex-wrap: nowrap !important; gap: 16px !important; }

/* Zones de texte / saisie */
.gradio-container textarea, .gradio-container input[type=text] {
  background: #f8fafc !important; 
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
}
.gradio-container textarea:focus, .gradio-container input[type=text]:focus {
  border-color: #1e3a8a !important;
  box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.15) !important;
}

/* Bouton principal Bleu Navy */
#btn_verifier button, #btn_verifier {
  background: #1e3a8a !important; 
  color: #ffffff !important;
  border: none !important; 
  font-weight: 600 !important;
  font-size: 16px !important;
  border-radius: 10px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 12px rgba(30, 58, 138, 0.25) !important;
}
#btn_verifier button:hover, #btn_verifier:hover { 
  background: #1d4ed8 !important; 
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(29, 78, 216, 0.35) !important;
}

/* Bouton Effacer */
#btn_effacer button, #btn_effacer {
  background: #ffffff !important; 
  color: #475569 !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
}
#btn_effacer button:hover, #btn_effacer:hover {
  background: #f1f5f9 !important;
  color: #0f172a !important;
  border-color: #94a3b8 !important;
}

/* Accordéon d'aide */
.gradio-container .accordion {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 10px !important;
}

#pied { color: #64748b; font-size: 13px; text-align: center; margin-top: 16px; }
"""

with gr.Blocks(css=CSS, title="Auditeur de Cohérence Médicale") as demo:
    gr.HTML(
        "<div id='entete'><h1>Auditeur de Cohérence Médicale</h1>"
        "<p>Vérifiez si un compte rendu clinique correspond bien à sa "
        "radiographie thoracique.</p></div>"
    )
    with gr.Group(elem_id="carte_blanche"):
        with gr.Row(elem_id="ligne_entrees", equal_height=True):
            with gr.Column(elem_classes="col-image", min_width=280):
                image_in = gr.Image(
                    label="Radiographie thoracique", type="filepath", height=300
                )
            with gr.Column(elem_classes="col-texte", min_width=280):
                texte_in = gr.Textbox(label="Compte rendu radiologique", lines=11)
        with gr.Row():
            bouton = gr.Button(
                "Lancer la vérification", elem_id="btn_verifier", scale=3
            )
            bouton_effacer = gr.Button("Effacer", elem_id="btn_effacer", scale=1)
        resultat = gr.HTML()

    with gr.Accordion("Comment ça marche ?", open=False):
        gr.Markdown(
            "L'outil encode la radiographie et le compte rendu avec le modèle "
            "public BioViL-T, puis mesure leur correspondance. Un score élevé "
            "indique une cohérence entre l'image et le texte. Outil d'aide, "
            "ne remplaçant pas l'avis d'un professionnel de santé."
        )

    gr.HTML("<div id='pied'>Auditeur de Cohérence Médicale - démonstration</div>")

    bouton.click(verifier, inputs=[image_in, texte_in], outputs=resultat)
    bouton_effacer.click(effacer, inputs=None, outputs=[image_in, texte_in, resultat])


if __name__ == "__main__":
    demo.launch()
