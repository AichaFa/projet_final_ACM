# ============================================================================
# Auditeur de Cohérence Médicale - Application AUTONOME (Gradio)
# Thème : Fond très sombre, accents cyan/bleu moderne, design très classe.
# Charge les encodeurs BioViL-T PUBLICS et calcule la cohérence localement.
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


ICONE_OK = (
    "<svg width='26' height='26' viewBox='0 0 24 24' fill='none'>"
    "<circle cx='12' cy='12' r='11' fill='#10b981'/>"
    "<path d='M7 12.5l3 3 7-7' stroke='#0f172a' stroke-width='2.2' "
    "fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>"
)
ICONE_NON = (
    "<svg width='26' height='26' viewBox='0 0 24 24' fill='none'>"
    "<circle cx='12' cy='12' r='11' fill='#f43f5e'/>"
    "<path d='M8 8l8 8M16 8l-8 8' stroke='#0f172a' stroke-width='2.2' "
    "stroke-linecap='round'/></svg>"
)


def _carte(accent, icone, titre, detail):
    return (
        f"<div style='background:#1e293b; border-left:6px solid {accent}; "
        f"border-radius:12px; padding:18px 20px; display:flex; align-items:center; "
        f"gap:14px; margin-top:8px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);'>{icone}"
        f"<div><div style='font-size:20px; font-weight:700; color:{accent};'>{titre}</div>"
        f"<div style='color:#f8fafc; margin-top:2px;'>{detail}</div></div></div>"
    )


@spaces.GPU
def verifier(image_path, texte):
    if image_path is None or not texte or not texte.strip():
        return _carte(
            "#f59e0b",
            "",
            "Informations manquantes",
            "Merci de fournir une radiographie ET un compte rendu.",
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
        return _carte("#f43f5e", "", "Erreur lors de l'analyse", str(e))

    if proba >= SEUIL:
        return _carte(
            "#10b981",
            ICONE_OK,
            "COHÉRENT",
            f"Le compte rendu correspond à la radiographie. "
            f"Score de confiance : {proba:.3f}",
        )
    return _carte(
        "#f43f5e",
        ICONE_NON,
        "INCOHÉRENT",
        f"Le compte rendu ne correspond pas à la radiographie. "
        f"Score de confiance : {proba:.3f}",
    )


def effacer():
    return None, "", ""


CSS = """
html, body, .gradio-container, .app { background: #0b0f17 !important; }
.gradio-container {
  max-width: 1400px !important;
  width: 95% !important;
  margin: 0 auto !important;
  color: #f8fafc !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.gradio-container label, .gradio-container .prose, .gradio-container .prose *,
.gradio-container span, .gradio-container p,
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  color: #f8fafc !important;
}
#entete {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border: 1px solid #334155;
  padding: 18px 26px; border-radius: 16px; margin-bottom: 12px;
  box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.8); text-align: center;
}
#entete h1 { 
  margin: 0; font-size: 27px; color: #ffffff !important; 
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
#entete p { margin: 6px 0 0 0; color: #94a3b8 !important; opacity: 0.95; }
#carte_blanche { background: transparent !important; border: none !important; padding: 6px 0 !important; }
.col-image, .col-texte {
  background: #111827 !important; border: 1px solid #1f2937 !important;
  border-radius: 12px; padding: 12px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}
#ligne_entrees { flex-wrap: nowrap !important; gap: 14px !important; }
.gradio-container textarea, .gradio-container input[type=text] {
  background: #030712 !important; color: #f3f4f6 !important;
  border: 1px solid #374151 !important;
  border-radius: 8px !important;
}
.gradio-container textarea:focus, .gradio-container input[type=text]:focus {
  border-color: #38bdf8 !important;
  box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}
#btn_verifier button, #btn_verifier {
  background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important; 
  color: #ffffff !important;
  border: none !important; font-weight: 600 !important;
  border-radius: 8px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
}
#btn_verifier button:hover, #btn_verifier:hover { 
  background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important; 
  box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
}
#btn_effacer button, #btn_effacer {
  background: #1e293b !important; color: #cbd5e1 !important;
  border: 1px solid #334155 !important;
  border-radius: 8px !important;
  transition: all 0.2s ease !important;
}
#btn_effacer button:hover, #btn_effacer:hover {
  background: #334155 !important;
  color: #ffffff !important;
}
.gradio-container .accordion {
  background: #111827 !important;
  border: 1px solid #1f2937 !important;
  border-radius: 8px !important;
}
#pied { color: #64748b; font-size: 12px; text-align: center; margin-top: 10px; }
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
