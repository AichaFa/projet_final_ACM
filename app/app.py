# ============================================================================
# Auditeur de Cohérence Médicale - Application AUTONOME (Gradio)
# Thème : fond noir, éléments bleus, textes blancs.
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
    "<circle cx='12' cy='12' r='11' fill='#3ddc84'/>"
    "<path d='M7 12.5l3 3 7-7' stroke='#0e2447' stroke-width='2.2' "
    "fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>"
)
ICONE_NON = (
    "<svg width='26' height='26' viewBox='0 0 24 24' fill='none'>"
    "<circle cx='12' cy='12' r='11' fill='#ff6b6b'/>"
    "<path d='M8 8l8 8M16 8l-8 8' stroke='#0e2447' stroke-width='2.2' "
    "stroke-linecap='round'/></svg>"
)


def _carte(accent, icone, titre, detail):
    return (
        f"<div style='background:#0e2447; border-left:6px solid {accent}; "
        f"border-radius:12px; padding:18px 20px; display:flex; align-items:center; "
        f"gap:14px; margin-top:8px;'>{icone}"
        f"<div><div style='font-size:20px; font-weight:700; color:{accent};'>{titre}</div>"
        f"<div style='color:#ffffff; margin-top:2px;'>{detail}</div></div></div>"
    )


@spaces.GPU
def verifier(image_path, texte):
    if image_path is None or not texte or not texte.strip():
        return _carte(
            "#f5c542",
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
        return _carte("#ff6b6b", "", "Erreur lors de l'analyse", str(e))

    if proba >= SEUIL:
        return _carte(
            "#3ddc84",
            ICONE_OK,
            "COHÉRENT",
            f"Le compte rendu correspond à la radiographie. "
            f"Score de confiance : {proba:.3f}",
        )
    return _carte(
        "#ff6b6b",
        ICONE_NON,
        "INCOHÉRENT",
        f"Le compte rendu ne correspond pas à la radiographie. "
        f"Score de confiance : {proba:.3f}",
    )


def effacer():
    return None, "", ""


CSS = """
html, body, gradio-app, .gradio-container, .app, .main, .wrap, .contain, .fillable {
  background: #000000 !important; background-image: none !important;
}
.gradio-container {
  max-width: 1400px !important;
  width: 95% !important;
  margin: 0 auto !important;
  color: #ffffff !important;
}
.gradio-container label, .gradio-container .prose, .gradio-container .prose *,
.gradio-container span, .gradio-container p,
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  color: #ffffff !important;
}
#entete {
  background: linear-gradient(135deg, #16306b, #2e6bb0);
  padding: 18px 26px; border-radius: 16px; margin-bottom: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.6); text-align: center;
}
#entete h1 { margin: 0; font-size: 27px; color: #ffffff !important; }
#entete p { margin: 6px 0 0 0; color: #ffffff !important; opacity: 0.95; }
#carte_blanche { background: transparent !important; border: none !important; padding: 6px 0 !important; }
.col-image, .col-texte {
  background: #0e2447 !important; border: 1px solid #1e3f6f !important;
  border-radius: 12px; padding: 12px;
}
#ligne_entrees { flex-wrap: nowrap !important; gap: 14px !important; }
.gradio-container textarea, .gradio-container input[type=text] {
  background: #0b1e3f !important; color: #ffffff !important;
  border: 1px solid #24457a !important;
}
#btn_verifier button, #btn_verifier {
  background: #2e8bd8 !important; color: #ffffff !important;
  border: none !important; font-weight: 600 !important;
}
#btn_verifier button:hover { background: #4a9fe0 !important; }
#btn_effacer button, #btn_effacer {
  background: #16306b !important; color: #ffffff !important;
  border: 1px solid #2e6bb0 !important;
}
#pied { color: #9fb3d6; font-size: 12px; text-align: center; margin-top: 10px; }
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
