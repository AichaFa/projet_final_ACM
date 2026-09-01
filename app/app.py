# ============================================================================
# Auditeur de Cohérence Médicale - interface Gradio (version enrichie)
# ----------------------------------------------------------------------------
# Améliorations : bandeau d'en-tête bleu marine, fond dégradé, mise en page plus
# large, bouton "Effacer", carte de résultat avec icône (coche / croix), pied de
# page. Le fonctionnement (appel API + @spaces.GPU) reste identique.
# ============================================================================

import io
import requests
import gradio as gr
import spaces
from PIL import Image

# Adresse de l'API de prédiction (à remplacer par la tienne le jour venu).
API_URL = "https://sammec-demoday-fastapi.hf.space/predict"

# --- Styles (CSS) : largeur, fond dégradé, coins arrondis ---
CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important;
  background: linear-gradient(160deg, #EAF1FB 0%, #F7FAFD 55%, #E8F0FA 100%) !important;}
#zone-saisie {background:#ffffff; border-radius:16px; padding:16px;
  box-shadow: 0 2px 12px rgba(22,64,107,0.08);}
"""

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

# --- Bandeau d'en-tête (bleu marine, dégradé) ---
HEADER_HTML = """
<div style="background: linear-gradient(135deg, #16406B 0%, #2E6BB0 100%);
            border-radius: 16px; padding: 30px 24px; text-align: center;
            box-shadow: 0 4px 16px rgba(22,64,107,0.28); margin: 6px 0 14px 0;">
  <div style="font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: 0.3px;">
    Auditeur de Cohérence Médicale
  </div>
  <div style="font-size: 1.06rem; color: #DCE8F6; margin-top: 8px;">
    Vérifiez si un compte rendu clinique correspond bien à sa radiographie thoracique
  </div>
</div>
"""

# --- Pied de page ---
FOOTER_HTML = """
<div style="text-align:center; color:#6B7B8C; font-size:0.9rem;
            margin-top:18px; padding:12px; border-top:1px solid #D6E3F2;">
  Démonstration à visée pédagogique - ne remplace pas l'avis d'un professionnel de santé.
</div>
"""

# --- Icônes (SVG, sans emoji) ---
ICONE_OK = (
    "<svg width='54' height='54' viewBox='0 0 52 52'><circle cx='26' cy='26' r='24' fill='#34A853'/>"
    "<path d='M16 27 l7 7 l13 -15' fill='none' stroke='#fff' stroke-width='4' "
    "stroke-linecap='round' stroke-linejoin='round'/></svg>"
)
ICONE_KO = (
    "<svg width='54' height='54' viewBox='0 0 52 52'><circle cx='26' cy='26' r='24' fill='#D93025'/>"
    "<path d='M18 18 L34 34 M34 18 L18 34' stroke='#fff' stroke-width='4' stroke-linecap='round'/></svg>"
)


def carte(fond, bord, texte_couleur, icone, titre, message, score=None):
    ligne_score = ""
    if score is not None:
        ligne_score = (
            f"<div style='margin-top:10px; font-weight:600; color:{texte_couleur};'>"
            f"Score de confiance : {score * 100:.1f}%</div>"
        )
    return (
        f"<div style='padding:24px; border-radius:16px; background:{fond}; border:1px solid {bord};"
        f" text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.05);'>"
        f"<div style='margin-bottom:8px;'>{icone}</div>"
        f"<div style='font-size:1.5rem; font-weight:700; color:{texte_couleur};'>{titre}</div>"
        f"<div style='color:#333; margin-top:6px;'>{message}</div>"
        f"{ligne_score}</div>"
    )


@spaces.GPU  # étiquette obligatoire pour ZeroGPU (Spaces gratuits)
def verifier_coherence(image, texte):
    if image is None or not texte or not texte.strip():
        return carte(
            "#FFF7E6",
            "#F0C36D",
            "#8A6D00",
            "",
            "Information manquante",
            "Merci de fournir une radiographie ET un compte rendu avant de lancer la vérification.",
        )

    tampon = io.BytesIO()
    image.save(tampon, format="PNG")
    tampon.seek(0)
    fichiers = {"image_file": ("image.png", tampon, "image/png")}
    donnees = {"text_input": texte}

    try:
        reponse = requests.post(API_URL, data=donnees, files=fichiers, timeout=60)
    except Exception as e:
        return carte(
            "#FDECEA",
            "#D93025",
            "#C5221F",
            "",
            "Serveur injoignable",
            f"Impossible de contacter l'API. Détails : {e}",
        )

    if reponse.status_code != 200:
        return carte(
            "#FDECEA",
            "#D93025",
            "#C5221F",
            "",
            "Erreur de l'API",
            f"Code {reponse.status_code}. Réessaie dans un instant.",
        )

    resultat = reponse.json()
    prediction = resultat.get("prediction")
    probabilite = resultat.get("probability", 0.0)
    score = probabilite if prediction == 1 else (1 - probabilite)

    if prediction == 1:
        return carte(
            "#E8F6EC",
            "#34A853",
            "#1E7E34",
            ICONE_OK,
            "COHÉRENT",
            "Le compte rendu correspond bien à la radiographie.",
            score,
        )
    return carte(
        "#FDECEA",
        "#D93025",
        "#C5221F",
        ICONE_KO,
        "INCOHÉRENT",
        "Le compte rendu ne correspond pas à la radiographie.",
        score,
    )


def effacer():
    return None, "", ""


with gr.Blocks(theme=THEME, css=CSS, title="Auditeur de Cohérence Médicale") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(elem_id="zone-saisie"):
        with gr.Column(scale=1):
            image_entree = gr.Image(
                type="pil", label="Radiographie thoracique", height=380
            )
        with gr.Column(scale=1):
            texte_entree = gr.Textbox(
                lines=15,
                label="Compte rendu radiologique",
                placeholder="ex. : Radiographie thoracique de face montrant un épanchement pleural bilatéral modéré...",
            )

    with gr.Row():
        bouton = gr.Button(
            "Lancer la vérification", variant="primary", size="lg", scale=3
        )
        bouton_effacer = gr.Button("Effacer", variant="secondary", size="lg", scale=1)

    sortie = gr.HTML()

    with gr.Accordion("Comment ça marche ?", open=False):
        gr.Markdown(
            "Déposez une radiographie thoracique, saisissez le compte rendu clinique associé, "
            "puis lancez l'analyse. L'outil compare l'image et le texte, et indique s'ils sont "
            "cohérents, accompagné d'un score de confiance."
        )

    gr.HTML(FOOTER_HTML)

    bouton.click(
        fn=verifier_coherence, inputs=[image_entree, texte_entree], outputs=sortie
    )
    bouton_effacer.click(
        fn=effacer, inputs=[], outputs=[image_entree, texte_entree, sortie]
    )


if __name__ == "__main__":
    demo.launch()
