# ============================================================================
# Auditeur de Cohérence Médicale - APP (interface) reliée à l'API
# ----------------------------------------------------------------------------
# L'App n'exécute PAS le modèle : elle appelle le Space API (Auditeur-API),
# récupère "VERDICT|score" et l'affiche dans une belle carte.
# ============================================================================

from gradio_client import Client, handle_file
import gradio as gr
import spaces


@spaces.GPU
def _reserve_gpu():
    # Fonction GPU factice : requise pour démarrer un Space ZeroGPU.
    # L'App n'utilise pas réellement le GPU (elle appelle l'API).
    return True


API_SPACE = "AichaFaHugFace/Auditeur-API"
SEUIL = 0.5

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Client(API_SPACE)
    return _client


ICONE_OK = (
    "<svg width='26' height='26' viewBox='0 0 24 24' fill='none'>"
    "<circle cx='12' cy='12' r='11' fill='#2ecc71'/>"
    "<path d='M7 12.5l3 3 7-7' stroke='white' stroke-width='2.2' "
    "fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>"
)
ICONE_NON = (
    "<svg width='26' height='26' viewBox='0 0 24 24' fill='none'>"
    "<circle cx='12' cy='12' r='11' fill='#e74c3c'/>"
    "<path d='M8 8l8 8M16 8l-8 8' stroke='white' stroke-width='2.2' "
    "stroke-linecap='round'/></svg>"
)


def _carte(accent, icone, titre, detail):
    return (
        f"<div style='background:#16233c; border-left:6px solid {accent}; "
        f"border-radius:12px; padding:18px 20px; display:flex; align-items:center; "
        f"gap:14px; margin-top:10px;'>{icone}"
        f"<div><div style='font-size:20px; font-weight:700; color:{accent};'>{titre}</div>"
        f"<div style='color:#ffffff; margin-top:2px;'>{detail}</div></div></div>"
    )


def verifier(image_path, texte):
    if image_path is None or not texte or not texte.strip():
        return _carte(
            "#f1c40f",
            "",
            "Informations manquantes",
            "Merci de fournir une radiographie ET un compte rendu.",
        )
    # Appel de l'API (le moteur)
    try:
        reponse = get_client().predict(
            handle_file(image_path), texte, api_name="/predict"
        )
    except Exception:
        return _carte(
            "#e74c3c",
            "",
            "Erreur de l'API",
            "L'API n'a pas répondu (Space en veille ?). Réessaie dans un instant.",
        )
    # Lecture de "VERDICT|score"
    try:
        verdict, score = str(reponse).split("|")
        proba = float(score)
    except Exception:
        return _carte("#e74c3c", "", "Réponse inattendue", str(reponse))

    if verdict == "ERREUR":
        return _carte(
            "#e74c3c",
            "",
            "Erreur lors de l'analyse",
            "Vérifie que l'image et le texte sont valides.",
        )

    pourcentage = f"{proba * 100:.1f} %"
    if verdict == "COHERENT":
        return _carte(
            "#2ecc71",
            ICONE_OK,
            "COHÉRENT",
            f"Le compte rendu correspond à la radiographie. "
            f"Score de confiance : {pourcentage}",
        )
    return _carte(
        "#e74c3c",
        ICONE_NON,
        "INCOHÉRENT",
        f"Le compte rendu ne correspond pas à la radiographie. "
        f"Score de confiance : {pourcentage}",
    )


def effacer():
    return None, "", ""


CSS = """
html, body, gradio-app, .gradio-container, .app, .main, .wrap, .contain, .fillable {
  background: linear-gradient(160deg, #0a1424 0%, #0e2036 100%) !important;
  background-attachment: fixed !important;
}
.gradio-container {
  max-width: 1400px !important; width: 95% !important; margin: 0 auto !important;
  color: #ffffff !important;
}
.gradio-container .prose, .gradio-container .prose *,
.gradio-container h1, .gradio-container h2, .gradio-container h3 { color: #ffffff !important; }
#entete {
  background: linear-gradient(135deg, #16306b, #3a5bd0);
  padding: 22px 26px; border-radius: 18px; margin-bottom: 14px;
  box-shadow: 0 8px 22px rgba(0,0,0,0.5); text-align: center;
}
#entete h1 { margin: 0; font-size: 28px; color: #ffffff !important; }
#entete p { margin: 6px 0 0 0; color: #ffffff !important; opacity: 0.95; }
#carte_blanche {
  background: #ffffff !important; border-radius: 18px !important;
  padding: 18px !important; box-shadow: 0 8px 22px rgba(0,0,0,0.35);
}
#carte_blanche .gr-group, #carte_blanche .form, #carte_blanche .wrap,
#carte_blanche .styler, #ligne_entrees, .col-image, .col-texte {
  background: transparent !important; border: none !important;
}
#ligne_entrees { flex-wrap: nowrap !important; gap: 18px !important; }
#carte_blanche .col-image .block, #carte_blanche .col-texte .block,
#carte_blanche textarea {
  background: #16233c !important; border: 1px solid #26375c !important;
  border-radius: 12px !important; color: #ffffff !important;
}
#carte_blanche textarea::placeholder { color: #9fb0cc !important; }
#carte_blanche .col-texte, #carte_blanche .col-texte .block,
#carte_blanche .col-texte label, #carte_blanche .col-texte textarea {
  width: 100% !important; max-width: 100% !important;
}
#carte_blanche .col-image label, #carte_blanche .col-image .block-label,
#carte_blanche .col-texte label > span:first-child,
#carte_blanche .col-texte .block-info, #carte_blanche .col-texte .block-title {
  background: #2f4bf0 !important; color: #ffffff !important; border-radius: 8px !important;
  padding: 4px 12px !important; font-weight: 600 !important;
  display: inline-block !important; width: auto !important;
  max-width: max-content !important; box-shadow: none !important; margin: 0 0 6px 0 !important;
}
#carte_blanche .col-texte label { display: block !important; width: 100% !important; max-width: 100% !important; }
#btn_verifier button, #btn_verifier {
  background: #2f4bf0 !important; color: #ffffff !important; border: none !important; font-weight: 700 !important;
}
#btn_verifier button:hover { background: #4661f5 !important; }
#btn_effacer button, #btn_effacer {
  background: #3b4252 !important; color: #ffffff !important; border: none !important;
}
#pied { color: #9fb0cc; font-size: 12px; text-align: center; margin-top: 10px; }
"""

with gr.Blocks(css=CSS, title="Auditeur de Cohérence Médicale") as demo:
    gr.HTML(
        "<div id='entete'><h1>Auditeur de Cohérence Médicale</h1>"
        "<p>Vérifiez si un compte rendu clinique correspond bien à sa "
        "radiographie thoracique</p></div>"
    )
    with gr.Group(elem_id="carte_blanche"):
        with gr.Row(elem_id="ligne_entrees", equal_height=True):
            with gr.Column(elem_classes="col-image", min_width=280):
                image_in = gr.Image(
                    label="Radiographie thoracique", type="filepath", height=300
                )
            with gr.Column(elem_classes="col-texte", min_width=280):
                texte_in = gr.Textbox(
                    label="Compte rendu radiologique",
                    lines=11,
                    placeholder="ex. : Radiographie thoracique de face montrant un "
                    "épanchement pleural bilatéral modéré...",
                )
        with gr.Row():
            bouton = gr.Button(
                "Lancer la vérification", elem_id="btn_verifier", scale=3
            )
            bouton_effacer = gr.Button("Effacer", elem_id="btn_effacer", scale=1)
        resultat = gr.HTML()

    with gr.Accordion("Comment ça marche ?", open=False):
        gr.Markdown(
            "Déposez une radiographie thoracique, saisissez le compte rendu clinique "
            "associé, puis lancez l'analyse. L'application envoie l'image et le texte "
            "à l'API de prédiction, qui renvoie un verdict et un score de confiance."
        )

    gr.HTML(
        "<div id='pied'>Démonstration à visée pédagogique - ne remplace pas "
        "l'avis d'un professionnel de santé.</div>"
    )

    bouton.click(verifier, inputs=[image_in, texte_in], outputs=resultat)
    bouton_effacer.click(effacer, inputs=None, outputs=[image_in, texte_in, resultat])


if __name__ == "__main__":
    demo.launch()
