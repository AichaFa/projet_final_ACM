# ============================================================================
# Auditeur de Cohérence Médicale - APP autonome (inférence locale) + journalisation
# ============================================================================

import gradio as gr

# Décorateur GPU du Space (ZeroGPU). Hors Hugging Face, il devient neutre,
# ce qui permet de lancer l'application en local sans le paquet "spaces".
try:
    import spaces
    gpu = spaces.GPU
except Exception:
    def gpu(fonction):
        return fonction

from inference import charger_modeles, predire

# Préchargement de BioViL-T et du classifieur au démarrage du Space
try:
    charger_modeles()
except Exception:
    pass


@gpu
def _inferer(image_path, texte):
    # Inférence exécutée sur le GPU lorsqu'il est disponible
    return predire(image_path, texte)


# --- Journalisation vers Neon (données de production, sans texte patient) ---
import os
from datetime import datetime
from sqlalchemy import create_engine, text as sql_text

DATABASE_URL = os.environ.get("DATABASE_URL")
_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None and DATABASE_URL:
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        _ENGINE = create_engine(url, pool_pre_ping=True)
        with _ENGINE.begin() as con:
            con.execute(
                sql_text(
                    "CREATE TABLE IF NOT EXISTS predictions ("
                    "horodatage TIMESTAMP, longueur_texte INTEGER,"
                    " luminosite_image DOUBLE PRECISION, prediction INTEGER,"
                    " probabilite DOUBLE PRECISION)"
                )
            )
    return _ENGINE


def journaliser(longueur_texte, luminosite, prediction, probabilite):
    eng = _engine()
    if eng is None:
        return
    with eng.begin() as con:
        con.execute(
            sql_text(
                "INSERT INTO predictions (horodatage, longueur_texte,"
                " luminosite_image, prediction, probabilite)"
                " VALUES (:h, :l, :lu, :p, :pr)"
            ),
            {
                "h": datetime.now().isoformat(timespec="seconds"),
                "l": int(longueur_texte),
                "lu": round(float(luminosite), 3),
                "p": int(prediction),
                "pr": round(float(probabilite), 4),
            },
        )


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

    # Inférence locale (modèle embarqué) au lieu d'un appel à une API externe
    try:
        res = _inferer(image_path, texte)
        prediction = int(res["prediction"])
        probability = float(res["probabilite"])
    except Exception:
        return _carte(
            "#e74c3c",
            "",
            "Erreur lors de l'analyse",
            "L'analyse n'a pas abouti. Réessaie dans quelques instants.",
        )

    # Journaliser la prédiction dans Neon (indicateurs seulement, jamais le texte)
    try:
        from PIL import Image

        vignette = Image.open(image_path).convert("L").resize((64, 64))
        pixels = list(vignette.getdata())
        luminosite = (sum(pixels) / len(pixels)) / 255.0
        journaliser(len(texte), luminosite, prediction, probability)
    except Exception:
        pass

    # Interprétation :
    # prediction == 1 -> COHÉRENT (score = probability)
    # prediction == 0 -> INCOHÉRENT (score = 1 - probability)
    if prediction == 1:
        pourcentage = f"{probability * 100:.1f} %"
        return _carte(
            "#2ecc71",
            ICONE_OK,
            "COHÉRENT",
            f"Le compte rendu correspond à la radiographie. "
            f"Score de confiance : {pourcentage}",
        )
    pourcentage = f"{(1 - probability) * 100:.1f} %"
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
#carte_blanche .col-image .block-label,
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
            "associé, puis lancez l'analyse. L'application analyse directement l'image "
            "et le texte avec le modèle embarqué, et renvoie un verdict accompagné d'un "
            "score de confiance."
        )

    gr.HTML(
        "<div id='pied'>Démonstration à visée pédagogique - ne remplace pas "
        "l'avis d'un professionnel de santé.</div>"
    )

    bouton.click(verifier, inputs=[image_in, texte_in], outputs=resultat)
    bouton_effacer.click(effacer, inputs=None, outputs=[image_in, texte_in, resultat])


if __name__ == "__main__":
    demo.launch()
