"""
Monitoring de la dérive sémantique - compare les embeddings des comptes rendus de
production (datalake Azure) à ceux de la référence, et signale la dérive du sens.

La dérive est mesurée sur les 128 dimensions de l'embedding BioViL-T : au-delà d'une
part de 0,55 de dimensions en dérive, une dérive sémantique est déclarée.

En cas de dérive, un signal repository_dispatch est envoyé à GitHub pour déclencher
le workflow de réentraînement (même mécanisme que l'architecture de référence).

Léger : lit des embeddings déjà calculés, sans recharger BioViL-T.
"""

import os
import pandas as pd
import requests
from evidently import Report
from evidently.presets import DataDriftPreset
from datalake_azure import lire_lignes

BASE = os.path.dirname(__file__)
RAPPORTS = os.path.join(BASE, "rapports")
DIM = 128
SEUIL_DERIVE = 0.55
COLS = [f"emb_{i:03d}" for i in range(DIM)]

# Dépôt à prévenir en cas de dérive, et nom de l'événement déclencheur.
DEPOT = "AichaFa/projet_final_ACM"
EVENEMENT = "derive-detectee"


def _en_dataframe(records):
    return pd.DataFrame([r["embedding"] for r in records if "embedding" in r], columns=COLS)


def charger_reference():
    records = lire_lignes("reference/")
    if not records:
        raise SystemExit("Référence absente dans le datalake Azure. Lance : python construire_reference.py")
    return _en_dataframe(records)


def charger_production():
    records = lire_lignes("production/")
    if not records:
        raise SystemExit("Aucune prédiction de production dans le datalake Azure. Lance des prédictions sur l'application.")
    return _en_dataframe(records)


def declencher_reentrainement():
    """Envoie un signal repository_dispatch à GitHub pour lancer le réentraînement.

    Le token est fourni automatiquement par GitHub Actions (variable GITHUB_TOKEN).
    L'événement 'derive-detectee' déclenchera le workflow d'entraînement dédié.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN absent : signal non envoyé (exécution hors GitHub Actions ?).")
        return
    reponse = requests.post(
        f"https://api.github.com/repos/{DEPOT}/dispatches",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"event_type": EVENEMENT},
    )
    if reponse.status_code == 204:
        print("Signal de réentraînement envoyé à GitHub.")
    else:
        print(f"Échec de l'envoi du signal ({reponse.status_code}) : {reponse.text}")


def analyser():
    os.makedirs(RAPPORTS, exist_ok=True)
    reference = charger_reference()
    production = charger_production()

    snapshot = Report([DataDriftPreset()]).run(current_data=production, reference_data=reference)
    snapshot.save_html(os.path.join(RAPPORTS, "rapport_derive_semantique.html"))

    d = snapshot.dict()
    resume = next(m for m in d["metrics"] if m["metric_name"].startswith("DriftedColumnsCount"))
    part = float(resume["value"]["share"])
    derive_detectee = part >= SEUIL_DERIVE

    # En cas de dérive, on déclenche le réentraînement via GitHub.
    if derive_detectee:
        declencher_reentrainement()

    return {
        "reference": len(reference),
        "production": len(production),
        "dimensions_en_derive": int(resume["value"]["count"]),
        "part_dimensions_en_derive": round(part, 3),
        "derive_semantique_detectee": derive_detectee,
    }


if __name__ == "__main__":
    print(analyser())
