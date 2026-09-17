"""
Monitoring de la dérive sémantique - compare les embeddings des comptes rendus de
production (datalake Azure) à ceux de la référence, et signale la dérive du sens.

La dérive est mesurée sur les 128 dimensions de l'embedding BioViL-T : au-delà d'une
part de 0,55 de dimensions en dérive, une dérive sémantique est déclarée.

Léger : lit des embeddings déjà calculés, sans recharger BioViL-T.
"""

import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from datalake_azure import lire_lignes

BASE = os.path.dirname(__file__)
RAPPORTS = os.path.join(BASE, "rapports")
DIM = 128
SEUIL_DERIVE = 0.55
COLS = [f"emb_{i:03d}" for i in range(DIM)]


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


def analyser():
    os.makedirs(RAPPORTS, exist_ok=True)
    reference = charger_reference()
    production = charger_production()

    snapshot = Report([DataDriftPreset()]).run(current_data=production, reference_data=reference)
    snapshot.save_html(os.path.join(RAPPORTS, "rapport_derive_semantique.html"))

    d = snapshot.dict()
    resume = next(m for m in d["metrics"] if m["metric_name"].startswith("DriftedColumnsCount"))
    part = float(resume["value"]["share"])
    return {
        "reference": len(reference),
        "production": len(production),
        "dimensions_en_derive": int(resume["value"]["count"]),
        "part_dimensions_en_derive": round(part, 3),
        "derive_semantique_detectee": part >= SEUIL_DERIVE,
    }


if __name__ == "__main__":
    print(analyser())
