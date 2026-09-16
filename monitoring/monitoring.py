"""
Monitoring de la dérive sémantique - compare les embeddings des comptes rendus de
production à ceux de la référence (Evidently), et signale la dérive du sens.

La dérive est mesurée sur les 128 dimensions de l'embedding BioViL-T : au-delà d'une
part de 0,55 de dimensions en dérive, une dérive sémantique est déclarée.
"""

import os
import glob
import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

BASE = os.path.dirname(__file__)
DATALAKE = os.path.join(BASE, "datalake")
RAPPORTS = os.path.join(BASE, "rapports")
DIM = 128
SEUIL_DERIVE = 0.55
COLS = [f"emb_{i:03d}" for i in range(DIM)]


def _lire_embeddings(chemins):
    lignes = []
    for chemin in chemins:
        with open(chemin, encoding="utf-8") as f:
            for ligne in f:
                if ligne.strip():
                    enr = json.loads(ligne)
                    if "embedding" in enr:
                        lignes.append(enr["embedding"])
    return pd.DataFrame(lignes, columns=COLS)


def charger_reference():
    chemin = os.path.join(DATALAKE, "reference_embeddings.jsonl")
    if not os.path.exists(chemin):
        raise SystemExit("Référence absente. Lance : python construire_reference.py")
    return _lire_embeddings([chemin])


def charger_production():
    chemins = sorted(glob.glob(os.path.join(DATALAKE, "prod_*.jsonl")))
    df = _lire_embeddings(chemins) if chemins else pd.DataFrame(columns=COLS)
    if df.empty:
        raise SystemExit("Aucun embedding de production. Lance l'app, ou : python journalisation.py --derive")
    return df


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
