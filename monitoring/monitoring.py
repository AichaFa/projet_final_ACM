# ============================================================================
# Monitoring - lit les données de prod (Neon) et détecte la dérive (Evidently)
# ============================================================================

import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from journalisation import get_engine, DATALAKE, COLONNES

BASE = os.path.dirname(__file__)
RAPPORTS = os.path.join(BASE, "rapports")


def charger_reference():
    chemin = os.path.join(DATALAKE, "reference.csv")
    if not os.path.exists(chemin):
        raise SystemExit("Référence absente. Lance : python mock_producer.py --reference")
    return pd.read_csv(chemin)[COLONNES]


def charger_production():
    engine = get_engine()
    df = pd.read_sql("SELECT " + ", ".join(COLONNES) + " FROM predictions", engine)
    if df.empty:
        raise SystemExit("Aucune prédiction dans la base Neon. Lance : python journalisation.py [--derive]")
    return df


def analyser():
    os.makedirs(RAPPORTS, exist_ok=True)
    reference = charger_reference()
    production = charger_production()
    snapshot = Report([DataDriftPreset()]).run(current_data=production, reference_data=reference)
    snapshot.save_html(os.path.join(RAPPORTS, "rapport_derive.html"))
    d = snapshot.dict()
    resume = next(m for m in d["metrics"] if m["metric_name"].startswith("DriftedColumnsCount"))
    part = float(resume["value"]["share"])
    return {
        "reference": len(reference),
        "production": len(production),
        "colonnes_en_derive": int(resume["value"]["count"]),
        "part_colonnes_en_derive": round(part, 3),
        "derive_detectee": part >= 0.5,
    }


if __name__ == "__main__":
    print(analyser())
