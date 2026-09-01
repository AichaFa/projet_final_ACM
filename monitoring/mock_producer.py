# ============================================================================
# Mock API / producteur de données - alimente le "datalake"
# ----------------------------------------------------------------------------
# Simule un flux de production : génère des enregistrements (comme si l'app avait
# fait des prédictions) et les dépose dans un dossier "datalake" (fichiers CSV).
# Ces données seront ensuite analysées par Evidently (voir monitoring.py).
#
# Usage :
#   python mock_producer.py --reference        # crée le jeu de RÉFÉRENCE (baseline)
#   python mock_producer.py                     # ajoute un lot de production "normal"
#   python mock_producer.py --derive            # ajoute un lot AVEC dérive (pour tester)
# ============================================================================

import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

DATALAKE = os.path.join(os.path.dirname(__file__), "datalake")

# Colonnes journalisées pour chaque prédiction (aucune donnée patient sensible) :
#  - longueur_texte   : nombre de caractères du compte rendu
#  - luminosite_image : luminosité moyenne de la radiographie (indicateur d'entrée)
#  - prediction       : 1 = cohérent, 0 = incohérent
#  - probabilite      : score du modèle


def generer(n, avec_derive=False, graine=0):
    rng = np.random.default_rng(graine)
    longueur = rng.normal(110 if avec_derive else 80, 22, n).clip(10).round().astype(int)
    luminosite = rng.normal(0.55 if avec_derive else 0.45, 0.10, n).clip(0, 1).round(3)
    proba = rng.beta(2, 3 if avec_derive else 5, n).round(3)
    prediction = (proba >= 0.5).astype(int)
    base = datetime.now()
    horodatage = [(base - timedelta(minutes=int(i))).isoformat(timespec="seconds") for i in range(n)]
    return pd.DataFrame({
        "horodatage": horodatage,
        "longueur_texte": longueur,
        "luminosite_image": luminosite,
        "prediction": prediction,
        "probabilite": proba,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", action="store_true", help="génère le jeu de référence (baseline)")
    p.add_argument("--derive", action="store_true", help="introduit une dérive (pour tester le monitoring)")
    p.add_argument("--n", type=int, default=400, help="nombre d'enregistrements")
    args = p.parse_args()

    os.makedirs(DATALAKE, exist_ok=True)

    if args.reference:
        df = generer(args.n, avec_derive=False, graine=1)
        chemin = os.path.join(DATALAKE, "reference.csv")
    else:
        df = generer(args.n, avec_derive=args.derive, graine=2 if args.derive else 3)
        nom = "production_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        chemin = os.path.join(DATALAKE, nom)

    df.to_csv(chemin, index=False)
    print("Écrit :", chemin, "(", len(df), "lignes)")


if __name__ == "__main__":
    main()
