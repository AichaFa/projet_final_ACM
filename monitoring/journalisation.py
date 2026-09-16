"""
Journalisation des prédictions - datalake + data warehouse (Neon / PostgreSQL).

Chaque prédiction est enregistrée :
  - dans le DATALAKE (fichiers .jsonl), avec l'embedding sémantique du compte rendu ;
  - dans le DATA WAREHOUSE Neon (table "predictions"), avec les indicateurs de surface.

L'embedding, riche et volumineux, vit dans le datalake ; l'entrepôt reste léger.
La dérive sémantique se calcule ensuite sur les embeddings du datalake.
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from embeddings import embedding_texte, DIM

BASE = os.path.dirname(__file__)
DATALAKE = os.path.join(BASE, "datalake")

load_dotenv(os.path.join(BASE, ".env"))
DATABASE_URL = os.environ.get("DATABASE_URL")

COLONNES = ["longueur_texte", "luminosite_image", "prediction", "probabilite"]

_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        if not DATABASE_URL:
            raise SystemExit("DATABASE_URL manquante : crée un fichier .env avec ta chaîne Neon.")
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        _ENGINE = create_engine(url, pool_pre_ping=True)
        with _ENGINE.begin() as con:
            con.execute(text(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "horodatage TIMESTAMP, longueur_texte INTEGER,"
                " luminosite_image DOUBLE PRECISION, prediction INTEGER,"
                " probabilite DOUBLE PRECISION)"
            ))
    return _ENGINE


def _ecrire_datalake(record):
    os.makedirs(DATALAKE, exist_ok=True)
    fichier = os.path.join(DATALAKE, "prod_" + datetime.now().strftime("%Y%m%d") + ".jsonl")
    with open(fichier, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ecrire_neon(record):
    engine = get_engine()
    surface = {c: record[c] for c in ["horodatage"] + COLONNES}
    with engine.begin() as con:
        con.execute(text(
            "INSERT INTO predictions (horodatage, longueur_texte, luminosite_image, prediction, probabilite)"
            " VALUES (:horodatage, :longueur_texte, :luminosite_image, :prediction, :probabilite)"
        ), surface)


def journaliser_prediction(compte_rendu, luminosite_image, prediction, probabilite):
    """À appeler après chaque prédiction de l'application."""
    record = {
        "horodatage": datetime.now().isoformat(timespec="seconds"),
        "longueur_texte": int(len(compte_rendu)),
        "luminosite_image": round(float(luminosite_image), 3),
        "prediction": int(prediction),
        "probabilite": round(float(probabilite), 3),
        "embedding": embedding_texte(compte_rendu),
    }
    _ecrire_datalake(record)
    _ecrire_neon(record)
    return record


def _simuler(n, avec_derive):
    import numpy as np
    rng = np.random.default_rng(2 if avec_derive else 3)
    longueur = rng.normal(110 if avec_derive else 80, 22, n).clip(10)
    luminosite = rng.normal(0.55 if avec_derive else 0.45, 0.10, n).clip(0, 1)
    proba = rng.beta(2, 3 if avec_derive else 5, n)
    # Un décalage de moyenne des embeddings représente une dérive du sens.
    centre = 0.35 if avec_derive else 0.0
    embeddings = rng.normal(centre, 1.0, (n, DIM))
    for i in range(n):
        record = {
            "horodatage": datetime.now().isoformat(timespec="seconds"),
            "longueur_texte": int(longueur[i]),
            "luminosite_image": round(float(luminosite[i]), 3),
            "prediction": int(proba[i] >= 0.5),
            "probabilite": round(float(proba[i]), 3),
            "embedding": embeddings[i].tolist(),
        }
        _ecrire_datalake(record)
        _ecrire_neon(record)
    print(f"{n} prédictions journalisées (datalake + Neon).",
          "Dérive sémantique simulée." if avec_derive else "")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--derive", action="store_true")
    p.add_argument("--n", type=int, default=400)
    a = p.parse_args()
    _simuler(a.n, a.derive)
