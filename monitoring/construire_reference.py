"""
Construit la référence sémantique : les embeddings des comptes rendus normaux
(cohérents), auxquels la production sera comparée pour détecter la dérive.

Source : chexpert_matches_sample_2.csv (colonne report), les vraies paires cohérentes.
"""

import os
import json
import pandas as pd
from embeddings import embedding_texte

BASE = os.path.dirname(__file__)
DATALAKE = os.path.join(BASE, "datalake")
SOURCE = os.path.join(BASE, "..", "2_preparation_donnees", "chexpert_matches_sample_2.csv")
N_REFERENCE = 200


def construire():
    os.makedirs(DATALAKE, exist_ok=True)
    df = pd.read_csv(SOURCE, usecols=["report"]).dropna()
    df = df[df["report"].astype(str).str.strip().str.len() >= 20].head(N_REFERENCE)

    chemin = os.path.join(DATALAKE, "reference_embeddings.jsonl")
    with open(chemin, "w", encoding="utf-8") as f:
        for rep in df["report"].astype(str):
            f.write(json.dumps({"embedding": embedding_texte(rep)}) + "\n")
    print(f"{len(df)} comptes rendus de référence transformés en embeddings -> {chemin}")


if __name__ == "__main__":
    construire()
