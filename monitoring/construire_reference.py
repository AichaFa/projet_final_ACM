"""
Construit la référence sémantique dans le datalake Azure : les embeddings des
comptes rendus normaux (cohérents), qui servent de point de comparaison stable
pour détecter la dérive.

Source : chexpert_matches_sample_2.csv (colonne report), les vraies paires cohérentes.
Destination : le conteneur "datalake" sur Azure Blob, blob "reference/reference_embeddings.jsonl".
"""

import os
import pandas as pd
from embeddings import embedding_texte
from datalake_azure import ecrire_lignes

BASE = os.path.dirname(__file__)
SOURCE = os.path.join(BASE, "..", "2_preparation_donnees", "chexpert_matches_sample_2.csv")
BLOB_REFERENCE = "reference/reference_embeddings.jsonl"
N_REFERENCE = 200


def construire():
    df = pd.read_csv(SOURCE, usecols=["report"]).dropna()
    df = df[df["report"].astype(str).str.strip().str.len() >= 20].head(N_REFERENCE)

    records = [{"embedding": embedding_texte(rep)} for rep in df["report"].astype(str)]
    ecrire_lignes(BLOB_REFERENCE, records)
    print(f"{len(records)} comptes rendus de référence déposés dans le datalake Azure : {BLOB_REFERENCE}")


if __name__ == "__main__":
    construire()
