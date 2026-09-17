"""
Accès au datalake sur Azure Blob Storage.

Range et relit les prédictions de production et la référence sémantique, sous
forme de lignes JSON (jsonl), dans le conteneur "datalake".

La chaîne de connexion est lue depuis le fichier .env (AZURE_STORAGE_CONNECTION_STRING),
jamais écrite en clair dans le code.
"""

import os
import json
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

BASE = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE, ".env"))

CONNEXION = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTENEUR = "datalake"

_client = None


def _conteneur():
    global _client
    if _client is None:
        if not CONNEXION:
            raise SystemExit("AZURE_STORAGE_CONNECTION_STRING manquante dans le fichier .env.")
        service = BlobServiceClient.from_connection_string(CONNEXION)
        _client = service.get_container_client(CONTENEUR)
    return _client


def ajouter_ligne(nom_blob, record):
    """Ajoute une ligne JSON à la fin d'un blob (le crée s'il n'existe pas)."""
    blob = _conteneur().get_blob_client(nom_blob)
    if not blob.exists():
        blob.create_append_blob()
    blob.append_block((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


def ecrire_lignes(nom_blob, records):
    """Écrit (ou remplace) un blob avec une liste de records, un par ligne."""
    contenu = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    _conteneur().upload_blob(nom_blob, contenu.encode("utf-8"), overwrite=True)


def lire_lignes(prefixe):
    """Lit tous les blobs dont le nom commence par le préfixe, et renvoie leurs lignes."""
    lignes = []
    for blob in _conteneur().list_blobs(name_starts_with=prefixe):
        contenu = _conteneur().download_blob(blob.name).readall().decode("utf-8")
        for ligne in contenu.splitlines():
            if ligne.strip():
                lignes.append(json.loads(ligne))
    return lignes


def tester_connexion():
    """Vérifie que la connexion au conteneur fonctionne."""
    noms = [b.name for b in _conteneur().list_blobs()]
    print(f"Connexion au conteneur '{CONTENEUR}' réussie. {len(noms)} blob(s) présent(s).")
    return noms


if __name__ == "__main__":
    tester_connexion()
