# ============================================================================
# Datalake sur Hugging Face - envoie le dossier datalake/ vers un Dataset HF
# ----------------------------------------------------------------------------
# Lit le jeton et le nom du Dataset depuis .env (HF_TOKEN, HF_DATASET), puis
# téléverse tout le contenu du dossier datalake/ vers ton Dataset Hugging Face.
#
# Usage : py datalake_hf.py
# ============================================================================

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

BASE = os.path.dirname(__file__)
DATALAKE = os.path.join(BASE, "datalake")

load_dotenv(os.path.join(BASE, ".env"))
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = os.environ.get("HF_DATASET")  # ex : AichaFaHugFace/auditeur-datalake


def pousser_datalake():
    if not HF_TOKEN or not HF_DATASET:
        raise SystemExit("HF_TOKEN ou HF_DATASET manquant dans le fichier .env.")
    if not os.path.isdir(DATALAKE):
        raise SystemExit("Dossier datalake/ introuvable. Lance d'abord journalisation.py.")
    api = HfApi()
    api.upload_folder(
        folder_path=DATALAKE,
        repo_id=HF_DATASET,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Mise a jour du datalake",
    )
    print(f"Datalake envoyé -> https://huggingface.co/datasets/{HF_DATASET}")


if __name__ == "__main__":
    pousser_datalake()
