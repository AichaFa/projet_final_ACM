"""
Calcul de l'embedding sémantique d'un compte rendu avec BioViL-T (128 dimensions).

Un embedding est un vecteur de 128 nombres qui capture le sens d'un compte rendu.
C'est sur ces vecteurs, et non sur des indicateurs de surface, que repose la
détection de dérive sémantique.
"""

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "microsoft/BiomedVLP-BioViL-T"
DIM = 128

_ressources = {}


def _charger():
    if not _ressources:
        _ressources["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        _ressources["text_model"] = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
    return _ressources


def embedding_texte(compte_rendu):
    """Renvoie l'embedding sémantique du compte rendu : une liste de 128 nombres."""
    r = _charger()
    inputs = r["tokenizer"](
        compte_rendu,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        emb = r["text_model"].get_projected_text_embeddings(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    return emb.squeeze(0).cpu().tolist()
