# ============================================================================
# Test de /predict avec une image + un texte (modèle simulé)
# ----------------------------------------------------------------------------
# Ce test vérifie que la route /predict :
#   - accepte bien une image ET un texte,
#   - passe par la chaîne de prédiction,
#   - renvoie la bonne structure de réponse (status / prediction / probability).
#
# Pour rester LÉGER dans la CI, on ne charge pas le vrai modèle (BioViL-T, MLflow,
# S3) : on le "simule" (monkeypatch). On teste donc la plomberie de l'endpoint,
# pas la justesse du modèle lui-même (qui, elle, est vérifiée manuellement via l'app).
#
# Ce test tourne dans le conteneur Docker de l'API (où torch est déjà installé).
# ============================================================================

import io
import torch
from PIL import Image
from fastapi.testclient import TestClient
import radio_check_app as rca


def test_predict_avec_image_et_texte(monkeypatch):
    # 1) Simuler des modèles "chargés" pour passer le garde-fou 503,
    #    sans charger le vrai modèle lourd.
    monkeypatch.setattr(rca, "text_model", object())
    monkeypatch.setattr(rca, "image_model", object())
    monkeypatch.setattr(
        rca, "cross_att_classifier", lambda img, txt: torch.tensor([[1.5]])
    )
    monkeypatch.setattr(rca, "get_text_embeddings", lambda t: torch.zeros(1, 300, 8))
    monkeypatch.setattr(
        rca, "get_image_embeddings_from_pil", lambda img: torch.zeros(1, 8)
    )

    client = TestClient(rca.app)

    # 2) Construire une petite image en mémoire + un texte
    tampon = io.BytesIO()
    Image.new("RGB", (32, 32), "gray").save(tampon, format="PNG")
    tampon.seek(0)

    # 3) Envoyer la requête à /predict (image + texte)
    reponse = client.post(
        "/predict",
        data={"text_input": "Radiographie thoracique sans anomalie."},
        files={"image_file": ("radio.png", tampon, "image/png")},
    )

    # 4) Vérifier la réponse
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["status"] == "success"
    assert corps["prediction"] in (0, 1)
    assert 0.0 <= corps["probability"] <= 1.0
