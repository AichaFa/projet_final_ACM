# ============================================================================
# Tests automatiques de l'API (avec pytest)
# ----------------------------------------------------------------------------
# But : vérifier que l'API répond correctement, SANS charger les gros modèles.
# On utilise le TestClient de FastAPI, qui "appelle" l'API en mémoire.
# Écrit ainsi (sans le mot-clé "with"), il ne déclenche pas le démarrage qui
# charge les modèles : les tests restent donc rapides et légers.
# ============================================================================

from fastapi.testclient import TestClient  # Outil pour appeler l'API dans les tests
import radio_check_app  # On importe ton fichier API pour récupérer "app"

# Client de test qui enverra des requêtes à ton API (en mémoire, sans vrai serveur)
client = TestClient(radio_check_app.app)


def test_health():
    # On appelle la route /health, comme le ferait un navigateur
    reponse = client.get("/health")
    # On vérifie que l'API répond "OK" (code 200 = tout va bien)
    assert reponse.status_code == 200
    # On vérifie que le contenu renvoyé contient bien status = "ok"
    assert reponse.json()["status"] == "ok"
