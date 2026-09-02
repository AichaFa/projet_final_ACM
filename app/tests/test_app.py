# ============================================================================
# Test léger de l'application Gradio
# ----------------------------------------------------------------------------
# On ne lance pas l'app (elle dépend de "spaces", propre à Hugging Face) :
# on vérifie que app.py est un fichier Python VALIDE et que c'est bien une
# application Gradio. C'est rapide et sans dépendance lourde.
# ============================================================================

import ast
import os

CHEMIN_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def _lire_app():
    with open(CHEMIN_APP, encoding="utf-8") as f:
        return f.read()


def test_app_syntaxe_valide():
    # ast.parse lève une erreur si la syntaxe est incorrecte
    ast.parse(_lire_app())


def test_app_est_bien_gradio():
    source = _lire_app()
    assert "gradio" in source.lower() or "gr." in source
