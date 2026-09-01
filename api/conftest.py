# ============================================================================
# conftest.py - petit fichier lu AUTOMATIQUEMENT par pytest
# ----------------------------------------------------------------------------
# Depuis que le code est rangé dans le dossier "src", on doit indiquer à Python
# d'aller aussi chercher les fichiers à importer là-dedans. Sans ce fichier,
# "import radio_check_app" (dans le test) ne trouverait plus le code, qui n'est
# plus à la racine.
#
# Ce fichier se place à la RACINE du dépôt (au même niveau que src/ et tests/).
# ============================================================================
import os
import sys

# On ajoute le dossier "src" à la liste des endroits où Python cherche les modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
