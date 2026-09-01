# ============================================================================
# Orchestrateur - enchaîne automatiquement TOUTE la boucle du bas
# ----------------------------------------------------------------------------
# 1) sauvegarde le datalake en ligne (Hugging Face)
# 2) lance l'analyse de dérive (monitoring.analyser(), lecture depuis Neon)
# 3) si dérive détectée -> déclenche le réentraînement.
#
# C'est ce script qu'on planifie (workflow) pour que tout s'enchaîne tout seul.
# ============================================================================

from monitoring import analyser
from datalake_hf import pousser_datalake


def declencher_reentrainement():
    # À BRANCHER sur la partie de Samer (déclencher le workflow d'entraînement, etc.)
    print(
        ">>> DÉRIVE DÉTECTÉE : déclenchement du réentraînement (à brancher sur le dépôt d'entraînement)."
    )


def main():
    # 1) Sauvegarder le datalake en ligne (Hugging Face).
    #    Si le jeton HF n'est pas configuré, on prévient mais on continue le monitoring.
    try:
        pousser_datalake()
    except (SystemExit, Exception) as e:
        print("Datalake non envoyé (on continue quand même) :", e)

    # 2) Monitoring de dérive (lecture des données de prod depuis Neon)
    verdict = analyser()
    print("Verdict :", verdict)

    # 3) Déclencheur
    if verdict["derive_detectee"]:
        declencher_reentrainement()
    else:
        print(">>> Pas de dérive significative : rien à faire.")


if __name__ == "__main__":
    main()
