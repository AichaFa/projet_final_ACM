# Auditeur de Cohérence Médicale

Vérifier la cohérence entre une radiographie thoracique et son compte rendu
radiologique - de la donnée jusqu'à une chaîne MLOps complète qui se surveille
et se réentraîne. Ce dépôt réunit l'ensemble du projet, entièrement autonome.

## Vue d'ensemble (parcours de la donnée)

    Ingestion / Entraînement -> Intégration continue (CI) ->
    Déploiement / Prédiction (CD) -> Monitoring / Réentraînement,
    avec Tracking / Audit (MLflow) et un Orchestrateur au-dessus.

## Organisation du dépôt

| Dossier | Rôle | Bloc de l'architecture |
|---|---|---|
| `1_exploration/` | Analyse exploratoire des données | Ingestion |
| `2_preparation_donnees/` | Construction du jeu équilibré (paires image + texte) | Ingestion |
| `3_modelisation/` | Notebooks du modèle (BioViL-T, BiomedCLIP) + suivi MLflow | Entraînement / Tracking |
| `api/` | API de prédiction (FastAPI) + Dockerfile + tests + CI | Prédiction |
| `app/` | Application (Gradio), déployée sur Hugging Face | Déploiement |
| `monitoring/` | Journalisation, détection de dérive (Evidently), orchestrateur planifié | Monitoring |
| `modele/` | Pointeur vers le modèle entraîné (hébergé sur le Dataset Hugging Face) | Tracking |
| `docs/` | Schéma d'architecture, présentation | - |

## Ce qui est à moi (autonome, sans dépendance externe)

- Application en ligne (Gradio, Hugging Face Space).
- API de prédiction (FastAPI).
- Intégration continue (GitHub Actions : style, tests dans Docker, alertes).
- Stockage : data warehouse (Neon, PostgreSQL) + datalake (Dataset Hugging Face).
- Monitoring automatisé (Evidently) et planifié.
- Modèle entraîné `model.pth`, hébergé sur le Dataset Hugging Face.

## Ce qui reste public et gratuit (aucune dépendance)

- Encodeurs BioViL-T (image et texte), téléchargés depuis Hugging Face.

## Données et gros fichiers (hors dépôt)

Les données CheXpert (plus de 10 Go) et le modèle `model.pth` (~20 Mo) ne sont
pas versionnés ici. Les données restent en local ; le modèle est hébergé sur le
Dataset Hugging Face et chargé au besoin.

## Suivi de mise en place

- [ ] Squelette du dépôt (structure + README)
- [ ] Intégration de l'application (app/)
- [ ] Intégration de l'API (api/)
- [ ] Intégration du monitoring (monitoring/)
- [ ] Workflows propres (CI + déploiement contrôlé)
- [ ] Artefacts sur stockage objet Neon (remplacement de S3)
- [ ] Application autonome (chargement du modèle sans dépendance externe)