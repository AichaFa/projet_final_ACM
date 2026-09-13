# Auditeur de Cohérence Médicale

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![Evidently](https://img.shields.io/badge/Evidently-ED0400?style=for-the-badge)
![PostgreSQL](https://img.shields.io/badge/Neon%20Postgres-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

## Présentation

Ce projet vérifie automatiquement qu'un compte rendu radiologique correspond bien à sa radiographie thoracique. Le système reçoit une paire image-texte et estime la probabilité qu'ils décrivent le même examen, afin de signaler les associations incohérentes pour une revue humaine. Il ne pose pas de diagnostic : c'est un outil d'audit multimodal.

## Approche

Le besoin métier est traduit en un problème de classification binaire : pour une paire (image, texte), prédire cohérent (1) ou incohérent (0), avec un score de confiance. Le modèle repose sur BioViL-T, des encodeurs image et texte spécialisés pour la radiographie thoracique, surmontés d'un classifieur à attention croisée où le texte interroge les régions de l'image.

## Architecture

La solution est industrialisée selon une chaîne MLOps complète et autonome :

- Inférence : BioViL-T (public, licence MIT) et classifieur entraîné, chargé depuis des poids portables au format safetensors.
- API : service FastAPI exposant les routes /health et /predict.
- Application : interface Gradio déployée sur Hugging Face.
- Entrepôt et datalake : PostgreSQL (Neon) et jeu de données Hugging Face.
- Intégration et déploiement continus : GitHub Actions.
- Monitoring : Evidently, détection de dérive sémantique sur les embeddings de texte.

## Structure du dépôt

- 1_exploration : analyses exploratoires.
- 2_preparation_donnees : construction du jeu équilibré (paires cohérentes et incohérences calibrées).
- 3_modelisation : entraînement du modèle et poids.
- api : service FastAPI d'inférence.
- app : application Gradio.
- monitoring : surveillance de la dérive avec Evidently.
- modele : poids du modèle.
- docs : documentation.
- .github : workflows d'intégration continue.

## Données

Les données proviennent de CheXpert (radiographies thoraciques et comptes rendus dé-identifiés). Volumineuses et sensibles, elles ne sont pas versionnées dans ce dépôt et se téléchargent séparément depuis leurs sources d'origine.

## Stack technique

Python, PyTorch, Transformers, BioViL-T, FastAPI, Gradio, safetensors, Evidently, PostgreSQL (Neon), Docker et GitHub Actions.
