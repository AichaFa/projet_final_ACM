# Auditeur de Cohérence Médicale

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Evidently](https://img.shields.io/badge/Evidently-ED0400?style=for-the-badge)
![PostgreSQL](https://img.shields.io/badge/Neon%20Postgres-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

## Présentation

Ce projet vérifie automatiquement qu'un compte rendu radiologique correspond bien à sa radiographie thoracique. Le système reçoit une paire image-texte et estime la probabilité qu'ils décrivent le même examen, afin de signaler les associations incohérentes pour une revue humaine. Il ne pose pas de diagnostic : c'est un outil d'audit multimodal.

## Démonstration et liens

- Application en ligne : [Auditeur de Cohérence Médicale (Hugging Face)](https://huggingface.co/spaces/AichaFaHugFace/Auditeur-App-Gradio)
- Suivi des expériences et registre de modèles : MLflow hébergé sur Azure App Service (serveur privé, non exposé publiquement).

Démonstration vidéo (glisser la vidéo ici depuis l'éditeur GitHub) :

https://github.com/user-attachments/assets/98fc7a8c-cfca-4345-96da-5cddeaeffac5

## Approche

Le besoin métier est traduit en un problème de classification binaire : pour une paire (image, texte), prédire cohérent (1) ou incohérent (0), avec un score de confiance. Le modèle repose sur BioViL-T, des encodeurs image et texte spécialisés pour la radiographie thoracique, surmontés d'un classifieur à attention croisée où le texte interroge les régions de l'image.

## Architecture
<img width="2089" height="1289" alt="architecture_auditeur" src="https://github.com/user-attachments/assets/54956cd0-3c18-4377-9a51-21a869402221" />

La solution est industrialisée selon une chaîne MLOps complète, couvrant l'entraînement, le suivi des modèles, le service de prédiction, la surveillance et le réentraînement automatique.

- Modèle : BioViL-T (public, licence MIT) et classifieur à attention croisée entraîné, au format portable safetensors.
- Suivi et registre de modèles : serveur MLflow hébergé sur Azure App Service, adossé à une base PostgreSQL (Neon) pour les métadonnées et à Azure Blob Storage pour les artefacts. Le registre versionne les modèles ; la version de production porte l'alias `prod`.
- Entraînement : réalisé sur GPU via un notebook Kaggle, qui journalise l'expérience dans MLflow, enregistre le modèle au registre et le promeut en `prod` uniquement s'il améliore la perte de validation.
- API : service FastAPI (routes `/health`, `/predict`, `/reload-model`), conteneurisé avec Docker et testé en intégration continue. Elle charge le modèle de production depuis MLflow, avec repli sur une copie locale si le serveur MLflow est momentanément indisponible.
- Application : interface Gradio de démonstration, déployée sur Hugging Face, qui réalise l'inférence localement à partir du modèle embarqué.
- Entrepôt et datalake : PostgreSQL (Neon) pour les prédictions de production ; Azure Blob Storage pour le datalake (embeddings de production et référence sémantique).
- Monitoring : Evidently, détection de dérive sémantique sur les embeddings de texte.
- Intégration et déploiement continus : GitHub Actions.

## Boucle de réentraînement automatique

Le cycle de vie du modèle est automatisé de bout en bout :

1. L'application journalise ses prédictions et leurs embeddings dans le datalake Azure.
2. Un workflow de monitoring planifié mesure la dérive sémantique avec Evidently.
3. Au-delà d'un seuil de dérive, le monitoring émet un signal (`repository_dispatch`).
4. Ce signal déclenche un workflow de réentraînement, qui relance l'entraînement sur Kaggle via son API.
5. Le nouveau modèle est journalisé dans MLflow, enregistré au registre et promu en `prod` s'il est meilleur.
6. L'API peut recharger le modèle de production à chaud, via sa route `/reload-model`.

Le calcul GPU est déporté sur Kaggle, faute d'accès à un GPU sur le cloud de production. La structure de la chaîne reste identique à une architecture de référence ; seule la ressource de calcul diffère.

## Organisation des fichiers

Le projet est organisé en dossiers numérotés suivant les étapes du pipeline, des données brutes jusqu'à la solution industrialisée.

```
projet_final_ACM/
├── 1_exploration/                          (analyses exploratoires, non officielles)
│   ├── 00 - EDA.ipynb
│   ├── 01_dataset_matching_image_report.ipynb
│   ├── analyse_json_csv.ipynb
│   ├── create_sample.ipynb
│   └── prepa_csv_image_json.ipynb
│
├── 2_preparation_donnees/                  (construction du jeu équilibré)
│   ├── create_dataset_sample.ipynb
│   ├── chexpert_matches_sample_2.csv
│   ├── chexpert_mismatches_swapping_2.csv
│   └── chexpert_mismatches_swap_cluster.csv
│
├── 3_modelisation/                         (entraînement, inférence, notebooks)
│   ├── train.py                            (entraînement + registre + promotion MLflow)
│   ├── inference.py                        (cœur d'inférence BioViL-T)
│   ├── models_biovil_t.ipynb               (modélisation BioViL-T retenue)
│   ├── models_biomed_clip.ipynb            (modélisation BiomedCLIP, comparaison)
│   ├── test_inference.ipynb
│   ├── Kaggle_notebook033e66b2c2.ipynb     (notebook d'entraînement Kaggle)
│   └── anciennes_versions/                 (itérations antérieures, trace de la démarche)
│       ├── create_pos_neg.ipynb
│       ├── models_1_biomed_log_reg_mlp.ipynb
│       ├── models_2_biomed_cross_attention.ipynb
│       ├── models_3_biomed_lora.ipynb
│       ├── models_4_biovil_cross_attention.ipynb
│       ├── models_biomed_clip_old.ipynb
│       ├── models_biovil_t_old.ipynb
│       └── Guide_explicatif_du_code_Final.pdf
│
├── api/                                    (service d'inférence FastAPI)
│   ├── src/
│   │   └── radio_check_app.py              (API, chargement du modèle prod MLflow)
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_predict.py
│   ├── Dockerfile
│   ├── conftest.py
│   └── requirements.txt
│
├── app/                                    (application Gradio de démonstration)
│   ├── app.py                              (interface et inférence embarquée)
│   ├── inference.py
│   ├── datalake_azure.py                   (journalisation vers le datalake Azure)
│   ├── tests/
│   │   └── test_app.py
│   ├── requirements.txt
│   └── README.md
│
├── monitoring/                             (surveillance de dérive et déclenchement)
│   ├── monitoring.py                       (dérive Evidently + signal de réentraînement)
│   ├── embeddings.py                       (calcul des embeddings BioViL-T)
│   ├── construire_reference.py             (référence sémantique)
│   ├── journalisation.py
│   ├── datalake_azure.py
│   ├── orchestrateur.py
│   └── requirements.txt
│
├── modele/                                 (documentation du modèle, poids gérés via MLflow)
│   └── README.md
│
├── .github/workflows/                      (intégration continue et automatisation)
│   ├── ci-api.yml                          (tests de l'API)
│   ├── ci-app.yml                          (tests de l'application)
│   ├── deploy-app.yml                      (déploiement de l'application)
│   ├── monitoring.yml                      (monitoring planifié quotidien)
│   └── reentrainement.yml                  (réentraînement sur signal de dérive)
│
├── presentation/                           (support de soutenance)
│   └── Presentation_Auditeur.pptx          (support de présentation)
│
├── .gitignore
└── README.md
```

Les données (images CheXpert, comptes rendus, étiquettes), les poids du modèle, le cache de caractéristiques et les fichiers de secrets ne sont pas versionnés : ils sont exclus du dépôt via `.gitignore`. Les poids sont gérés par le registre MLflow, et les données se téléchargent depuis leurs sources d'origine.

## Données

Les données proviennent de CheXpert (radiographies thoraciques et comptes rendus dé-identifiés). Volumineuses et sensibles, elles ne sont pas versionnées dans ce dépôt. Le jeu d'entraînement (un sous-ensemble équilibré de 20 000 paires) est hébergé comme jeu de données Kaggle, où s'exécute l'entraînement sur GPU.

## Données et conformité

Les données proviennent de CheXpert, un jeu de recherche public et dé-identifié (anonymisé), ce qui écarte le traitement de données personnelles directement identifiables. Elles ne sont pas versionnées dans ce dépôt et se téléchargent depuis leur source d'origine.

Le serveur MLflow ne stocke que des métadonnées d'expériences (paramètres, métriques) et les poids du modèle, à l'exclusion de toute donnée patient. Les identifiants et secrets (accès cloud, jetons d'API) sont gérés hors du dépôt, au moyen de secrets de déploiement et de variables d'environnement.

## Stack technique

Python, PyTorch, Transformers, BioViL-T, FastAPI, Gradio, safetensors, MLflow, Azure (App Service, Blob Storage), Kaggle, Evidently, PostgreSQL (Neon), Docker et GitHub Actions.
