# Modèle entraîné

Le modèle entraîné `model.pth` (~20 Mo, BioViL-T attention croisée) n'est pas
versionné dans ce dépôt car trop volumineux pour Git.

Il est hébergé sur le Dataset Hugging Face :
https://huggingface.co/datasets/AichaFaHugFace/auditeur-datalake

Pour l'utiliser en local, le placer dans :

    modele/data/model.pth

Ce fichier est chargé au démarrage de l'application (avec les encodeurs BioViL-T
publics) pour réaliser les prédictions.