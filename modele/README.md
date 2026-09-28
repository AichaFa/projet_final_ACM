# Modèle

Les poids du modèle (BioViL-T à attention croisée, format safetensors, ~20 Mo)
ne sont pas versionnés dans ce dépôt : ils sont trop volumineux pour Git et
sont gérés par le registre de modèles.

## Source de vérité : le registre MLflow

Le modèle est suivi et versionné dans le registre MLflow (serveur hébergé sur
Azure App Service). Chaque entraînement produit une version numérotée ; la
version de production porte l'alias `prod`. L'API récupère automatiquement le
modèle `prod` depuis le registre, avec repli sur une copie locale des poids si
le serveur MLflow est momentanément indisponible.

## Copie locale de travail

Une copie de travail des poids (`model.safetensors`) peut exister en local, dans
`3_modelisation`, pour l'inférence et les tests hors ligne. Elle sert de repli et
n'est pas la source de référence : c'est le registre MLflow qui fait foi.
