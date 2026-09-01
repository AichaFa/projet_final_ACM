---
title: Auditeur App Gradio
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

Version Gradio (gratuite) de l'application de vérification de cohérence entre
une radiographie thoracique et son compte rendu. L'interface envoie l'image et
le texte à l'API de prédiction, puis affiche le verdict.

<!--
Note (français) : le bloc entre les lignes --- est la configuration du Space
Hugging Face. Ici, sdk: gradio indique un Space Gradio (gratuit), et
app_file: app.py désigne le fichier lancé au démarrage. On n'a PAS de Dockerfile
dans cette version : c'est justement ce qui la rend gratuite.
gradio n'est pas listé dans requirements.txt car il est fourni par le SDK du Space.
-->