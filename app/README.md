---
title: Auditeur de Cohérence Médicale
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

Auditeur de Cohérence Médicale - interface (App) autonome.
Analyse directement la radiographie et le compte rendu avec le modèle embarqué
(BioViL-T et classifieur à attention croisée), affiche le verdict avec un score
de confiance, et journalise la prédiction dans l'entrepôt de production.
