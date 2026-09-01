
# Image de base : une image légère contenant déjà Conda (gestion d'environnements Python)
FROM continuumio/miniconda3
 
# Mise à jour des paquets, puis installation de nano (éditeur) et curl (outil réseau)
RUN apt-get update -y 
RUN apt-get install nano curl -y
 
# On fixe Python en version 3.10 (version stable et éprouvée pour le machine learning)
RUN conda install -y python=3.10
 
# --- Partie spécifique à Hugging Face ---
# On crée un utilisateur nommé "user" (identifiant 1000) au lieu de travailler en "root" : plus sûr
RUN useradd -m -u 1000 user
# On bascule de "root" (utilisateur par défaut) vers cet utilisateur "user"
USER user
# Variables d'environnement :
#  - HOME : le dossier personnel de l'utilisateur
#  - PATH : on y ajoute le dossier des programmes installés par "user"
#  - PORT : le port sur lequel l'API écoutera (7860 est le port attendu par Hugging Face)
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860
 
# Dossier de travail à l'intérieur du conteneur (équivaut à /home/user/app)
WORKDIR $HOME/app
 
# On copie d'abord requirements.txt seul : Docker réutilise ainsi son cache
# tant que les dépendances ne changent pas (la construction est plus rapide)
COPY --chown=user requirements.txt $HOME/app/requirements.txt
 
# Installation des dépendances Python listées dans requirements.txt
RUN pip install -r requirements.txt
 
# On copie ensuite tout le reste du projet dans le conteneur (dont le dossier src/)
COPY --chown=user . $HOME/app
 
# On déclare que le conteneur rend le port 7860 accessible
EXPOSE 7860
 
# Commande lancée au démarrage : démarrer l'API FastAPI.
# NOUVEAU : le code est désormais dans src/, on pointe donc vers src/radio_check_app.py.
CMD ["sh", "-c", "fastapi run src/radio_check_app.py --port ${PORT}"]
