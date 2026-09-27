"""
Script d'entraînement (et de réentraînement) de l'Auditeur de Cohérence Médicale.

Étapes :
  1. Charge le jeu déjà étiqueté (une ligne par paire, colonne target : 1 = cohérent, 0 = incohérent).
  2. Extrait les caractéristiques BioViL-T et les enregistre sur le disque, une par une.
  3. Entraîne le classifieur à attention croisée (perte BCE, AdamW, arrêt anticipé).
  4. Enregistre l'expérience dans MLflow.
  5. Sauvegarde les meilleurs poids au format portable model.safetensors.

Source des données : un fichier unique déjà assemblé et étiqueté
(chexpert_sample_dataset.csv), avec la colonne target prête et la colonne report
(compte rendu complet). C'est la même recette que le notebook de modélisation
d'origine, qui lisait lui aussi un seul fichier déjà étiqueté.

Environnement : le script fonctionne à deux endroits sans modification. Il détecte
s'il tourne sur Kaggle (où les données sont montées sous /kaggle/input) ou en local
(dossier CheXpert du projet), et choisit les chemins correspondants.

Mémoire : les caractéristiques extraites sont lues depuis le disque par petits lots,
au moment de l'entraînement, et jamais toutes chargées en mémoire d'un coup. Cela
permet d'entraîner sur l'ensemble des paires sans saturer la mémoire.

Robustesse et reprise : l'extraction sauvegarde chaque résultat sur le disque au
fur et à mesure. Si le programme s'interrompt, il suffit de le relancer : il repart
où il s'était arrêté, sans refaire ce qui est déjà en cache.
"""

import os
import gc
import hashlib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from safetensors.torch import save_file
from tqdm import tqdm
import mlflow
from PIL import ImageFile

# Tolère les images tronquées, au lieu de faire planter le programme.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from inference import (
    charger_modeles,
    encoder_image,
    encoder_texte,
    CrossAttentionClassifierBiovil,
    peripherique,
    LONGUEUR_TEXTE,
)


# ---------------------------------------------------------------------------
# Emplacement des données : détecté automatiquement selon l'environnement
# ---------------------------------------------------------------------------
# Le dossier /kaggle/input n'existe que sur Kaggle. Sa présence sert donc de
# repère : si je le trouve, je suis sur Kaggle ; sinon, je suis en local.
SUR_KAGGLE = os.path.exists("/kaggle/input")

if SUR_KAGGLE:
    # Sur Kaggle, mon jeu de données est monté en lecture seule sous ce chemin.
    # Le chemin inclut datasets/<identifiant>/<nom-du-dataset>, et mes images se
    # trouvent sous un double dossier images/images (issu de la compression du dossier).
    BASE_DONNEES = "/kaggle/input/datasets/aichajedha/auditeur-chexpert-20000"
    IMG_BASE = os.path.join(BASE_DONNEES, "images", "images")
else:
    # En local, les données sont dans le dossier CheXpert du projet.
    BASE_DONNEES = "../CheXpert"
    IMG_BASE = os.path.join(BASE_DONNEES, "CheXpert-v1.0-small")

# Le fichier de paires déjà étiqueté, au bon endroit dans les deux cas.
CSV_PAIRES = os.path.join(BASE_DONNEES, "chexpert_sample_dataset.csv")


# ---------------------------------------------------------------------------
# Réglages d'entraînement (mêmes valeurs que le notebook de modélisation)
# ---------------------------------------------------------------------------
N_SAMPLES = None         # None = tout le jeu (les 20 000 paires). Mettre un petit nombre (ex. 400) pour un test rapide.
EPOCHS = 100             # Nombre maximal d'époques, comme dans mon notebook (l'arrêt anticipé coupe souvent avant).
BATCH_SIZE = 64          # Taille de lot du classifieur, identique à mon notebook.
LR = 1e-5               # Taux d'apprentissage
WEIGHT_DECAY = 1e-2     # Régularisation L2
PATIENCE = 3            # Nombre d'époques sans amélioration avant l'arrêt anticipé
SEUIL = 0.5            # Seuil de décision cohérent / incohérent
NUM_WORKERS = 2         # Nombre de processus qui lisent le cache disque en parallèle
SORTIE_POIDS = "model.safetensors"
EXPERIENCE = "auditeur-coherence-medicale"
NOM_MODELE = "auditeur-coherence-medicale"   # nom du modèle dans le registre MLflow

CACHE = "features_demo"                          # dossier des caractéristiques déjà extraites
FICHIER_EXCLUSIONS = "_images_a_ignorer.txt"     # images à sauter (corrompues)
FICHIER_EN_COURS = "_image_en_cours.txt"         # image en cours (témoin de plantage)


def cle(chemin, texte):
    """Identifiant stable d'une paire, pour nommer son fichier de cache."""
    # Une empreinte md5 du couple (chemin, texte) donne un nom de fichier unique
    # et reproductible : la même paire retombe toujours sur le même cache.
    return hashlib.md5((chemin + "||" + texte).encode("utf-8")).hexdigest()


def charger_exclusions():
    """Charge la liste des images à ignorer ; si un plantage a laissé un témoin, l'ajoute."""
    exclus = set()
    if os.path.exists(FICHIER_EXCLUSIONS):
        with open(FICHIER_EXCLUSIONS, encoding="utf-8") as f:
            exclus = {ligne.strip() for ligne in f if ligne.strip()}
    if os.path.exists(FICHIER_EN_COURS):
        with open(FICHIER_EN_COURS, encoding="utf-8") as f:
            coupable = f.read().strip()
        if coupable and coupable not in exclus:
            exclus.add(coupable)
            with open(FICHIER_EXCLUSIONS, "a", encoding="utf-8") as f:
                f.write(coupable + "\n")
            print(f"Image écartée (plantage précédent) :\n  {coupable}")
        os.remove(FICHIER_EN_COURS)
    return exclus


def construire_jeu(exclus):
    """Charge le jeu déjà étiqueté (colonne target) et compose les chemins d'images."""
    df = pd.read_csv(CSV_PAIRES)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    lignes = []
    for _, r in df.iterrows():
        rel = str(r["path_to_image"])   # chemin relatif de l'image
        rep = str(r["report"])          # compte rendu complet (colonne report, comme à l'entraînement d'origine)

        # Garde-fou : j'écarte les comptes rendus vides ou quasi vides (peu exploitables).
        if len(rep.strip()) < 20:
            continue

        chemin = os.path.join(IMG_BASE, rel)

        # J'ignore les images exclues (corrompues) ou absentes du disque.
        if chemin in exclus or not os.path.exists(chemin):
            continue

        lignes.append((chemin, rep, int(r["target"])))

        # Si une limite est fixée (N_SAMPLES non None), on s'arrête une fois atteinte.
        if N_SAMPLES is not None and len(lignes) >= N_SAMPLES:
            break
    return lignes


def extraire_vers_cache(lignes, res, dev):
    """Extrait et sauvegarde sur disque les caractéristiques encore manquantes."""
    os.makedirs(CACHE, exist_ok=True)
    nouveaux = 0
    for chemin, texte, y in tqdm(lignes, desc="Extraction des caractéristiques"):
        dest = os.path.join(CACHE, cle(chemin, texte) + ".pt")
        if os.path.exists(dest):
            continue  # déjà extrait lors d'un lancement précédent

        # On note l'image en cours : en cas de plantage, elle sera écartée au prochain lancement.
        with open(FICHIER_EN_COURS, "w", encoding="utf-8") as f:
            f.write(chemin)

        # Encodage BioViL-T : patches de l'image et séquence de jetons du texte.
        patches = encoder_image(chemin, res, dev).squeeze(0).detach().cpu()
        seq = encoder_texte(texte, res, dev)[:, :LONGUEUR_TEXTE, :].squeeze(0).detach().cpu()
        torch.save({"img": patches, "txt": seq, "y": torch.tensor(float(y))}, dest)

        del patches, seq
        nouveaux += 1
        if nouveaux % 20 == 0:
            gc.collect()

    if os.path.exists(FICHIER_EN_COURS):
        os.remove(FICHIER_EN_COURS)
    print(f"{nouveaux} nouvelles paires extraites ; les autres étaient déjà en cache.")


class CacheDataset(Dataset):
    """Lit les caractéristiques d'une paire depuis le cache disque, à la demande.

    Ne garde en mémoire que la liste des références (chemin, texte, cible), pas les
    données elles-mêmes. Chaque lot est lu du disque au moment où l'entraînement en a
    besoin, puis libéré. La mémoire ne contient donc jamais qu'un lot à la fois.
    """

    def __init__(self, lignes):
        self.lignes = lignes

    def __len__(self):
        return len(self.lignes)

    def __getitem__(self, idx):
        chemin, texte, y = self.lignes[idx]
        dest = os.path.join(CACHE, cle(chemin, texte) + ".pt")
        d = torch.load(dest, weights_only=True)
        img = d["img"].float()
        txt = d["txt"].float()
        cible = torch.tensor(float(y))
        return img, txt, cible


def entrainer():
    dev = peripherique()
    print("Périphérique :", dev)

    # Chargement des encodeurs BioViL-T, nécessaires uniquement à l'extraction.
    res = charger_modeles()
    res["text_model"].to(dev)
    res["image_model"].to(dev)

    # Préparation du jeu : exclusions, puis sélection des paires exploitables.
    exclus = charger_exclusions()
    lignes = construire_jeu(exclus)
    print(f"{len(lignes)} paires sélectionnées")

    # Extraction (avec cache). Si tout est déjà en cache, cette étape est quasi instantanée.
    extraire_vers_cache(lignes, res, dev)

    # Les encodeurs ne servent plus après l'extraction : on rend la mémoire.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Découpe entraînement / validation sur les indices, stratifiée pour garder
    # l'équilibre des classes. On ne manipule que des références, pas les données.
    indices = list(range(len(lignes)))
    cibles = [y for (_, _, y) in lignes]
    idx_tr, idx_va = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=cibles
    )
    lignes_tr = [lignes[i] for i in idx_tr]
    lignes_va = [lignes[i] for i in idx_va]

    # Chargeurs par lots : ils lisent le cache disque au fur et à mesure (mémoire basse).
    train_loader = DataLoader(
        CacheDataset(lignes_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        CacheDataset(lignes_va), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Classifieur à attention croisée : c'est la partie entraînée (BioViL-T reste gelé).
    model = CrossAttentionClassifierBiovil().to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Adresse du serveur MLflow où envoyer les résultats.
    # Lue depuis la variable d'environnement MLFLOW_TRACKING_URI (secret Kaggle) si elle existe ;
    # sinon, on utilise par défaut l'adresse de mon serveur MLflow hébergé sur Azure.
    adresse_mlflow = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "https://auditeur-mlflow-hbb4ambmcnhhfjcz.swedencentral-01.azurewebsites.net",
    )
    mlflow.set_tracking_uri(adresse_mlflow)
    print("Serveur MLflow :", adresse_mlflow)

    mlflow.set_experiment(EXPERIENCE)
    with mlflow.start_run():
        mlflow.log_params({
            "n_samples": len(lignes),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "text_length": LONGUEUR_TEXTE,
        })

        best_val = float("inf")
        best_state = None
        patience_ct = 0

        for epoch in range(EPOCHS):
            # --- Phase d'entraînement ---
            model.train()
            total, correct = 0.0, 0
            for xi, xt, yb in train_loader:
                xi, xt, yb = xi.to(dev), xt.to(dev), yb.to(dev)
                optimizer.zero_grad()
                out = model(xi, xt).squeeze(1)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total += loss.item()
                correct += ((torch.sigmoid(out) >= SEUIL).float() == yb).sum().item()
            tr_loss = total / len(train_loader)
            tr_acc = correct / len(train_loader.dataset)

            # --- Phase de validation (sans mise à jour des poids) ---
            model.eval()
            v_total, v_correct = 0.0, 0
            with torch.no_grad():
                for xi, xt, yb in val_loader:
                    xi, xt, yb = xi.to(dev), xt.to(dev), yb.to(dev)
                    out = model(xi, xt).squeeze(1)
                    v_total += criterion(out, yb).item()
                    v_correct += ((torch.sigmoid(out) >= SEUIL).float() == yb).sum().item()
            v_loss = v_total / len(val_loader)
            v_acc = v_correct / len(val_loader.dataset)
            scheduler.step(v_loss)

            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            mlflow.log_metric("val_loss", v_loss, step=epoch)
            mlflow.log_metric("train_accuracy", tr_acc, step=epoch)
            mlflow.log_metric("val_accuracy", v_acc, step=epoch)
            print(
                f"Epoch {epoch + 1}/{EPOCHS}  "
                f"train_loss={tr_loss:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}"
            )

            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ct = 0
            else:
                patience_ct += 1
                if patience_ct >= PATIENCE:
                    print("Arrêt anticipé : la perte de validation ne s'améliore plus.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        mlflow.log_metric("best_val_loss", best_val)

        # Sauvegarde des poids au format portable, puis dépôt comme artefact du run.
        poids = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(poids, SORTIE_POIDS)
        mlflow.log_artifact(SORTIE_POIDS)
        print(f"Meilleurs poids sauvegardés dans {SORTIE_POIDS}")

        # -------------------------------------------------------------------
        # Registre de modèles et promotion en production
        # -------------------------------------------------------------------
        # J'enregistre les poids de ce run comme une nouvelle version du modèle dans
        # le registre MLflow. Chaque entraînement produit ainsi une version numérotée
        # et traçable, au lieu d'un simple fichier isolé.
        # Comme je dépose un fichier de poids (un artefact), et non un modèle MLflow
        # complet, je crée la version directement via le client, en la faisant pointer
        # vers l'emplacement réel de l'artefact du run. Le registre est créé s'il n'existe pas.
        client = mlflow.MlflowClient()
        run_id = mlflow.active_run().info.run_id
        try:
            client.create_registered_model(NOM_MODELE)
        except Exception:
            pass  # le modèle existe déjà dans le registre : rien à créer
        source = f"{mlflow.get_artifact_uri()}/{SORTIE_POIDS}"
        version = client.create_model_version(
            name=NOM_MODELE, source=source, run_id=run_id
        ).version
        print(f"Modèle enregistré : {NOM_MODELE}, version {version}")

        # Je récupère la perte de validation du modèle actuellement en production
        # (celui qui porte l'alias "prod"). S'il n'y en a pas encore, je récupère None.
        try:
            version_prod = client.get_model_version_by_alias(NOM_MODELE, "prod")
            perte_prod = client.get_run(version_prod.run_id).data.metrics.get("best_val_loss")
        except Exception:
            perte_prod = None

        # Promotion : je place l'alias "prod" sur cette version uniquement si sa perte
        # de validation est plus basse (meilleure), ou s'il n'existe pas encore de
        # modèle en production. Cette condition protège la production d'un modèle moins bon.
        if perte_prod is None or best_val < perte_prod:
            client.set_registered_model_alias(NOM_MODELE, "prod", version)
            print(f"Nouveau modèle promu en production (val_loss {best_val:.4f}).")
        else:
            print(
                f"Modèle conservé hors production : sa val_loss {best_val:.4f} "
                f"n'améliore pas la production actuelle ({perte_prod:.4f})."
            )


if __name__ == "__main__":
    entrainer()
