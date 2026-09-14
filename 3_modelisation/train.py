"""
Script d'entraînement (et de réentraînement) de l'Auditeur de Cohérence Médicale.

Étapes :
  1. Reconstruit le jeu étiqueté (paires cohérentes = 1, incohérences calibrées = 0).
  2. Extrait les caractéristiques BioViL-T et les enregistre sur le disque, une par une.
  3. Entraîne le classifieur à attention croisée (perte BCE, AdamW, arrêt anticipé).
  4. Enregistre l'expérience dans MLflow.
  5. Sauvegarde les meilleurs poids au format portable model.safetensors.

Robustesse et reprise : l'extraction sauvegarde chaque résultat sur le disque au
fur et à mesure. La mémoire reste donc basse, et si le programme s'interrompt
(mémoire saturée ou image corrompue), il suffit de le relancer : il repart où il
s'était arrêté, sans refaire ce qui est déjà en cache.

Pour une démonstration locale sans GPU, l'entraînement porte sur un sous-ensemble
(N_SAMPLES). Un entraînement complet reprend le même script sur l'ensemble des
données, sur une machine dotée d'un GPU.
"""

import os
import gc
import hashlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
# Configuration (mêmes réglages que le notebook de modélisation)
# ---------------------------------------------------------------------------
PREP = "../2_preparation_donnees"
IMG_BASE = "../CheXpert/CheXpert-v1.0-small"
MATCHES = os.path.join(PREP, "chexpert_matches_sample_2.csv")
MISMATCHES = os.path.join(PREP, "chexpert_mismatches_swap_cluster.csv")

N_SAMPLES = 400
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-5
WEIGHT_DECAY = 1e-2
PATIENCE = 3
SEUIL = 0.5
SORTIE_POIDS = "model.safetensors"
EXPERIENCE = "auditeur-coherence-medicale"

CACHE = "features_demo"                          # dossier des caractéristiques déjà extraites
FICHIER_EXCLUSIONS = "_images_a_ignorer.txt"     # images à sauter (corrompues)
FICHIER_EN_COURS = "_image_en_cours.txt"         # image en cours (témoin de plantage)


def cle(chemin, texte):
    """Identifiant stable d'une paire, pour nommer son fichier de cache."""
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
    """Assemble un jeu étiqueté : cohérent (1) et incohérent calibré (0)."""
    m = pd.read_csv(MATCHES, usecols=["path_to_image", "report"])
    m["target"] = 1
    n = pd.read_csv(MISMATCHES, usecols=["path_to_image", "report"])
    n["target"] = 0
    df = pd.concat([m, n], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    lignes = []
    for _, r in df.iterrows():
        rel, rep = str(r["path_to_image"]), str(r["report"])
        if "frontal" not in rel.lower() or len(rep.strip()) < 20:
            continue
        chemin = os.path.join(IMG_BASE, rel)
        if chemin in exclus or not os.path.exists(chemin):
            continue
        lignes.append((chemin, rep, int(r["target"])))
        if len(lignes) >= N_SAMPLES:
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


def charger_cache(lignes):
    """Charge depuis le disque les caractéristiques de toutes les paires."""
    imgs, txts, ys = [], [], []
    for chemin, texte, y in lignes:
        dest = os.path.join(CACHE, cle(chemin, texte) + ".pt")
        d = torch.load(dest, weights_only=True)
        # Passage par NumPy : remet les données dans une disposition mémoire propre
        # (contiguë), comme le faisait le notebook d'origine.
        imgs.append(d["img"].detach().contiguous().numpy())
        txts.append(d["txt"].detach().contiguous().numpy())
        ys.append(d["y"].item())
    X_img = torch.from_numpy(np.array(imgs)).float()
    X_txt = torch.from_numpy(np.array(txts)).float()
    y = torch.tensor(ys)
    return X_img, X_txt, y


def entrainer():
    dev = peripherique()
    print("Périphérique :", dev)

    res = charger_modeles()
    res["text_model"].to(dev)
    res["image_model"].to(dev)

    exclus = charger_exclusions()
    lignes = construire_jeu(exclus)
    print(f"{len(lignes)} paires sélectionnées")

    extraire_vers_cache(lignes, res, dev)
    X_img, X_txt, y = charger_cache(lignes)

    Xi_tr, Xi_va, Xt_tr, Xt_va, y_tr, y_va = train_test_split(
        X_img, X_txt, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = DataLoader(
        TensorDataset(Xi_tr, Xt_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(Xi_va, Xt_va, y_va), batch_size=BATCH_SIZE)

    model = CrossAttentionClassifierBiovil().to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

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

        poids = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(poids, SORTIE_POIDS)
        mlflow.log_artifact(SORTIE_POIDS)
        print(f"Meilleurs poids sauvegardés dans {SORTIE_POIDS}")


if __name__ == "__main__":
    entrainer()
