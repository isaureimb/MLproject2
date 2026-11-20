"""Contrôle de l'heure et comptage des valeurs bizarres (Cohort B).

Ce script répond à deux questions :
1. Les heures écrites dans glucose.csv sont-elles en heure mondiale (UTC)
   ou déjà en heure suisse ?
2. Combien de valeurs de glucose semblent bizarres ?

Il ne modifie aucun fichier de données. Il affiche des chiffres et sauvegarde
un graphique dans le dossier figures/.

Lancer depuis le dossier MLproject2 :
    python src/datasets/controle_heure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # permet de sauvegarder le graphique sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import pandas as pd

# --- Chemins des fichiers -----------------------------------------------
# DATA : le dossier "data", situé à côté du dossier MLproject2.
# FIG : le dossier "figures" du projet, où on sauvegarde le graphique.
DATA = Path(__file__).resolve().parents[3] / "data"
FIG = Path(__file__).resolve().parents[2] / "figures"
FIG.mkdir(exist_ok=True)  # crée le dossier figures s'il n'existe pas encore

# --- 1. Charger les mesures de la Cohort B ------------------------------
# glucose.csv contient les mesures : user_id (la personne), read_at (le moment)
# et val (la valeur de glucose).
glu = pd.read_csv(DATA / "glucose.csv")

# metadata.csv dit à quelle cohorte appartient chaque personne.
# On ne garde que deux colonnes : id (la personne) et cohort.
meta = pd.read_csv(DATA / "metadata.csv", usecols=["id", "cohort"])

# On garde seulement les personnes de la Cohort B.
ids_b = meta.loc[meta["cohort"] == "cohort_b", "id"]
glu = glu[glu["user_id"].isin(ids_b)].copy()

# On transforme le texte de la colonne read_at en vraie date et heure.
# utc=True veut dire : "on suppose que ces heures sont en heure mondiale".
# C'est justement ce qu'on veut tester plus bas.
glu["ts"] = pd.to_datetime(glu["read_at"], format="ISO8601", utc=True)
print(f"Participants : {glu['user_id'].nunique()} | Mesures : {len(glu)}")

# --- 2. Contrôle de l'heure ---------------------------------------------
# Idée : la nuit, le glucose est au plus bas (vers 3 h à 5 h du matin).
# On calcule le glucose moyen pour chaque heure de la journée (0 h, 1 h... 23 h)
# de deux façons, puis on regarde laquelle donne un creux à une heure
# qui a du sens.

# Façon A : on lit l'heure telle qu'elle est écrite dans le fichier.
heure_telle_quelle = glu["ts"].dt.hour
# Façon B : on suppose que l'heure est mondiale et on la convertit en heure suisse
# (+1 h en hiver, +2 h en été).
heure_suisse = glu["ts"].dt.tz_convert("Europe/Zurich").dt.hour

# Glucose moyen pour chaque heure de la journée, pour A et pour B.
prof_telle_quelle = glu.groupby(heure_telle_quelle)["val"].mean()
prof_suisse = glu.groupby(heure_suisse)["val"].mean()

# Contre-vérification : on compare l'hiver (décembre à février) et l'été
# (juin à août), toujours avec les heures telles quelles.
# Si les heures sont mondiales, la vie des gens ne change pas entre hiver et été
# mais l'heure mondiale, elle, change : le creux d'été arrive environ 1 h plus tôt.
# Si les heures sont déjà suisses, le creux est à la même heure toute l'année.
mois = glu["ts"].dt.month
hiver = glu[mois.isin([12, 1, 2])]
ete = glu[mois.isin([6, 7, 8])]
prof_hiver = hiver.groupby(hiver["ts"].dt.hour)["val"].mean()
prof_ete = ete.groupby(ete["ts"].dt.hour)["val"].mean()


def creux(profil):
    """Renvoie l'heure (entre 0 h et 9 h) où le glucose moyen est le plus bas."""
    return int(profil.loc[0:9].idxmin())


print("\nHeure du creux de la nuit :")
print(f"  heures telles quelles dans le fichier : {creux(prof_telle_quelle)} h")
print(f"  heures converties en heure suisse     : {creux(prof_suisse)} h")
print(f"  en hiver (heures telles quelles)      : {creux(prof_hiver)} h")
print(f"  en été (heures telles quelles)        : {creux(prof_ete)} h")

# Graphique : deux courbes côte à côte.
# À gauche : heures telles quelles contre heures converties.
# À droite : hiver contre été.
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
ax[0].plot(prof_telle_quelle.index, prof_telle_quelle.values, marker="o", label="heures telles quelles")
ax[0].plot(prof_suisse.index, prof_suisse.values, marker="o", label="converties en heure suisse")
ax[0].set_title("Telles quelles contre converties")
ax[1].plot(prof_hiver.index, prof_hiver.values, marker="o", label="hiver")
ax[1].plot(prof_ete.index, prof_ete.values, marker="o", label="été")
ax[1].set_title("Hiver contre été (heures telles quelles)")
for a in ax:
    a.set_xlabel("heure de la journée")
    a.set_xticks(range(0, 24, 2))
    a.legend()
ax[0].set_ylabel("glucose moyen (mmol/L)")
plt.tight_layout()
fig.savefig(FIG / "controle_heure.png", dpi=150)
plt.close(fig)

# --- 3. Valeurs bizarres -------------------------------------------------
# Ici on COMPTE seulement, on ne supprime rien. Les seuils (2 et 25 mmol/L)
# sont indicatifs : ils servent à repérer ce qui sort de l'ordinaire,
# pas à décider quoi retirer. La règle finale se décide en équipe.
v = glu["val"]
print("\nValeurs bizarres :")
print(f"  plus petite / plus grande valeur : {v.min():.2f} / {v.max():.2f}")
print(f"  valeurs inférieures ou égales à 0 : {(v <= 0).sum()}")  # impossibles
print(f"  valeurs inférieures à 2 : {(v < 2).sum()} | supérieures à 25 : {(v > 25).sum()}")

# Séries de valeurs identiques : si le capteur affiche exactement la même
# valeur pendant 12 mesures de suite (3 heures), c'est suspect (capteur bloqué ?).
s = glu.sort_values(["user_id", "ts"])  # mesures dans l'ordre, personne par personne
# On repère le début d'une nouvelle série : la valeur change, ou on change de personne.
nouvelle_serie = (s["val"] != s["val"].shift()) | (s["user_id"] != s["user_id"].shift())
# cumsum() numérote les séries, puis on compte combien de mesures contient chacune.
longueur = s.groupby(nouvelle_serie.cumsum())["val"].size()
print(f"  séries d'au moins 12 mesures identiques de suite (3 h ou plus) : {(longueur >= 12).sum()}")

# Unités : glucose moyen de chaque participant.
# En mmol/L, les moyennes tournent autour de 5 à 8. En mg/dL, autour de 90 à 150.
moyennes = glu.groupby("user_id")["val"].mean()
print(f"\nMoyenne par participant, plus petite / plus grande : {moyennes.min():.2f} / {moyennes.max():.2f}")
print(f"Participants dont la moyenne dépasse 30 (probablement des mg/dL) : {(moyennes > 30).sum()}")

# Trous : temps écoulé entre deux mesures successives d'une même personne.
# Une mesure est prévue toutes les 15 minutes, donc un écart de plus de 30 minutes
# veut dire qu'au moins une mesure manque.
s = glu.sort_values(["user_id", "ts"])
ecart_min = s.groupby("user_id")["ts"].diff().dt.total_seconds() / 60
gros_trous = ecart_min[ecart_min > 30]
print(f"\nTrous de plus de 30 minutes : {len(gros_trous)}")
# On range ces trous par durée
tranches = pd.cut(gros_trous, [30, 60, 120, 360, 1440, float("inf")],
                  labels=["30 min à 1 h", "1 à 2 h", "2 à 6 h", "6 à 24 h", "plus de 24 h"])
print(tranches.value_counts().sort_index().to_string())
# Plus long trou de chaque participant, en heures
plus_long = ecart_min.groupby(s["user_id"]).max() / 60
print(f"Participants avec un trou de plus de 24 h : {(plus_long > 24).sum()}")
print(f"Plus long trou, médiane / maximum : {plus_long.median():.1f} h / {plus_long.max():.1f} h")