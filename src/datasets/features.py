import numpy as np
import pandas as pd


def build_xy_cgm_only(
    split_df: pd.DataFrame,
    k_days: int = 3,
    target_start_day: int = 7,  # Jour 8 en indexation 0 (jours 7, 8, 9, 10, 11)
    timezone: str = "Europe/Zurich",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix X (from days 1..k) and target matrix y (from days 8 to 12)
    using only CGM glucose readings, with 1 row per participant.
    """
    if not (1 <= k_days <= target_start_day):
        raise ValueError(f"k_days must be between 1 and {target_start_day}, got {k_days}")

    df = split_df.copy()
    ts_local = pd.to_datetime(df["read_at"], format="ISO8601", utc=True).dt.tz_convert(timezone)
    df["local_date"] = ts_local.dt.date

    # Numérotation relative des jours 0..11 dans le bloc de 12 jours de chaque participant
    df = df.sort_values(["user_id", "local_date", "read_at"])
    df["day_idx"] = df.groupby("user_id")["local_date"].transform(
        lambda s: pd.factorize(s)[0]
    )

    X_rows = []
    y_rows = []

    for uid, g in df.groupby("user_id", sort=True):
        past = g.loc[g["day_idx"] < k_days, "val"].to_numpy()
        future = g.loc[g["day_idx"] >= target_start_day, "val"].to_numpy()

        past_mean = float(np.mean(past))
        past_std = float(np.std(past, ddof=1))
        past_cv = 100.0 * past_std / past_mean

        X_rows.append({
            "user_id": uid,
            "cgm_mean": past_mean,
            "cgm_std": past_std,
            "cgm_cv": past_cv,
            "cgm_min": float(np.min(past)),
            "cgm_q25": float(np.percentile(past, 25)),
            "cgm_median": float(np.median(past)),
            "cgm_q75": float(np.percentile(past, 75)),
            "cgm_max": float(np.max(past)),
            "cgm_iqr": float(np.percentile(past, 75) - np.percentile(past, 25)),
            "cgm_tir_39_100": float(np.mean((past >= 3.9) & (past <= 10.0)) * 100.0),
            "cgm_coverage": float(len(past) / (k_days * 96.0)),
        })

        fut_mean = float(np.mean(future))
        fut_std = float(np.std(future, ddof=1))
        fut_cv = 100.0 * fut_std / fut_mean

        y_rows.append({
            "user_id": uid,
            "y_mean": fut_mean,
            "y_cv": fut_cv,
        })

    X_df = pd.DataFrame(X_rows).set_index("user_id")
    y_df = pd.DataFrame(y_rows).set_index("user_id")
    return X_df, y_df




"""Caractéristiques supplémentaires, calculées sur les jours d'entrée.

Ce fichier ajoute trois chiffres par personne, qui ne sont pas dans windowings.py :
  1. le glucose moyen la nuit
  2. le glucose moyen la journée
  3. le plus long moment sans mesure (en heures)

Découpage d'un bloc de 12 jours (on numérote les jours de 0 à 11) :
  - les jours d'entrée sont ceux que le modèle a le droit de voir ;
  - les jours 7 à 11 (jours 8 à 12) servent à la CIBLE et ne sont jamais utilisés ici.

Les jours d'entrée se terminent toujours juste avant la cible, quelle que soit la durée :
  1 jour  = jour 7                  (numéro 6)
  3 jours = jours 5 à 7             (numéros 4 à 6)
  5 jours = jours 3 à 7             (numéros 2 à 6)
  7 jours = jours 1 à 7             (numéros 0 à 6)
Ainsi, quand la durée change, la distance à la cible ne change pas.
"""
import pandas as pd

# Plages horaires, en heure suisse. Ce sont des choix de départ, à faire valider.
# Attention : "nuit" veut dire seulement "entre 0 h et 6 h sur l'horloge",
# ce n'est pas une mesure du sommeil ni du jeûne.
NUIT = (0, 6)  # de 0 h inclus à 6 h exclu
JOUR = (8, 20)  # de 8 h inclus à 20 h exclu


def build_features_extra(
    split_df: pd.DataFrame,
    k_days: int = 3,
    target_start_day: int = 7,
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
    """Calcule les trois chiffres pour chaque personne.

    Paramètres :
        split_df : tableau des mesures (une ligne par mesure), avec les colonnes
                   user_id (la personne), read_at (le moment) et val (le glucose).
        k_days : nombre de jours d'entrée (1, 3, 5 ou 7).
        target_start_day : numéro du premier jour de la cible (7 = jour 8).
        timezone : fuseau horaire dans lequel on découpe les jours.

    Renvoie un tableau avec une ligne par personne et trois colonnes.
    """
    # Garde-fou : on ne peut pas prendre plus de jours qu'il n'y en a avant la cible.
    if not (1 <= k_days <= target_start_day):
        raise ValueError(f"k_days doit être entre 1 et {target_start_day}, reçu {k_days}")

    df = split_df.copy()

    # Les heures du fichier sont en heure mondiale (UTC). On les convertit en heure
    # suisse pour que "la nuit" corresponde vraiment à la nuit des participants.
    df["ts_local"] = pd.to_datetime(df["read_at"], format="ISO8601", utc=True).dt.tz_convert(timezone)
    df["local_date"] = df["ts_local"].dt.date

    # On met les mesures dans l'ordre, personne par personne.
    df = df.sort_values(["user_id", "ts_local"])

    # On numérote les jours de chaque personne : 0 pour son premier jour, 1 pour le suivant...
    # Chaque personne a un bloc de 12 jours de suite, donc les numéros vont de 0 à 11.
    df["day_idx"] = df.groupby("user_id")["local_date"].transform(lambda s: pd.factorize(s)[0])

    # On ne garde que les jours d'entrée : ceux juste avant la cible.
    premier_jour = target_start_day - k_days
    fenetre = df[(df["day_idx"] >= premier_jour) & (df["day_idx"] < target_start_day)]

    lignes = []
    for uid, g in fenetre.groupby("user_id", sort=True):
        # Heure de chaque mesure (0 à 23)
        heure = g["ts_local"].dt.hour

        # Valeurs de glucose mesurées la nuit, puis la journée
        nuit = g.loc[(heure >= NUIT[0]) & (heure < NUIT[1]), "val"]
        jour = g.loc[(heure >= JOUR[0]) & (heure < JOUR[1]), "val"]

        # Temps écoulé entre deux mesures qui se suivent, en heures.
        # Normalement 0,25 h (15 minutes). Un grand nombre veut dire un trou.
        ecarts = g["ts_local"].diff().dt.total_seconds() / 3600

        lignes.append({
            "user_id": uid,
            "cgm_night_mean": nuit.mean(),  # vaut "vide" si aucune mesure la nuit
            "cgm_day_mean": jour.mean(),
            "cgm_longest_gap_h": ecarts.max() if len(ecarts) > 1 else 0.0,
        })

    return pd.DataFrame(lignes).set_index("user_id")