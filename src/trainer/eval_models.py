"""Compare trois méthodes de prédiction sur le groupe de validation.

Les trois méthodes :
  - Ridge : une formule simple (une somme pondérée des caractéristiques)
  - RandomForest : beaucoup d'arbres de décision, on fait leur moyenne
  - GradientBoosting : des petits arbres qui corrigent les erreurs des précédents

On les essaie pour 1, 3, 5 et 7 jours d'entrée, avec et sans les caractéristiques
de features_extra.py, et pour deux cibles : la moyenne et la variabilité (CV).

Règles pour ne pas tricher :
  - le modèle apprend UNIQUEMENT sur le groupe d'entraînement ;
  - on juge sur le groupe de validation, des personnes que le modèle n'a jamais vues ;
  - le groupe de test n'est jamais lu ici ;
  - les réglages sont les mêmes pour toutes les durées.

Lancer depuis le dossier :
    python src/trainer/eval_models.py
"""
import sys
from pathlib import Path

# Permet d'importer les fichiers du dossier src (même astuce que eval_baselines.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.features import build_xy


def make_models() -> dict:
    """Crée trois modèles neufs, avec des réglages volontairement modestes.

    Réglages modestes = moins de risque que le modèle apprenne les détails par cœur.
    Ils sont fixés d'avance et on ne les change pas en regardant les résultats.

    SimpleImputer : si une valeur manque (par exemple pas de mesure la nuit),
    il met la valeur médiane de l'entraînement à la place.
    StandardScaler : remet les chiffres à la même échelle (utile pour Ridge).
    """
    return {
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,  # nombre d'arbres
                min_samples_leaf=5,  # chaque réponse repose sur au moins 5 personnes
                random_state=42,  # même tirage au sort à chaque lancement
                n_jobs=-1,  # utilise tous les cœurs de l'ordinateur
            ),
        ),
        "GradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                n_estimators=200,  # nombre de petits arbres
                learning_rate=0.05,  # chaque arbre corrige peu : plus prudent
                max_depth=3,  # arbres peu profonds
                random_state=42,
            ),
        ),
    }


def get_xy(raw: pd.DataFrame, k: int, add_extra: bool):
    """Prépare les tableaux : X (ce que le modèle voit) et y (ce qu'il doit prédire).

    add_extra=True : on ajoute les trois chiffres de features_extra.py à ceux de windowings.py.
    """
    X, y = build_xy(raw, k_days=k)
    if not add_extra:
        X = X.drop(columns=["cgm_night_mean", "cgm_day_mean"])
    return X, y


def evaluate_models(horizons=(1, 3, 5, 7)) -> pd.DataFrame:
    """Entraîne chaque modèle et mesure ses erreurs. Sauvegarde le tableau des résultats."""
    project_root = Path(__file__).resolve().parents[3]
    processed = project_root / "data" / "processed"

    # On lit seulement l'entraînement et la validation (jamais le test).
    train_raw = pd.read_csv(processed / "train_12d.csv")
    val_raw = pd.read_csv(processed / "val_12d.csv")

    jeux = {"base": False, "base+extra": True}
    records = []

    for k in horizons:  # 1, 3, 5, puis 7 jours d'entrée
        for nom_jeu, add_extra in jeux.items():
            X_tr, y_tr = get_xy(train_raw, k, add_extra)
            X_va, y_va = get_xy(val_raw, k, add_extra)

            for target in ("y_mean", "y_cv"):  # un modèle séparé pour chaque cible
                for nom_modele, modele in make_models().items():
                    # 1. Le modèle apprend sur l'entraînement.
                    modele.fit(X_tr, y_tr[target])

                    # 2. Il prédit pour les personnes de la validation.
                    preds = modele.predict(X_va)
                    # On mesure aussi son erreur sur l'entraînement, pour détecter un apprentissage par cœur.
                    preds_train = modele.predict(X_tr)

                    records.append({
                        "k_days": k,
                        "features": nom_jeu,
                        "target": target,
                        "model": nom_modele,
                        # Erreur moyenne en valeur absolue (plus petit = mieux)
                        "MAE": round(float(mean_absolute_error(y_va[target], preds)), 4),
                        # Pareil, mais les grosses erreurs comptent davantage
                        "RMSE": round(float(root_mean_squared_error(y_va[target], preds)), 4),
                        # Biais : positif si le modèle prédit trop haut en moyenne,
                        # négatif s'il prédit trop bas
                        "bias": round(float((preds - y_va[target]).mean()), 4),
                        # Erreur sur les personnes d'ENTRAÎNEMENT
                        "MAE_train": round(float(mean_absolute_error(y_tr[target], preds_train)), 4),
                    })

    results = pd.DataFrame(records)
    print(results.to_string(index=False))

    # Sauvegarde à côté des résultats de ton amie
    out_dir = project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "models_val_results.csv", index=False)
    print(f"\nRésultats sauvegardés dans {out_dir / 'models_val_results.csv'}")
    return results


if __name__ == "__main__":
    evaluate_models()
