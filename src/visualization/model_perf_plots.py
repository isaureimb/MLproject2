import matplotlib.pyplot as plt

TARGETS = ("y_mean", "y_cv")
TARGET_LABELS = {"y_mean": "mean glucose (mmol/L)", "y_cv": "CV (% points)"}


def _baseline_lines(ax, baselines, target):
    """Draw the baselines (if given) as dashed grey reference lines."""
    if baselines is None:
        return
    b = baselines[baselines["target"] == target]
    for (name, g), style in zip(b.groupby("model"), ["--", ":", "-."]):
        g = g.sort_values("k_days")
        ax.plot(g["k_days"], g["MAE"], color="grey", linestyle=style, marker=".", label=name)


def plot_mae_vs_k(results, features="base", models=None, baselines=None, targets=TARGETS):
    """
    MAE of each model as a function of the number of input days k, one curve per model.
    One panel per target, side by side (mean glucose, then CV).

    Args:
        results (pd.DataFrame): output of evaluate_models 
        features (str): feature set to show (e.g. "base" or "base+extra").
        models (list[str] | None): models to show (default: all models in `results`).
        baselines (pd.DataFrame | None): optional baseline results (columns k_days,
            target, model, MAE), drawn as dashed grey lines for reference.
        targets (tuple[str]): targets to show, one panel each.

    """
    r = results[results["features"] == features]
    models = models or list(r["model"].unique())

    fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 4.5), squeeze=False)
    for ax, target in zip(axes[0], targets):
        rt = r[r["target"] == target]
        for name in models:
            g = rt[rt["model"] == name].sort_values("k_days")
            ax.plot(g["k_days"], g["MAE"], marker="o", label=name)
        _baseline_lines(ax, baselines, target)

        ax.set_xticks(sorted(rt["k_days"].unique()))
        ax.set_xlabel("number of input days k")
        ax.set_ylabel(TARGET_LABELS.get(target, "MAE"))
        ax.set_title(target)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Validation MAE vs input duration")
    fig.tight_layout()



def plot_mae_vs_k_extra_features(results, models=None, feature_sets=("base", "base+extra"),
                              baselines=None, targets=TARGETS):
    """
    Grid of panels: one row per target (mean glucose on top, CV below), one column
    per model. Each panel shows MAE as a function of k, with one curve per feature
    set (without and with the extra features).

    Args:
        results (pd.DataFrame): output of evaluate_models.
        models (list[str] | None): models to show, one column each (default: all).
        feature_sets (tuple[str]): feature sets to compare in each panel.
        baselines (pd.DataFrame | None): optional baseline results, drawn in grey.
        targets (tuple[str]): targets to show, one row each.

    """
    models = models or list(results["model"].unique())

    fig, axes = plt.subplots(len(targets), len(models),
                             figsize=(4.5 * len(models), 3.8 * len(targets)),
                             sharey="row", sharex=True, squeeze=False)
    for row, target in enumerate(targets):
        rt = results[results["target"] == target]
        for col, name in enumerate(models):
            ax = axes[row, col]
            for feat, marker in zip(feature_sets, ["o", "s", "^"]):
                g = rt[(rt["model"] == name) & (rt["features"] == feat)].sort_values("k_days")
                ax.plot(g["k_days"], g["MAE"], marker=marker, label=feat)
            _baseline_lines(ax, baselines, target)

            ax.set_xticks(sorted(rt["k_days"].unique()))
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(name)
            if row == len(targets) - 1:
                ax.set_xlabel("number of input days k")
            if col == 0:
                ax.set_ylabel(TARGET_LABELS.get(target, "MAE"))
                ax.legend(fontsize=8)

    fig.suptitle("Validation MAE vs input duration, with and without extra features")
    fig.tight_layout()