import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluation_metrics(y_true, y_pred):
    """
    Evaluation metrics used for every model and baseline.

    Returns:
        dict: MAE, RMSE and signed bias (mean of prediction - truth;
            > 0 means the model overestimates on average).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "Bias": float(np.mean(y_pred - y_true)),
    }