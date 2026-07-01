from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(np.asarray(y_true) == 0, 1, np.asarray(y_true))
    mape = float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / denom)) * 100)
    return {"mae": float(mae), "rmse": float(rmse), "mape": mape, "r2": float(r2_score(y_true, y_pred))}
