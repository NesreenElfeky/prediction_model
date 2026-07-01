from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet


def random_forest(**kwargs) -> RandomForestRegressor:
    params = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
    params.update(kwargs)
    return RandomForestRegressor(**params)


def elastic_net(**kwargs) -> ElasticNet:
    params = {"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42}
    params.update(kwargs)
    return ElasticNet(**params)

