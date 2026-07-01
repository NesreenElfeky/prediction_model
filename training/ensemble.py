from __future__ import annotations

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from models.baselines import elastic_net, random_forest


def default_stacking_regressor() -> StackingRegressor:
    return StackingRegressor(
        estimators=[("rf", random_forest(n_estimators=200)), ("en", elastic_net())],
        final_estimator=Ridge(),
    )

