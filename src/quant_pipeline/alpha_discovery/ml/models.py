from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline


def fit_predict_models(x: np.ndarray, y: np.ndarray, train: np.ndarray, validation: np.ndarray, seed: int = 1729) -> dict[str, np.ndarray]:
    models = {
        "ridge": make_pipeline(SimpleImputer(), Ridge(alpha=10.0)),
        "elastic_net": make_pipeline(SimpleImputer(), ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=5000)),
        "shallow_boosting": make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(max_depth=3, max_iter=100, random_state=seed)),
        "random_forest": make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=seed)),
        "extra_trees": make_pipeline(SimpleImputer(), ExtraTreesRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=seed)),
    }
    output = {}
    for name, model in models.items():
        model.fit(x[train], y[train]); output[name] = model.predict(x[validation])
    return output
