from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from synthetic_data import generate_synthetic_loans


def _binary_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def _binary_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_prob).rank(method="average").to_numpy()
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train_and_save_pd_model(
    output_path: str = "pd_model.json",
    n_samples: int = 50_000,
    seed: int | None = None,
    return_stats: bool = False,
) -> Union[str, Tuple[str, Dict[str, float], Dict[str, float]]]:
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 1_000_000))
    features, target = generate_synthetic_loans(n_samples, seed)

    rng = np.random.default_rng(seed + 7)
    n_estimators = int(rng.choice([80, 100, 120, 150, 200]))
    max_depth = int(rng.choice([3, 4, 5, 6]))
    learning_rate = float(rng.choice([0.05, 0.1, 0.15, 0.2, 0.3, 0.5]))
    subsample = float(rng.choice([0.7, 0.8, 0.9, 1.0]))
    colsample_bytree = float(rng.choice([0.7, 0.8, 0.9, 1.0]))

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    n_train = int(0.8 * len(features))
    X_train, X_test = features.iloc[:n_train], features.iloc[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]

    model.fit(X_train, y_train)
    output = Path(output_path)
    model.save_model(str(output))

    if not return_stats:
        return str(output)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_auc = _binary_auc(y_test, test_probs)
    metrics = {
        "test_logloss": _binary_logloss(y_test, test_probs),
        "test_auc": test_auc,
        "test_gini": 2 * test_auc - 1,
    }
    params = {
        "n_estimators": model.get_params().get("n_estimators"),
        "max_depth": model.get_params().get("max_depth"),
        "learning_rate": model.get_params().get("learning_rate"),
        "subsample": model.get_params().get("subsample"),
        "colsample_bytree": model.get_params().get("colsample_bytree"),
        "objective": model.get_params().get("objective"),
        "eval_metric": model.get_params().get("eval_metric"),
        "train_rows": float(n_train),
        "test_rows": float(len(X_test)),
        "training_seed": float(seed),
        "feature_count": float(features.shape[1]),
    }
    return str(output), metrics, params


if __name__ == "__main__":
    train_and_save_pd_model()
