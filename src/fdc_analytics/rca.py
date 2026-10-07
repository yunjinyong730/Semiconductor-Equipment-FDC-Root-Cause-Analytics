from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


@dataclass
class RCAModel:
    classifier: LogisticRegression
    ranking: pd.DataFrame
    metrics: dict[str, float]


def _minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def fit_rca_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    pca_residual_eval: pd.DataFrame,
    pca_loading: pd.DataFrame,
    random_state: int = 42,
) -> RCAModel:
    y_train_bin = (y_train.to_numpy() == 1).astype(int)
    y_eval_bin = (y_eval.to_numpy() == 1).astype(int)

    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )
    classifier.fit(X_train, y_train_bin)

    probability = classifier.predict_proba(X_eval)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_eval_bin, prediction)),
        "average_precision": float(average_precision_score(y_eval_bin, probability)),
        "roc_auc": float(roc_auc_score(y_eval_bin, probability))
        if len(np.unique(y_eval_bin)) == 2
        else float("nan"),
    }

    mutual_info = mutual_info_classif(X_train, y_train_bin, random_state=random_state)
    coefficient = np.abs(classifier.coef_[0])

    pass_mean = X_train.loc[y_train_bin == 0].mean()
    fail_mean = X_train.loc[y_train_bin == 1].mean()
    pooled_std = X_train.std(ddof=0).replace(0, 1.0)
    effect_size = ((fail_mean - pass_mean) / pooled_std).abs()

    try:
        permutation = permutation_importance(
            classifier,
            X_eval,
            y_eval_bin,
            scoring="average_precision",
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
        ).importances_mean
        permutation = np.maximum(permutation, 0.0)
    except ValueError:
        permutation = np.zeros(X_train.shape[1])

    fail_mask = y_eval_bin == 1
    if fail_mask.any():
        residual = pca_residual_eval.loc[fail_mask].mean()
    else:
        residual = pca_residual_eval.mean()

    ranking = pd.DataFrame(
        {
            "signal": X_train.columns,
            "effect_size": effect_size.reindex(X_train.columns).to_numpy(),
            "mutual_information": mutual_info,
            "logistic_abs_coef": coefficient,
            "permutation_importance": permutation,
            "fail_residual_contribution": residual.reindex(X_train.columns).fillna(0.0).to_numpy(),
        }
    )
    ranking = ranking.merge(pca_loading, on="signal", how="left").fillna(0.0)

    evidence_columns = [
        "effect_size",
        "mutual_information",
        "logistic_abs_coef",
        "permutation_importance",
        "fail_residual_contribution",
        "pca_loading_importance",
    ]

    normalized = []
    for column in evidence_columns:
        norm_column = f"{column}_norm"
        ranking[norm_column] = _minmax(ranking[column])
        normalized.append(norm_column)

    ranking["rca_score"] = ranking[normalized].mean(axis=1)
    ranking["evidence_count"] = (ranking[normalized] >= 0.5).sum(axis=1).astype(int)
    ranking["evidence_ratio"] = ranking["evidence_count"] / len(normalized)

    ranking = ranking.sort_values(
        ["rca_score", "evidence_count"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    return RCAModel(classifier, ranking, metrics)
