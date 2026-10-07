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


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def fit_rca_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    pca_residual_eval: pd.DataFrame,
    pca_loading: pd.DataFrame,
    random_state: int = 42,
) -> RCAModel:
    yb_train = (y_train.to_numpy() == 1).astype(int)
    yb_eval = (y_eval.to_numpy() == 1).astype(int)

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )
    clf.fit(X_train, yb_train)
    prob = clf.predict_proba(X_eval)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(yb_eval, pred)),
        "average_precision": float(average_precision_score(yb_eval, prob)),
        "roc_auc": float(roc_auc_score(yb_eval, prob)) if len(np.unique(yb_eval)) == 2 else float("nan"),
    }

    mi = mutual_info_classif(X_train, yb_train, random_state=random_state)
    coef = np.abs(clf.coef_[0])

    train_df = X_train.copy()
    pass_mean = train_df.loc[yb_train == 0].mean()
    fail_mean = train_df.loc[yb_train == 1].mean()
    pooled_std = train_df.std(ddof=0).replace(0, 1.0)
    effect = ((fail_mean - pass_mean) / pooled_std).abs()

    try:
        perm = permutation_importance(
            clf,
            X_eval,
            yb_eval,
            scoring="average_precision",
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
        ).importances_mean
        perm = np.maximum(perm, 0.0)
    except ValueError:
        perm = np.zeros(X_train.shape[1])

    fail_eval_mask = yb_eval == 1
    if fail_eval_mask.any():
        residual = pca_residual_eval.loc[fail_eval_mask].mean()
    else:
        residual = pca_residual_eval.mean()

    ranking = pd.DataFrame({
        "signal": X_train.columns,
        "effect_size": effect.reindex(X_train.columns).to_numpy(),
        "mutual_information": mi,
        "logistic_abs_coef": coef,
        "permutation_importance": perm,
        "fail_residual_contribution": residual.reindex(X_train.columns).fillna(0.0).to_numpy(),
    })
    ranking = ranking.merge(pca_loading, on="signal", how="left").fillna(0.0)
    components = [
        "effect_size",
        "mutual_information",
        "logistic_abs_coef",
        "permutation_importance",
        "fail_residual_contribution",
        "pca_loading_importance",
    ]
    weights = {
        "effect_size": 0.20,
        "mutual_information": 0.15,
        "logistic_abs_coef": 0.20,
        "permutation_importance": 0.15,
        "fail_residual_contribution": 0.20,
        "pca_loading_importance": 0.10,
    }
    score = 0.0
    for col in components:
        ranking[f"{col}_norm"] = _minmax(ranking[col])
        score = score + weights[col] * ranking[f"{col}_norm"]
    ranking["rca_score"] = score

    # 한 개의 importance 결과만으로 RCA 후보를 정하지 않고,
    # 서로 다른 분석 근거가 같은 signal을 반복해서 지목하는지도 같이 본다.
    normalized_columns = [f"{col}_norm" for col in components]
    ranking["evidence_count"] = (ranking[normalized_columns] >= 0.5).sum(axis=1).astype(int)
    ranking["evidence_agreement"] = ranking["evidence_count"] / len(normalized_columns)
    ranking["evidence_level"] = np.select(
        [ranking["evidence_count"] >= 4, ranking["evidence_count"] >= 3],
        ["strong", "moderate"],
        default="limited",
    )

    ranking = ranking.sort_values(
        ["rca_score", "evidence_count"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return RCAModel(clf, ranking, metrics)
