from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class FDCModel:
    pca: PCA
    t2_threshold: float
    q_threshold: float
    feature_names: list[str]

    def score(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        Z = X_scaled[self.feature_names].to_numpy()
        scores = self.pca.transform(Z)
        eig = np.maximum(self.pca.explained_variance_, 1e-12)
        t2 = np.sum((scores ** 2) / eig, axis=1)
        recon = self.pca.inverse_transform(scores)
        residual = Z - recon
        q = np.sum(residual ** 2, axis=1)
        out = pd.DataFrame({"t2": t2, "q": q}, index=X_scaled.index)
        out["t2_ratio"] = out["t2"] / max(self.t2_threshold, 1e-12)
        out["q_ratio"] = out["q"] / max(self.q_threshold, 1e-12)
        out["fdc_score"] = out[["t2_ratio", "q_ratio"]].max(axis=1)
        out["fdc_alarm"] = (out["t2"] > self.t2_threshold) | (out["q"] > self.q_threshold)
        return out

    def residual_contribution(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        Z = X_scaled[self.feature_names].to_numpy()
        recon = self.pca.inverse_transform(self.pca.transform(Z))
        residual_sq = (Z - recon) ** 2
        return pd.DataFrame(residual_sq, columns=self.feature_names, index=X_scaled.index)


def fit_fdc_model(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    variance: float = 0.95,
    threshold_quantile: float = 0.99,
) -> FDCModel:
    normal_mask = y_train.to_numpy() == -1
    if normal_mask.sum() < 10:
        raise ValueError("Too few normal Pass samples to establish an FDC baseline.")
    baseline = X_train_scaled.loc[normal_mask]
    pca = PCA(n_components=variance, svd_solver="full")
    pca.fit(baseline.to_numpy())

    tmp = FDCModel(pca=pca, t2_threshold=1.0, q_threshold=1.0, feature_names=list(X_train_scaled.columns))
    base_scores = tmp.score(baseline)
    t2_threshold = float(base_scores["t2"].quantile(threshold_quantile))
    q_threshold = float(base_scores["q"].quantile(threshold_quantile))
    return FDCModel(pca, t2_threshold, q_threshold, list(X_train_scaled.columns))


def pca_loading_ranking(model: FDCModel) -> pd.DataFrame:
    weights = np.abs(model.pca.components_).T * model.pca.explained_variance_ratio_
    importance = weights.sum(axis=1)
    return pd.DataFrame({"signal": model.feature_names, "pca_loading_importance": importance}).sort_values(
        "pca_loading_importance", ascending=False
    )
