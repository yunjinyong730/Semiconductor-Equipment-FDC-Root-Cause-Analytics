from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    quality_report: pd.DataFrame

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = self.imputer.transform(X[self.columns])
        values = self.scaler.transform(values)
        return pd.DataFrame(values, columns=self.columns, index=X.index)


def chronological_split(n: int, train_frac: float = 0.60, val_frac: float = 0.20) -> tuple[slice, slice, slice]:
    train_end = max(1, int(n * train_frac))
    val_end = max(train_end + 1, int(n * (train_frac + val_frac)))
    val_end = min(val_end, n - 1) if n > 2 else n
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def fit_preprocessor(
    X_train: pd.DataFrame,
    missing_ratio_threshold: float = 0.40,
    correlation_threshold: float = 0.98,
) -> Preprocessor:
    missing_ratio = X_train.isna().mean()
    nunique = X_train.nunique(dropna=True)
    usable = (missing_ratio <= missing_ratio_threshold) & (nunique > 1)
    candidate_cols = X_train.columns[usable].tolist()
    if not candidate_cols:
        raise ValueError("No usable signals remain after data-quality filtering.")

    imputer_probe = SimpleImputer(strategy="median")
    probe = pd.DataFrame(
        imputer_probe.fit_transform(X_train[candidate_cols]),
        columns=candidate_cols,
        index=X_train.index,
    )
    corr = probe.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_drop = [col for col in upper.columns if (upper[col] > correlation_threshold).any()]
    selected = [c for c in candidate_cols if c not in corr_drop]

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(X_train[selected])
    scaler = StandardScaler()
    scaler.fit(train_imputed)

    quality = pd.DataFrame({
        "signal": X_train.columns,
        "missing_ratio": missing_ratio.values,
        "nunique": nunique.values,
    })
    quality["status"] = "selected"
    quality.loc[quality["missing_ratio"] > missing_ratio_threshold, "status"] = "high_missing"
    quality.loc[quality["nunique"] <= 1, "status"] = "constant"
    quality.loc[quality["signal"].isin(corr_drop), "status"] = "correlated_removed"

    return Preprocessor(selected, imputer, scaler, quality)
