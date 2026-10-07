import numpy as np
import pandas as pd

from src.fdc_analytics.drift import rolling_drift_score
from src.fdc_analytics.fdc import fit_fdc_model
from src.fdc_analytics.preprocessing import fit_preprocessor


def make_data(seed=7):
    rng = np.random.default_rng(seed)
    n = 160
    X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"signal_{i:03d}" for i in range(12)])
    y = pd.Series(np.where(rng.random(n) < 0.1, 1, -1))
    X.loc[100:, "signal_002"] += 1.5
    X.loc[::17, "signal_004"] = np.nan
    return X, y


def test_preprocess_and_fdc_scores():
    X, y = make_data()
    prep = fit_preprocessor(X.iloc[:100])
    Z = prep.transform(X)
    fdc = fit_fdc_model(Z.iloc[:100], y.iloc[:100], variance=0.90, threshold_quantile=0.98)
    scores = fdc.score(Z)
    assert {"t2", "q", "fdc_score", "fdc_alarm"}.issubset(scores.columns)
    assert np.isfinite(scores[["t2", "q"]].to_numpy()).all()


def test_drift_increases_after_shift():
    X, _ = make_data()
    prep = fit_preprocessor(X.iloc[:80])
    Z = prep.transform(X)
    drift = rolling_drift_score(Z, Z.iloc[:80], ["signal_002", "signal_003"], window=20)
    assert drift.iloc[120:]["drift_score"].mean() > drift.iloc[40:80]["drift_score"].mean()


def test_rca_has_multi_evidence_columns():
    from src.fdc_analytics.rca import fit_rca_model
    from src.fdc_analytics.fdc import pca_loading_ranking

    X, y = make_data(seed=11)
    prep = fit_preprocessor(X.iloc[:100])
    Z = prep.transform(X)
    fdc = fit_fdc_model(Z.iloc[:100], y.iloc[:100], variance=0.90, threshold_quantile=0.98)
    residuals = fdc.residual_contribution(Z.iloc[100:])
    loading = pca_loading_ranking(fdc)
    model = fit_rca_model(
        Z.iloc[:100],
        y.iloc[:100],
        Z.iloc[100:],
        y.iloc[100:],
        residuals,
        loading,
        random_state=11,
    )
    assert {"evidence_count", "evidence_ratio"}.issubset(model.ranking.columns)
    assert model.ranking["evidence_count"].between(0, 6).all()
    assert model.ranking["evidence_ratio"].between(0, 1).all()
