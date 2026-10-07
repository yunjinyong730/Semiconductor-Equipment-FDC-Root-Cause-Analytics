import numpy as np
import pandas as pd

from src.fdc_analytics.drift import rolling_drift_score
from src.fdc_analytics.fdc import fit_fdc_model, pca_loading_ranking
from src.fdc_analytics.preprocessing import fit_preprocessor
from src.fdc_analytics.rca import fit_rca_model


def make_data(seed=7):
    rng = np.random.default_rng(seed)
    n = 160
    X = pd.DataFrame(
        rng.normal(size=(n, 12)),
        columns=[f"signal_{i:03d}" for i in range(12)],
    )
    y = pd.Series(np.where(rng.random(n) < 0.1, 1, -1))
    X.loc[100:, "signal_002"] += 1.5
    X.loc[::17, "signal_004"] = np.nan
    return X, y


def test_preprocess_and_fdc_scores():
    X, y = make_data()
    preprocessor = fit_preprocessor(X.iloc[:100])
    X_scaled = preprocessor.transform(X)

    fdc = fit_fdc_model(
        X_scaled.iloc[:100],
        y.iloc[:100],
        variance=0.90,
        threshold_quantile=0.98,
    )
    scores = fdc.score(X_scaled)

    assert {"t2", "q", "fdc_score", "fdc_alarm"}.issubset(scores.columns)
    assert np.isfinite(scores[["t2", "q"]].to_numpy()).all()


def test_drift_increases_after_shift():
    X, _ = make_data()
    preprocessor = fit_preprocessor(X.iloc[:80])
    X_scaled = preprocessor.transform(X)

    drift = rolling_drift_score(
        X_scaled,
        X_scaled.iloc[:80],
        ["signal_002", "signal_003"],
        window=20,
    )

    before = drift.iloc[40:80]["drift_score"].mean()
    after = drift.iloc[120:]["drift_score"].mean()
    assert after > before


def test_rca_evidence_fields():
    X, y = make_data(seed=11)
    preprocessor = fit_preprocessor(X.iloc[:100])
    X_scaled = preprocessor.transform(X)

    fdc = fit_fdc_model(
        X_scaled.iloc[:100],
        y.iloc[:100],
        variance=0.90,
        threshold_quantile=0.98,
    )
    residuals = fdc.residual_contribution(X_scaled.iloc[100:])
    loadings = pca_loading_ranking(fdc)

    rca = fit_rca_model(
        X_scaled.iloc[:100],
        y.iloc[:100],
        X_scaled.iloc[100:],
        y.iloc[100:],
        residuals,
        loadings,
        random_state=11,
    )

    assert {"evidence_count", "evidence_ratio"}.issubset(rca.ranking.columns)
    assert rca.ranking["evidence_count"].between(0, 6).all()
    assert rca.ranking["evidence_ratio"].between(0, 1).all()
