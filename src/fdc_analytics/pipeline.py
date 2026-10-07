from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from .config import PipelineConfig
from .data import load_secom
from .drift import rolling_drift_score
from .fdc import fit_fdc_model, pca_loading_ranking
from .preprocessing import chronological_split, fit_preprocessor
from .rca import fit_rca_model


def run_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "outputs",
    config: PipelineConfig | None = None,
) -> dict:
    config = config or PipelineConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, timestamp = load_secom(data_dir, auto_download=True)
    train_sl, val_sl, test_sl = chronological_split(len(X))

    X_train_raw = X.iloc[train_sl]
    y_train = y.iloc[train_sl]
    y_val = y.iloc[val_sl]
    y_test = y.iloc[test_sl]

    preprocessor = fit_preprocessor(
        X_train_raw,
        missing_ratio_threshold=config.missing_ratio_threshold,
        correlation_threshold=config.correlation_threshold,
    )
    X_scaled = preprocessor.transform(X)
    X_train = X_scaled.iloc[train_sl]
    X_val = X_scaled.iloc[val_sl]
    X_test = X_scaled.iloc[test_sl]

    fdc = fit_fdc_model(
        X_train,
        y_train,
        variance=config.pca_variance,
        threshold_quantile=config.fdc_quantile,
    )
    fdc_scores = fdc.score(X_scaled)
    residuals = fdc.residual_contribution(X_scaled)
    loadings = pca_loading_ranking(fdc)

    monitored = loadings.head(config.top_monitored_signals)["signal"].tolist()
    pass_baseline = X_train.loc[y_train.to_numpy() == -1]
    drift = rolling_drift_score(
        X_scaled,
        pass_baseline,
        monitored,
        window=config.drift_window,
    )

    rca = fit_rca_model(
        X_train,
        y_train,
        X_val,
        y_val,
        residuals.iloc[val_sl],
        loadings,
        random_state=config.random_state,
    )

    y_test_bin = (y_test.to_numpy() == 1).astype(int)
    test_probability = rca.classifier.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= 0.5).astype(int)
    test_metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test_bin, test_prediction)),
        "average_precision": float(average_precision_score(y_test_bin, test_probability)),
        "roc_auc": float(roc_auc_score(y_test_bin, test_probability))
        if len(np.unique(y_test_bin)) == 2
        else float("nan"),
    }

    timeline = pd.DataFrame(
        {
            "timestamp": timestamp,
            "label": y,
            "is_fail": y == 1,
        }
    )
    timeline = pd.concat(
        [
            timeline,
            fdc_scores.reset_index(drop=True),
            drift.reset_index(drop=True),
        ],
        axis=1,
    )
    timeline["fail_risk"] = rca.classifier.predict_proba(X_scaled)[:, 1]

    metrics = {
        "samples": int(len(X)),
        "raw_signals": int(X.shape[1]),
        "selected_signals": int(len(preprocessor.columns)),
        "pca_components": int(fdc.pca.n_components_),
        "pca_explained_variance": float(fdc.pca.explained_variance_ratio_.sum()),
        "t2_threshold": float(fdc.t2_threshold),
        "q_threshold": float(fdc.q_threshold),
        "fdc_alarm_rate": float(timeline["fdc_alarm"].mean()),
        "fail_rate": float(timeline["is_fail"].mean()),
        "validation_classifier": rca.metrics,
        "classifier": test_metrics,
        "split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
    }

    preprocessor.quality_report.to_csv(output_dir / "data_quality.csv", index=False)
    loadings.to_csv(output_dir / "pca_loading_ranking.csv", index=False)
    rca.ranking.head(config.rca_top_k).to_csv(output_dir / "rca_ranking.csv", index=False)
    timeline.to_csv(output_dir / "timeline.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")
    joblib.dump(fdc, artifacts_dir / "fdc_pca.joblib")
    joblib.dump(rca.classifier, artifacts_dir / "fail_risk_logistic.joblib")

    return metrics
