from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_drift_score(
    X_scaled: pd.DataFrame,
    baseline_scaled: pd.DataFrame,
    monitored_signals: list[str],
    window: int = 30,
) -> pd.DataFrame:
    signals = [s for s in monitored_signals if s in X_scaled.columns]
    if not signals:
        raise ValueError("No monitored signals are available for drift scoring.")

    base_mean = baseline_scaled[signals].mean()
    base_std = baseline_scaled[signals].std(ddof=0).replace(0, 1.0)
    roll_mean = X_scaled[signals].rolling(window=window, min_periods=max(5, window // 3)).mean()
    roll_std = X_scaled[signals].rolling(window=window, min_periods=max(5, window // 3)).std(ddof=0)

    mean_shift = ((roll_mean - base_mean) / base_std).abs()
    variance_shift = (roll_std / base_std).replace([np.inf, -np.inf], np.nan)
    variance_penalty = np.log(variance_shift.clip(lower=1e-6)).abs()

    out = pd.DataFrame(index=X_scaled.index)
    out["mean_shift_score"] = mean_shift.median(axis=1)
    out["variance_shift_score"] = variance_penalty.median(axis=1)
    out["drift_score"] = out["mean_shift_score"] + 0.5 * out["variance_shift_score"]
    out["drift_warning"] = out["drift_score"] > 1.0
    return out.fillna(0.0)
