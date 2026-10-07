from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_pm_scenario(
    X_scaled: pd.DataFrame,
    suspect_signals: list[str],
    window: int = 30,
    offset_sigma: float = 0.5,
    variance_multiplier: float = 1.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a clearly labeled synthetic pre/post-PM workflow validation.

    A temporary offset + variance increase is injected into suspect signals.
    After the synthetic PM point, the original baseline data is restored.
    """
    rng = np.random.default_rng(random_state)
    scenario = X_scaled.copy()
    n = len(scenario)
    start = int(n * 0.55)
    pm = int(n * 0.78)
    signals = [s for s in suspect_signals if s in scenario.columns][:5]
    if not signals or pm <= start:
        raise ValueError("Not enough data/signals to build the synthetic PM scenario.")

    segment = scenario.loc[start:pm - 1, signals].to_numpy(copy=True)
    noise = rng.normal(0.0, np.sqrt(max(variance_multiplier - 1.0, 0.0)), size=segment.shape)
    scenario.loc[start:pm - 1, signals] = segment + offset_sigma + noise

    rolling_mean = scenario[signals].rolling(window, min_periods=max(5, window // 3)).mean().abs().median(axis=1)
    rolling_var = scenario[signals].rolling(window, min_periods=max(5, window // 3)).std(ddof=0).median(axis=1)
    score = (rolling_mean + 0.5 * (rolling_var - 1.0).abs()).fillna(0.0)
    phase = np.where(np.arange(n) < start, "baseline", np.where(np.arange(n) < pm, "pre_pm_drift", "post_pm_recovery"))
    return pd.DataFrame({
        "synthetic_pm_score": score,
        "phase": phase,
        "synthetic_pm_event": np.arange(n) == pm,
    }, index=X_scaled.index)
