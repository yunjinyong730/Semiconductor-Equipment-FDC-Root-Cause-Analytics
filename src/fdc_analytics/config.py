from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    missing_ratio_threshold: float = 0.40
    correlation_threshold: float = 0.98
    pca_variance: float = 0.95
    fdc_quantile: float = 0.99
    drift_window: int = 30
    top_monitored_signals: int = 20
    rca_top_k: int = 20
    random_state: int = 42
