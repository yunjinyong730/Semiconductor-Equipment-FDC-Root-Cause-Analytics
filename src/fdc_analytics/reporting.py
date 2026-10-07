from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_engineering_report(metrics: dict, rca: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    top = rca.head(5)
    rows = "\n".join(
        f"| {int(r['rank'])} | {r['signal']} | {r['rca_score']:.3f} |" for _, r in top.iterrows()
    )
    text = f"""# One-page Engineering Report

## Semiconductor Equipment Health Analytics

**Objective**  
Detect multivariate equipment/process excursions early and prioritize suspect signals for engineering investigation using the UCI SECOM manufacturing dataset.

## Health Summary

- Samples: **{metrics['samples']}**
- Raw signals: **{metrics['raw_signals']}**
- Monitoring signals after data-quality screening: **{metrics['selected_signals']}**
- PCA components: **{metrics['pca_components']}**
- FDC alarm rate: **{metrics['fdc_alarm_rate']:.1%}**
- Fail rate: **{metrics['fail_rate']:.1%}**
- Fail-risk balanced accuracy: **{metrics['classifier']['balanced_accuracy']:.3f}**
- Fail-risk average precision: **{metrics['classifier']['average_precision']:.3f}**

## Top Root-Cause Candidates

| Rank | Suspect signal | RCA score |
|---:|---|---:|
{rows}

## Recommended Engineering Actions

1. Review the top suspect signals against the same time window as FDC T²/Q excursions.
2. Compare correlated/neighboring process variables before attributing a physical cause.
3. Check whether the suspect ranking is stable across repeated excursions and product/recipe context when available.
4. After corrective action or PM, re-baseline and verify recovery of T², Q, drift score, and fail trend.

> **Interpretation boundary:** SECOM feature names are anonymized. This report prioritizes signals for investigation; it does not claim a physical hardware/process root cause.

> **Synthetic PM experiment:** The PM section is a workflow-validation scenario created by injecting temporary shift/variance changes. It is not a recorded SECOM maintenance event.
"""
    output_path.write_text(text, encoding="utf-8")
