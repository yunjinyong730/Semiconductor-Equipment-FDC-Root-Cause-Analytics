from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_engineering_report(metrics: dict, rca: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    top = rca.head(5)
    rows = "\n".join(
        f"| {int(r['rank'])} | {r['signal']} | {r['rca_score']:.3f} | {int(r.get('evidence_count', 0))}/6 |"
        for _, r in top.iterrows()
    )

    text = f"""# 반도체 장비 상태 분석 요약

## 목적

UCI SECOM 공정 데이터를 이용해 정상 상태에서 벗어나는 다변량 이상을 찾고, Fail과 함께 확인해야 할 signal 후보를 우선순위화했습니다.

## 현재 상태

- 전체 sample: {metrics['samples']:,}
- 원본 signal: {metrics['raw_signals']:,}
- 전처리 후 사용 signal: {metrics['selected_signals']:,}
- PCA component: {metrics['pca_components']:,}
- FDC alarm rate: {metrics['fdc_alarm_rate']:.1%}
- Fail rate: {metrics['fail_rate']:.1%}
- Test balanced accuracy: {metrics['classifier']['balanced_accuracy']:.3f}
- Test average precision: {metrics['classifier']['average_precision']:.3f}

## 우선 확인할 signal

| 순위 | Signal | RCA score | 근거 일치 수 |
| ---: | --- | ---: | ---: |
{rows}

## 확인 순서

1. FDC T²/Q가 상승한 구간에서 상위 signal이 실제로 같이 변했는지 확인합니다.
2. 한 개의 중요도 결과만 보지 않고 effect size, MI, regression coefficient, permutation importance, PCA residual/loading이 같은 방향을 지목하는지 같이 봅니다.
3. 실제 장비 데이터라면 Tool, Chamber, Recipe, Alarm, PM history와 연결해 물리적인 원인을 확인합니다.
4. 조치나 PM 이후에는 같은 baseline 기준으로 T², Q, drift score가 회복됐는지 다시 확인합니다.

> SECOM의 변수명은 익명화되어 있습니다. 여기서 제시하는 결과는 실제 고장 원인의 확정값이 아니라, 엔지니어가 먼저 확인할 signal 후보입니다.

> PM 분석은 SECOM의 실제 유지보수 이력이 아니라 분석 흐름을 검증하기 위해 만든 synthetic scenario입니다.
"""
    output_path.write_text(text, encoding="utf-8")
