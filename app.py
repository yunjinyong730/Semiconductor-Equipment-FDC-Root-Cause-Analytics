from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.fdc_analytics.pipeline import run_pipeline

st.set_page_config(page_title="반도체 장비 FDC & RCA", layout="wide")
st.title("반도체 장비 FDC & Root Cause Analytics")
st.caption("SECOM 공정 데이터를 이용한 정상 상태 모니터링, Drift 탐지, RCA 후보 분석")

OUTPUT_DIR = Path("outputs")
metrics_path = OUTPUT_DIR / "metrics.json"

if not metrics_path.exists():
    st.info("분석 결과가 없습니다. 아래 버튼을 누르면 SECOM 데이터를 받아 분석을 실행합니다.")
    if st.button("분석 실행"):
        with st.spinner("분석 중..."):
            run_pipeline()
        st.rerun()
    st.stop()

if st.sidebar.button("분석 다시 실행"):
    with st.spinner("분석 결과를 갱신하고 있습니다..."):
        run_pipeline()
    st.rerun()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
timeline = pd.read_csv(OUTPUT_DIR / "timeline.csv", parse_dates=["timestamp"])
rca = pd.read_csv(OUTPUT_DIR / "rca_ranking.csv")
quality = pd.read_csv(OUTPUT_DIR / "data_quality.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 Sample", f"{metrics['samples']:,}")
c2.metric("분석 Signal", f"{metrics['selected_signals']}", f"원본 {metrics['raw_signals']}")
c3.metric("FDC Alarm", f"{metrics['fdc_alarm_rate']:.1%}")
c4.metric("Test ROC-AUC", f"{metrics['classifier']['roc_auc']:.3f}")

overview_tab, fdc_tab, drift_tab, rca_tab = st.tabs(
    ["Overview", "FDC", "Drift", "RCA"]
)

with overview_tab:
    st.subheader("분석 결과")
    st.line_chart(
        timeline.set_index("timestamp")[["fdc_score", "drift_score", "fail_risk"]]
    )

    st.subheader("데이터 전처리")
    counts = quality["status"].value_counts().rename_axis("status").to_frame("count")
    st.dataframe(counts, use_container_width=True)

    st.caption(
        "Fail Risk는 보조 지표입니다. SECOM의 Fail 비율이 낮고 시간 순 분할에서 "
        "분류 성능이 제한적이어서 FDC와 RCA 결과를 함께 확인합니다."
    )

with fdc_tab:
    st.subheader("PCA 기반 FDC")
    st.line_chart(timeline.set_index("timestamp")[["t2_ratio", "q_ratio"]])
    st.caption("1을 넘으면 Train Pass baseline의 99 percentile 기준을 초과한 것입니다.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        timeline["t2_ratio"],
        timeline["q_ratio"],
        c=timeline["is_fail"].astype(int),
        alpha=0.55,
        s=12,
    )
    ax.axvline(1.0, linestyle="--")
    ax.axhline(1.0, linestyle="--")
    ax.set_xlabel("T² / threshold")
    ax.set_ylabel("Q / threshold")
    ax.set_title("FDC Operating Space")
    st.pyplot(fig)

with drift_tab:
    st.subheader("Rolling Drift")
    st.line_chart(
        timeline.set_index("timestamp")[
            ["mean_shift_score", "variance_shift_score", "drift_score"]
        ]
    )
    st.caption(
        "PCA loading이 큰 signal을 대상으로 정상 Pass 구간 대비 평균과 분산 변화를 계산했습니다."
    )

with rca_tab:
    st.subheader("Root Cause Candidate")
    columns = [
        "rank",
        "signal",
        "rca_score",
        "evidence_count",
        "evidence_ratio",
        "effect_size",
        "mutual_information",
        "logistic_abs_coef",
        "permutation_importance",
        "fail_residual_contribution",
        "pca_loading_importance",
    ]
    columns = [column for column in columns if column in rca.columns]
    st.dataframe(rca[columns], use_container_width=True)

    st.caption(
        "RCA score는 6개 근거를 각각 0~1로 정규화한 뒤 평균한 값입니다. "
        "evidence_count는 그중 0.5 이상인 근거의 개수입니다."
    )
    st.warning(
        "SECOM 변수명은 익명화되어 있으므로 실제 물리적 원인을 확정할 수 없습니다. "
        "여기서는 우선 확인할 signal을 좁히는 데까지만 사용합니다."
    )
