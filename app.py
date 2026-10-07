from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.fdc_analytics.pipeline import run_pipeline

st.set_page_config(page_title="반도체 장비 FDC & RCA", layout="wide")
st.title("반도체 장비 FDC & Root Cause Analytics")
st.caption("PCA 기반 FDC, Drift Detection, Yield Fail Risk, Root Cause Candidate 분석")

OUT = Path("outputs")
metrics_path = OUT / "metrics.json"
if not metrics_path.exists():
    st.info("No analysis outputs found. Run the pipeline to download SECOM and build the analytics artifacts.")
    if st.button("Run analysis pipeline"):
        with st.spinner("Running SECOM analytics..."):
            run_pipeline()
        st.rerun()
    st.stop()

if st.sidebar.button("Re-run analysis"):
    with st.spinner("Refreshing analytics..."):
        run_pipeline()
    st.rerun()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
timeline = pd.read_csv(OUT / "timeline.csv", parse_dates=["timestamp"])
rca = pd.read_csv(OUT / "rca_ranking.csv")
quality = pd.read_csv(OUT / "data_quality.csv")
report = (OUT / "engineering_report.md").read_text(encoding="utf-8")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{metrics['samples']:,}")
c2.metric("Monitoring signals", f"{metrics['selected_signals']}", f"from {metrics['raw_signals']}")
c3.metric("FDC alarm rate", f"{metrics['fdc_alarm_rate']:.1%}")
c4.metric("Test AP", f"{metrics['classifier']['average_precision']:.3f}")

tabs = st.tabs(["장비 상태", "FDC", "Drift", "Yield / RCA", "Synthetic PM", "분석 요약"])

with tabs[0]:
    st.subheader("장비 상태 Timeline")
    st.line_chart(timeline.set_index("timestamp")[["fdc_score", "drift_score", "fail_risk"]])
    st.dataframe(timeline[["timestamp", "health_state", "fdc_alarm", "drift_warning", "fail_risk", "is_fail"]].tail(30), use_container_width=True)

with tabs[1]:
    st.subheader("Multivariate FDC: Hotelling T² and SPE/Q")
    chart = timeline.set_index("timestamp")[["t2_ratio", "q_ratio"]]
    st.line_chart(chart)
    st.caption("Ratio > 1 means the empirical 99th-percentile normal-baseline threshold is exceeded.")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(timeline["t2_ratio"], timeline["q_ratio"], c=timeline["is_fail"].astype(int), alpha=0.55, s=12)
    ax.axvline(1.0, linestyle="--")
    ax.axhline(1.0, linestyle="--")
    ax.set_xlabel("T² / threshold")
    ax.set_ylabel("Q / threshold")
    ax.set_title("FDC operating-space view (fail samples highlighted by class value)")
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Rolling drift indicators")
    st.line_chart(timeline.set_index("timestamp")[["mean_shift_score", "variance_shift_score", "drift_score"]])
    st.caption("Drift is computed from rolling mean/variance changes of high-loading monitoring signals relative to the Pass baseline.")

with tabs[3]:
    st.subheader("Yield Fail Risk와 RCA 후보")
    st.line_chart(timeline.set_index("timestamp")[["fail_risk"]])
    rca_cols = [
        "rank", "signal", "rca_score", "evidence_count", "evidence_agreement",
        "effect_size", "mutual_information", "logistic_abs_coef", "fail_residual_contribution"
    ]
    rca_cols = [col for col in rca_cols if col in rca.columns]
    st.dataframe(rca[rca_cols], use_container_width=True)
    st.warning("SECOM 변수는 익명화되어 있습니다. RCA 결과는 실제 고장 원인의 확정값이 아니라 먼저 확인할 signal 후보입니다.")
    if "evidence_count" in rca.columns:
        st.caption("evidence_count는 서로 다른 6개 RCA 근거 중 정규화 값이 0.5 이상인 항목 수입니다. 확률값이 아니라 교차 확인용 지표입니다.")
    st.subheader("Data-quality disposition")
    st.dataframe(quality["status"].value_counts().rename_axis("status").to_frame("count"), use_container_width=True)

with tabs[4]:
    st.subheader("Synthetic pre/post-PM workflow validation")
    st.line_chart(timeline.set_index("timestamp")[["synthetic_pm_score"]])
    pm_rows = timeline[timeline["synthetic_pm_event"]]
    if not pm_rows.empty:
        st.write(f"Synthetic PM 시점: {pm_rows.iloc[0]['timestamp']}")
    st.info("Temporary offset/variance changes are injected only to validate the analytical workflow. This is not a recorded SECOM PM event.")

with tabs[5]:
    st.markdown(report)
    st.download_button("분석 요약 다운로드", report, file_name="engineering_report.md")
