from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "docs" / "images"
RESULT_DIR = ROOT / "docs" / "results"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fdc(timeline: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = pd.to_datetime(timeline["timestamp"])
    ax.plot(x, timeline["fdc_score"], linewidth=1.1, label="FDC score")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Alarm threshold")
    fail = timeline["is_fail"].astype(bool)
    ax.scatter(x[fail], timeline.loc[fail, "fdc_score"], s=18, marker="x", label="Fail sample")
    ax.set_title("PCA-based FDC monitoring")
    ax.set_ylabel("Normalized FDC score")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    _save(fig, "fdc_timeline.png")


def plot_drift_and_risk(timeline: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 4.6))
    x = pd.to_datetime(timeline["timestamp"])
    ax1.plot(x, timeline["drift_score"], linewidth=1.1, label="Drift score")
    ax1.axhline(1.0, linestyle="--", linewidth=1.0, label="Drift warning")
    ax1.set_ylabel("Drift score")
    ax1.set_xlabel("Time")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, timeline["fail_risk"], linewidth=1.0, alpha=0.75, label="Fail risk")
    ax2.set_ylabel("Estimated fail risk")
    ax2.set_ylim(0, 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("Distribution drift and yield-fail risk")
    _save(fig, "drift_fail_risk.png")


def plot_rca(rca: pd.DataFrame) -> None:
    top = rca.head(10).sort_values("rca_score")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(top["signal"], top["rca_score"])
    ax.set_title("Top 10 suspect signals by RCA score")
    ax.set_xlabel("Combined RCA score")
    ax.set_ylabel("Anonymized SECOM signal")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, "rca_top_signals.png")


def plot_pm(timeline: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = pd.to_datetime(timeline["timestamp"])
    ax.plot(x, timeline["synthetic_pm_score"], linewidth=1.1, label="Synthetic PM health score")

    pre_mask = timeline["phase"].eq("pre_pm_drift")
    if pre_mask.any():
        ax.axvspan(x[pre_mask].iloc[0], x[pre_mask].iloc[-1], alpha=0.12, label="Injected pre-PM drift")

    pm_mask = timeline["synthetic_pm_event"].astype(bool)
    if pm_mask.any():
        pm_time = x[pm_mask].iloc[0]
        ax.axvline(pm_time, linestyle="--", linewidth=1.1, label="Synthetic PM point")

    ax.set_title("Synthetic pre/post-PM workflow validation")
    ax.set_ylabel("Synthetic health score")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    _save(fig, "synthetic_pm.png")


def build_summary(metrics: dict, rca: pd.DataFrame) -> None:
    top = rca.head(5)[["rank", "signal", "rca_score"]].copy()
    rows = ["| 순위 | Signal | RCA score |", "| ---: | --- | ---: |"]
    rows.extend(
        f"| {int(row.rank)} | `{row.signal}` | {row.rca_score:.4f} |"
        for row in top.itertuples(index=False)
    )
    lines = [
        "# 실행 결과 요약",
        "",
        f"- 전체 샘플: {metrics['samples']:,}",
        f"- 원본 신호 수: {metrics['raw_signals']:,}",
        f"- 전처리 후 사용 신호: {metrics['selected_signals']:,}",
        f"- PCA component 수: {metrics['pca_components']:,}",
        f"- PCA 설명 분산: {metrics['pca_explained_variance']:.3f}",
        f"- 전체 Fail 비율: {metrics['fail_rate']:.3%}",
        f"- 전체 FDC alarm 비율: {metrics['fdc_alarm_rate']:.3%}",
        f"- Test balanced accuracy: {metrics['classifier']['balanced_accuracy']:.3f}",
        f"- Test average precision: {metrics['classifier']['average_precision']:.3f}",
        f"- Test ROC-AUC: {metrics['classifier']['roc_auc']:.3f}",
        "",
        "## Top suspect signals",
        "",
        *rows,
        "",
        "> SECOM 변수는 익명화되어 있으므로 위 결과는 물리적 고장 원인이 아니라 우선 확인할 signal 후보입니다.",
    ]
    (RESULT_DIR / "result_summary_ko.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rca.head(20).to_csv(RESULT_DIR / "rca_top20.csv", index=False)


def main() -> None:
    timeline = pd.read_csv(OUTPUT_DIR / "timeline.csv")
    rca = pd.read_csv(OUTPUT_DIR / "rca_ranking.csv")
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    plot_fdc(timeline)
    plot_drift_and_risk(timeline)
    plot_rca(rca)
    plot_pm(timeline)
    build_summary(metrics, rca)
    print(f"Saved README figures to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
