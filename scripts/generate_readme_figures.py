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

EVIDENCE_COLUMNS = [
    ("effect_size_norm", "Effect size"),
    ("mutual_information_norm", "MI"),
    ("logistic_abs_coef_norm", "Logit coef"),
    ("permutation_importance_norm", "Permutation"),
    ("fail_residual_contribution_norm", "PCA residual"),
    ("pca_loading_importance_norm", "PCA loading"),
]


def _save(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fdc(timeline: pd.DataFrame) -> None:
    x = pd.to_datetime(timeline["timestamp"])
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(x, timeline["fdc_score"], linewidth=1.1, label="FDC score")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Threshold")

    fail = timeline["is_fail"].astype(bool)
    ax.scatter(
        x[fail],
        timeline.loc[fail, "fdc_score"],
        s=18,
        marker="x",
        label="Fail sample",
    )
    ax.set_title("PCA-based FDC")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized score")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    _save(fig, "fdc_timeline.png")


def plot_drift(timeline: pd.DataFrame) -> None:
    x = pd.to_datetime(timeline["timestamp"])
    fig, ax1 = plt.subplots(figsize=(12, 4.6))
    ax1.plot(x, timeline["drift_score"], linewidth=1.1, label="Drift score")
    ax1.axhline(1.0, linestyle="--", linewidth=1.0, label="Drift warning")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Drift score")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, timeline["fail_risk"], linewidth=1.0, alpha=0.7, label="Fail risk")
    ax2.set_ylabel("Fail risk")
    ax2.set_ylim(0, 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("Drift and Fail Risk")
    _save(fig, "drift_fail_risk.png")


def plot_rca(rca: pd.DataFrame) -> None:
    top = rca.head(10).sort_values("rca_score")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(top["signal"], top["rca_score"])
    ax.set_title("Top 10 RCA candidates")
    ax.set_xlabel("RCA score")
    ax.set_ylabel("SECOM signal")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, "rca_top_signals.png")


def _cell_color(value: float) -> str:
    low = (244, 247, 251)
    high = (32, 84, 147)
    value = max(0.0, min(1.0, float(value)))
    rgb = tuple(round(a + value * (b - a)) for a, b in zip(low, high))
    return "#%02x%02x%02x" % rgb


def write_evidence_matrix(rca: pd.DataFrame) -> None:
    top = rca.head(10)
    width, height = 1040, 650
    left, top_y = 150, 100
    cell_w, cell_h = 130, 46

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#1f2328}.title{font-size:24px;font-weight:700}.label{font-size:15px}.cell{font-size:13px;font-weight:600}</style>',
        f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">RCA evidence matrix</text>',
    ]

    for j, (_, label) in enumerate(EVIDENCE_COLUMNS):
        x = left + j * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top_y - 18}" text-anchor="middle" class="label">{label}</text>'
        )

    for i, (_, row) in enumerate(top.iterrows()):
        y = top_y + i * cell_h
        parts.append(
            f'<text x="{left - 18}" y="{y + cell_h / 2 + 5}" text-anchor="end" class="label">{row["signal"]}</text>'
        )
        for j, (column, _) in enumerate(EVIDENCE_COLUMNS):
            value = float(row[column])
            x = left + j * cell_w
            text_color = "#ffffff" if value > 0.62 else "#1f2328"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" rx="4" fill="{_cell_color(value)}"/>'
            )
            parts.append(
                f'<text x="{x + cell_w / 2 - 1}" y="{y + cell_h / 2 + 5}" text-anchor="middle" class="cell" style="fill:{text_color}">{value:.2f}</text>'
            )

    parts.append("</svg>")
    (IMAGE_DIR / "rca_evidence_matrix.svg").write_text(
        "\n".join(parts),
        encoding="utf-8",
    )


def save_result_snapshot(metrics: dict, rca: pd.DataFrame) -> None:
    (RESULT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    rca.head(20).to_csv(RESULT_DIR / "rca_top20.csv", index=False)


def main() -> None:
    timeline = pd.read_csv(OUTPUT_DIR / "timeline.csv")
    rca = pd.read_csv(OUTPUT_DIR / "rca_ranking.csv")
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))

    plot_fdc(timeline)
    plot_drift(timeline)
    plot_rca(rca)
    write_evidence_matrix(rca)
    save_result_snapshot(metrics, rca)


if __name__ == "__main__":
    main()
