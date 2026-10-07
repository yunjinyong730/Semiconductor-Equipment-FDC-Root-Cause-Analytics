from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RCA_PATH = ROOT / "outputs" / "rca_ranking.csv"
OUTPUT_PATH = ROOT / "docs" / "images" / "rca_evidence_matrix.svg"

EVIDENCE_COLUMNS = [
    ("effect_size_norm", "Effect size"),
    ("mutual_information_norm", "MI"),
    ("logistic_abs_coef_norm", "Logit coef"),
    ("permutation_importance_norm", "Permutation"),
    ("fail_residual_contribution_norm", "PCA residual"),
    ("pca_loading_importance_norm", "PCA loading"),
]


def _color(value: float) -> str:
    low = (244, 247, 251)
    high = (32, 84, 147)
    value = max(0.0, min(1.0, float(value)))
    rgb = tuple(round(a + value * (b - a)) for a, b in zip(low, high))
    return "#%02x%02x%02x" % rgb


def build_svg(rca: pd.DataFrame, top_k: int = 10) -> str:
    top = rca.head(top_k).copy()
    width, height = 1040, 650
    left, top_y = 150, 100
    cell_w, cell_h = 130, 46

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR","Malgun Gothic",sans-serif;fill:#1f2328}.title{font-size:24px;font-weight:700}.label{font-size:15px}.cell{font-size:13px;font-weight:600}</style>',
        f'<text x="{width/2}" y="42" text-anchor="middle" class="title">RCA 근거 교차검증</text>',
        f'<text x="{width/2}" y="70" text-anchor="middle" class="label">같은 signal을 여러 분석이 같이 지목하는지 확인</text>',
    ]

    for j, (_, label) in enumerate(EVIDENCE_COLUMNS):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="{top_y-18}" text-anchor="middle" class="label">{label}</text>')

    for i, (_, row) in enumerate(top.iterrows()):
        y = top_y + i * cell_h
        parts.append(f'<text x="{left-18}" y="{y+cell_h/2+5}" text-anchor="end" class="label">{row["signal"]}</text>')
        for j, (column, _) in enumerate(EVIDENCE_COLUMNS):
            value = float(row[column])
            x = left + j * cell_w
            text_color = "#ffffff" if value > 0.62 else "#1f2328"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" rx="4" fill="{_color(value)}"/>')
            parts.append(f'<text x="{x+cell_w/2-1}" y="{y+cell_h/2+5}" text-anchor="middle" class="cell" style="fill:{text_color}">{value:.2f}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    if not RCA_PATH.exists():
        raise FileNotFoundError("outputs/rca_ranking.csv가 없습니다. 먼저 python run_pipeline.py를 실행하세요.")
    rca = pd.read_csv(RCA_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_svg(rca), encoding="utf-8")
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
