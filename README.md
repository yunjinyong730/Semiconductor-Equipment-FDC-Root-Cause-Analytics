# Semiconductor Equipment FDC & Root Cause Analytics

반도체 공정 센서 데이터를 이용해 정상 상태에서 벗어나는 변화를 찾고, Yield Fail이 발생했을 때 어떤 signal을 먼저 확인해야 하는지 좁혀가는 프로젝트입니다.

새로운 반도체 ML 모델을 하나 더 만드는 것보다는, 제가 석사과정에서 계속 다뤄 온 Sensor Calibration, Drift, Distribution Shift 문제를 반도체 장비 분석 관점으로 옮겨보는 쪽이 더 의미 있다고 생각했습니다.

기존 연구에서는 센서 값이 기준값에서 얼마나 벗어났는지와 시간이 지나거나 측정 환경이 바뀌었을 때 보정 성능이 어떻게 달라지는지를 봤다면, 이번 프로젝트에서는 그 흐름을 다음과 같이 확장했습니다.

```text
Sensor Reliability / Calibration
            ↓
Drift & Distribution Shift
            ↓
Normal Operating Space
            ↓
PCA-based FDC
            ↓
Yield Fail Risk
            ↓
Root Cause Candidate
            ↓
Engineering Check Priority
```

## 프로젝트를 만든 이유

제가 해 온 연구와 반도체 장비 데이터 분석은 사용하는 데이터와 목적은 다르지만, 문제를 보는 방식은 꽤 비슷하다고 생각했습니다.

센서 보정 연구에서는 현재 측정값이 기존 상태와 얼마나 달라졌는지, 그 변화가 일시적인 노이즈인지 장기적인 drift인지 구분하는 과정이 중요했습니다. 반도체 장비에서도 정상 상태를 먼저 정의하고 여러 공정 signal이 그 상태에서 얼마나 벗어났는지를 보는 과정이 필요합니다.

다만 이번에는 분석의 끝을 조금 다르게 잡았습니다. 이상을 찾는 것에서 끝내지 않고, 실제 엔지니어 입장에서 다음에 무엇을 확인할지까지 이어지도록 구성했습니다.

| 기존 연구에서 다룬 문제 | 이번 프로젝트에서 옮긴 방식 |
| --- | --- |
| Sensor Calibration | Equipment Health Monitoring |
| 시간에 따른 Sensor Drift | Process Drift / Early Warning |
| 위치 및 환경 변화에 따른 Distribution Shift | Normal Operating Space 이탈 탐지 |
| Reference 대비 보정 오차 | PCA T² / SPE(Q) 기반 FDC |
| 여러 센서의 변화 비교 | Root Cause Candidate Ranking |
| 모델 성능 및 Edge 적용 | Engineer가 확인할 signal 우선순위 |

제가 기존 연구에서 계속 다뤄 온 핵심은 결국 센서 값 자체를 그대로 믿는 것이 아니라, 기준 상태와 비교해서 언제부터 달라졌고 어떤 변화가 중요한지를 판단하는 일이었습니다. 이 프로젝트는 그 경험을 반도체 공정 데이터에 적용해 본 것입니다.

## Dataset

[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) 데이터를 사용했습니다.

- Sample: 1,567
- Process / Sensor Signal: 590
- Pass: 1,463
- Fail: 104
- Fail 비율: 6.64%
- Timestamp 포함
- 결측치 포함
- Label: `-1 = Pass`, `1 = Fail`

SECOM의 변수명은 모두 익명화되어 있습니다. 따라서 `signal_059`가 압력, 온도, RF Power와 같은 특정 물리량이라고 임의로 해석하지 않았습니다.

이 프로젝트에서 Root Cause 결과는 실제 고장 원인을 확정하는 값이 아니라, 분석 결과상 먼저 확인할 suspect signal 후보입니다.

## 전체 분석 흐름

```text
Raw SECOM Signals
        ↓
Data Quality Check
        ↓
Chronological Split
        ↓
Pass-only Normal Baseline
        ↓
PCA FDC ── Drift Detection
        ↓
Yield Fail Risk
        ↓
RCA Candidate Ranking
        ↓
Multi-evidence Cross Check
        ↓
Engineering Action
```

Random split으로 성능을 높여 보이기보다 실제 운영 상황에 조금 더 가깝게 보려고 시간 순서대로 Train, Validation, Test를 나눴습니다. 이후 시점의 정보가 정상 baseline을 만드는 데 들어가지 않도록 한 것이 중요하다고 봤습니다.

## 1. Data Quality & Preprocessing

590개 signal을 바로 모델에 넣지 않고 먼저 데이터 자체를 확인했습니다.

- Missing ratio가 높은 signal 제거
- Constant signal 제거
- Train 구간 median으로 결측치 보간
- 상관관계가 지나치게 높은 중복 signal 제거
- Standard Scaling
- 시간 순서 기준 Train / Validation / Test 분할

최종적으로 590개 중 321개 signal을 분석에 사용했습니다.

## 2. PCA-based FDC

Train 구간의 Pass sample만 이용해 Normal Operating Space를 구성했습니다.

PCA는 누적 설명 분산 95%를 기준으로 155개 component를 사용했고, FDC 지표는 Hotelling T²와 SPE(Q)를 함께 사용했습니다.

- Hotelling T²: 기존 정상 PCA 공간 안에서 평소와 다른 방향으로 이동했는지 확인
- SPE(Q): 기존 정상 PCA 공간으로 설명되지 않는 residual이 얼마나 커졌는지 확인

정상 Train 구간의 99 percentile을 threshold로 두고 T² 또는 Q가 기준을 넘으면 FDC alarm으로 처리했습니다.

![PCA 기반 FDC 결과](docs/images/fdc_timeline.png)

전체 구간의 FDC alarm 비율은 28.0%였습니다.

이 값을 Fail 적중률로 해석하지는 않았습니다. Train 초반 구간으로 만든 정상 baseline과 이후 데이터 사이에 상태 변화가 꽤 있다는 의미로 보고, Drift 결과와 같이 확인했습니다.

## 3. Drift Detection

이 부분은 제가 해 온 센서 보정 연구와 가장 직접적으로 연결됩니다.

기존 연구에서도 시간이 지나면서 센서의 평균값이 이동하거나 분산이 달라지는 문제가 보정 성능에 영향을 줬기 때문에, 단순 point anomaly보다 지속적인 분포 변화를 중요하게 봤습니다.

이번에는 PCA loading이 큰 주요 signal을 대상으로 정상 Pass baseline과 비교해 rolling mean shift와 variance shift를 계산했습니다.

```text
Normal Pass Baseline
        ↓
Rolling Mean Shift
Rolling Variance Shift
        ↓
Drift Score
        ↓
Warning Candidate
```

![Drift 및 Fail Risk 결과](docs/images/drift_fail_risk.png)

센서 보정에서는 drift가 calibration error를 키우는 원인이었다면, 여기서는 장비 또는 공정 상태가 기존 정상 조건에서 벗어나기 시작했다는 신호로 봤습니다.

## 4. Yield Fail Risk

Fail sample이 전체의 6.64%로 적기 때문에 class-balanced Logistic Regression을 사용했습니다.

복잡한 모델을 추가해서 Accuracy를 높이는 것보다, 어떤 signal이 Fail 판단에 영향을 줬는지 이후 RCA에서 같이 확인할 수 있는 구조를 우선했습니다.

| Metric | Test |
| --- | ---: |
| Balanced Accuracy | 0.480 |
| Average Precision | 0.068 |
| ROC-AUC | 0.581 |

분류 성능 자체는 높지 않았습니다.

오히려 이 결과 때문에 Fail 여부를 맞히는 모델 하나에 프로젝트를 기대면 안 된다고 판단했습니다. 특히 시간 순서대로 분할했을 때 성능이 제한적이었기 때문에, Yield Fail Risk는 최종 판단값이 아니라 FDC와 RCA를 보조하는 값으로 사용했습니다.

랜덤 분할을 사용하면 더 좋은 숫자가 나올 수 있지만, 실제 적용 상황을 생각하면 현재 결과를 그대로 보여주는 편이 맞다고 봤습니다.

## 5. Root Cause Candidate Analysis

이 프로젝트에서 가장 중요하게 본 부분입니다.

이상이 발생했다는 것만 보여주면 실제 점검으로 이어지기 어렵기 때문에, Fail과 함께 변화한 signal 중 무엇을 먼저 확인할지 우선순위를 만들었습니다.

각 signal에 대해 다음 값을 같이 사용했습니다.

- Pass / Fail Effect Size
- Mutual Information
- Logistic Regression Coefficient
- Permutation Importance
- PCA Residual Contribution
- PCA Loading Importance

![Root Cause Candidate 결과](docs/images/rca_top_signals.png)

현재 실행 결과의 상위 후보는 다음과 같습니다.

| Rank | Signal | RCA Score |
| ---: | --- | ---: |
| 1 | `signal_059` | 0.5993 |
| 2 | `signal_122` | 0.4716 |
| 3 | `signal_127` | 0.4133 |
| 4 | `signal_301` | 0.4051 |
| 5 | `signal_365` | 0.3906 |

이 결과 역시 `signal_059`가 실제 장비 고장의 원인이라는 의미는 아닙니다.

실제 현장 데이터라면 해당 signal이 어느 Tool, Chamber, Recipe, Alarm, PM 이력과 연결되는지를 확인한 뒤 물리적인 원인을 판단해야 합니다. 여기서는 그 전 단계인 분석상 확인 우선순위를 만드는 것까지 구현했습니다.

## 6. 제가 한 단계 더 넣은 부분: Multi-evidence RCA

일반적인 프로젝트처럼 feature importance 하나만 뽑아서 Root Cause라고 적고 싶지는 않았습니다.

센서 데이터를 연구하면서 느낀 점은 하나의 지표만 보면 환경 변화, sensor drift, 순간적인 이상을 서로 잘못 해석할 수 있다는 것이었습니다. 그래서 이번 RCA에서도 한 가지 방법이 높게 나온 signal보다 서로 다른 분석이 반복해서 지목하는 signal을 더 주의해서 보도록 만들었습니다.

6개의 RCA 근거를 0~1 범위로 정규화하고, 각 근거가 0.5 이상인 경우를 하나의 strong evidence로 계산했습니다. `evidence_count`는 이 여섯 가지 분석 중 몇 개가 같은 signal을 비교적 강하게 지목했는지를 나타냅니다.

![RCA 근거 교차검증](docs/images/rca_evidence_matrix.svg)

이 값은 통계적인 confidence probability가 아닙니다. 제가 보기 편하게 만든 교차 확인 지표입니다.

예를 들어 RCA score가 높더라도 한 가지 분석 결과에만 크게 의존한다면 바로 원인 후보로 확정하지 않고, 다른 근거와 실제 장비 이력을 더 확인하도록 했습니다. 반대로 Effect Size, PCA Residual, Logistic Coefficient처럼 서로 다른 관점의 결과가 같이 올라오면 점검 우선순위를 높게 보는 방식입니다.

이 부분이 제가 해 온 센서 연구 경험을 이번 프로젝트에 가장 직접적으로 넣은 포인트입니다. 센서 값 하나를 그대로 믿기보다 여러 관점에서 같은 변화가 반복해서 확인되는지를 보는 습관을 RCA에 그대로 가져왔습니다.

## 7. Synthetic PM Scenario

SECOM에는 실제 Preventive Maintenance 시점 정보가 없습니다.

없는 PM 이력을 실제 데이터처럼 만들 수는 없기 때문에, 분석 방법을 검증하기 위한 Synthetic Scenario를 별도로 만들었습니다.

상위 suspect signal 일부에 일정 기간 offset과 variance 변화를 주고, PM을 수행했다고 가정한 시점 이후 원래 baseline으로 돌아오도록 구성했습니다.

```text
Baseline
   ↓
Offset / Variance Drift 주입
   ↓
Synthetic PM
   ↓
Baseline Recovery
```

![Synthetic PM 결과](docs/images/synthetic_pm.png)

이 실험으로 PM 효과를 주장하는 것은 아닙니다. 실제 장비 데이터에 PM 이력이 주어졌을 때 PM 전후 상태를 어떤 지표로 비교할지 분석 흐름을 먼저 구현해 본 것입니다.

## 현재 결과 정리

| 항목 | 결과 |
| --- | ---: |
| 전체 Sample | 1,567 |
| Raw Signals | 590 |
| 전처리 후 Signals | 321 |
| PCA Components | 155 |
| PCA Explained Variance | 95.0% |
| Fail Rate | 6.64% |
| FDC Alarm Rate | 28.0% |
| Test ROC-AUC | 0.581 |
| Top RCA Candidate | `signal_059` |

결과를 보면서 가장 중요하게 본 것은 높은 분류 성능을 만드는 것보다 정상 상태가 언제 달라지는지와 이상 발생 후 어떤 signal을 먼저 확인할지를 한 흐름으로 연결하는 것이었습니다.

그래서 이 프로젝트의 결과물도 모델 파일 하나가 아니라 FDC, Drift, RCA, PM 전후 확인, Engineering Report가 같이 나오도록 구성했습니다.

## Dashboard

Streamlit에서 주요 결과를 한 화면에서 볼 수 있도록 정리했습니다.

```bash
streamlit run app.py
```

Dashboard에서는 다음 내용을 확인할 수 있습니다.

- 장비 상태 Timeline
- Hotelling T² / SPE(Q)
- Drift Score
- Yield Fail Risk
- RCA Candidate
- Multi-evidence RCA
- Synthetic PM Scenario
- Engineering Report

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py
python scripts/generate_readme_figures.py
python scripts/generate_rca_evidence.py
streamlit run app.py
```

`data/raw/`에 SECOM 데이터가 없으면 `run_pipeline.py` 실행 시 UCI에서 자동으로 다운로드합니다.

분석을 다시 실행하면 `outputs/`에 다음 결과가 생성됩니다.

```text
outputs/
├── data_quality.csv
├── timeline.csv
├── pca_loading_ranking.csv
├── rca_ranking.csv
├── metrics.json
├── engineering_report.md
└── artifacts/
```

README에 사용한 결과 이미지와 요약 파일은 아래 명령으로 다시 만들 수 있습니다.

```bash
python scripts/generate_readme_figures.py
python scripts/generate_rca_evidence.py
```

## Repository Structure

```text
.
├── app.py
├── run_pipeline.py
├── scripts/
│   ├── generate_readme_figures.py
│   └── generate_rca_evidence.py
├── data/
│   └── raw/
├── docs/
│   ├── images/
│   └── results/
├── outputs/
├── src/
│   └── fdc_analytics/
│       ├── data.py
│       ├── preprocessing.py
│       ├── fdc.py
│       ├── drift.py
│       ├── rca.py
│       ├── pm_scenario.py
│       ├── reporting.py
│       └── pipeline.py
└── tests/
    └── test_core.py
```

## 실제 장비 데이터가 있다면 다음으로 해보고 싶은 것

SECOM은 익명화된 정적 공정 데이터라 여기까지가 가능한 범위라고 봤습니다.

실제 Lam 장비 데이터처럼 Tool, Chamber, Recipe, Alarm, PM, OES, Wafer sequence가 같이 있다면 다음 단계에서는 동일한 분석을 장비 단위로 확장하고 싶습니다.

1. Chamber별 Normal Operating Space 분리
2. Recipe별 baseline 비교
3. PM 전후 recovery 검증
4. Alarm과 FDC excursion 동시 분석
5. OES spectrum과 sensor signal의 시점별 이상 연계
6. 반복적으로 나타나는 suspect signal과 실제 corrective action 결과 비교

현재 프로젝트에서는 실제로 없는 정보를 가정해서 채우기보다, 공개 데이터에서 확인할 수 있는 범위와 확인할 수 없는 범위를 구분하는 데 신경 썼습니다.

## Limitations

- SECOM signal은 익명화되어 있어 실제 공정 변수의 물리적 의미를 알 수 없습니다.
- 실제 장비의 고주파 waveform이나 OES spectrum 데이터가 아닙니다.
- FDC threshold는 본 데이터의 Train Pass baseline에서 설정한 값이며 실제 Fab 관리 기준이 아닙니다.
- Yield Fail 분류 성능이 제한적이어서 Fail Risk는 보조 지표로 사용했습니다.
- `evidence_count`는 교차 확인을 위한 heuristic이며 통계적인 신뢰도 값이 아닙니다.
- PM 분석은 실제 유지보수 이력이 아니라 workflow 검증용 synthetic scenario입니다.
- 실제 적용에서는 Tool, Chamber, Recipe, Lot, Alarm, PM, Metrology 정보가 추가로 필요합니다.

## Dataset Citation

McCann, M. & Johnston, A. (2008). SECOM. UCI Machine Learning Repository.  
DOI: `10.24432/C54305`