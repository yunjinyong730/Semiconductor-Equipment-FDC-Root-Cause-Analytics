# Semiconductor Equipment Health Analytics

## FDC, Drift Detection and Root Cause Analysis for Yield Excursions

Data-driven semiconductor equipment/process health analytics using the UCI SECOM manufacturing dataset. This project translates sensor calibration, drift analysis, and anomaly-detection experience into an engineering workflow centered on multivariate FDC, early warning, yield risk, suspect-signal prioritization, and corrective-action validation.

> Portfolio message: I have worked on improving sensor-data reliability; this project extends that experience to semiconductor process/equipment signals for FDC, drift detection, and root-cause investigation.

## Architecture

~~~text
SECOM process signals
        |
        v
Data Quality + Chronological Split
        |
        v
Pass-only Normal Operating Space
        |
        +--> PCA --> Hotelling T² / SPE(Q) --> FDC alarms
        +--> Rolling mean/variance shift --> Drift early warning
        +--> Balanced Logistic Regression --> Yield-fail risk
        +--> Multi-evidence RCA ranking --> Suspect signals
                                          |
                                          v
                         Synthetic pre/post-PM validation
                                          |
                                          v
                         Streamlit + One-page report
~~~

## Dataset

UCI SECOM contains 1,567 semiconductor manufacturing samples with missing values, timestamps, and Pass/Fail labels. The raw secom.data file used by this repository has 590 process/sensor columns. The current UCI landing-page metadata reports 591 features, so this project reports the raw-file dimensionality explicitly.

- UCI SECOM: https://archive.ics.uci.edu/dataset/179/secom
- DOI: https://doi.org/10.24432/C54305
- Dataset license: CC BY 4.0
- Label convention: -1 = Pass, 1 = Fail
- Raw data is downloaded at runtime and is not committed to this repository.

## Engineering workflow

### 1. Equipment data quality

The pipeline measures missingness and cardinality, removes high-missing and constant signals, median-imputes with training-only statistics, and prunes nearly duplicate signals using a high-correlation threshold. Samples are sorted by their actual SECOM timestamps and split chronologically 60/20/20.

### 2. PCA-based FDC

PCA is fitted only on Pass samples from the training period to define the normal operating space.

- Hotelling T² measures abnormal movement inside the retained PCA subspace.
- SPE/Q measures reconstruction residual outside the modeled normal subspace.
- An FDC alarm occurs when T² or Q exceeds the empirical 99th percentile of the normal Pass baseline.

This is intentionally an engineering monitoring design rather than a black-box classification-only project.

### 3. Drift and early warning

High-loading monitoring signals are tracked with rolling mean shift and variance shift relative to the Pass baseline. This extends sensor-drift experience to equipment/process health monitoring.

### 4. Yield excursion risk

A class-balanced logistic model estimates Fail risk. The model is kept interpretable so coefficients and permutation importance can be used as supporting evidence during investigation.

### 5. Root-cause candidate ranking

SECOM variables are anonymized, so this repository does not claim a physical hardware or process root cause. Instead, it ranks suspect signals using independent evidence:

- standardized Pass/Fail effect size
- mutual information
- absolute logistic coefficient
- permutation importance
- PCA residual contribution on Fail samples
- PCA loading importance

The final RCA score is a prioritization signal for engineering investigation.

### 6. Synthetic pre/post-PM validation

SECOM has no preventive-maintenance event label. A clearly labeled synthetic scenario temporarily injects offset and variance changes into top suspect signals, then restores the original baseline after the synthetic PM point. This validates the before/after analytical workflow without pretending that SECOM contains real PM history.

## Repository structure

~~~text
.
├── app.py
├── run_pipeline.py
├── requirements.txt
├── data/raw/
├── outputs/
├── src/fdc_analytics/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── fdc.py
│   ├── drift.py
│   ├── rca.py
│   ├── pm_scenario.py
│   ├── reporting.py
│   └── pipeline.py
└── tests/
    ├── conftest.py
    └── test_core.py
~~~

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
~~~

On Windows, activate the virtual environment with .venv\Scripts\activate.

The CLI automatically downloads SECOM from UCI if data/raw/secom.data and data/raw/secom_labels.data are absent. If the current UCI zip endpoint is unavailable, the loader also attempts the legacy UCI raw-file endpoint.

## Generated deliverables

After python run_pipeline.py finishes, outputs contains:

- metrics.json: FDC thresholds, split sizes, model metrics
- timeline.csv: timestamped T², Q, drift, fail-risk, health state, and synthetic-PM scores
- data_quality.csv: disposition of raw signals
- pca_loading_ranking.csv: monitoring candidate signals
- rca_ranking.csv: root-cause candidate ranking
- engineering_report.md: one-page engineering summary
- artifacts/*.joblib: fitted preprocessing, FDC/PCA, and fail-risk models

## Dashboard

The Streamlit app is organized around engineering decisions.

1. Equipment Health: Normal / Warning / Critical timeline
2. FDC: T² and Q threshold ratios
3. Drift: rolling mean and variance change
4. Yield / RCA: Fail risk and suspect-signal ranking
5. Synthetic PM: pre/post maintenance workflow validation
6. Engineering Report: downloadable one-page summary

## Engineering interpretation

An anonymized variable such as signal_104 can be ranked as a strong suspect, but it cannot be renamed as chamber pressure, RF power, gas flow, or another physical quantity without external metadata. In a real fab workflow, the next step is to map the signal ID to tool, chamber, recipe, lot, alarm, maintenance, and metrology context; inspect correlated signals; compare neighboring wafers/lots; and verify recovery after corrective action.

## Portfolio wording

> Built an end-to-end semiconductor equipment health analytics pipeline on UCI SECOM data. Established a Pass-only PCA normal operating space, monitored Hotelling T² and SPE/Q for FDC, detected rolling distribution drift, estimated yield-excursion risk, and ranked suspect process signals using multi-evidence root-cause analytics. Added a synthetic pre/post-PM scenario and one-page engineering report while explicitly separating analytical candidates from unverified physical root causes.

## Limitations

- SECOM variables are anonymized, so physical mechanisms cannot be confirmed.
- Each row is a production entity at a test point, not a dense chamber waveform; this is not an OES/plasma trace project.
- FDC thresholds are empirical baseline quantiles, not production control limits qualified for a real tool.
- The PM experiment is synthetic and exists only to validate the analysis workflow.
- Production deployment would require tool, chamber, recipe, lot, alarm, maintenance, and metrology context.

## Validation

The repository includes synthetic regression tests that verify finite PCA/FDC statistics and confirm that the drift score increases after an injected signal shift. GitHub Actions runs compilation and pytest on every push and pull request.

## Citation

McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305
