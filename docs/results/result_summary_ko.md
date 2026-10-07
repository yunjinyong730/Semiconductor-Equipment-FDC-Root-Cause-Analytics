# 실행 결과 요약

- 전체 샘플: 1,567
- 원본 신호 수: 590
- 전처리 후 사용 신호: 321
- PCA component 수: 155
- PCA 설명 분산: 0.950
- 전체 Fail 비율: 6.637%
- 전체 FDC alarm 비율: 28.015%
- Test balanced accuracy: 0.480
- Test average precision: 0.068
- Test ROC-AUC: 0.581

## Top suspect signals

| 순위 | Signal | RCA score |
| ---: | --- | ---: |
| 1 | `signal_059` | 0.5993 |
| 2 | `signal_122` | 0.4716 |
| 3 | `signal_127` | 0.4133 |
| 4 | `signal_301` | 0.4051 |
| 5 | `signal_365` | 0.3906 |

> SECOM 변수는 익명화되어 있으므로 위 결과는 물리적 고장 원인이 아니라 우선 확인할 signal 후보입니다.