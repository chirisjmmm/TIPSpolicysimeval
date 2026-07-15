# Macro-Accuracy 리포트 — MAPE / Calibration (§7.3)

> `mape_per_scenario = mean_s(|ŷ(s)−gt|/|gt|)` (시나리오별 개별 정확도) · `mape_aggregate = |mean_s(ŷ(s))−gt|/|gt|` (논문 Eq.7 정책수준 점추정, **Table 1 대응**).

## 1. 전체 결과 (세 set)

| set | target | GT | MAPE(agg, 논문식) | MAPE(per-scenario) | mean_est | bias% | coverage[p10,p90] | sharpness |
|---|---|---:|---:|---:|---:|---:|:--:|---:|
| 100E_gemini | approved_projects | 115 | **0.98** | 30.91 | 113.87 | -1.0 | ✅ | 114.5 |
| 100E_gemini | deployed_projects | 73 | **5.95** | 39.87 | 77.34 | +5.9 | ✅ | 80.9 |
| 100E_deepseek | approved_projects | 115 | **32.44** | 32.44 | 77.69 | -32.4 | ❌ | 59.4 |
| 100E_deepseek | deployed_projects | 73 | **41.72** | 41.72 | 42.55 | -41.7 | ❌ | 40.0 |
| BK21_deepseek | employment_rate | 82.2 | **0.85** | 4.00 | 82.90 | +0.9 | ✅ | 8.6 |
| BK21_deepseek | intl_collaboration_rate | 36.0 | **4.33** | 32.11 | 34.44 | -4.3 | ✅ | 30.1 |

## 2. 싱가포르 100E 백본 모델 성능차이 (DeepSeek vs Gemini) — 논문 Table 1 대비

논문 값은 CIKM26 Table 1의 full-TIPS 행, 100E=Policy A(ŷ1_A/ŷ2_A). 우리 파이프라인의 **aggregate MAPE**가 논문 정의와 일치하며 값이 재현된다.

| target | GT | Gemini agg-MAPE (논문) | DeepSeek agg-MAPE (논문) | Gemini per-scen | DeepSeek per-scen |
|---|---:|---:|---:|---:|---:|
| approved_projects (ŷ1_A) | 115 | **0.98** (2.87) | **32.44** (32.61) | 30.91 | 32.44 |
| deployed_projects (ŷ2_A) | 73 | **5.95** (4.08) | **41.72** (43.09) | 39.87 | 41.72 |

논문 6-타겟 평균(참고): Gemini **13.62** / DeepSeek **25.72**.

## 3. 해석 — 왜 두 MAPE가 모델별로 갈리는가 (분산 vs 편향)

- **Gemini**: 추정치가 GT 주변에 넓게 흩어짐(approved sharpness 114, coverage ✅). 편향 거의 없음(bias -1.0%) → 앙상블 평균이 GT에 수렴 → **agg-MAPE 1.0%** 인데 per-scenario는 30.9%. '다양성이 진실 주변에서 발생'.
- **DeepSeek**: 체계적 과소추정(bias -32.4%, mean_est 78 vs GT 115), coverage ❌. 평균 자체가 GT에서 벗어나 → agg-MAPE와 per-scenario가 모두 큼(32.4% ≈ 32.4%). 앙상블 평균으로도 편향이 상쇄되지 않음.
- **결론**: 싱가포르 100E에서 Gemini가 DeepSeek을 크게 앞서며(논문과 동일 방향·유사 크기), 이는 per-scenario MAPE만 보면 드러나지 않는다. 정책수준 예측 정확도(논문 지표)는 반드시 aggregate MAPE로 봐야 하고, calibration(coverage/sharpness)과 bias를 함께 봐야 '분산이 커서 per-scenario가 나쁜 것'과 '편향돼서 근본적으로 부정확한 것'을 구분할 수 있다.

## 4. 가드레일 준수 메모

- 이상치 제거/클리핑 없음(§6): sharpness가 큰 것은 그대로 신호로 보고. bias/coverage로 분산과 편향을 분리 보고.
- GT는 §5 TIGRIS Appendix E, Table 4 값만 사용(임의 생성 없음). GT 없는 지표는 skip.
- MAPE는 robust 병기: per-scenario는 median_robust도 저장. aggregate와 per-scenario를 둘 다 보고(한쪽만으로 결론 금지).
