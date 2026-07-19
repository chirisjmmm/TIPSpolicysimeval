# TIPS 정책 시뮬레이션 발화 평가 파이프라인

TIPS(싱가포르 100 Experiments / 한국 BK21) 정책 시뮬레이션에서 나온 **페르소나 발화(narrative)**를
[Accuracy, Diversity, and Reflection: Purpose-driven Evaluation for Social Simulation](#) 프레임워크의
ADR(Accuracy·Diversity·Reflection) 축으로 평가하는 API-free 파이프라인이다.

예측 정확도(MAPE)는 보조 지표이고, **faithfulness(규범 준수·근거성)와 diversity가 무게중심**이다.
전체 마일스톤·스키마·수식의 원본 명세는 [`CODING_GUIDE.md`](./CODING_GUIDE.md)에 있다 — 이 문서는 그
요약이자 "지금 뭐가 되어 있는지"에 대한 안내다.

## 절대 제약

- **외부 LLM API 호출 금지.** judge가 필요한 지표는 전부 rule/lexical/numeric로 구현한다. 로컬
  `sentence-transformers`(§11 허용, API 아님)만 예외.
- **Ground truth를 지어내지 않는다.** GT는 TIGRIS Appendix E 값만 쓴다(§5). 불확실하면 skip.
- **이상치를 임의로 바꾸지 않는다.** clip/winsorize/삭제 대신 flag만 남긴다.
- **파싱은 무손실.** raw 필드는 수정하지 않고, 파생 필드만 덧붙인다.
- **rule/model 기반 지표는 human gold로 신뢰도(Krippendorff α ≥ 0.667)를 검증하기 전엔 "unvalidated"로
  표기**하고 본문 수치로 확정하지 않는다.

## 데이터

세 개의 독립적인 50-시나리오 세트(정책 × 모델 backbone). 지표는 항상 **세트별로 따로** 산출한다.

| policy_id | model_id | 시나리오 수 | 비고 |
|---|---|---|---|
| BK21 | deepseek | 50 | 한국 BK21(대학원 인재양성) |
| 100E | deepseek | 50 | 싱가포르 100 Experiments |
| 100E | gemini | 50 | 5개 폴더 × 10개 파일 → 전역 재색인(§4) |

각 시나리오는 5단계 ToC(Theory of Change) phase(Inputs → Activities → Outputs → Outcomes → Impact)를
거치며, 매 phase마다 5명의 페르소나가 initial/revised 두 라운드에 걸쳐 발화(narrative)·수치 예측
(prediction_values)·근거(evidence)를 남긴다.

## 저장소 구조

```
config/policies.yaml        역할 매핑(role_map), GT 매핑(gt_map), flag_policyrole 키워드
src/io/
  load_scenarios.py          원본 JSON -> 공통 IR(Scenario, Utterance). scenario_uid 부여,
                              Gemini 폴더 재색인, kg_role 크로스워크, flag_policyrole 판정
  load_policy_text.py         {policy}_policy_graph.json 파싱 -> PolicyKG
                              (norms_by_role_type, authority_phrases, grounding_corpus, norm_units)
  run_m1.py                   M1 파서 + 수용기준 assert 실행, data/ir/*.jsonl 생성
  qa_sample.py                 human 스팟체크용 조회 전용 표(qa1~qa3)
src/metrics/
  accuracy_micro.py            §7.1 위반률 · grounding(semantic 본 지표 + lexical baseline) · fabrication
  accuracy_meso.py              §7.2 anchoring β · convergence_rate · responsiveness · cross-phase coherence
  accuracy_macro.py             §7.3 MAPE + calibration
  diversity.py                  §7.4 micro(집단 분산·distinct-n·self-BLEU) + macro(outcome range·clustering)
  run_m2.py / run_m4.py / run_m5.py   각 마일스톤 실행 스크립트(results/에 저장)
  qa_m2_sample.py               M2 조회용 진단 표
src/reliability/
  gold_pool_sampling.py         §8 계층 표본(random/boundary gold pool)
  independent_rater.py          accuracy_micro.py와 코드 독립적인 2차 판정기(삼각검증용)
  violation_gold_pool.py        violation 축 전용 라벨링 표(applicable_norms 참고 컬럼 포함)
  agreement.py                   Krippendorff α, 혼동행렬, initial/revised 판정 불일치 점검
  threshold_analysis.py         grounding 임계값 precision/recall 스윕(gold 라벨 필요)
  resample_and_suggest.py       gold pool 재표본 + 블라인드 처리
data/ir/                        파싱된 IR(scenarios.jsonl, utterances.jsonl)
data/gold/                      human 라벨링 표(일부 라벨링 완료, 일부 진행 중)
results/                        지표 산출물(JSON/CSV)
tests/                           더미 IR 단위테스트
```

## 평가 프레임워크 ↔ 구현 매핑

아래 표는 원본 ADR 프레임워크(이미지)의 4개 셀을 이 저장소의 실제 구현에 대응시킨 것이다. `micro`/`macro`
외에 `meso`(상호작용) 레벨은 이미지에는 없지만 `CODING_GUIDE.md` §7.2가 별도로 요구하는 확장이며, 이는
정책 연구(policy simulation)의 특성상 "합의/앵커링" 과정 자체가 중요한 신호이기 때문이다.

| Axis | Level | 평가 요인 | 정책 연구 적용 | 구현 지표 | 코드 |
|---|---|---|---|---|---|
| Accuracy | micro | 개별 에이전트 발화가 현실적 행동과 얼마나 일치하는가 | 발화가 구조화된 정책 문서(KG)의 제약을 얼마나 잘 따르는가 | **규칙 위반률**(violation_rate, 이진분류) + **grounding**(semantic support, 본 지표) + lexical baseline + **fabrication_rate**(수치 날조) | `accuracy_micro.py` |
| Accuracy | meso *(확장)* | 상호작용을 거치며 발화가 어떻게 수렴/변화하는가 | 또래 신호에 대한 앵커링, herding vs 근거 있는 합의 | anchoring β · convergence_rate · responsiveness · cross-phase coherence | `accuracy_meso.py` |
| Accuracy | macro | 집합적 사회 현상이 시뮬레이션에서 출현하는가 | ToC chain을 거친 최종 예측값이 실제 정책 성과 지표에 근접하는가 | **MAPE** + calibration(10–90 백분위 coverage/sharpness) | `accuracy_macro.py` |
| Diversity | micro | 동일 자극에 개별 에이전트가 얼마나 다양하게 반응하는가 | 이해관계자별 우려·요구·전략 차이가 예측값 분산으로 드러나는가 | 집단 간 분산(z-정규화) + distinct-n + self-BLEU(어휘 다양성) | `diversity.py` |
| Diversity | macro | 동일 초기조건에서 서로 다른 사회 수준 결과가 나오는가 | assumption-conditioned pathway가 단일 pathway보다 넓은 결과 공간을 보이는가 | outcome range/IQR/bin coverage·entropy + trajectory clustering(k-means) | `diversity.py` |
| Reflection | — | 유저스터디(코드 아님) | 연구자 관점의 과정/결과 평가 | 자동화 대상 아님. (선택) pathway_completeness 프록시만 sanity check용 | — |

### Accuracy/micro grounding의 변천 — 한 줄 요약

같은 세트 자신의 median으로 임계값을 잡는 **자기참조**(BK21이 늘 정확히 0.500) → 다른 정책 코퍼스와
대조하는 **cross-policy null**(사실상 "정책 간 어휘 구별"을 잼) → 같은 정책 코퍼스를 뒤섞은
**within-policy decoy**(lexical 한계로 separation_gap이 세 세트 전부 음수) → 현재의
**semantic support**(로컬 sentence-transformers + coherent null, separation_gap이 전부 양수로 전환)
순으로 반복 수정되었다. 자세한 이유는 `accuracy_micro.py`의 모듈 docstring과 아래 "알려진 한계"를 참고.

## 실행 순서

```bash
# M1: IR 파싱 + 수용기준 검증 (세 세트 -> data/ir/scenarios.jsonl, utterances.jsonl)
python src/io/run_m1.py

# M2: 위반률 · grounding(semantic+lexical) · fabrication  -> results/m2_*
python src/metrics/run_m2.py

# M4: anchoring β · convergence · responsiveness · cross-phase coherence -> results/m4_*
python src/metrics/run_m4.py

# M5: MAPE/calibration + diversity -> results/m5_*
python src/metrics/run_m5.py

# 단위테스트
python tests/test_grounding_threshold.py
```

필수 라이브러리: `numpy pandas scikit-learn statsmodels nltk krippendorff pyyaml sentence-transformers`.
API 키가 필요한 구성요소는 없다.

## 마일스톤 현황 (§10 기준)

| 마일스톤 | 상태 | 비고 |
|---|---|---|
| M0 데이터 인벤토리 | ✅ | `policies.yaml` gt_map 확정 |
| M1 IR 파서 + 검증 | ✅ | 세 세트 150 시나리오, uid 유일, Gemini 재색인·중복 0 |
| M2 Accuracy/micro | ✅ (수치는 unvalidated) | grounding은 semantic 본 지표로 전환 완료 |
| M3 신뢰성 검증(§8) | 🔶 진행 중 | violation 축 α 계산 완료(α=-0.0011, 임계값 미달) — 아래 참고. grounding/fabrication 축 라벨링 진행 중 |
| M4 Accuracy/meso | ✅ (수치는 unvalidated) | 아래 "알려진 한계" 참고(revised≡initial) |
| M5 Diversity + macro-accuracy | ✅ (수치는 unvalidated) | `plausible_diversity_filter`는 TODO(passthrough) |
| M6 리포트(axis×level 표 + unvalidated 워터마크) | ⬜ 미착수 | |

## 신뢰성 검증(§8) 현재 상태

- `data/gold/`에 층화 표본(`random_gold_pool.csv`, `boundary_gold_pool.csv`)과 축별 라벨링 표
  (`violation_gold_pool.csv` 등)가 있다. **violation 축만 라벨링이 끝나 α를 계산했다**
  (`src/reliability/agreement.py`): α = **-0.0011**, `ALPHA_THRESHOLD = 0.667` 미달 →
  `violation_rate`는 여전히 **UNVALIDATED**.
  - 원인: gold 표본에 진짜 위반(TP)이 0건이라 표본 자체가 과소검정(underpowered)됨. 관측된 오탐 2건은
    전부 `cross_role_authority_claim`의 "oversee" 트리거가 원인으로 특정됨.
  - 이 트리거를 좁힐지 여부는 **의도적으로 보류 중**(사용자 결정: "그대로 두고 라벨링만 진행") — 더
    많은 라벨이 모여 진짜 TP가 나오는지 먼저 확인하기로 함.
- grounding(semantic)·fabrication 축은 아직 gold 대조 전. `threshold_analysis.py`가 grounding
  precision/recall 스윕을 준비해두었으나 라벨링 완료 전까지는 실행해도 "0/N labeled"로 종료됨.

## 알려진 한계 (숨기지 않고 명시)

- **`revised_posts == initial_posts`가 코퍼스 전체에서 100% 동일**하다(15,000개 수치 쌍, 3,750개 텍스트
  쌍 전수 확인). 이 때문에 M4의 `responsiveness`(moved_ratio)와 `anchoring_beta`는 세 세트 전부 정확히
  `0.0`이 나오는데, 이는 "반응이 없다"는 실질적 신호가 아니라 **정의상 항상 나오는 값**이다.
  `convergence_rate`만 "1라운드 시점의 페르소나 간 우연한 일치율"이라는 축소된 의미로 여전히 유효하다.
- **grounding(semantic)의 grounded_ratio가 매우 높다(92~98%)** — lexical 한계(모든 세트 separation_gap
  음수)는 해결됐지만, 이 높은 비율 자체가 아직 gold로 검증되지 않았다. `results/
  m2_semantic_grounding_traceability_sample.csv`에서 top1 매칭 근거를 직접 확인할 수 있다.
- 100E deepseek vs gemini의 **grounded_ratio 상대순서가 fabrication_rate 방향과 일치하지 않는다**
  (gemini가 수치는 더 많이 날조하면서 질적 근거성은 더 높게 나옴). 모순이라기보다 두 지표가 애초에
  독립적인(상보적인) 축이기 때문일 수 있으나, 미화하지 않고 그대로 보고한다.
- fabrication_rate가 65~90%로 높게 나오는데 아직 gold 미검증 — grounding 축과 함께 라벨링 우선순위.
- violation 감지 규칙은 recall이 낮다(§7.1에 명시된 한계). 영어 표현 위주 단서라 한국어 발화에서는
  거의 못 잡는다.
- 100E의 `PolicyRole:AI for Public Good Principle` 페르소나(173명)는 KG상 deontic edge의
  source가 아니라서 `no_institutional_position_in_kg`로 violation_rate 분모에서만 제외된다(다른
  지표에는 포함). BK21의 동일 유형 오배정(`flag_policyrole=True`)은 GraduateStudent로 재배정한다 —
  자세한 이유는 `config/policies.yaml` 주석 참고.

## 참고

전체 스키마·수식·가드레일 체크리스트는 [`CODING_GUIDE.md`](./CODING_GUIDE.md)를 따른다. 이 README는
그 요약이며, 두 문서가 어긋나면 `CODING_GUIDE.md`가 우선한다.
