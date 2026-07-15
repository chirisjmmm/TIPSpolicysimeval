# TIPS 발화 평가 파이프라인 — Claude Code 실행 가이드 (자체 완결형)

> 이 파일 하나로 전체 구현이 가능하도록 정리했다. **LLM API를 전혀 쓰지 않는다**(로컬 임베딩만 선택적으로 사용). Claude Code는 아래 마일스톤을 **순서대로, 한 번에 하나씩** 구현하고 각 수용기준(✅)을 통과한 뒤 다음으로 넘어간다.

---

## 0. 목적과 절대 제약 (먼저 읽을 것)

- **목적**: TIPS 시뮬레이션의 **페르소나 발화(narrative)** 를 ADR(Accuracy·Diversity·Reflection) 축으로 평가한다. 예측 정확도(MAPE)는 보조이고, faithfulness·diversity가 무게중심이다.
- **절대 제약 (위반 금지)**:
  1. **외부 LLM API 호출 금지.** judge가 필요했던 지표는 전부 rule/lexical/numeric로 구현한다. `sentence-transformers`(로컬 모델, API 아님)는 선택적 강화로만 허용.
  2. **Ground truth를 지어내지 말 것.** GT는 §5 표의 값만 쓴다. 값이 불확실하면 skip하고 리포트에 "GT 미입력"으로 남긴다.
  3. **이상치(원본 값·발화)를 임의로 바꾸지 말 것.** clip/winsorize/삭제 금지(§6).
  4. **파싱은 무손실.** raw 필드를 수정하지 않는다.
  5. rule 기반 지표도 human gold로 신뢰도(Krippendorff α)를 검증하기 전에는 "미검증(unvalidated)"으로 표기하고 본문 수치로 확정하지 않는다(§8).

---

## 1. 데이터 구성 (세 개의 독립 50-set)

| policy_id | model_id | 개수 | 특이사항 |
|---|---|---|---|
| 100E | deepseek | 50 | 파일 1~50 |
| 100E | gemini | 50 | **5개 하위폴더 × 각 10개(파일명 1~10 중복)** → 전역 재색인 필요(§4) |
| BK21 | deepseek | 50 | 파일 1~50 |

지표는 **(policy_id, model_id) set별로 따로** 산출한다. 100E는 deepseek vs gemini 비교로 백본 강건성 결과가 덤으로 나온다.

> **M0에서 먼저 할 일**: 실제 폴더 트리를 `find . -name '*.json' | sort` 등으로 스캔해 위 가정(폴더명·개수)을 **확인/수정**한다. 아래 경로는 예시이므로 실제 구조에 맞춘다.

---

## 2. 저장소 구조 (생성 대상)

```
<data_folder>/
├── CODING_GUIDE.md                 # 이 문서
├── config/
│   └── policies.yaml               # role_map, gt_map, outcome bin, stance lexicon
├── src/
│   ├── io/load_scenarios.py        # 원본 JSON → 공통 IR (§3,§4)
│   ├── io/load_policy_text.py      # 정책 원문/KG 텍스트 로드(§7 grounding·fabrication용)
│   ├── metrics/accuracy_micro.py   # §7.1
│   ├── metrics/accuracy_meso.py    # §7.2
│   ├── metrics/accuracy_macro.py   # §7.3
│   ├── metrics/diversity.py        # §7.4
│   ├── metrics/reflection.py       # §7.5
│   ├── reliability/agreement.py    # §8
│   └── report/build_report.py      # §9
├── data/ir/                        # 파싱된 IR (jsonl/parquet)
├── data/gold/                      # human 라벨 (§8)
├── results/                        # 지표 산출물 + 리포트
└── tests/                          # 더미 IR 단위테스트
```

---

## 3. 공통 IR 스키마 & 파서 (scenario_34 실구조 기준)

### 실제 파일 구조

```
scenario.json (dict)
├── scenario_id : str            # 지역명 섞인 긴 문자열 → 그대로 쓰지 말 것(canonical uid 재부여)
├── participants : list[5]
│    ├── stakeholder_type         # EducationResearchGroup|GraduateStudent|PolicyRole|EarlyCareerResearcher
│    ├── name, institution, role, background   # institution/role/background 는 빈 값(신뢰 X)
│    ├── entity_name              # "EducationResearchGroup | KG-Gen-xxxx" (KG 노드 태그)
│    ├── persona_id, professional_persona, skills_and_expertise, career_goals_and_ambitions
│    └── sex, age, education_level, occupation, bachelors_field, ...   # 인구통계(채워짐)
├── forward_pass : list[5 phases]
│    └── phase(dict)
│         ├── phase              # Inputs|Activities|Outputs|Outcomes|Impact  (순서 고정)
│         ├── direction          # "forward"
│         ├── posting_order       # [persona_name...]  turn 순서
│         ├── initial_posts:list[5]   # 1라운드
│         ├── revised_posts:list[5]   # 2라운드(남들 본 뒤)
│         └── refined_posts:None|list # 있을 수도
│              post(dict): persona_name, stakeholder_type,
│                          prediction_values{var:num}, narrative(★발화), evidence[list], judgment
├── backward_pass, fwd_bwd_fwd_pass, fwd_bwd_fwd_bwd_pass : 보통 None → non-null만 순회
├── cross_checks : list[{phase,variable,forward_agg,aggregated_value,...}]  # 시나리오 집계(MAPE용)
├── scenario_impact : {impact_var:num}
└── scenario_confidence : {impact_var:null}
```

### IR 레코드(발화 단위)

```
Utterance = {
  utterance_id, scenario_uid, policy_id, model_id,
  pass, direction, phase, round, turn_index,
  persona_name, kg_role,
  text(=narrative), evidence(list), judgment,
  prediction_values(dict)
}
```

### 파서 핵심 규칙

- `scenario_uid = f"{policy_id}_{model_id}_{global_index:02d}"`. 원본 `scenario_id`는 `source_id`로 보존.
- role 배정: 빈 `institution/role/background` 무시. `stakeholder_type`을 `policies.yaml`의 `role_map[policy]`으로 정규 KG 역할에 매핑. `entity_name`·`professional_persona`는 보조 근거. **`PolicyRole`은 실제 서술이 학생인 오배정이 잦음(scenario_34 3번)** → `flag_policyrole=True`로 표시하고 human 스팟체크 대상(§8).
- 모든 pass(non-null)·모든 round(initial/revised/refined)의 모든 post를 Utterance로 전개. `turn_index = posting_order.index(persona_name)`.
- 시나리오 집계값은 `cross_checks`의 `(phase,variable)→aggregated_value` 우선, 없으면 revised 평균.

✅ **M1 수용기준**: 세 set 각각 IR 50개 생성, uid 유일, 각 시나리오 participants 5·forward_pass 5-phase, phase 순서 = [Inputs,Activities,Outputs,Outcomes,Impact], 각 phase initial/revised 각 5 post, posting_order와 post의 persona 일치. 실패 시 assert로 중단.

---

## 4. 싱가포르 Gemini 폴더 충돌 해소 & 중복 방지

- Gemini 100E: 5개 폴더 × 파일명 `1.json`~`10.json`(중복) → 전역 인덱스로 1~50 재색인.
- 폴더/파일은 **자연 정렬**(`1,2,...,10`; 문자열 정렬 금지). `global_index = (folder_rank-1)*10 + local_index`.
- 중복 방지 3중: (a) 전역 인덱스로 uid 유일화, (b) 파일 내용 md5로 물리적 중복 탐지 후 assert, (c) 로드 끝에 `len(unique uid)==50` assert.
- set 간 충돌은 uid 접두어(`100E_deepseek_`,`100E_gemini_`,`BK21_deepseek_`)로 차단.

✅ **M1 수용기준(추가)**: Gemini set에서 폴더 5·각 10파일 확인, 내용 해시 중복 0, uid 1~50 연속.

---

## 5. Ground Truth (TIGRIS Appendix E, Table 4) — **Outcomes 단계에 매칭**

GT는 두 정책 모두 **Outcomes** 단계 값에 매칭한다. **Output 단계의 유사 변수명에 매칭하지 말 것.**

### 100E (Policy A, Singapore) — 보고연도 2023
| GT 지표 | 값 | 매칭 대상(Outcomes) | 혼동 금지(매칭 X) |
|---|---|---|---|
| Approved 100E projects (누적) | **115** | Outcomes의 approved projects 변수 | Output `ai_prototypes_developed` |
| Deployed 100E projects (누적) | **73** | Outcomes의 deployed projects 변수 | Output `trainees_completed_program` |

### BK21 (Policy B) — 보고연도 2022
| GT 지표 | 값 | 매칭 대상(Outcomes) | 혼동 금지(매칭 X) |
|---|---|---|---|
| Trainee employment rate (%) | **82.2** | `employment_rate` (≈80) | `graduate_employment_rate_of_bk21_participants_in_2022` (=800) |
| International collaboration rate (%) | **36.0** | `international_collaboration_rate` (≈35) | `international_research_collaboration_rate_of_bk21_participants_in_2022` (=350) |

### 방어적 매핑 규칙 (키 이름 헷갈릴 때 값 스케일로 확정)
- **phase == "Outcomes"** 에서만 변수 선택.
- 비율(%) GT는 값이 **[0,100]** 인 변수를 택함 → 스케일 변형(800/350)은 자동 배제.
- `policies.yaml`의 `gt_map`에 `{phase, var, gt, range}`로 명시하고 로드 시 `range` assert.
- **100E 파일의 정확한 변수 키 문자열은 100E deepseek 파일 하나를 열어 확인**해 `gt_map`에 고정한다(가이드 작성자는 BK21 샘플만 확인함).

```yaml
gt_map:
  BK21:
    employment_rate:         {phase: Outcomes, var: employment_rate,                    gt: 82.2, range: [0,100]}
    intl_collaboration_rate: {phase: Outcomes, var: international_collaboration_rate,    gt: 36.0, range: [0,100]}
  100E:
    approved_projects:       {phase: Outcomes, var: <100E파일에서_확인>, gt: 115, range: [0,1000]}
    deployed_projects:       {phase: Outcomes, var: <100E파일에서_확인>, gt: 73,  range: [0,1000]}
```

---

## 6. 이상치 정책 — 임의 변경 금지

발화 평가에서 이상치는 노이즈가 아니라 **신호**(다양성·반대·herding 이탈·날조)다.
1. **원본 불변**: `prediction_values`·`narrative` 수정 금지. 파싱 무손실.
2. 발화 단위 지표(diversity 분산, anchoring, fabrication)에서 **이상치 제거·클리핑 금지** — 퍼짐 자체가 측정값.
3. MAPE에서 이상치를 지워 숫자를 "개선"하지 말 것. 강건성은 **평균 + 중앙값/robust 병기**, 필요 시 "포함/제외" 민감도를 **양쪽 다** 보고. 데이터는 유지.
4. 스케일 인공물(800 vs 80)은 **값을 고치지 말고** §5 규칙으로 *올바른 변수를 선택*해 회피.
5. 삭제 대신 **`anomaly_flag`** 만 부여(관찰용, 자동 제거 트리거 아님).

---

## 7. 지표 정의 (전부 API-free)

> 통일 프레이밍: diversity 계열은 "엔트로피 → 유효 개수(exp(H))" 한 뿌리의 변주. Plausible Diversity — diversity 지표는 §7.1 fabrication/grounding 필터 통과 발화에만 집계.

### 7.1 Accuracy / micro — 규칙·근거 (rule + lexical)

- **Deontic compliance = 규칙 위반률(이진분류)**: KG 규범 `(role,must/can/cannot,action,cond)`에서 역할별 금지/의무 행위 lexicon 구축 → 발화(형태소 정규화; 한국어 `kiwipiepy`/`konlpy`) 매칭. `cannot` 표현 등장+조건 성립 또는 타역할 전용 authority 주장 = 위반=1. `violation_rate = 위반 발화 / 적용규범 있는 발화`. 한계(recall↓) 명시.
- **Grounding = 근거 귀속(lexical)**: 정책 원문·KG 라벨을 TF-IDF 인덱싱 → 발화(또는 evidence)와의 코사인 최댓값/Jaccard. 임계 이상 grounded. `grounded_ratio`. (선택: 로컬 임베딩 코사인.)
- **Fabrication = 날조(가장 견고)**: 발화에서 정규식으로 수치(퍼센트·금액·연도·개수)·고유개체 추출 → 정책 원문에 문자열/근사 존재 검사. 원문에 없는 구체 수치 = 날조. `fabrication_rate`. 완전 결정적.
- (선택) **Profile consistency(내부지표)**: `education_level/occupation` 등 자격과 발화 모순 rule 검사. 비교 축 아님.

### 7.2 Accuracy / meso — 상호작용 (순수 numeric/로그)

- **Anchoring β**: 각 (scenario,phase,target)에서 `Δ_i=revised_i−initial_i`를 `peer_signal_i=(타인 initial 평균)−initial_i`에 OLS 회귀 → β(또래 이동 정도). **수렴 시리즈(값 전부 동일) 제외**(β=1 인공물 방지). `statsmodels`.
- **Convergence rate**: `(scenario,phase,target)`에서 revised 값이 전부 동일(±ε)인 비율. herding 직접 증거. faithfulness와 교차(근거 있는 합의 vs 맹목 herding).
- **Responsiveness**: revised가 initial에서 바뀐 비율(`moved_ratio`). (선택: 또래 텍스트와 term-overlap.)
- **Cross-phase coherence**: 연속 `phase_summary` 간 TF-IDF 코사인 / 개체 carry-over. ToC 체인 단절 탐지.

### 7.3 Accuracy / macro — MAPE + Calibration (numeric)

- **MAPE**: 시나리오 추정 = `cross_checks (Outcomes,var) aggregated_value` 우선(없으면 revised 평균). `mean(|gt−pred|/|gt|)*100`. GT 미입력 skip.
- **Calibration**: 50개 추정 분포의 10–90 백분위 구간이 GT를 감싸는지(coverage)·폭(sharpness). `numpy`.

### 7.4 Diversity

- **micro (numeric+lexical)**: 이해관계자 집단별 예측값 분산(z-정규화 후 집단 간 거리) + 어휘 다양성 `distinct-n`, `self-BLEU`(높을수록 다양성↓ 주의; 형태소 토큰). (선택: 로컬 임베딩 Vendi score.) + 발화 정성 확인(사람).
- **macro (numeric)**: 타겟 field `outcome range`(min–max, IQR) + bin coverage/entropy(도메인 구간, `policies.yaml`) + trajectory clustering(5-phase 상태 벡터 → `sklearn` k-means/HDBSCAN → pathway 유형 수). assumption-conditioned pathway가 단일 대비 넓은지 비교.

### 7.5 Reflection — 유저스터디(코드 아님)

- process/outcome 모두 연구자 관점·유저스터디. 자동 지표 아님.
- (선택) 구조 완결성 프록시: pathway가 5-phase 모두 존재+각 non-null 추정/evidence = `pathway_completeness`(sanity check).

---

## 8. 신뢰성 검증 (API-free여도 필수)

- rule/lexical 판정(위반·grounding·fabrication·분류)을 human gold 10~20%(층화; 정책·역할·phase 고르게)와 대조해 **Krippendorff α**(`krippendorff` 패키지) 산출.
- **α ≥ 0.667 통과 지표만 리포트 본문 확정.** 통과 전엔 "unvalidated" 표기.
- 사람-사람 α를 먼저(과제 정의 명확성) → rule-vs-human α(휴리스틱 성능). 낮으면 lexicon/임계값 수정 후 재측정.
- role 배정(특히 `flag_policyrole`)도 소량 human 검증.

---

## 9. 리포트

- **axis×level 표**(set별: 100E-deepseek, 100E-gemini, BK21-deepseek)에 §7 지표를 채움. 비율엔 95% CI(Wilson/bootstrap), 판정 지표엔 α 병기. α 미통과·GT 미입력·이상치 플래그를 각주로 표기.
- 선택/제외 근거 박스: Diversity/meso 제외, Accuracy/meso 유지, micro=규칙준수 프록시, Reflection=유저스터디, API-free 한계.
- (선택) 시각화: axis×level 히트맵, set별 지표 막대+CI, 수렴 분포, 100E deepseek vs gemini 대비. 미검증 값엔 "unvalidated" 워터마크.

---

## 10. 마일스톤 (순서대로, 하나씩, 각 ✅ 통과 후 진행)

- **M0 데이터 인벤토리**: 실제 폴더 트리 스캔, §1 가정 확인/수정, 100E 변수 키 확인 → `policies.yaml` gt_map 확정. ✅ 세 set 파일 수·경로 확정, 100E GT 키 채움.
- **M1 IR 파서 + 검증**(§3,§4): 세 set → IR, 모든 assert 통과, Gemini 재색인·중복0. ✅ IR 150개(50×3), uid 유일.
- **M2 Accuracy/micro**(§7.1): 규칙 위반률·grounding·fabrication. ✅ set별 수치 산출 + per-utterance 라벨 저장.
- **M3 신뢰성**(§8): gold set + α. **M2를 믿을 수 있는지 먼저 확인.** ✅ 태스크별 α 보고, 미통과 지표 unvalidated 표기.
- **M4 Accuracy/meso**(§7.2): anchoring β(수렴 제외)·convergence·responsiveness·coherence. ✅ set별 β·수렴률.
- **M5 Diversity + macro-accuracy**(§7.3,§7.4): 분산·distinct-n·self-BLEU·outcome range·clustering·MAPE·calibration. ✅ GT 있는 set만 MAPE, 이상치 정책 준수.
- **M6 리포트**(§9): axis×level 표 + 근거 박스(+선택 시각화). ✅ 세 set 표 완성.

각 metric은 `tests/`에 더미 IR 단위테스트(예: Vendi가 동일 발화에 1.0, anchoring이 복사 시나리오에 β≈1, fabrication이 원문에 없는 숫자를 잡는지).

---

## 11. 라이브러리 (전부 API 불필요)

필수: `numpy pandas scikit-learn statsmodels nltk krippendorff pyyaml` + 한국어 `kiwipiepy`(또는 `konlpy`).
선택(로컬, API 아님): `sentence-transformers`.
**API 키 필요 구성요소: 없음.**

---

## 12. 하지 말 것 (가드레일 체크리스트)

- [ ] 외부 LLM API 호출(금지). judge 대신 rule/lexical/numeric.
- [ ] GT 값 임의 생성(금지). §5 값만, 불확실하면 skip.
- [ ] 이상치 clip/winsorize/삭제(금지). flag만.
- [ ] raw 필드 수정(금지). 파싱 무손실.
- [ ] 스케일 변형 숫자 덮어쓰기(금지). 올바른 변수 선택으로 회피.
- [ ] α 미검증 지표를 확정 수치로 보고(금지). unvalidated 표기.
- [ ] 파싱 로직을 metric 코드에 섞기(금지). metric은 IR만 입력.
- [ ] set 혼합(금지). (policy,model)별 독립 산출.
- [ ] self-BLEU 방향 착각(높을수록 다양성↓).
- [ ] 마일스톤 건너뛰기(금지). 하나씩 ✅ 통과.
