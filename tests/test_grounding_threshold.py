"""§7.1 grounding 지표: semantic support(로컬 sentence-transformers) 전환 더미 IR 단위테스트.

본 지표(grounded_ratio)는 이제 lexical(TF-IDF)이 아니라 semantic 유사도다 — "발화의 질적
주장이 이 policy KG의 규범 unit(norm_units) 중 하나에 의미상 뒷받침되는가." null은 word-salad
(토큰 셔플)가 아니라 coherent null(같은 policy의 "무관한" 실제 규범 unit)이어야 한다.
검증 항목:
  (i)   규범을 그대로 재진술한 발화 -> semantic grounded=True.
  (ii)  규범을 정당하게 패러프레이즈한 발화(단어 거의 안 겹침) -> semantic grounded=True,
        lexical은 놓칠 수 있음(이게 semantic 전환의 정당화).
  (iii) 정책 주제 어휘는 쓰지만 실제 규범 내용과 무관한 더미 발화 -> 대부분 semantic grounded=False.
  (iv)  coherent null이 word-salad가 아니라 실제 norm_units 기반인지(구조적으로 확인).
  (v)   임베딩 모델이 고정된 버전이고 로컬 sentence-transformers인지.
  (vi)  lexical baseline이 semantic과 함께 산출되는지.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "metrics"))
sys.path.insert(0, str(REPO_ROOT / "src" / "io"))

from accuracy_micro import (  # noqa: E402
    EMBEDDING_MODEL_NAME_DEFAULT,
    NULL_PERCENTILE_DEFAULT,
    compute_max_cosine,
    compute_null_max_cosine,
    compute_null_threshold,
    compute_semantic_coherent_null,
    compute_semantic_matches,
    compute_set_metrics,
    get_embedder,
)
from load_policy_text import PolicyKG  # noqa: E402

# 대학원 펠로우십 규정 도메인의 규범 unit(실제 KG norm_units 형태를 흉내냄).
NORM_UNITS = [
    {"text": "Graduate students must submit an annual progress report to the department by September.",
     "kind": "deontic_norm", "ref_id": "e1"},
    {"text": "Doctoral fellows may request travel funding for international conferences with advisor approval.",
     "kind": "deontic_norm", "ref_id": "e2"},
    {"text": "Research fellows cannot hold outside employment while receiving the full stipend.",
     "kind": "deontic_norm", "ref_id": "e3"},
    {"text": "The fellowship committee reviews applications every autumn semester before the funding cycle begins.",
     "kind": "edge_fact", "ref_id": "e4"},
    {"text": "The graduate office disburses stipends on a quarterly basis to enrolled doctoral students.",
     "kind": "node_summary", "ref_id": "n1"},
    {"text": "Fellows must maintain a minimum GPA of 3.0 to remain eligible for the stipend.",
     "kind": "deontic_norm", "ref_id": "e5"},
    {"text": "The department may extend the reporting deadline by two weeks upon written request.",
     "kind": "deontic_norm", "ref_id": "e6"},
    {"text": "Doctoral students cannot transfer their fellowship award to another institution.",
     "kind": "deontic_norm", "ref_id": "e7"},
    {"text": "The selection committee interviews shortlisted applicants each spring before making final offers.",
     "kind": "edge_fact", "ref_id": "e8"},
    {"text": "Fellows must attend the mandatory orientation session in their first semester.",
     "kind": "deontic_norm", "ref_id": "e9"},
    {"text": "The university publishes the list of funded fellows on its research portal each year.",
     "kind": "node_summary", "ref_id": "n2"},
    {"text": "Advisors must co-sign the annual progress report before submission to the department.",
     "kind": "deontic_norm", "ref_id": "e10"},
]

EXACT_RESTATEMENT = NORM_UNITS[0]["text"]
GENUINE_PARAPHRASE = (
    "Every year before autumn begins, doctoral trainees are required to hand in a report "
    "describing their progress to the academic office."
)
# 규범 관련 어휘(fellowship/department/graduate/stipend/autumn 등)는 쓰지만 실제로는 아무 규범도
# 서술하지 않는 잡담 — coherent(문법적으로 말이 되는) 문장이되 내용은 무관하다.
TOPICAL_DUMMIES = [
    "I really enjoyed the fellowship reception where department staff served coffee and pastries "
    "to graduate students last autumn.",
    "The graduate department building has a lovely garden where doctoral students like to relax "
    "near the stipend office.",
    "My favorite fellow at the department told me a funny story about the graduate student parking "
    "lot last September.",
    "I heard the campus cafe near the graduate office has great coffee in autumn.",
    "The department hallway was recently repainted a cheerful shade of blue for the fellows.",
    "Graduate students organized a fun autumn picnic near the fellowship office last week.",
    "The stipend office moved to a bigger room down the hall from the department library.",
    "A doctoral student left an umbrella in the fellowship committee meeting room last autumn.",
    "The department cafeteria added a new autumn menu that graduate fellows seem to enjoy.",
    "Someone in the graduate office plays music every September during the lunch break.",
]


def _mk_utt(uid: str, text: str) -> dict:
    return {"utterance_id": uid, "text": text}


def _mk_kg(policy_id: str, norm_units: list[dict]) -> PolicyKG:
    grounding_corpus = [u["text"] for u in norm_units]
    return PolicyKG(
        policy_id=policy_id, nodes=[], edges=[],
        norms_by_role_type={}, authority_phrases_by_role_type={},
        grounding_corpus=grounding_corpus, norm_units=norm_units,
    )


def test_embedding_model_pinned_and_local():
    """(v) 임베딩 모델 버전이 고정돼 있고, 실제로 로컬 sentence-transformers 라이브러리로 로드되는지."""
    assert EMBEDDING_MODEL_NAME_DEFAULT == "paraphrase-multilingual-MiniLM-L12-v2"
    model = get_embedder(EMBEDDING_MODEL_NAME_DEFAULT)
    assert model.__class__.__module__.startswith("sentence_transformers")


def test_coherent_null_is_real_norm_units_not_word_salad():
    """(iv) coherent null이 토큰 셔플(word-salad)이 아니라 실제 norm_units의 코사인 값에서
    뽑힌 것인지 구조적으로 확인한다: compute_semantic_coherent_null은 코퍼스 텍스트를 전혀
    받지 않고 이미 계산된 sims 행렬(발화 x 실제 norm_units)만 입력으로 받으며, 반환값은
    그 행렬 안에 실제로 존재하는 코사인 값들의 부분집합이어야 한다."""
    import inspect

    sig = inspect.signature(compute_semantic_coherent_null)
    assert set(sig.parameters.keys()) == {"sims", "seed", "topk_exclude", "samples_per_utt"}
    assert "corpus" not in sig.parameters  # word-salad 코퍼스를 받을 자리가 없음

    model = get_embedder()
    probe = [_mk_utt(f"p_{i}", t) for i, t in enumerate([EXACT_RESTATEMENT, GENUINE_PARAPHRASE] + TOPICAL_DUMMIES)]
    _, sims = compute_semantic_matches(probe, NORM_UNITS, model)
    null_vals = compute_semantic_coherent_null(sims, seed=1, topk_exclude=3, samples_per_utt=5)

    assert len(null_vals) > 0
    flat_real_values = set(round(float(v), 6) for v in sims.flatten())
    assert all(round(v, 6) in flat_real_values for v in null_vals)


def test_i_exact_restatement_is_semantic_grounded():
    """(i) 규범을 그대로 재진술한 발화는 semantic grounded=True."""
    model = get_embedder()
    probe = [_mk_utt(f"p_{i}", t) for i, t in enumerate([EXACT_RESTATEMENT, GENUINE_PARAPHRASE] + TOPICAL_DUMMIES)]
    per_utt, sims = compute_semantic_matches(probe, NORM_UNITS, model)
    null_vals = compute_semantic_coherent_null(sims, seed=1)
    tau = compute_null_threshold(null_vals, NULL_PERCENTILE_DEFAULT)

    assert per_utt["p_0"]["sem_max_cosine"] > tau


def test_ii_genuine_paraphrase_semantic_grounded_lexical_misses():
    """(ii) 정당한 패러프레이즈(단어 거의 안 겹침) -> semantic grounded=True.
    lexical(TF-IDF, within-policy decoy)은 이 케이스를 놓칠 수 있음을 직접 재현한다 —
    이게 lexical에서 semantic으로 전환한 정당화의 핵심 근거다.
    """
    model = get_embedder()
    probe = [_mk_utt(f"p_{i}", t) for i, t in enumerate([EXACT_RESTATEMENT, GENUINE_PARAPHRASE] + TOPICAL_DUMMIES)]
    per_utt, sims = compute_semantic_matches(probe, NORM_UNITS, model)
    null_vals = compute_semantic_coherent_null(sims, seed=1)
    tau_sem = compute_null_threshold(null_vals, NULL_PERCENTILE_DEFAULT)

    assert per_utt["p_1"]["sem_max_cosine"] > tau_sem  # semantic: 잡아냄

    kg = _mk_kg("TESTA", NORM_UNITS * 10)  # decoy용 토큰 풀 확보
    lex_probe = [_mk_utt("u_paraphrase", GENUINE_PARAPHRASE)]
    lex_real = compute_max_cosine(lex_probe, kg)
    lex_null = list(compute_null_max_cosine(lex_probe * 20, kg, seed=5).values())
    tau_lex = compute_null_threshold(lex_null, NULL_PERCENTILE_DEFAULT)

    assert lex_real["u_paraphrase"] <= tau_lex  # lexical: 놓침(단어가 거의 안 겹침)


def test_iii_topical_but_irrelevant_dummies_mostly_not_grounded():
    """(iii) 정책 주제 어휘(fellowship/department/graduate/stipend/autumn 등)는 쓰지만 실제
    규범 내용과는 무관한 더미 발화 10건 -> 대부분 semantic grounded=False로 떨어져야 한다
    (topical confound 통제 확인 — semantic이 그냥 '같은 주제 단어 개수'로 흔들리지 않는지)."""
    model = get_embedder()
    probe = [_mk_utt(f"p_{i}", t) for i, t in enumerate([EXACT_RESTATEMENT, GENUINE_PARAPHRASE] + TOPICAL_DUMMIES)]
    per_utt, sims = compute_semantic_matches(probe, NORM_UNITS, model)
    null_vals = compute_semantic_coherent_null(sims, seed=1)
    tau = compute_null_threshold(null_vals, NULL_PERCENTILE_DEFAULT)

    dummy_ids = [f"p_{i}" for i in range(2, 2 + len(TOPICAL_DUMMIES))]
    n_grounded = sum(1 for uid in dummy_ids if per_utt[uid]["sem_max_cosine"] > tau)
    assert n_grounded <= len(dummy_ids) // 2, (
        f"주제 어휘만 쓰는 무관 더미 {len(dummy_ids)}건 중 {n_grounded}건이 grounded=True (기대: 대부분 False)"
    )


def test_semantic_and_lexical_baseline_both_present_in_set_metrics():
    """(vi) compute_set_metrics 결과에 semantic(본 지표)과 lexical_baseline이 함께 산출되는지,
    그리고 grounded_ratio가 여전히 하나의 키(축 분리 없음)로 유지되는지 확인한다."""
    kg = _mk_kg("TESTA", NORM_UNITS)
    policy_kgs = {"TESTA": kg}

    # null 위양성률 sanity assert(코드 정확성 체크)는 동일 문장 12종의 단순 반복만으로는 통계적으로
    # 불안정하다(코사인 값이 12가지로만 묶여 동점이 몰림) — 실제 코퍼스처럼 연속적인 변이를 주기
    # 위해 어휘를 무작위 재조합한 filler 발화를 넉넉히 섞는다(이 filler들의 grounded 값 자체는
    # 이 테스트의 관심사가 아니다. 스키마·핵심 케이스 확인이 목적).
    vocab = sorted(set(" ".join(u["text"] for u in NORM_UNITS).split()))
    rng = random.Random(2)
    base_texts = [EXACT_RESTATEMENT, GENUINE_PARAPHRASE] + TOPICAL_DUMMIES
    filler_texts = []
    for _ in range(100):
        k = rng.randint(4, 10)
        words = rng.sample(vocab, min(k, len(vocab)))
        rng.shuffle(words)
        filler_texts.append(" ".join(words))

    utterances = []
    idx = 0
    for t in base_texts + filler_texts:
        utterances.append({
            "utterance_id": f"u_{idx}", "policy_id": "TESTA", "model_id": "m1",
            "scenario_uid": "TESTA_m1_01", "kg_role": "TestRole:TestRole",
            "deontic_status": "applicable", "phase": "Inputs", "round": "initial", "text": t,
        })
        idx += 1

    result = compute_set_metrics(utterances, policy_kgs)
    gr = result["set_results"]["TESTA_m1"]["grounded_ratio"]

    assert gr["basis"] == "semantic"
    assert "lexical_baseline" in gr
    assert "rate" in gr and "rate" in gr["lexical_baseline"]

    label = result["per_utterance"]["u_0"]
    assert label["grounded"] is True  # 본 지표(semantic) 필드명은 여전히 "grounded"
    for key in ("sem_max_cosine", "top1_norm_ref", "top1_norm_text", "top1_norm_kind",
                "lex_grounded", "lex_max_cosine"):
        assert key in label


if __name__ == "__main__":
    test_embedding_model_pinned_and_local()
    test_coherent_null_is_real_norm_units_not_word_salad()
    test_i_exact_restatement_is_semantic_grounded()
    test_ii_genuine_paraphrase_semantic_grounded_lexical_misses()
    test_iii_topical_but_irrelevant_dummies_mostly_not_grounded()
    test_semantic_and_lexical_baseline_both_present_in_set_metrics()
    print("모든 semantic grounding 테스트 통과")
