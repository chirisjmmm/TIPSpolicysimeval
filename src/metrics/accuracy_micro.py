"""§7.1 Accuracy/micro: deontic compliance(위반률) · grounding · fabrication.
violation/fabrication은 rule/lexical(API-free). grounding은 semantic support가 본 지표이고
로컬 sentence-transformers(§11 허용 — 외부 LLM API 아님)를 쓴다. lexical(TF-IDF)은 baseline
진단값으로만 병기한다.

입력은 IR(data/ir/utterances.jsonl)만 받는다 — 파싱 로직은 여기 섞지 않는다(§12 가드레일).
"""
from __future__ import annotations

import random
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from load_policy_text import PolicyKG  # noqa: E402

STOPWORDS = {
    "with", "that", "this", "from", "have", "must", "will", "shall", "into", "their", "they",
    "them", "which", "under", "over", "such", "than", "when", "where", "what", "each", "other",
    "across", "through", "during", "while", "about", "budget",
}

NEGATION_CUES = [
    r"\bdid not\b", r"\bdidn't\b", r"\bhave not\b", r"\bhaven't\b", r"\bhas not\b", r"\bhasn't\b",
    r"\bnever\b", r"\bfailed to\b", r"\brefuse(?:d)? to\b", r"\bunable to\b", r"\bwithout completing\b",
    r"\bskip(?:ped)? the\b",
]
NEGATION_RE = re.compile("|".join(NEGATION_CUES), re.IGNORECASE)

AUTHORITY_CLAIM_CUES = [
    r"\bI\s+(?:will\s+|can\s+|have\s+|am\s+going\s+to\s+|personally\s+)?"
    r"(decide[ds]?|approve[ds]?|allocate[ds]?|reallocate[ds]?|authoriz(?:e|ed|es)|oversee(?:s|n)?|"
    r"sanction(?:ed|s)?|assign(?:ed|s)?|recommend(?:ed|s)?|coordinate[ds]?|launch(?:ed|es)?|"
    r"drive(?:s)?|issue[ds]?)\b",
    r"\bwe\s+(?:will\s+|can\s+|have\s+|personally\s+)?"
    r"(decide[ds]?|approve[ds]?|allocate[ds]?|reallocate[ds]?|authoriz(?:e|ed|es)|oversee(?:s|n)?|"
    r"sanction(?:ed|s)?|assign(?:ed|s)?|recommend(?:ed|s)?|coordinate[ds]?|launch(?:ed|es)?)\b",
]
AUTHORITY_CLAIM_RE = re.compile("|".join(AUTHORITY_CLAIM_CUES), re.IGNORECASE)


def significant_words(phrase: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{4,}", phrase.lower())
    return {w for w in words if w not in STOPWORDS}


def strip_subject_and_deontic(fact: str) -> str:
    """'Graduate students must complete X' -> 'complete X' (아주 단순한 rule, NLP 아님)."""
    m = re.search(r"\b(must|can|cannot)\b\s+(.*)", fact, re.IGNORECASE)
    return m.group(2) if m else fact


def _role_type_of(kg_role: str) -> str:
    return kg_role.split(":", 1)[0]


def check_deontic_violation(utterance: dict, kg: PolicyKG) -> dict:
    """규칙: (a) 자기 역할의 must-의무를 명시적으로 부정, (b) 다른 역할 전용 authority를 자기 것처럼 주장.
    한계(recall 낮음, §7.1 명시 요건): 부정/authority 단서는 영어 표현 위주라 한국어 발화에서는
    거의 못 잡는다. cannot 규범은 원본 KG에 2건뿐이고 둘 다 조직(EducationResearchGroup) 단위 제외
    기준이라 개인 발화 자기주장으로 위반되는 경우가 사실상 없다(관측된 그대로 보고, 임의로 만들지 않음).
    """
    if utterance["deontic_status"] != "applicable":
        return {"n_applicable": False, "violation": None, "reason": None}

    role_type = _role_type_of(utterance["kg_role"])
    own_norms = kg.norms_by_role_type.get(role_type, [])
    if not own_norms:
        return {"n_applicable": False, "violation": None, "reason": None}

    text = utterance["text"] or ""
    window = 80  # 문장 분리에 기대지 않고 단서어 주변 ±80자로 국소 근접성만 본다(아래 이유)

    def _local_window(pos_start: int, pos_end: int) -> str:
        return text[max(0, pos_start - window): pos_end + window]

    def _overlap_strong_enough(phrase_words: set[str], window_words: set[str]) -> bool:
        """단어 1개 겹침은 약한 신호(예: "evaluation"처럼 코퍼스에 흔한 단어 하나만 우연히 근처에
        있어도 걸림 — 실제로 스팟체크에서 발견한 오탐: "I have never experienced delays..." 근처에
        전혀 다른 맥락의 "Evaluation panels..."가 있어서 "submit evaluation reports"와 오매칭됨).
        구문 단어의 과반 또는 최소 2개가 겹쳐야 인정한다.
        """
        overlap = phrase_words & window_words
        if not phrase_words:
            return False
        return len(overlap) >= min(2, len(phrase_words)) if len(phrase_words) > 1 else len(overlap) == 1

    # (a) own-role must 의무의 명시적 부정
    # 문장 단위(마침표 기준) 근접성은 실제로 오탐을 냈다: 이 코퍼스의 발화는 LLM이 마침표 없이
    # 길게 이어 쓰는 경우가 많아 "I have not seen other posts at this phase..."(시뮬레이션 진행에 대한
    # 메타 발언)와 한참 뒤에 나오는 "eligibility criteria... program" 같은 무관한 단어가 같은 "문장"으로
    # 묶여 오매칭됐다(직접 확인함). 그래서 문장 대신 단서어 앞뒤 ±80자 고정 윈도우로 좁혔다.
    own_must_phrases = [strip_subject_and_deontic(n["fact"]) for n in own_norms if n["deontic"] == "must" and n["fact"]]
    for m in NEGATION_RE.finditer(text):
        window_text = _local_window(m.start(), m.end())
        window_words = significant_words(window_text)
        for phrase in own_must_phrases:
            if _overlap_strong_enough(significant_words(phrase), window_words):
                return {"n_applicable": True, "violation": True, "reason": f"must_negation:{phrase[:60]}"}

    # (b) 타 역할 전용 authority를 자기 것처럼 주장
    other_role_phrases: list[tuple[str, str]] = [
        (other_type, phrase)
        for other_type, phrases in kg.authority_phrases_by_role_type.items()
        if other_type != role_type
        for phrase in phrases
    ]
    for m in AUTHORITY_CLAIM_RE.finditer(text):
        window_text = _local_window(m.start(), m.end())
        window_words = significant_words(window_text)
        for other_type, phrase in other_role_phrases:
            if _overlap_strong_enough(significant_words(phrase), window_words):
                return {
                    "n_applicable": True,
                    "violation": True,
                    "reason": f"cross_role_authority_claim:{other_type}:{phrase[:60]}",
                }

    return {"n_applicable": True, "violation": False, "reason": None}


# 숫자 하나당 레코드 하나. 화폐 기호가 숫자 "앞"(S$500)이든 배수 단위가 숫자 "뒤"(500 million)든
# 같은 숫자에 대한 것이면 한 번만 정규화한다 — 예전에는 접두형 패턴과 접미형 패턴이 "S$500 million"의
# 앞부분/뒷부분을 따로 잘라내 "S$500"(=500)과 "500 million"(=5억)을 별개 토큰 두 개로 만들었고,
# 배수 없는 "S$500" 쪽이 코퍼스의 무관한 작은 숫자와 우연히 일치/불일치하며 오판정을 냈다(발견한 버그).
NUMBER_CORE_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
CURRENCY_TOKENS = ("usd", "sgd", "krw", "s$", "₩", "원")
SCALE_MULTIPLIERS = {"billion": 1e9, "million": 1e6, "thousand": 1e3, "k": 1e3, "억": 1e8, "만": 1e4}
SCALE_AFTER_RE = re.compile(r"\s?(billion|million|thousand|k(?![a-z])|억|만)", re.IGNORECASE)
# "S$500M"/"S$1.2B"처럼 화폐 기호+숫자 뒤 대문자 축약(M/B, 공백 없이)도 스케일로 인정한다.
# 통화 문맥이 있을 때만 적용(그냥 "50m"이 미터/분 등을 뜻하는 오탐 방지).
SCALE_ABBREV_RE = re.compile(r"(M|B)(?![a-zA-Z])")
SCALE_ABBREV_MULTIPLIERS = {"M": 1e6, "B": 1e9}


def extract_numeric_candidates(text: str) -> list[dict]:
    """퍼센트·화폐/스케일 금액·연도·콤마 구분 개수만 "체크 대상"으로 추출한다(사소한 맨숫자는 제외 —
    기존 범위 유지). 각 항목: {raw, class(percent/amount/year), value(정규화된 값)}.
    """
    if not text:
        return []
    results = []
    for m in NUMBER_CORE_RE.finditer(text):
        num_str = m.group(0)
        num = float(num_str.replace(",", ""))
        before = text[max(0, m.start() - 8): m.start()].lower()
        after_raw = text[m.end(): m.end() + 15]
        after = after_raw.lower()

        is_percent = after_raw.lstrip().startswith("%")
        scale_match = SCALE_AFTER_RE.match(after)
        scale = scale_match.group(1).lower() if scale_match else None
        has_currency = any(c in before for c in CURRENCY_TOKENS) or any(c in after[:6] for c in CURRENCY_TOKENS)
        is_year = bool(re.fullmatch(r"(19|20)\d{2}", num_str))
        is_comma_count = "," in num_str

        abbrev_match = None
        if has_currency and not scale:
            abbrev_match = SCALE_ABBREV_RE.match(after_raw)

        if is_percent:
            cls, value = "percent", num
        elif scale or has_currency:
            cls = "amount"
            if abbrev_match:
                value = num * SCALE_ABBREV_MULTIPLIERS[abbrev_match.group(1)]
            else:
                value = num * SCALE_MULTIPLIERS.get(scale, 1.0)
        elif is_year:
            cls, value = "year", num
        elif is_comma_count:
            cls, value = "amount", num
        else:
            continue  # 체크 대상 아님(사소한 맨숫자)

        if is_percent:
            percent_match = re.match(r"\s*%", after_raw)
            raw_end = m.end() + percent_match.end()
        elif abbrev_match:
            raw_end = m.end() + abbrev_match.end()
        else:
            raw_end = m.end() + (scale_match.end() if scale_match else 0)
        results.append({"raw": text[m.start():raw_end].strip(), "class": cls, "value": value})
    return results


def build_corpus_numeric_index(kg: PolicyKG) -> dict[str, list[float]]:
    index: dict[str, list[float]] = defaultdict(list)
    for doc in kg.grounding_corpus:
        for cand in extract_numeric_candidates(doc):
            index[cand["class"]].append(cand["value"])
    return index


def _is_supported(cls: str, value: float, corpus_index: dict[str, list[float]], rel_tol: float = 0.01) -> bool:
    for corpus_value in corpus_index.get(cls, []):
        if corpus_value == 0:
            continue
        if abs(value - corpus_value) <= rel_tol * max(abs(value), abs(corpus_value)):
            return True
    return False


def _is_derived_estimate(value: float, same_utt_supported: list[float], rel_tol: float = 0.03) -> bool:
    """같은 발화 안에서 이미 근거가 확인된(grounded) 값들끼리 곱/나눗셈/덧셈/뺄셈으로
    이 값을 재구성할 수 있으면 "원문 근거값에서 산술로 도출된 추정치"로 본다(§7.1 fabrication_rate와
    분리해서 report — 정의만 바뀌는 것, 위반/그라운딩 로직은 그대로).
    """
    vals = [v for v in same_utt_supported if v != 0]
    for a in vals:
        for b in vals:
            for candidate in (a * b, a / b if b else None, a + b, a - b):
                if candidate is None:
                    continue
                if abs(value - candidate) <= rel_tol * max(abs(value), abs(candidate), 1.0):
                    return True
    return False


def check_fabrication(utterance: dict, corpus_index: dict[str, list[float]]) -> dict:
    candidates = extract_numeric_candidates(utterance["text"] or "")
    if not candidates:
        return {
            "n_applicable": False, "fabricated": None, "unsupported_tokens": [],
            "derived_estimate_tokens": [], "has_unsupported": None, "has_derived_estimate": None,
        }

    supported_amounts = [
        c["value"] for c in candidates if c["class"] == "amount" and _is_supported(c["class"], c["value"], corpus_index)
    ]

    unsupported_tokens = []
    derived_estimate_tokens = []
    for c in candidates:
        if _is_supported(c["class"], c["value"], corpus_index):
            continue
        if c["class"] == "amount" and _is_derived_estimate(c["value"], supported_amounts):
            derived_estimate_tokens.append(c["raw"])
        else:
            unsupported_tokens.append(c["raw"])

    return {
        "n_applicable": True,
        "fabricated": len(unsupported_tokens) > 0,
        "unsupported_tokens": unsupported_tokens,
        "derived_estimate_tokens": derived_estimate_tokens,
        "has_unsupported": len(unsupported_tokens) > 0,
        "has_derived_estimate": len(derived_estimate_tokens) > 0,
    }


def compute_max_cosine(utterances: list[dict], kg: PolicyKG) -> dict[str, float]:
    """TF-IDF 코사인 최댓값. 형태소분석기(kiwipiepy 등) 없이 sklearn 기본 토크나이저 사용 —
    한국어는 어절 단위로만 쪼개져 정밀도가 낮다(§8 전까지 unvalidated로만 취급할 것).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = kg.grounding_corpus
    vectorizer = TfidfVectorizer(min_df=1)
    corpus_matrix = vectorizer.fit_transform(corpus)

    texts = [u["text"] or "" for u in utterances]
    utt_matrix = vectorizer.transform(texts)
    sims = cosine_similarity(utt_matrix, corpus_matrix)
    max_sims = sims.max(axis=1)
    return {u["utterance_id"]: float(sim) for u, sim in zip(utterances, max_sims)}


# grounded/fabrication은 정책 원문과 대조 가능한 phase에서만 판정한다. Outputs/Outcomes/Impact는
# 시뮬레이션 참가자 스스로의 예측/추정이라 애초에 "원문에 있었는지" 대조할 근거 텍스트가 없다
# (원문은 Inputs/Activities류의 기존 사실만 서술함). 이 phase들의 근거 검증은 이번 라운드에서
# 하지 않고 M4의 cross-phase coherence(§7.2, phase_summary 간 일관성)로 넘긴다.
GROUNDING_FABRICATION_PHASES = ("Inputs", "Activities")
PHASE_OUT_OF_SCOPE_FOR_GROUNDING = "phase_out_of_scope_for_grounding"


# ---------------------------------------------------------------------------
# grounding 임계값: gold-free within-policy decoy(귀무) 보정
#
# 이전 방식(측정 대상 세트 자신의 max_cosine 중앙값)은 자기참조였다 — 정의상 그 세트 발화의
# 딱 절반이 "중앙값 이상"이 되므로, 실제 grounding 품질과 무관하게 BK21_deepseek처럼 policy 내
# 모델이 하나뿐인 세트는 항상 정확히 0.500이 나왔다(측정이 아니라 항등식).
#
# 그다음 시도(cross-policy null: 다른 policy 코퍼스와 대조)는 자기참조는 없앴지만 다른 문제를
# 만들었다 — 그건 사실상 "이 발화가 100E 어휘를 쓰는가 vs BK21 어휘를 쓰는가"를 재는 것이지,
# "이 발화가 이 policy의 KG 규범 내용과 실제로 닮았는가"를 재는 게 아니다(두 정책 어휘가 서로
# 다르기만 하면 무엇을 재든 grounded로 보일 수 있음).
#
# 그래서 within-policy decoy로 바꾼다: 같은 policy KG 코퍼스의 토큰을 문서 경계 넘어 무작위
# 재배치해 "이 policy 어휘는 그대로 쓰지만 실제 문서의 term 공기(co-occurrence) 구조는 파괴된"
# 가짜 코퍼스를 만들고, 그것과의 max_cosine 분포 상위 백분위를 τ로 삼는다. 이러면 τ는
# "이 policy의 일반 어휘 수준에서 우연히 나올 수 있는 유사도"를 대표하고, grounded 판정은
# "그 우연 수준을 넘어 실제 규범 내용과 닮았는가"를 잰다. cross-policy 코퍼스는 쓰지 않는다.
NULL_PERCENTILE_DEFAULT = 95  # τ = 귀무분포의 이 백분위값. 5%만 "우연히" 넘도록 잡음.
GROUNDING_NULL_MODE = "within_policy_decoy"  # 유일한 null 소스 — cross-policy 코퍼스는 쓰지 않음
SHUFFLE_SEED = 20260719  # 재현성 고정 시드(임의 날짜 아님 — 이 작업을 실행한 날짜)
GROUNDING_THRESHOLD_SWEEP = [round(0.10 + 0.05 * i, 2) for i in range(10)]  # 0.10..0.55
NULL_FPR_SANITY_TOLERANCE = 0.03  # null 위양성률이 (1-p/100) 근처인지 확인하는 허용오차(코드 정확성 체크용, 검증 아님)


def build_shuffled_corpus(corpus: list[str], seed: int) -> list[str]:
    """코퍼스 전체 토큰 풀을 문서 경계를 넘어 무작위로 재배치해, 전역 어휘분포·문서별 길이는
    보존하되 문서별 term 공기(co-occurrence) 구조만 파괴한 decoy(귀무) 코퍼스를 만든다.
    TF-IDF는 bag-of-words라 문서 '내부' 토큰 순서 셔플은 벡터에 영향이 없으므로(무효), 반드시
    '문서 간' 재배치여야 한다 — 이 함수가 그렇게 한다: 전체 토큰을 한 풀에 모아 셔플한 뒤
    원본과 같은 길이로 다시 잘라 담는다.
    """
    rng = random.Random(seed)
    doc_tokens = [doc.split() for doc in corpus]
    lengths = [len(toks) for toks in doc_tokens]
    pool: list[str] = [t for toks in doc_tokens for t in toks]
    rng.shuffle(pool)
    shuffled = []
    idx = 0
    for length in lengths:
        shuffled.append(" ".join(pool[idx: idx + length]))
        idx += length
    return shuffled


def compute_null_max_cosine(
    utterances: list[dict],
    own_kg: PolicyKG,
    seed: int = SHUFFLE_SEED,
) -> dict[str, float]:
    """같은 policy KG 코퍼스의 decoy(문서 간 토큰 재배치, 전역 어휘·문서 길이 보존, term 공기
    구조만 파괴)와 대조한 max_cosine — "이 policy 어휘를 쓰긴 하지만 실제 규범 내용과는 무관한
    발화"가 우연히 얻을 수 있는 유사도 분포를 만든다. cross-policy(다른 policy) 코퍼스는 쓰지
    않는다 — 그건 "정책 간 구별"을 재는 것이지 "이 policy 규범 근거성"을 재는 게 아니기 때문.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    decoy_corpus = build_shuffled_corpus(own_kg.grounding_corpus, seed)
    vectorizer = TfidfVectorizer(min_df=1)
    corpus_matrix = vectorizer.fit_transform(decoy_corpus)
    texts = [u["text"] or "" for u in utterances]
    utt_matrix = vectorizer.transform(texts)
    sims = cosine_similarity(utt_matrix, corpus_matrix)
    max_sims = sims.max(axis=1)
    return {u["utterance_id"]: float(sim) for u, sim in zip(utterances, max_sims)}


def compute_null_threshold(null_values: list[float], null_percentile: float = NULL_PERCENTILE_DEFAULT) -> float:
    if not null_values:
        return 0.0
    return float(np.percentile(np.array(null_values), null_percentile))


def _distribution_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    arr = np.array(values)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _threshold_sweep(real_values: list[float], sweep: list[float] = GROUNDING_THRESHOLD_SWEEP) -> list[dict]:
    """τ 후보값별 grounded_ratio 표(보고 투명성 — 단일 임계값 의존을 줄이기 위해 병기)."""
    arr = np.array(real_values)
    n = len(arr)
    out = []
    for tau in sweep:
        n_grounded = int(np.sum(arr > tau)) if n else 0
        out.append({"tau": tau, "n_grounded": n_grounded, "n": n, "grounded_ratio": (n_grounded / n) if n else None})
    return out


def _null_false_positive_rate(null_values: list[float], tau: float) -> float | None:
    if not null_values:
        return None
    return float(np.mean(np.array(null_values) > tau))


def _separation_gap(real_summary: dict, null_summary: dict) -> float | None:
    if real_summary["median"] is None or null_summary["median"] is None:
        return None
    return real_summary["median"] - null_summary["median"]


def _assert_null_fpr_sanity(policy_id: str, label: str, null_fpr: float | None) -> None:
    """null 위양성률이 (1-p/100) 근처인지 확인 — percentile 정의상 항상 참에 가까운 코드
    정확성 체크일 뿐, grounded 판정 자체의 검증이 아니다(§8 gold precision/recall이 담당)."""
    if null_fpr is None:
        return
    expected = 1.0 - NULL_PERCENTILE_DEFAULT / 100.0
    assert abs(null_fpr - expected) < NULL_FPR_SANITY_TOLERANCE, (
        f"{policy_id}[{label}]: null 위양성률 sanity check 실패 — 관측 {null_fpr:.4f}, "
        f"기대 {expected:.4f} (허용오차 {NULL_FPR_SANITY_TOLERANCE})"
    )


# ---------------------------------------------------------------------------
# grounding 본 지표: semantic support (로컬 sentence-transformers, §11 허용 — 외부 LLM API 아님)
#
# lexical(TF-IDF)은 "같은 단어를 쓰는가"만 잡는다 — 정당한 패러프레이즈(단어는 다르지만 같은
# 규범 내용)를 놓치고, within-policy decoy 실험에서 separation_gap이 세 세트 전부 음수로 나온 것도
# 이 한계와 무관하지 않다(어휘 수준에서는 실제 발화가 decoy보다 딱히 더 KG를 안 닮았다는 뜻).
# 그래서 grounded_ratio(하나의 지표, 축 분리 없음)의 정의를 semantic으로 바꾼다: "발화의 질적
# 주장이 이 policy KG 규범(norm_units) 중 하나에 의미상 뒷받침되는가." lexical은 지우지 않고
# baseline 진단값(lex_*)으로 병기한다.
EMBEDDING_MODEL_NAME_DEFAULT = "paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_NULL_TOPK_EXCLUDE = 3  # 발화의 top-k 근접 norm unit은 "실제로 근거했을 후보"라 null에서 제외
SEMANTIC_NULL_SAMPLES_PER_UTT = 5  # 발화당 "무관한 규범" 무작위 추출 개수(귀무 표본 크기 확보)
TRACEABILITY_SAMPLE_SIZE = 200


@lru_cache(maxsize=4)
def get_embedder(model_name: str = EMBEDDING_MODEL_NAME_DEFAULT):
    """로컬 sentence-transformers 모델 로더(프로세스당 1회, lru_cache로 재사용).
    HuggingFace Hub에서 가중치를 최초 1회 내려받아 로컬에 캐시해두고 그 뒤로는 전부 로컬
    추론이다 — 매 호출마다 외부 서버에 텍스트를 보내는 LLM API 호출이 아니다(§11 허용 항목).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    assert model.__class__.__module__.startswith("sentence_transformers"), (
        "임베딩은 로컬 sentence-transformers로만 계산해야 한다(외부 LLM API 금지, §11/§12)."
    )
    return model


def embed_texts(model, texts: list[str]) -> np.ndarray:
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim))
    return np.asarray(
        model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    )


def compute_semantic_matches(
    utterances: list[dict], norm_units: list[dict], model
) -> tuple[dict[str, dict], np.ndarray]:
    """발화별로 norm_units 전체와의 semantic cosine을 구해 최상위 매칭(top1)과 그 값을 남긴다.
    pooled 코퍼스 전체에 대한 단일 max가 아니라 "어느 규범 unit에 근거했는가"(kind/ref_id/text)가
    traceability로 남아야 하므로 per-utterance 결과에 top1 정보를 함께 기록한다.
    반환: (utterance_id -> {sem_max_cosine, top1_norm_ref, top1_norm_text, top1_norm_kind}, sims 행렬).
    sims 행렬은 coherent null 표본 추출에도 재사용한다(임베딩 재계산 방지).
    """
    if not utterances:
        return {}, np.zeros((0, len(norm_units)))
    if not norm_units:
        empty = {
            u["utterance_id"]: {
                "sem_max_cosine": 0.0, "top1_norm_ref": None, "top1_norm_text": None, "top1_norm_kind": None,
            }
            for u in utterances
        }
        return empty, np.zeros((len(utterances), 0))

    unit_texts = [nu["text"] for nu in norm_units]
    unit_emb = embed_texts(model, unit_texts)
    utt_texts = [u["text"] or "" for u in utterances]
    utt_emb = embed_texts(model, utt_texts)
    sims = utt_emb @ unit_emb.T  # 정규화된 임베딩이므로 내적 = 코사인

    top1_idx = sims.argmax(axis=1)
    per_utt: dict[str, dict] = {}
    for i, u in enumerate(utterances):
        j = int(top1_idx[i])
        per_utt[u["utterance_id"]] = {
            "sem_max_cosine": float(sims[i, j]),
            "top1_norm_ref": norm_units[j]["ref_id"],
            "top1_norm_text": norm_units[j]["text"],
            "top1_norm_kind": norm_units[j]["kind"],
        }
    return per_utt, sims


def compute_semantic_coherent_null(
    sims: np.ndarray,
    seed: int,
    topk_exclude: int = SEMANTIC_NULL_TOPK_EXCLUDE,
    samples_per_utt: int = SEMANTIC_NULL_SAMPLES_PER_UTT,
) -> list[float]:
    """coherent null: word-salad(토큰 셔플)이 아니라, 각 발화의 top-k 근접 norm unit(=그 발화가
    실제로 근거했을 수 있는 후보)을 제외한 나머지 "같은 policy의 무관한 규범"들 중에서 무작위로
    뽑은 cosine을 귀무 표본으로 쓴다. 이러면 "무관한 규범보다 유의하게 더 가깝다"가 grounded의
    의미가 되어, 단순히 같은 주제(topical) 영역이라서 생기는 우연한 유사도까지 통제된다.
    """
    rng = random.Random(seed)
    n_utts, n_units = sims.shape
    null_values: list[float] = []
    if n_units <= topk_exclude:
        return null_values  # norm unit이 너무 적어 "무관한 나머지"를 만들 수 없는 방어적 케이스
    for i in range(n_utts):
        row = sims[i]
        top_idx = set(np.argsort(row)[-topk_exclude:].tolist())
        remaining = [j for j in range(n_units) if j not in top_idx]
        if not remaining:
            continue
        k = min(samples_per_utt, len(remaining))
        for j in rng.sample(remaining, k):
            null_values.append(float(row[j]))
    return null_values


def compute_set_metrics(
    utterances: list[dict],
    policy_kgs: dict[str, PolicyKG],
    embedding_model_name: str = EMBEDDING_MODEL_NAME_DEFAULT,
) -> dict:
    """(policy_id, model_id)별로 묶어서 §7.1 세 지표를 산출.
    grounding 지표의 변천: (1) 자기 세트 max_cosine 중앙값(자기참조, BK21 항상 0.500) ->
    (2) cross-policy null(사실상 "정책 간 어휘 구별"을 잼) -> (3) within-policy decoy null
    (자기참조·정책 간 오염은 없앴지만 lexical(TF-IDF)이라 정당한 패러프레이즈를 놓치고,
    separation_gap이 세 세트 전부 음수로 나옴 — 어휘 수준에서는 실제 발화가 decoy보다 딱히
    KG를 더 안 닮았다는 뜻). 지금은 semantic support로 바꾼다: 로컬 sentence-transformers
    임베딩으로 "발화가 이 policy KG의 개별 규범 unit 중 하나에 의미상 뒷받침되는가"를 재고,
    coherent null(같은 policy의 "무관한" 규범 unit — word-salad 아님)로 τ를 보정한다.
    lexical은 지우지 않고 baseline 진단값(lex_*)으로 병기. 여전히 §8 gold 검증 전까지 unvalidated.
    """
    embedder = get_embedder(embedding_model_name)

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for u in utterances:
        by_policy[u["policy_id"]].append(u)

    lex_max_cosine_by_uid: dict[str, float] = {}
    lex_threshold_by_policy: dict[str, float] = {}
    lex_diagnostics_by_policy: dict[str, dict] = {}
    sem_match_by_uid: dict[str, dict] = {}
    sem_threshold_by_policy: dict[str, float] = {}
    sem_diagnostics_by_policy: dict[str, dict] = {}

    for policy_id, policy_utts in by_policy.items():
        kg = policy_kgs[policy_id]
        in_scope_utts = [u for u in policy_utts if u["phase"] in GROUNDING_FABRICATION_PHASES]

        # --- lexical(TF-IDF) baseline: grounded 판정에는 쓰지 않고 참조/비교용으로만 병기 ---
        lex_cosines = compute_max_cosine(policy_utts, kg)
        lex_max_cosine_by_uid.update(lex_cosines)
        lex_in_scope_vals = [lex_cosines[u["utterance_id"]] for u in in_scope_utts]

        lex_null_vals = list(compute_null_max_cosine(in_scope_utts, kg, seed=SHUFFLE_SEED).values())
        lex_tau = compute_null_threshold(lex_null_vals, NULL_PERCENTILE_DEFAULT)
        lex_null_fpr = _null_false_positive_rate(lex_null_vals, lex_tau)
        _assert_null_fpr_sanity(policy_id, "lexical", lex_null_fpr)

        lex_real_summary = _distribution_summary(lex_in_scope_vals)
        lex_null_summary = _distribution_summary(lex_null_vals)

        lex_threshold_by_policy[policy_id] = lex_tau
        lex_diagnostics_by_policy[policy_id] = {
            "threshold_used": lex_tau,
            "threshold_source": f"within_policy_decoy_null_percentile_p{NULL_PERCENTILE_DEFAULT}",
            "real_distribution": lex_real_summary,
            "null_distribution": lex_null_summary,
            "separation_gap_median_real_minus_null": _separation_gap(lex_real_summary, lex_null_summary),
            "null_false_positive_rate_at_tau": lex_null_fpr,
            "threshold_sweep": _threshold_sweep(lex_in_scope_vals),
        }

        # --- semantic support: 본 지표(grounded_ratio 정의) ---
        sem_per_utt, sims = compute_semantic_matches(in_scope_utts, kg.norm_units, embedder)
        sem_match_by_uid.update(sem_per_utt)
        sem_in_scope_vals = [sem_per_utt[u["utterance_id"]]["sem_max_cosine"] for u in in_scope_utts]

        sem_null_vals = compute_semantic_coherent_null(sims, seed=SHUFFLE_SEED)
        sem_tau = compute_null_threshold(sem_null_vals, NULL_PERCENTILE_DEFAULT)
        sem_null_fpr = _null_false_positive_rate(sem_null_vals, sem_tau)
        _assert_null_fpr_sanity(policy_id, "semantic", sem_null_fpr)

        sem_real_summary = _distribution_summary(sem_in_scope_vals)
        sem_null_summary = _distribution_summary(sem_null_vals)

        sem_threshold_by_policy[policy_id] = sem_tau
        sem_diagnostics_by_policy[policy_id] = {
            "threshold_used": sem_tau,
            "threshold_source": f"semantic_coherent_null_percentile_p{NULL_PERCENTILE_DEFAULT}",
            "embedding_model": embedding_model_name,
            "real_distribution": sem_real_summary,
            "null_distribution": sem_null_summary,
            "separation_gap_median_real_minus_null": _separation_gap(sem_real_summary, sem_null_summary),
            "null_false_positive_rate_at_tau": sem_null_fpr,
            "threshold_sweep": _threshold_sweep(sem_in_scope_vals),
        }

    by_set: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in utterances:
        by_set[(u["policy_id"], u["model_id"])].append(u)

    per_utterance_labels: dict[str, dict] = {}
    set_results: dict[str, dict] = {}

    for (policy_id, model_id), utts in by_set.items():
        kg = policy_kgs[policy_id]
        corpus_index = build_corpus_numeric_index(kg)
        sem_threshold = sem_threshold_by_policy[policy_id]
        lex_threshold = lex_threshold_by_policy[policy_id]

        n_no_institutional_position = 0
        n_applicable = 0
        n_violation = 0
        n_fab_applicable = 0
        n_fab_unsupported = 0
        n_fab_derived_estimate = 0
        n_grounded = 0
        n_lex_grounded = 0
        n_phase_out_of_scope = 0

        for u in utts:
            viol = check_deontic_violation(u, kg)  # phase 무관 — role 기반이라 그대로 유지
            lex_max_cosine = lex_max_cosine_by_uid[u["utterance_id"]]
            in_scope = u["phase"] in GROUNDING_FABRICATION_PHASES

            if in_scope:
                fab = check_fabrication(u, corpus_index)
                sem_info = sem_match_by_uid[u["utterance_id"]]
                sem_max_cosine = sem_info["sem_max_cosine"]
                grounded = sem_max_cosine > sem_threshold  # 본 지표: semantic 기준
                lex_grounded = lex_max_cosine > lex_threshold  # baseline 진단용
                grounding_status = "applicable"
            else:
                fab = {
                    "n_applicable": False, "fabricated": None, "unsupported_tokens": [],
                    "derived_estimate_tokens": [], "has_unsupported": None, "has_derived_estimate": None,
                }
                sem_info = {"sem_max_cosine": None, "top1_norm_ref": None, "top1_norm_text": None, "top1_norm_kind": None}
                sem_max_cosine = None
                grounded = None
                lex_grounded = None
                grounding_status = PHASE_OUT_OF_SCOPE_FOR_GROUNDING
                n_phase_out_of_scope += 1

            if u["deontic_status"] == "no_institutional_position_in_kg":
                n_no_institutional_position += 1
            if viol["n_applicable"]:
                n_applicable += 1
                if viol["violation"]:
                    n_violation += 1
            if fab["n_applicable"]:
                n_fab_applicable += 1
                if fab["has_unsupported"]:
                    n_fab_unsupported += 1
                if fab["has_derived_estimate"]:
                    n_fab_derived_estimate += 1
            if grounded:
                n_grounded += 1
            if lex_grounded:
                n_lex_grounded += 1

            per_utterance_labels[u["utterance_id"]] = {
                "utterance_id": u["utterance_id"],
                "policy_id": policy_id,
                "model_id": model_id,
                "scenario_uid": u["scenario_uid"],
                "kg_role": u["kg_role"],
                "deontic_status": u["deontic_status"],
                "violation_applicable": viol["n_applicable"],
                "violation": viol["violation"],
                "violation_reason": viol["reason"],
                "grounding_status": grounding_status,
                "grounded": grounded,
                "sem_max_cosine": sem_max_cosine,
                "top1_norm_ref": sem_info["top1_norm_ref"],
                "top1_norm_text": sem_info["top1_norm_text"],
                "top1_norm_kind": sem_info["top1_norm_kind"],
                "lex_grounded": lex_grounded,
                "lex_max_cosine": lex_max_cosine,
                "fabrication_applicable": fab["n_applicable"],
                "fabricated": fab["has_unsupported"],
                "unsupported_tokens": fab["unsupported_tokens"],
                "derived_estimate": fab["has_derived_estimate"],
                "derived_estimate_tokens": fab["derived_estimate_tokens"],
            }

        n_in_scope = len(utts) - n_phase_out_of_scope
        set_name = f"{policy_id}_{model_id}"
        sem_diag = sem_diagnostics_by_policy[policy_id]
        lex_diag = lex_diagnostics_by_policy[policy_id]
        set_results[set_name] = {
            "policy_id": policy_id,
            "model_id": model_id,
            "n_utterances": len(utts),
            "n_no_institutional_position_in_kg": n_no_institutional_position,
            "n_phase_out_of_scope_for_grounding": n_phase_out_of_scope,
            "violation_rate": {
                "n_applicable": n_applicable,
                "n_violation": n_violation,
                "rate": (n_violation / n_applicable) if n_applicable else None,
            },
            "grounded_ratio": {
                "basis": "semantic",
                "n": n_in_scope,
                "n_grounded": n_grounded,
                "rate": (n_grounded / n_in_scope) if n_in_scope else None,
                "threshold_used": sem_diag["threshold_used"],
                "threshold_source": sem_diag["threshold_source"],
                "embedding_model": sem_diag["embedding_model"],
                "threshold_note": "본 지표는 semantic support다: 발화가 이 policy KG의 개별 규범 unit "
                                   "(norm_units) 중 하나에 로컬 sentence-transformers 임베딩 기준으로 의미상 "
                                   "뒷받침되는가. τ는 coherent null(같은 policy의 '무관한' 규범 unit — 발화의 "
                                   f"top-{SEMANTIC_NULL_TOPK_EXCLUDE} 근접 후보 제외 후 무작위 샘플, word-salad 아님) "
                                   f"cosine 분포 상위 {NULL_PERCENTILE_DEFAULT}백분위. unvalidated(§8 gold 대조 전).",
                "scope_note": "Outputs/Outcomes/Impact는 phase_out_of_scope_for_grounding으로 분모 제외 "
                              "(원문과 대조할 근거가 없는 미래 예측 phase — 근거 검증은 M4 cross-phase coherence로 위임)",
                "real_distribution": sem_diag["real_distribution"],
                "null_distribution": sem_diag["null_distribution"],
                "separation_gap_median_real_minus_null": sem_diag["separation_gap_median_real_minus_null"],
                "null_false_positive_rate_at_tau": sem_diag["null_false_positive_rate_at_tau"],
                "null_false_positive_rate_note": "이 값은 percentile 정의상 항상 ≈(1-null_percentile)에 수렴한다 — "
                                                  "임계값 계산 코드가 올바르게 동작했는지 확인하는 정확성 체크일 뿐, "
                                                  "grounded 판정 자체의 실제 정확도 검증이 아니다(그건 §8 gold precision/recall의 역할).",
                "threshold_sweep": sem_diag["threshold_sweep"],
                "lexical_baseline": {
                    "n_grounded": n_lex_grounded,
                    "rate": (n_lex_grounded / n_in_scope) if n_in_scope else None,
                    "threshold_used": lex_diag["threshold_used"],
                    "threshold_source": lex_diag["threshold_source"],
                    "real_distribution": lex_diag["real_distribution"],
                    "null_distribution": lex_diag["null_distribution"],
                    "separation_gap_median_real_minus_null": lex_diag["separation_gap_median_real_minus_null"],
                    "null_false_positive_rate_at_tau": lex_diag["null_false_positive_rate_at_tau"],
                    "threshold_sweep": lex_diag["threshold_sweep"],
                    "note": "lexical(TF-IDF), within-policy decoy null 기준 — 참조/비교용 baseline이며 "
                            "grounded 판정에는 쓰지 않는다(본 지표는 위 semantic).",
                },
            },
            "fabrication_rate": {
                "n_applicable": n_fab_applicable,
                "n_fabricated": n_fab_unsupported,
                "rate": (n_fab_unsupported / n_fab_applicable) if n_fab_applicable else None,
                "note": "무근거 주장만(derived_estimate 제외)",
            },
            "derived_estimate_rate": {
                "n_applicable": n_fab_applicable,
                "n_derived": n_fab_derived_estimate,
                "rate": (n_fab_derived_estimate / n_fab_applicable) if n_fab_applicable else None,
                "note": "원문 근거값에서 산술로 도출 가능(합/곱/나눗셈/뺄셈, 오차 3%). fabrication_rate와 분리 표시, 합산 안 함",
            },
        }

    return {"set_results": set_results, "per_utterance": per_utterance_labels}
