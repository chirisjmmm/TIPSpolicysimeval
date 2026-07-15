"""§7.1 Accuracy/micro: deontic compliance(위반률) · grounding · fabrication. 전부 rule/lexical, API-free.

입력은 IR(data/ir/utterances.jsonl)만 받는다 — 파싱 로직은 여기 섞지 않는다(§12 가드레일).
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

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


def compute_set_metrics(utterances: list[dict], policy_kgs: dict[str, PolicyKG]) -> dict:
    """(policy_id, model_id)별로 묶어서 §7.1 세 지표를 산출.
    grounding 임계값은 policy_id별로 GROUNDING_FABRICATION_PHASES 범위 발화(모델 풀링)의
    max_cosine 중앙값을 써서 데이터 기반으로 잡는다(고정값 0.12는 이 코퍼스 최솟값 0.164보다 낮아
    전부 grounded=True가 되는 무의미한 임계값이었음 — 실행 후 발견해 수정). 이 임계값 자체도
    §8 검증 전까지 unvalidated.
    """
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for u in utterances:
        by_policy[u["policy_id"]].append(u)

    max_cosine_by_uid: dict[str, float] = {}
    grounding_threshold_by_policy: dict[str, float] = {}
    for policy_id, policy_utts in by_policy.items():
        kg = policy_kgs[policy_id]
        cosines = compute_max_cosine(policy_utts, kg)
        max_cosine_by_uid.update(cosines)
        phase_by_uid = {u["utterance_id"]: u["phase"] for u in policy_utts}
        in_scope_vals = sorted(
            v for uid, v in cosines.items() if phase_by_uid[uid] in GROUNDING_FABRICATION_PHASES
        )
        median = in_scope_vals[len(in_scope_vals) // 2] if in_scope_vals else 0.0
        grounding_threshold_by_policy[policy_id] = median

    by_set: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in utterances:
        by_set[(u["policy_id"], u["model_id"])].append(u)

    per_utterance_labels: dict[str, dict] = {}
    set_results: dict[str, dict] = {}

    for (policy_id, model_id), utts in by_set.items():
        kg = policy_kgs[policy_id]
        corpus_index = build_corpus_numeric_index(kg)
        threshold = grounding_threshold_by_policy[policy_id]

        n_no_institutional_position = 0
        n_applicable = 0
        n_violation = 0
        n_fab_applicable = 0
        n_fab_unsupported = 0
        n_fab_derived_estimate = 0
        n_grounded = 0
        n_phase_out_of_scope = 0

        for u in utts:
            viol = check_deontic_violation(u, kg)  # phase 무관 — role 기반이라 그대로 유지
            max_cosine = max_cosine_by_uid[u["utterance_id"]]
            in_scope = u["phase"] in GROUNDING_FABRICATION_PHASES

            if in_scope:
                fab = check_fabrication(u, corpus_index)
                grounded = max_cosine >= threshold
                grounding_status = "applicable"
            else:
                fab = {
                    "n_applicable": False, "fabricated": None, "unsupported_tokens": [],
                    "derived_estimate_tokens": [], "has_unsupported": None, "has_derived_estimate": None,
                }
                grounded = None
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
                "max_cosine": max_cosine,
                "fabrication_applicable": fab["n_applicable"],
                "fabricated": fab["has_unsupported"],
                "unsupported_tokens": fab["unsupported_tokens"],
                "derived_estimate": fab["has_derived_estimate"],
                "derived_estimate_tokens": fab["derived_estimate_tokens"],
            }

        n_in_scope = len(utts) - n_phase_out_of_scope
        set_name = f"{policy_id}_{model_id}"
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
                "n": n_in_scope,
                "n_grounded": n_grounded,
                "rate": (n_grounded / n_in_scope) if n_in_scope else None,
                "threshold_used": threshold,
                "threshold_note": f"policy 내 {GROUNDING_FABRICATION_PHASES} phase 발화 max_cosine의 중앙값(데이터 기반, unvalidated)",
                "scope_note": "Outputs/Outcomes/Impact는 phase_out_of_scope_for_grounding으로 분모 제외 "
                              "(원문과 대조할 근거가 없는 미래 예측 phase — 근거 검증은 M4 cross-phase coherence로 위임)",
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
