"""§8 "제안 라벨"용 독립 판정기. accuracy_micro.py의 규칙 엔진과 다른 알고리즘으로
violation/grounded/fabrication을 다시 처음부터 판단한다(코드 재사용 없음, 상수/정규식도 별도).
이건 사람 라벨링을 돕는 "제안"일 뿐 최종 판정이 아니다 — accuracy_micro.py는 건드리지 않는다.

방법이 의도적으로 다른 점:
- violation: decision_authority 구문이 아니라 각 노드 type의 전체 summary 텍스트에서
  "다른 role에만 있고 자기 role엔 없는 단어(집합 차)"를 뽑아 비교한다. 부정 탐지도 문자 윈도우가
  아니라 절(., !, ?, ; 로 나눔) 단위로 본다.
- grounded: TF-IDF 코사인이 아니라 단어 4-gram 겹침 비율(이산적 포함 관계)을 쓴다.
- fabrication: 클래스별 단위 환산(억/만/million/billion/K) 없이, 숫자를 콤마만 제거한 순수
  문자열로 코퍼스 원문에 그대로 있는지만 본다(스케일 인식 없음 — 의도적으로 더 단순/문자적인 방법).
"""
from __future__ import annotations

import re
from collections import defaultdict

CLAUSE_SPLIT_RE = re.compile(r"[.!?;]")
NEGATION_WORDS = ("not ", "never ", "didn't", "did not", "failed to", "no longer", "unable to", "refused")
SELF_PRONOUNS = ("i ", "i'", "we ", "our team", "our group")
AUTHORITY_VERBS = ("decide", "approve", "allocate", "oversee", "authorize", "authorise", "sanction", "assign", "recommend")


def _role_type(kg_role: str) -> str:
    return kg_role.split(":", 1)[0]


def _sig_words(text: str, min_len: int = 5) -> set[str]:
    return set(w for w in re.findall(r"[a-zA-Z]{%d,}" % min_len, text.lower()))


def build_role_vocab(kg) -> dict[str, set[str]]:
    """type별로 그 type에 속한 모든 노드의 summary 텍스트를 합쳐 유의어 집합을 만든다."""
    by_type_text: dict[str, list[str]] = defaultdict(list)
    for node in kg.nodes:
        t = node.get("type", "")
        if node.get("summary"):
            by_type_text[t].append(node["summary"])
    return {t: _sig_words(" ".join(texts)) for t, texts in by_type_text.items()}


def suggest_violation(text: str, kg_role: str, kg, role_vocab: dict[str, set[str]]) -> tuple[bool | None, str]:
    role_type = _role_type(kg_role)
    if role_type not in kg.norms_by_role_type and role_type not in role_vocab:
        return None, "no_role_context"

    own_words = role_vocab.get(role_type, set())
    exclusive_by_other: dict[str, set[str]] = {}
    for t, words in role_vocab.items():
        if t == role_type:
            continue
        exclusive_by_other[t] = words - own_words

    own_must_texts = [n["fact"] for n in kg.norms_by_role_type.get(role_type, []) if n["deontic"] == "must" and n["fact"]]
    own_must_words = _sig_words(" ".join(own_must_texts))

    clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(text or "") if c.strip()]
    for clause in clauses:
        low = clause.lower()
        has_negation = any(neg in low for neg in NEGATION_WORDS)
        if has_negation and own_must_words:
            clause_words = _sig_words(clause)
            if clause_words & own_must_words:
                return True, "own_duty_negation_in_clause"

    for clause in clauses:
        low = clause.lower()
        has_self = any(p in low for p in SELF_PRONOUNS)
        has_verb = any(v in low for v in AUTHORITY_VERBS)
        if has_self and has_verb:
            clause_words = _sig_words(clause)
            for other_type, excl_words in exclusive_by_other.items():
                if clause_words & excl_words:
                    return True, f"exclusive_vocab_claim:{other_type}"

    return False, "no_cue_found"


NGRAM_N = 4
WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def _word_ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    words = [w.lower() for w in WORD_RE.findall(text or "")]
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_corpus_ngram_set(kg) -> set[tuple[str, ...]]:
    ngrams: set[tuple[str, ...]] = set()
    for doc in kg.grounding_corpus:
        ngrams |= _word_ngrams(doc)
    return ngrams


def suggest_grounded(text: str, corpus_ngrams: set[tuple[str, ...]], threshold: float = 0.08) -> tuple[bool | None, float]:
    utt_ngrams = _word_ngrams(text)
    if not utt_ngrams:
        return None, 0.0
    overlap = len(utt_ngrams & corpus_ngrams) / len(utt_ngrams)
    return overlap >= threshold, round(overlap, 4)


PLAIN_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def build_corpus_raw_blob(kg) -> str:
    return re.sub(r",", "", " ".join(kg.grounding_corpus))


def suggest_fabrication(text: str, corpus_raw_blob: str) -> tuple[bool | None, list[str]]:
    numbers = [m.group(0) for m in PLAIN_NUMBER_RE.finditer(text or "") if "," in m.group(0) or "." in m.group(0)]
    if not numbers:
        return None, []
    unmatched = []
    for n in numbers:
        plain = n.replace(",", "")
        if plain not in corpus_raw_blob:
            unmatched.append(n)
    return len(unmatched) > 0, unmatched
