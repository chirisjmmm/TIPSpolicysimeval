"""M3(신뢰성 검증) 전 human gold 표본 추출 스크립트. 채점/분류 로직을 accuracy_micro.py에
반영하지 않는다 — 여기서 하는 계산은 전부 "표 추출을 위한 후보 선정 보조 휴리스틱"이며
최종 판단은 사람이 CSV를 읽고 한다.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from accuracy_micro import (  # noqa: E402
    AUTHORITY_CLAIM_RE,
    NEGATION_RE,
    _role_type_of,
    significant_words,
    strip_subject_and_deontic,
)
from load_policy_text import load_all_policy_kgs  # noqa: E402
from load_scenarios import REPO_ROOT  # noqa: E402

GOLD_DIR = REPO_ROOT / "data" / "gold"
IR_DIR = REPO_ROOT / "data" / "ir"
RESULTS_DIR = REPO_ROOT / "results"
SEED = 20260715


def load_utterances() -> dict[str, dict]:
    path = IR_DIR / "utterances.jsonl"
    return {(u := json.loads(line))["utterance_id"]: u for line in open(path, encoding="utf-8")}


def load_labels() -> dict[str, dict]:
    path = RESULTS_DIR / "m2_per_utterance_labels.csv"
    return {r["utterance_id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


# ---------------------------------------------------------------------------
# 1) violation: 강화 전(단어 1개 겹침) vs 강화 후(단어 2개 이상 겹침) 재구성 후 diff
# ---------------------------------------------------------------------------
WINDOW = 80


def _local_window(text: str, s: int, e: int) -> str:
    return text[max(0, s - WINDOW): e + WINDOW]


def check_violation_pre_hardening(utterance: dict, kg) -> dict:
    """§ 강화 전 규칙: ±80자 윈도우 + 유의어 '1개 이상' 겹치면 위반(과거에 실제로 쓰던 조건)."""
    if utterance["deontic_status"] != "applicable":
        return {"applicable": False, "violation": None, "reason": None, "window_text": None}
    role_type = _role_type_of(utterance["kg_role"])
    own_norms = kg.norms_by_role_type.get(role_type, [])
    if not own_norms:
        return {"applicable": False, "violation": None, "reason": None, "window_text": None}

    text = utterance["text"] or ""
    own_must_phrases = [strip_subject_and_deontic(n["fact"]) for n in own_norms if n["deontic"] == "must" and n["fact"]]
    for m in NEGATION_RE.finditer(text):
        w_text = _local_window(text, m.start(), m.end())
        w_words = significant_words(w_text)
        for phrase in own_must_phrases:
            if significant_words(phrase) & w_words:
                return {"applicable": True, "violation": True, "reason": f"must_negation:{phrase}", "window_text": w_text}

    other_role_phrases = [
        (t, p) for t, ps in kg.authority_phrases_by_role_type.items() if t != role_type for p in ps
    ]
    for m in AUTHORITY_CLAIM_RE.finditer(text):
        w_text = _local_window(text, m.start(), m.end())
        w_words = significant_words(w_text)
        for other_type, phrase in other_role_phrases:
            if significant_words(phrase) & w_words:
                return {
                    "applicable": True, "violation": True,
                    "reason": f"cross_role_authority_claim:{other_type}:{phrase}", "window_text": w_text,
                }
    return {"applicable": True, "violation": False, "reason": None, "window_text": None}


def check_violation_post_hardening(utterance: dict, kg) -> dict:
    """§ 강화 후 규칙: 위와 동일 + 유의어 '2개 이상'(구문 유의어가 1개뿐이면 1개) 겹쳐야 인정."""
    if utterance["deontic_status"] != "applicable":
        return {"applicable": False, "violation": None, "reason": None, "window_text": None}
    role_type = _role_type_of(utterance["kg_role"])
    own_norms = kg.norms_by_role_type.get(role_type, [])
    if not own_norms:
        return {"applicable": False, "violation": None, "reason": None, "window_text": None}

    def strong_enough(phrase_words: set[str], window_words: set[str]) -> bool:
        overlap = phrase_words & window_words
        if not phrase_words:
            return False
        return len(overlap) >= min(2, len(phrase_words)) if len(phrase_words) > 1 else len(overlap) == 1

    text = utterance["text"] or ""
    own_must_phrases = [strip_subject_and_deontic(n["fact"]) for n in own_norms if n["deontic"] == "must" and n["fact"]]
    for m in NEGATION_RE.finditer(text):
        w_text = _local_window(text, m.start(), m.end())
        w_words = significant_words(w_text)
        for phrase in own_must_phrases:
            if strong_enough(significant_words(phrase), w_words):
                return {"applicable": True, "violation": True, "reason": f"must_negation:{phrase}", "window_text": w_text}

    other_role_phrases = [
        (t, p) for t, ps in kg.authority_phrases_by_role_type.items() if t != role_type for p in ps
    ]
    for m in AUTHORITY_CLAIM_RE.finditer(text):
        w_text = _local_window(text, m.start(), m.end())
        w_words = significant_words(w_text)
        for other_type, phrase in other_role_phrases:
            if strong_enough(significant_words(phrase), w_words):
                return {
                    "applicable": True, "violation": True,
                    "reason": f"cross_role_authority_claim:{other_type}:{phrase}", "window_text": w_text,
                }
    return {"applicable": True, "violation": False, "reason": None, "window_text": None}


def build_violation_qa_table(utterances: dict[str, dict], policy_kgs: dict) -> list[dict]:
    kg = policy_kgs["BK21"]
    rows = []
    for uid, u in utterances.items():
        if u["policy_id"] != "BK21" or u["model_id"] != "deepseek":
            continue
        pre = check_violation_pre_hardening(u, kg)
        post = check_violation_post_hardening(u, kg)
        if not pre["violation"] and not post["violation"]:
            continue
        status = "dropped_by_hardening" if pre["violation"] and not post["violation"] else "confirmed_after_hardening"
        rows.append(
            {
                "status": status,
                "utterance_id": uid,
                "scenario_uid": u["scenario_uid"],
                "kg_role": u["kg_role"],
                "pre_reason": pre["reason"],
                "post_reason": post["reason"],
                "trigger_window_text": pre["window_text"] or post["window_text"],
                "full_text": u["text"],
                "evidence": " | ".join(u.get("evidence") or []),
            }
        )
    rows.sort(key=lambda r: (r["status"], r["scenario_uid"]))
    return rows


# ---------------------------------------------------------------------------
# 2) grounding: policy별 threshold(median) 근처 ±0.05 경계 발화 20건
# ---------------------------------------------------------------------------
def build_grounding_qa_table(utterances: dict[str, dict], labels: dict[str, dict]) -> list[dict]:
    by_policy_thresh: dict[str, float] = {}
    with open(RESULTS_DIR / "m2_set_results.json", encoding="utf-8") as f:
        set_results = json.load(f)
    for set_name, r in set_results.items():
        by_policy_thresh[r["policy_id"]] = r["grounded_ratio"]["threshold_used"]

    by_policy_candidates: dict[str, list[dict]] = {p: [] for p in by_policy_thresh}
    for uid, r in labels.items():
        u = utterances[uid]
        policy_id = u["policy_id"]
        thresh = by_policy_thresh[policy_id]
        cos = float(r["max_cosine"])
        if abs(cos - thresh) <= 0.05:
            by_policy_candidates[policy_id].append(
                {
                    "policy_id": policy_id,
                    "model_id": u["model_id"],
                    "scenario_uid": u["scenario_uid"],
                    "utterance_id": uid,
                    "kg_role": u["kg_role"],
                    "max_cosine": cos,
                    "threshold": thresh,
                    "grounded": r["grounded"],
                    "full_text": u["text"],
                }
            )

    rng = random.Random(SEED)
    rows = []
    for policy_id, candidates in by_policy_candidates.items():
        sample = rng.sample(candidates, min(20, len(candidates)))
        sample.sort(key=lambda r: r["max_cosine"])
        rows.extend(sample)
        print(f"  [grounding] {policy_id}: 경계(±0.05) 후보 {len(candidates)}건 중 {len(sample)}건 샘플")
    return rows


# ---------------------------------------------------------------------------
# 3) fabrication: "산술로 도출 가능해 보임" vs "근거 전혀 안 보임" 후보 분리(선정 보조 휴리스틱일 뿐,
#    최종 판단 아님)
# ---------------------------------------------------------------------------
ARITHMETIC_CUE_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*[\*x×]\s*\d+(?:\.\d+)?)"      # "30 * 615"
    r"|(\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?)"           # "2.478 / 3"
    r"|\b(per group|per university|per student|per capita)\b"
    r"|\b(average|averaging|proportional|proportion|share of|out of)\b"
    r"|\b(divided by|times|multiplied)\b"
    r"|(\bso\b.{0,15}\babout\b)"                       # "so about X"
    r"|(≈)",
    re.IGNORECASE,
)


def build_fabrication_qa_table(utterances: dict[str, dict], labels: dict[str, dict]) -> list[dict]:
    fabricated_uids = [uid for uid, r in labels.items() if r["fabricated"] == "True"]
    with_cue, without_cue = [], []
    for uid in fabricated_uids:
        u = utterances[uid]
        r = labels[uid]
        text = u["text"] or ""
        row = {
            "utterance_id": uid,
            "policy_id": u["policy_id"],
            "model_id": u["model_id"],
            "scenario_uid": u["scenario_uid"],
            "unsupported_tokens": r["unsupported_tokens"],
            "heuristic_bucket_guess": None,  # 아래서 채움, 최종 라벨 아님(사람이 다시 판단)
            "full_text": text,
        }
        if ARITHMETIC_CUE_RE.search(text):
            row["heuristic_bucket_guess"] = "a_arithmetic_cue_nearby"
            with_cue.append(row)
        else:
            row["heuristic_bucket_guess"] = "b_no_visible_basis"
            without_cue.append(row)

    rng = random.Random(SEED)
    sample_a = rng.sample(with_cue, min(15, len(with_cue)))
    sample_b = rng.sample(without_cue, min(15, len(without_cue)))
    print(f"  [fabrication] 산술단서 있음(휴리스틱) {len(with_cue)}건 중 {len(sample_a)}건 샘플")
    print(f"  [fabrication] 산술단서 없음(휴리스틱) {len(without_cue)}건 중 {len(sample_b)}건 샘플")
    rows = sample_a + sample_b
    rows.sort(key=lambda r: (r["heuristic_bucket_guess"], r["scenario_uid"]))
    return rows


def write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def print_preview(rows: list[dict], columns: list[str], title: str, n: int = 8) -> None:
    print(f"--- {title} (n={len(rows)}, 미리보기 최대 {n}행) ---")
    for r in rows[:n]:
        parts = []
        for c in columns:
            v = str(r.get(c, ""))
            if len(v) > 90:
                v = v[:90] + "..."
            parts.append(f"{c}={v}")
        print(" | ".join(parts))
    print()


def main() -> None:
    utterances = load_utterances()
    labels = load_labels()
    policy_kgs = load_all_policy_kgs()

    print("=== 1) violation: 강화 전/후 diff (BK21_deepseek) ===")
    viol_rows = build_violation_qa_table(utterances, policy_kgs)
    n_dropped = sum(1 for r in viol_rows if r["status"] == "dropped_by_hardening")
    n_confirmed = sum(1 for r in viol_rows if r["status"] == "confirmed_after_hardening")
    print(f"  dropped_by_hardening={n_dropped}, confirmed_after_hardening={n_confirmed}")
    viol_cols = ["status", "utterance_id", "scenario_uid", "kg_role", "pre_reason", "post_reason",
                 "trigger_window_text", "full_text", "evidence"]
    write_csv(viol_rows, GOLD_DIR / "qa_m2_violation_diff.csv", viol_cols)
    print_preview(viol_rows, ["status", "scenario_uid", "kg_role", "post_reason", "trigger_window_text"], "violation diff")

    print("=== 2) grounding: policy별 threshold ±0.05 경계 발화 ===")
    ground_rows = build_grounding_qa_table(utterances, labels)
    for r in ground_rows:
        r["gold_grounded"] = ""  # 사람이 채울 빈 컬럼(true/false) — src/reliability/threshold_analysis.py가 읽음
    ground_cols = ["policy_id", "model_id", "scenario_uid", "utterance_id", "kg_role",
                   "max_cosine", "threshold", "grounded", "gold_grounded", "full_text"]
    write_csv(ground_rows, GOLD_DIR / "qa_m2_grounding_boundary.csv", ground_cols)
    print_preview(ground_rows, ["policy_id", "scenario_uid", "max_cosine", "threshold", "grounded"], "grounding boundary")

    print("=== 3) fabrication: 산술단서 있음/없음 분리 ===")
    fab_rows = build_fabrication_qa_table(utterances, labels)
    fab_cols = ["heuristic_bucket_guess", "utterance_id", "policy_id", "model_id", "scenario_uid",
                "unsupported_tokens", "full_text"]
    write_csv(fab_rows, GOLD_DIR / "qa_m2_fabrication_split.csv", fab_cols)
    print_preview(fab_rows, ["heuristic_bucket_guess", "scenario_uid", "unsupported_tokens"], "fabrication split")

    print("저장 위치:", GOLD_DIR)
    print("  - qa_m2_violation_diff.csv")
    print("  - qa_m2_grounding_boundary.csv")
    print("  - qa_m2_fabrication_split.csv")
    print(f"random seed = {SEED}")


if __name__ == "__main__":
    main()
