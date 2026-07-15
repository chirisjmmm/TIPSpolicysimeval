"""random_gold_pool(300)을 120건(set당 40)으로 서브샘플링하고, accuracy_micro.py와는 다른
독립 판정기(independent_rater.py)로 violation/grounded/fabrication "제안 라벨"을 채운다.
무작위 20%는 제안을 비워 블라인드로 남긴다. 최종 판단은 사람이 한다.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from independent_rater import (  # noqa: E402
    build_corpus_ngram_set,
    build_corpus_raw_blob,
    build_role_vocab,
    suggest_fabrication,
    suggest_grounded,
    suggest_violation,
)
from load_policy_text import load_all_policy_kgs  # noqa: E402
from load_scenarios import REPO_ROOT  # noqa: E402

GOLD_DIR = REPO_ROOT / "data" / "gold"
SOURCE_POOL = GOLD_DIR / "random_gold_pool.csv"
SUBSAMPLE_SEED = 20260717  # 300 -> 120 서브샘플용 시드
BLIND_SEED = 20260718  # 블라인드 20% 선정용 시드
PER_SET_N = 40
BLIND_FRACTION = 0.2

OUTPUT_COLUMNS = [
    "utterance_id", "policy_id", "model_id", "scenario_uid", "phase", "round", "persona_name",
    "kg_role", "deontic_status",
    "machine_violation_applicable", "machine_violation", "machine_violation_reason",
    "machine_grounded", "machine_max_cosine",
    "machine_fabrication_applicable", "machine_fabricated", "machine_derived_estimate",
    "full_text",
    "is_blind",
    "suggested_violation", "suggested_violation_reason",
    "suggested_grounded", "suggested_grounded_score",
    "suggested_fabrication", "suggested_fabrication_tokens",
    "gold_violation", "gold_grounded", "gold_fabrication",
]


def load_source_pool() -> list[dict]:
    with open(SOURCE_POOL, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def subsample_120(rows: list[dict]) -> list[dict]:
    by_set: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_set[f"{r['policy_id']}_{r['model_id']}"].append(r)

    rng = random.Random(SUBSAMPLE_SEED)
    result = []
    for set_name, group_rows in by_set.items():
        k = min(PER_SET_N, len(group_rows))
        sample = rng.sample(group_rows, k)
        print(f"  [{set_name}] {len(group_rows)}건 중 {k}건 서브샘플")
        result.extend(sample)
    return result


def main() -> None:
    print("=== 300 -> 120 서브샘플 (set당 40, seed 고정) ===")
    source_rows = load_source_pool()
    sampled = subsample_120(source_rows)
    print(f"합계 {len(sampled)}건")

    policy_kgs = load_all_policy_kgs()
    role_vocab_by_policy = {pid: build_role_vocab(kg) for pid, kg in policy_kgs.items()}
    corpus_ngrams_by_policy = {pid: build_corpus_ngram_set(kg) for pid, kg in policy_kgs.items()}
    corpus_blob_by_policy = {pid: build_corpus_raw_blob(kg) for pid, kg in policy_kgs.items()}

    print("=== 독립 판정기로 제안 라벨 계산 ===")
    for r in sampled:
        policy_id = r["policy_id"]
        kg = policy_kgs[policy_id]
        text = r["full_text"]

        v_label, v_reason = suggest_violation(text, r["kg_role"], kg, role_vocab_by_policy[policy_id])
        g_label, g_score = suggest_grounded(text, corpus_ngrams_by_policy[policy_id])
        f_label, f_tokens = suggest_fabrication(text, corpus_blob_by_policy[policy_id])

        r["suggested_violation"] = "" if v_label is None else str(v_label)
        r["suggested_violation_reason"] = v_reason
        r["suggested_grounded"] = "" if g_label is None else str(g_label)
        r["suggested_grounded_score"] = g_score
        r["suggested_fabrication"] = "" if f_label is None else str(f_label)
        r["suggested_fabrication_tokens"] = "; ".join(f_tokens)

    print("=== 무작위 20% 블라인드 처리 ===")
    rng_blind = random.Random(BLIND_SEED)
    n_blind = round(len(sampled) * BLIND_FRACTION)
    blind_uids = set(rng_blind.sample([r["utterance_id"] for r in sampled], n_blind))
    print(f"  블라인드 {len(blind_uids)}/{len(sampled)}건")

    for r in sampled:
        is_blind = r["utterance_id"] in blind_uids
        r["is_blind"] = str(is_blind)
        if is_blind:
            r["suggested_violation"] = ""
            r["suggested_violation_reason"] = ""
            r["suggested_grounded"] = ""
            r["suggested_grounded_score"] = ""
            r["suggested_fabrication"] = ""
            r["suggested_fabrication_tokens"] = ""

    for r in sampled:
        r.setdefault("gold_violation", "")
        r.setdefault("gold_grounded", "")
        r.setdefault("gold_fabrication", "")

    sampled.sort(key=lambda r: (r["policy_id"], r["model_id"], r["scenario_uid"]))

    with open(SOURCE_POOL, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for r in sampled:
            w.writerow({c: r.get(c, "") for c in OUTPUT_COLUMNS})

    print(f"\n저장(덮어씀): {SOURCE_POOL} ({len(sampled)}행)")
    print(f"subsample_seed={SUBSAMPLE_SEED}, blind_seed={BLIND_SEED}")


if __name__ == "__main__":
    main()
