"""random_gold_pool + boundary_gold_pool에서 violation_applicable=True & deontic_status=applicable인
행만 모아 violation 축 전용 라벨링 표를 만든다. 중복 제거(utterance_id 기준), 판정 로직 없음 —
표만 뽑고 gold_violation은 빈 칸으로 둔다.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from load_policy_text import load_all_policy_kgs  # noqa: E402
from load_scenarios import REPO_ROOT  # noqa: E402

GOLD_DIR = REPO_ROOT / "data" / "gold"
IR_DIR = REPO_ROOT / "data" / "ir"
RESULTS_DIR = REPO_ROOT / "results"

OUTPUT_COLUMNS = [
    "utterance_id", "policy_id", "model_id", "scenario_uid", "persona_name", "kg_role",
    "phase", "round", "applicable_norms", "machine_violation", "machine_violation_reason",
    "full_text", "gold_violation",
]


def load_utterances() -> dict[str, dict]:
    path = IR_DIR / "utterances.jsonl"
    return {(u := json.loads(line))["utterance_id"]: u for line in open(path, encoding="utf-8")}


def load_labels() -> dict[str, dict]:
    path = RESULTS_DIR / "m2_per_utterance_labels.csv"
    return {r["utterance_id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def collect_source_uids() -> dict[str, set[str]]:
    sources = {}
    for name, path in [("random_gold_pool", GOLD_DIR / "random_gold_pool.csv"),
                        ("boundary_gold_pool", GOLD_DIR / "boundary_gold_pool.csv")]:
        with open(path, encoding="utf-8-sig") as f:
            sources[name] = {r["utterance_id"] for r in csv.DictReader(f)}
    return sources


def _role_type(kg_role: str) -> str:
    return kg_role.split(":", 1)[0]


def format_applicable_norms(kg_role: str, policy_id: str, policy_kgs: dict) -> str:
    role_type = _role_type(kg_role)
    norms = policy_kgs[policy_id].norms_by_role_type.get(role_type, [])
    if not norms:
        return "(등록된 must/can/cannot 규범 없음)"
    parts = []
    for n in norms:
        fact = n["fact"] or ""
        parts.append(f"[{n['deontic']}] {fact}")
    return " | ".join(parts)


def main() -> None:
    utterances = load_utterances()
    labels = load_labels()
    policy_kgs = load_all_policy_kgs()
    sources = collect_source_uids()

    all_uids = sources["random_gold_pool"] | sources["boundary_gold_pool"]
    print(f"random_gold_pool: {len(sources['random_gold_pool'])}건, "
          f"boundary_gold_pool: {len(sources['boundary_gold_pool'])}건, "
          f"합집합(중복 제거): {len(all_uids)}건")

    rows = []
    for uid in all_uids:
        r = labels.get(uid)
        if r is None:
            continue
        if r["violation_applicable"] != "True" or r["deontic_status"] != "applicable":
            continue
        u = utterances[uid]
        rows.append(
            {
                "utterance_id": uid,
                "policy_id": u["policy_id"],
                "model_id": u["model_id"],
                "scenario_uid": u["scenario_uid"],
                "persona_name": u["persona_name"],
                "kg_role": u["kg_role"],
                "phase": u["phase"],
                "round": u["round"],
                "applicable_norms": format_applicable_norms(u["kg_role"], u["policy_id"], policy_kgs),
                "machine_violation": r["violation"],
                "machine_violation_reason": r["violation_reason"],
                "full_text": u["text"],
                "gold_violation": "",
            }
        )

    rows.sort(key=lambda r: (r["policy_id"], r["model_id"], r["scenario_uid"], r["phase"]))

    out_path = GOLD_DIR / "violation_gold_pool.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"violation_applicable=True & deontic_status=applicable 필터 후: {len(rows)}건")
    from collections import Counter
    print("policy/model 분포:", dict(Counter(f"{r['policy_id']}_{r['model_id']}" for r in rows)))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
