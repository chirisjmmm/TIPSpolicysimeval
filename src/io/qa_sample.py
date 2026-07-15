"""M3 이전 human QA용 조회/추출 스크립트. 채점 로직 없음 — 표만 뽑아서 CSV로 저장하고
콘솔에 미리보기를 출력한다. 판정은 사람이 CSV를 열어서 직접 한다.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_scenarios import REPO_ROOT, SET_SPECS, load_config, load_set, resolve_participant_role  # noqa: E402

GOLD_DIR = REPO_ROOT / "data" / "gold"
SEED = 20260714  # 고정 시드 (재현 가능)


def build_participant_rows(config: dict) -> list[dict]:
    rows = []
    for set_name, policy_id, model_id in SET_SPECS:
        scenarios, _ = load_set(set_name, policy_id, model_id, config)
        for s in scenarios:
            for p in s.participants:
                role_info = resolve_participant_role(p, policy_id, config)
                rows.append(
                    {
                        "set_name": set_name,
                        "policy_id": policy_id,
                        "model_id": model_id,
                        "scenario_uid": s.scenario_uid,
                        "persona_name": p["name"],
                        "stakeholder_type": role_info["stakeholder_type"],
                        "kg_role": role_info["kg_role"],
                        "mislabel_source": role_info["mislabel_source"],
                        "deontic_status": role_info["deontic_status"],
                        "entity_name": p.get("entity_name", ""),
                        "professional_persona": p.get("professional_persona", ""),
                        "occupation": p.get("occupation", ""),
                        "education_level": p.get("education_level", ""),
                        "flag_policyrole": role_info["flag_policyrole"],
                    }
                )
    return rows


def write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in columns})


def print_preview(rows: list[dict], columns: list[str], title: str, truncate_col: str = "professional_persona", width: int = 70) -> None:
    print(f"--- {title} (n={len(rows)}, 미리보기 최대 10행, 전체는 CSV 참조) ---")
    header = " | ".join(columns)
    print(header)
    print("-" * len(header))
    for r in rows[:10]:
        vals = []
        for c in columns:
            v = str(r[c])
            if c == truncate_col and len(v) > width:
                v = v[:width] + "..."
            vals.append(v)
        print(" | ".join(vals))
    print()


def main() -> None:
    config = load_config()
    rows = build_participant_rows(config)
    rng = random.Random(SEED)

    # v2: BK21 policyrole-merge + 100E lexicon 협소화 반영 재검증. kg_role/mislabel_source 컬럼 추가.
    cols_persona = [
        "scenario_uid", "persona_name", "entity_name", "professional_persona",
        "flag_policyrole", "kg_role", "mislabel_source",
    ]

    # 1) 100E_gemini PolicyRole 전원 (flag 여부 무관)
    q1 = [r for r in rows if r["set_name"] == "100E_gemini" and r["stakeholder_type"] == "PolicyRole"]
    write_csv(q1, GOLD_DIR / "qa1_v2_100E_gemini_policyrole_all.csv", cols_persona)
    print(f"[1] 100E_gemini PolicyRole 전원: {len(q1)}명 (v1과 동일 모수, flag_policyrole만 재계산)")
    print_preview(q1, cols_persona, "100E_gemini PolicyRole 전원 (v2)")

    # 2) BK21_deepseek flag_policyrole=True 중 무작위 15명 (모두 kg_role=GraduateStudent:GraduateStudent로 병합됐는지 확인)
    q2_pool = [r for r in rows if r["set_name"] == "BK21_deepseek" and r["flag_policyrole"]]
    q2 = rng.sample(q2_pool, min(15, len(q2_pool)))
    q2.sort(key=lambda r: r["scenario_uid"])
    write_csv(q2, GOLD_DIR / "qa2_v2_BK21_deepseek_flagged_sample15.csv", cols_persona)
    print(f"[2] BK21_deepseek flag_policyrole=True 전체 {len(q2_pool)}명 중 무작위 {len(q2)}명 (seed={SEED})")
    print_preview(q2, cols_persona, "BK21_deepseek flagged sample15 (v2)")

    # 3) 세 set 전체, stakeholder_type별 kg_role 배정 무작위 10명씩
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["stakeholder_type"]].append(r)
    cols_kgrole = [
        "policy_id", "model_id", "scenario_uid", "persona_name", "stakeholder_type",
        "kg_role", "mislabel_source", "entity_name",
    ]
    q3 = []
    for st in sorted(by_type):
        sample = rng.sample(by_type[st], min(10, len(by_type[st])))
        sample.sort(key=lambda r: r["scenario_uid"])
        q3.extend(sample)
    write_csv(q3, GOLD_DIR / "qa3_v2_kgrole_sample_by_stakeholder_type.csv", cols_kgrole)
    print("[3] stakeholder_type별 전체 모수 및 샘플 수:")
    for st in sorted(by_type):
        print(f"    {st}: 모수 {len(by_type[st])}명 -> 샘플 {min(10, len(by_type[st]))}명")
    print_preview(q3, cols_kgrole, "kg_role 배정 샘플(전체 stakeholder_type 묶음) (v2)", truncate_col="entity_name")

    print(f"CSV 저장 위치: {GOLD_DIR}")
    print("  - qa1_v2_100E_gemini_policyrole_all.csv")
    print("  - qa2_v2_BK21_deepseek_flagged_sample15.csv")
    print("  - qa3_v2_kgrole_sample_by_stakeholder_type.csv")
    print(f"random seed = {SEED}")


if __name__ == "__main__":
    main()
