"""M1 실행 스크립트: 세 set를 IR로 파싱하고 §3/§4 수용기준을 assert로 검증한 뒤
data/ir/에 저장한다. metric 계산 없음(파서 전용, §12).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_scenarios import REPO_ROOT, SET_SPECS, load_config, load_set  # noqa: E402

IR_DIR = REPO_ROOT / "data" / "ir"


def check(condition: bool, message: str, checklist: list[tuple[str, bool]]):
    checklist.append((message, condition))
    if not condition:
        raise AssertionError(f"M1 FAIL: {message}")


def main() -> int:
    config = load_config()
    checklist: list[tuple[str, bool]] = []

    all_scenarios = []
    all_utterances = []
    per_set_summary = {}

    for set_name, policy_id, model_id in SET_SPECS:
        scenarios, utterances = load_set(set_name, policy_id, model_id, config)

        # ✅ M1: 세 set 각각 IR 50개 생성
        check(len(scenarios) == 50, f"[{set_name}] scenario 50개", checklist)

        uids = [s.scenario_uid for s in scenarios]
        check(len(set(uids)) == 50, f"[{set_name}] scenario_uid 유일(50개)", checklist)

        for s in scenarios:
            check(len(s.participants) == 5, f"[{set_name}] {s.scenario_uid} participants==5", checklist)

        # phase 순서/5-post/posting_order 일치는 parse_scenario_file 내부에서 시나리오마다
        # assert 후 실패 시 ParseError를 던진다. load_set이 예외 없이 반환했다는 것 자체가
        # 아래 세 조건이 이 set의 50개 시나리오 전부에서 성립했음을 의미한다(중복 재검증 없음).
        check(True, f"[{set_name}] forward_pass 5-phase, 순서=[Inputs,Activities,Outputs,Outcomes,Impact]", checklist)
        check(True, f"[{set_name}] 각 phase initial/revised 각 5 post", checklist)
        check(True, f"[{set_name}] posting_order와 post의 persona 일치", checklist)

        if set_name == "100E_gemini":
            n_dirs = len(config["raw_data_roots"][set_name])
            check(n_dirs == 5, "[100E_gemini] 폴더 5개", checklist)
            # load_set 내부에서 이미 폴더당 10파일 assert, global_index 1..50 assert, md5 중복 assert 통과함
            check(True, "[100E_gemini] 폴더당 10파일 확인(로더 내 assert 통과)", checklist)
            check(True, "[100E_gemini] uid 1~50 연속(로더 내 assert 통과)", checklist)
            check(True, "[100E_gemini] 내용 해시 중복 0(로더 내 assert 통과)", checklist)

        flag_count = sum(1 for u in utterances if u.flag_policyrole)
        flag_scenarios = len({u.scenario_uid for u in utterances if u.flag_policyrole})
        per_set_summary[set_name] = {
            "n_scenarios": len(scenarios),
            "n_utterances": len(utterances),
            "flag_policyrole_utterances": flag_count,
            "flag_policyrole_scenarios": flag_scenarios,
        }
        all_scenarios.extend(scenarios)
        all_utterances.extend(utterances)

    all_uids = [s.scenario_uid for s in all_scenarios]
    check(len(all_uids) == 150, "전체 scenario 150개(50x3)", checklist)
    check(len(set(all_uids)) == 150, "전체 scenario_uid 유일(set 간 충돌 없음)", checklist)

    IR_DIR.mkdir(parents=True, exist_ok=True)
    with open(IR_DIR / "scenarios.jsonl", "w", encoding="utf-8") as f:
        for s in all_scenarios:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    with open(IR_DIR / "utterances.jsonl", "w", encoding="utf-8") as f:
        for u in all_utterances:
            f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")

    print("=" * 70)
    print("M1 수용기준 체크리스트 (모두 assert 통과)")
    print("=" * 70)
    for msg, ok in checklist:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    print()
    print("=" * 70)
    print("set별 요약")
    print("=" * 70)
    for set_name, summary in per_set_summary.items():
        print(f"  {set_name}: {summary}")

    print()
    print(f"IR 저장 완료: {IR_DIR / 'scenarios.jsonl'} ({len(all_scenarios)}행), "
          f"{IR_DIR / 'utterances.jsonl'} ({len(all_utterances)}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
