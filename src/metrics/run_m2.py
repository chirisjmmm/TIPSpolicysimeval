"""M2 실행 스크립트: data/ir/utterances.jsonl + policy_graph.json -> §7.1 세 지표.
아직 human gold 대조(§8/M3) 전이므로 모든 수치는 unvalidated로만 취급한다(§12 가드레일).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from accuracy_micro import compute_set_metrics  # noqa: E402
from load_policy_text import load_all_policy_kgs  # noqa: E402
from load_scenarios import REPO_ROOT  # noqa: E402

RESULTS_DIR = REPO_ROOT / "data" / "ir"
OUT_DIR = REPO_ROOT / "results"


def load_utterances() -> list[dict]:
    path = RESULTS_DIR / "utterances.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    utterances = load_utterances()
    policy_kgs = load_all_policy_kgs()
    result = compute_set_metrics(utterances, policy_kgs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "m2_set_results.json", "w", encoding="utf-8") as f:
        json.dump(result["set_results"], f, ensure_ascii=False, indent=2)

    per_utt_cols = [
        "utterance_id", "policy_id", "model_id", "scenario_uid", "kg_role", "deontic_status",
        "violation_applicable", "violation", "violation_reason",
        "grounding_status", "grounded", "max_cosine",
        "fabrication_applicable", "fabricated", "unsupported_tokens",
        "derived_estimate", "derived_estimate_tokens",
    ]
    with open(OUT_DIR / "m2_per_utterance_labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_utt_cols)
        w.writeheader()
        for row in result["per_utterance"].values():
            r = dict(row)
            r["unsupported_tokens"] = "; ".join(r["unsupported_tokens"])
            r["derived_estimate_tokens"] = "; ".join(r["derived_estimate_tokens"])
            w.writerow(r)

    print("=" * 78)
    print("M2 (§7.1 Accuracy/micro) — 모든 수치는 UNVALIDATED (human gold 대조 전, §8/M3에서 검증)")
    print("메모: grounded/fabrication은 Inputs·Activities phase에서만 판정한다. Outputs/Outcomes/Impact는")
    print("     참가자 스스로의 미래 예측이라 원문 대조 근거가 없어 phase_out_of_scope_for_grounding으로")
    print("     분모 제외하고 별도 카운트만 남긴다 — 이 phase들의 근거 검증은 이번 라운드에서 하지 않고")
    print("     M4의 cross-phase coherence(phase_summary 간 일관성)로 넘긴다.")
    print("=" * 78)
    for set_name, r in sorted(result["set_results"].items()):
        vr = r["violation_rate"]
        gr = r["grounded_ratio"]
        fr = r["fabrication_rate"]
        print(f"\n[{set_name}]  n_utterances={r['n_utterances']}  "
              f"no_institutional_position_in_kg={r['n_no_institutional_position_in_kg']}  "
              f"phase_out_of_scope_for_grounding={r['n_phase_out_of_scope_for_grounding']}")
        vr_str = f"{vr['rate']:.3f}" if vr["rate"] is not None else "N/A"
        print(f"  violation_rate   = {vr_str}  (violation={vr['n_violation']} / applicable={vr['n_applicable']})")
        gr_str = f"{gr['rate']:.3f}" if gr["rate"] is not None else "N/A"
        print(f"  grounded_ratio   = {gr_str}  (grounded={gr['n_grounded']} / n={gr['n']}, "
              f"threshold={gr['threshold_used']:.3f} [{gr['threshold_note']}])")
        fr_str = f"{fr['rate']:.3f}" if fr["rate"] is not None else "N/A"
        print(f"  fabrication_rate = {fr_str}  (unsupported={fr['n_fabricated']} / applicable={fr['n_applicable']}) [{fr['note']}]")
        der = r["derived_estimate_rate"]
        der_str = f"{der['rate']:.3f}" if der["rate"] is not None else "N/A"
        print(f"  derived_estimate_rate = {der_str}  (derived={der['n_derived']} / applicable={der['n_applicable']}) [{der['note']}]")

    print(f"\n저장 완료: {OUT_DIR / 'm2_set_results.json'}, {OUT_DIR / 'm2_per_utterance_labels.csv'}")


if __name__ == "__main__":
    main()
