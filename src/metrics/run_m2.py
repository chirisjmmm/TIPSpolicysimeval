"""M2 실행 스크립트: data/ir/utterances.jsonl + policy_graph.json -> §7.1 세 지표.
아직 human gold 대조(§8/M3) 전이므로 모든 수치는 unvalidated로만 취급한다(§12 가드레일).
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from accuracy_micro import TRACEABILITY_SAMPLE_SIZE, compute_set_metrics  # noqa: E402
from load_policy_text import load_all_policy_kgs  # noqa: E402
from load_scenarios import REPO_ROOT  # noqa: E402

RESULTS_DIR = REPO_ROOT / "data" / "ir"
OUT_DIR = REPO_ROOT / "results"
TRACEABILITY_SAMPLE_SEED = 20260719


def load_utterances() -> list[dict]:
    path = RESULTS_DIR / "utterances.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    utterances = load_utterances()
    text_by_uid = {u["utterance_id"]: u["text"] for u in utterances}
    policy_kgs = load_all_policy_kgs()
    result = compute_set_metrics(utterances, policy_kgs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "m2_set_results.json", "w", encoding="utf-8") as f:
        json.dump(result["set_results"], f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "m2_grounding_threshold_sweep.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set_name", "basis", "tau", "n_grounded", "n", "grounded_ratio"])
        for set_name, r in sorted(result["set_results"].items()):
            for row in r["grounded_ratio"]["threshold_sweep"]:
                w.writerow([set_name, "semantic", row["tau"], row["n_grounded"], row["n"], row["grounded_ratio"]])
            for row in r["grounded_ratio"]["lexical_baseline"]["threshold_sweep"]:
                w.writerow([set_name, "lexical_baseline", row["tau"], row["n_grounded"], row["n"], row["grounded_ratio"]])

    per_utt_cols = [
        "utterance_id", "policy_id", "model_id", "scenario_uid", "kg_role", "deontic_status",
        "violation_applicable", "violation", "violation_reason",
        "grounding_status", "grounded", "sem_max_cosine", "top1_norm_kind", "top1_norm_ref", "top1_norm_text",
        "lex_grounded", "lex_max_cosine",
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

    # traceability: grounded=True(semantic) 표본 최대 200건에 top1_norm_text/cosine 로깅.
    grounded_rows = [row for row in result["per_utterance"].values() if row["grounded"] is True]
    rng = random.Random(TRACEABILITY_SAMPLE_SEED)
    sample = rng.sample(grounded_rows, min(TRACEABILITY_SAMPLE_SIZE, len(grounded_rows)))
    with open(OUT_DIR / "m2_semantic_grounding_traceability_sample.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "utterance_id", "policy_id", "model_id", "kg_role",
            "sem_max_cosine", "top1_norm_kind", "top1_norm_ref", "top1_norm_text", "utterance_text",
        ])
        for row in sample:
            w.writerow([
                row["utterance_id"], row["policy_id"], row["model_id"], row["kg_role"],
                row["sem_max_cosine"], row["top1_norm_kind"], row["top1_norm_ref"], row["top1_norm_text"],
                text_by_uid.get(row["utterance_id"], ""),
            ])

    print("=" * 78)
    print("M2 (§7.1 Accuracy/micro) — 모든 수치는 UNVALIDATED (human gold 대조 전, §8/M3에서 검증)")
    print("메모: grounded/fabrication은 Inputs·Activities phase에서만 판정한다. Outputs/Outcomes/Impact는")
    print("     참가자 스스로의 미래 예측이라 원문 대조 근거가 없어 phase_out_of_scope_for_grounding으로")
    print("     분모 제외하고 별도 카운트만 남긴다 — 이 phase들의 근거 검증은 이번 라운드에서 하지 않고")
    print("     M4의 cross-phase coherence(phase_summary 간 일관성)로 넘긴다.")
    print("메모: grounded_ratio 본 지표는 이제 semantic support다(로컬 sentence-transformers, §11).")
    print("     lexical(TF-IDF) within-policy decoy에서는 세 세트 전부 separation_gap이 음수였다 —")
    print("     즉 실제 발화가 KG 코퍼스와 갖는 어휘 유사도가, 뒤섞은 decoy보다 딱히 높지 않았다는")
    print("     뜻이라 lexical 정의로는 \"규범 근거성\" 신호를 못 냈다. semantic은 발화-규범 unit 쌍의")
    print("     의미 유사도를 coherent null(같은 policy의 '무관한' 규범 unit, word-salad 아님) 대비")
    print("     상위 95백분위와 비교한다. lexical은 지우지 않고 baseline으로 병기(grounded_ratio.lexical_baseline).")
    print("     null 위양성률 assert는 임계값 계산 코드의 정확성 체크일 뿐 grounded 판정 자체의")
    print("     검증이 아니다(§8 gold precision/recall이 담당, model-based라 더더욱 unvalidated).")
    print("=" * 78)
    for set_name, r in sorted(result["set_results"].items()):
        vr = r["violation_rate"]
        gr = r["grounded_ratio"]
        lb = gr["lexical_baseline"]
        fr = r["fabrication_rate"]
        print(f"\n[{set_name}]  n_utterances={r['n_utterances']}  "
              f"no_institutional_position_in_kg={r['n_no_institutional_position_in_kg']}  "
              f"phase_out_of_scope_for_grounding={r['n_phase_out_of_scope_for_grounding']}")
        vr_str = f"{vr['rate']:.3f}" if vr["rate"] is not None else "N/A"
        print(f"  violation_rate   = {vr_str}  (violation={vr['n_violation']} / applicable={vr['n_applicable']})")

        gr_str = f"{gr['rate']:.3f}" if gr["rate"] is not None else "N/A"
        print(f"  grounded_ratio(semantic, 본 지표) = {gr_str}  (grounded={gr['n_grounded']} / n={gr['n']}, "
              f"tau={gr['threshold_used']:.3f}, model={gr['embedding_model']})")
        rd, nd = gr["real_distribution"], gr["null_distribution"]
        print(f"    sem real(vs 실제 규범 unit) cosine: mean={rd['mean']:.3f} median={rd['median']:.3f} "
              f"p25={rd['p25']:.3f} p75={rd['p75']:.3f} min={rd['min']:.3f} max={rd['max']:.3f}")
        print(f"    sem null(vs 무관한 규범 unit) cosine: mean={nd['mean']:.3f} median={nd['median']:.3f} "
              f"p25={nd['p25']:.3f} p75={nd['p75']:.3f} min={nd['min']:.3f} max={nd['max']:.3f}")
        gap = gr["separation_gap_median_real_minus_null"]
        gap_str = f"{gap:.3f}" if gap is not None else "N/A"
        print(f"    separation_gap(median real-null) = {gap_str}  <- τ 선택과 무관한 직접 요약")
        nfpr = gr["null_false_positive_rate_at_tau"]
        print(f"    null_false_positive_rate_at_tau={nfpr:.4f} (기대치 ~0.05, 코드 정확성 체크용)")

        lb_str = f"{lb['rate']:.3f}" if lb["rate"] is not None else "N/A"
        lb_gap = lb["separation_gap_median_real_minus_null"]
        lb_gap_str = f"{lb_gap:.3f}" if lb_gap is not None else "N/A"
        print(f"  grounded_ratio(lexical baseline)  = {lb_str}  tau={lb['threshold_used']:.3f}  "
              f"separation_gap={lb_gap_str}")

        fr_str = f"{fr['rate']:.3f}" if fr["rate"] is not None else "N/A"
        print(f"  fabrication_rate = {fr_str}  (unsupported={fr['n_fabricated']} / applicable={fr['n_applicable']}) [{fr['note']}]")
        der = r["derived_estimate_rate"]
        der_str = f"{der['rate']:.3f}" if der["rate"] is not None else "N/A"
        print(f"  derived_estimate_rate = {der_str}  (derived={der['n_derived']} / applicable={der['n_applicable']}) [{der['note']}]")

    print(f"\n저장 완료: {OUT_DIR / 'm2_set_results.json'}, {OUT_DIR / 'm2_per_utterance_labels.csv'}, "
          f"{OUT_DIR / 'm2_grounding_threshold_sweep.csv'}, "
          f"{OUT_DIR / 'm2_semantic_grounding_traceability_sample.csv'}")


if __name__ == "__main__":
    main()
