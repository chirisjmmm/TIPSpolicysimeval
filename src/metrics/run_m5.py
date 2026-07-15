"""M5 실행 스크립트: §7.3 accuracy/macro(MAPE+calibration, GT 있는 지표만) +
§7.4 diversity(micro/macro, plausible-diversity 필터는 아직 TODO/passthrough).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from accuracy_macro import compute_mape_and_calibration  # noqa: E402
from diversity import compute_set_diversity_metrics  # noqa: E402
from load_scenarios import REPO_ROOT, load_config  # noqa: E402

IR_DIR = REPO_ROOT / "data" / "ir"
OUT_DIR = REPO_ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    utterances = load_jsonl(IR_DIR / "utterances.jsonl")
    scenarios = load_jsonl(IR_DIR / "scenarios.jsonl")
    config = load_config()

    macro_acc = compute_mape_and_calibration(scenarios, utterances, config)
    diversity = compute_set_diversity_metrics(utterances, scenarios, m2_gold_results=None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "m5_accuracy_macro.json", "w", encoding="utf-8") as f:
        json.dump(macro_acc, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "m5_diversity.json", "w", encoding="utf-8") as f:
        json.dump(diversity, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("M5-a (§7.3 Accuracy/macro): MAPE + Calibration — GT 있는 지표만")
    print("=" * 78)
    for set_name, r in sorted(macro_acc.items()):
        print(f"\n[{set_name}]")
        for metric_name, m in r["metrics"].items():
            if m["mape"] is None:
                print(f"  {metric_name}: {m['note']}")
                continue
            cal = m["calibration"]
            print(f"  {metric_name} (gt={m['gt']}, phase={m['phase']}, var={m['var']}, n={m['n_scenarios']}, "
                  f"fallback_used={m['n_fallback_used']}, skipped={m['n_skipped']})")
            print(f"    MAPE(per-scenario) = {m['mape']:.2f}%  (median_robust={m['mape_median_robust']:.2f}%)")
            print(f"    MAPE(aggregate, 논문 Eq.7) = {m['mape_aggregate']:.2f}%  "
                  f"[mean_est={m['mean_estimate']:.2f}, bias={m['bias']:+.2f} ({m['bias_pct']:+.1f}%)]")
            print(f"    calibration: p10={cal['p10']:.2f} p90={cal['p90']:.2f} "
                  f"coverage(gt in [p10,p90])={cal['coverage_gt_in_p10_p90']} "
                  f"sharpness={cal['sharpness_p90_minus_p10']:.2f}")
            if m["n_out_of_declared_range"] > 0:
                print(f"    주의: 선언된 range 밖 추정치 {m['n_out_of_declared_range']}건")

    print("\n" + "=" * 78)
    print("M5-b (§7.4 Diversity) — plausible_diversity_filter는 TODO(현재 passthrough, 필터 미적용)")
    print("=" * 78)
    for set_name, r in sorted(diversity.items()):
        micro, macro = r["micro"], r["macro"]
        print(f"\n[{set_name}]")
        gv = micro["group_variance"]
        gv_str = f"{gv['mean_between_group_variance']:.4f}" if gv["mean_between_group_variance"] is not None else "N/A"
        print(f"  micro.group_variance(between-role, z-정규화) = {gv_str}  (n_cells={gv['n_cells']})")
        d1, d2 = micro["distinct_1"], micro["distinct_2"]
        print(f"  micro.distinct-1 = {d1['distinct_n']:.4f}  distinct-2 = {d2['distinct_n']:.4f}"
              if d1["distinct_n"] is not None else "  micro.distinct-n = N/A")
        sb = micro["self_bleu"]
        sb_str = f"{sb['self_bleu']:.4f}" if sb["self_bleu"] is not None else "N/A"
        print(f"  micro.self_bleu(높을수록 다양성↓) = {sb_str}  (n_pairs={sb['n_pairs']})")

        print("  macro.outcome_range_and_bins:")
        for var_key, v in macro["outcome_range_and_bins"].items():
            print(f"    {var_key}: range=[{v['min']:.2f},{v['max']:.2f}] iqr={v['iqr']:.2f} "
                  f"bin_coverage={v['bin_coverage']:.2f} bin_entropy_norm={v['bin_entropy_normalized']:.2f} (n={v['n']})")

        tc = macro["trajectory_clustering"]
        if tc.get("best_k") is not None:
            print(f"  macro.trajectory_clustering: best_k={tc['best_k']} silhouette={tc['silhouette_score']:.3f} "
                  f"cluster_sizes={tc['cluster_sizes']} (n_features={tc['n_features']})")
        else:
            print(f"  macro.trajectory_clustering: {tc.get('note')}")

    print(f"\n저장 완료: {OUT_DIR / 'm5_accuracy_macro.json'}, {OUT_DIR / 'm5_diversity.json'}")


if __name__ == "__main__":
    main()
