"""M4 실행 스크립트: data/ir/{scenarios,utterances}.jsonl -> §7.2 네 지표, 세 set 각각."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from accuracy_meso import compute_set_meso_metrics  # noqa: E402
from load_scenarios import REPO_ROOT  # noqa: E402

IR_DIR = REPO_ROOT / "data" / "ir"
OUT_DIR = REPO_ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    utterances = load_jsonl(IR_DIR / "utterances.jsonl")
    scenarios = load_jsonl(IR_DIR / "scenarios.jsonl")
    result = compute_set_meso_metrics(utterances, scenarios)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # betas 리스트는 파일이 커지므로 요약만 json에 남기고 원본은 별도 저장
    summary = {}
    for set_name, r in result.items():
        summary[set_name] = {
            "policy_id": r["policy_id"],
            "model_id": r["model_id"],
            "anchoring_beta": {k: v for k, v in r["anchoring_beta"].items() if k != "betas"},
            "convergence_rate": r["convergence_rate"],
            "responsiveness": r["responsiveness"],
            "cross_phase_coherence": r["cross_phase_coherence"],
        }
    with open(OUT_DIR / "m4_set_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    import csv
    with open(OUT_DIR / "m4_anchoring_beta_detail.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set_name", "scenario_uid", "phase", "target", "beta", "n"])
        for set_name, r in result.items():
            for b in r["anchoring_beta"]["betas"]:
                scenario_uid, phase, target = b["key"]
                w.writerow([set_name, scenario_uid, phase, target, b["beta"], b["n"]])

    print("=" * 78)
    print("M4 (§7.2 Accuracy/meso) — anchoring β · convergence_rate · responsiveness · cross-phase coherence")
    print("=" * 78)
    for set_name, r in sorted(result.items()):
        ab = r["anchoring_beta"]
        cr = r["convergence_rate"]
        rp = r["responsiveness"]
        cc = r["cross_phase_coherence"]
        print(f"\n[{set_name}]")
        mean_beta_str = f"{ab['mean_beta']:.3f}" if ab["mean_beta"] is not None else "N/A"
        median_beta_str = f"{ab['median_beta']:.3f}" if ab["median_beta"] is not None else "N/A"
        print(f"  anchoring_beta: mean={mean_beta_str} median={median_beta_str} "
              f"(used={ab['n_used']}/{ab['n_cells_total']} cells; "
              f"excluded_converged={ab['n_excluded_converged']}, excluded_degenerate={ab['n_excluded_degenerate']})")
        cr_str = f"{cr['rate']:.3f}" if cr["rate"] is not None else "N/A"
        print(f"  convergence_rate: {cr_str}  (converged={cr['n_converged']}/{cr['n_cells']})")
        rp_str = f"{rp['moved_ratio']:.3f}" if rp["moved_ratio"] is not None else "N/A"
        print(f"  responsiveness (moved_ratio): {rp_str}  (moved={rp['n_moved']}/{rp['n']})")
        cos_str = f"{cc['mean_cosine']:.3f}" if cc["mean_cosine"] is not None else "N/A"
        jac_str = f"{cc['mean_word_overlap_jaccard']:.3f}" if cc["mean_word_overlap_jaccard"] is not None else "N/A"
        print(f"  cross_phase_coherence: mean_cosine={cos_str} mean_word_overlap_jaccard={jac_str} "
              f"(scenarios={cc['n_scenarios']}, phase-pairs={cc['n_pairs']})")

    print(f"\n저장 완료: {OUT_DIR / 'm4_set_results.json'}, {OUT_DIR / 'm4_anchoring_beta_detail.csv'}")


if __name__ == "__main__":
    main()
