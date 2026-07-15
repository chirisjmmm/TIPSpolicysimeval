"""§7.3 Accuracy/macro: MAPE + Calibration. GT 있는 지표만 계산한다(§5, §6 — GT 임의 생성/이상치
제거 금지). 입력은 IR(scenarios.jsonl) + policies.yaml gt_map만 받는다(§12).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from load_scenarios import load_config  # noqa: E402


def _scenario_estimate(scenario: dict, phase: str, var: str) -> float | None:
    """§3 규칙: cross_checks의 (phase,variable)->aggregated_value 우선, 없으면 해당 phase revised 평균."""
    for cc in scenario["cross_checks"]:
        if cc["phase"] == phase and cc["variable"] == var and cc.get("aggregated_value") is not None:
            return float(cc["aggregated_value"])
    return None  # fallback은 utterances가 있어야 계산 가능 -> 호출부에서 처리


def _fallback_revised_mean(utterances_for_scenario_phase: list[dict], var: str) -> float | None:
    vals = [
        u["prediction_values"][var]
        for u in utterances_for_scenario_phase
        if u["round"] == "revised" and var in u["prediction_values"]
        and isinstance(u["prediction_values"][var], (int, float))
    ]
    return float(np.mean(vals)) if vals else None


def compute_mape_and_calibration(
    scenarios: list[dict], utterances: list[dict], config: dict | None = None
) -> dict:
    config = config or load_config()
    gt_map = config["gt_map"]

    by_scenario_phase_utts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in utterances:
        by_scenario_phase_utts[(u["scenario_uid"], u["phase"])].append(u)

    by_set_scenarios: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in scenarios:
        by_set_scenarios[(s["policy_id"], s["model_id"])].append(s)

    results = {}
    for (policy_id, model_id), set_scenarios in by_set_scenarios.items():
        set_name = f"{policy_id}_{model_id}"
        policy_gt = gt_map.get(policy_id, {})
        set_metrics = {}

        for metric_name, spec in policy_gt.items():
            phase, var, gt, value_range = spec["phase"], spec["var"], spec["gt"], spec["range"]
            estimates = []
            n_fallback = 0
            n_skipped = 0
            for s in set_scenarios:
                est = _scenario_estimate(s, phase, var)
                if est is None:
                    est = _fallback_revised_mean(by_scenario_phase_utts.get((s["scenario_uid"], phase), []), var)
                    if est is not None:
                        n_fallback += 1
                if est is None:
                    n_skipped += 1
                    continue
                estimates.append(est)

            if not estimates:
                set_metrics[metric_name] = {"gt": gt, "n_scenarios": 0, "mape": None, "note": "GT 미입력/추정 불가 skip"}
                continue

            estimates_arr = np.array(estimates)
            ape = np.abs(estimates_arr - gt) / abs(gt) * 100
            mape = float(np.mean(ape))
            mape_median = float(np.median(ape))

            p10, p90 = float(np.percentile(estimates_arr, 10)), float(np.percentile(estimates_arr, 90))
            coverage = bool(p10 <= gt <= p90)
            sharpness = p90 - p10

            out_of_range = int(np.sum((estimates_arr < value_range[0]) | (estimates_arr > value_range[1])))

            set_metrics[metric_name] = {
                "gt": gt,
                "phase": phase,
                "var": var,
                "n_scenarios": len(estimates),
                "n_fallback_used": n_fallback,
                "n_skipped": n_skipped,
                "mape": mape,
                "mape_median_robust": mape_median,
                "calibration": {
                    "p10": p10, "p90": p90, "coverage_gt_in_p10_p90": coverage, "sharpness_p90_minus_p10": sharpness,
                },
                "n_out_of_declared_range": out_of_range,
            }

        results[set_name] = {"policy_id": policy_id, "model_id": model_id, "metrics": set_metrics}

    return results
