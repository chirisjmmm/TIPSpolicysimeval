"""§7.3 Accuracy/macro: MAPE + Calibration. GT 있는 지표만 계산한다(§5, §6 — GT 임의 생성/이상치
제거 금지). 입력은 IR(scenarios.jsonl) + policies.yaml gt_map만 받는다(§12).

두 가지 MAPE를 함께 보고한다(둘은 다른 질문에 답한다):
  1) mape_per_scenario  = mean_s( |ŷ(s) - gt| / |gt| ) * 100
     → 시나리오별 예측이 개별적으로 얼마나 정확한가(퍼짐에 민감).
  2) mape_aggregate     = |mean_s( ŷ(s) ) - gt| / |gt| * 100    ← TIPS 논문 Eq.(7) 정의
     → 시나리오 앙상블을 평균낸 정책수준 점추정 ŷ_p 가 GT에 얼마나 가까운가(논문 Table 1과 대응).
싱가포르 100E에서 두 값의 괴리가 모델별로 크게 갈린다: Gemini는 추정치가 GT 주변에 폭넓게
흩어져(분산↑) 평균이 GT에 수렴 → aggregate MAPE는 작지만 per-scenario MAPE는 큼. DeepSeek은
체계적 편향(bias)으로 평균 자체가 GT에서 벗어나 → 두 MAPE가 모두 크고 서로 비슷.
mean_estimate/bias/bias_pct를 같이 저장해 이 차이(분산 vs 편향)를 드러낸다.
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
            mape = float(np.mean(ape))            # per-scenario 평균(퍼짐 민감)
            mape_median = float(np.median(ape))

            # TIPS Eq.(7): 시나리오 앙상블 평균낸 점추정 ŷ_p 후 MAPE(논문 Table 1과 대응)
            mean_estimate = float(np.mean(estimates_arr))
            mape_aggregate = float(abs(mean_estimate - gt) / abs(gt) * 100)
            bias = float(mean_estimate - gt)                 # 부호 있는 평균 오차(양수=과대추정)
            bias_pct = float(bias / abs(gt) * 100)

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
                # 하위호환: "mape" == per-scenario 평균. 명시적 alias도 함께 저장.
                "mape": mape,
                "mape_per_scenario": mape,
                "mape_median_robust": mape_median,
                "mape_aggregate": mape_aggregate,   # ← 논문 Eq.(7)/Table 1 대응
                "mean_estimate": mean_estimate,
                "bias": bias,
                "bias_pct": bias_pct,
                "calibration": {
                    "p10": p10, "p90": p90, "coverage_gt_in_p10_p90": coverage, "sharpness_p90_minus_p10": sharpness,
                },
                "n_out_of_declared_range": out_of_range,
            }

        results[set_name] = {"policy_id": policy_id, "model_id": model_id, "metrics": set_metrics}

    return results
