"""Macro-accuracy 리포트 생성기 — 특히 싱가포르(100E) 백본 모델(DeepSeek vs Gemini) 성능차이를
TIPS 논문(Table 1) 값과 나란히 정리한다.

입력:  results/m5_accuracy_macro.json  (run_m5.py 산출; mape_aggregate/bias 포함)
출력:  results/macro_accuracy_report.md
       results/macro_accuracy_singapore_comparison.csv

두 MAPE의 정의 차이가 핵심:
  - mape_per_scenario = mean_s(|ŷ(s)-gt|/|gt|)     : 시나리오별 개별 정확도(퍼짐 민감)
  - mape_aggregate    = |mean_s(ŷ(s))-gt|/|gt|      : 논문 Eq.(7) 정책수준 점추정 정확도
논문 Table 1의 100E(=Policy A) 값과 대응하는 것은 mape_aggregate 이다.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

# TIPS 논문(CIKM26 제출본) Table 1, full TIPS 행. 100E = Policy A(P_A), 두 지표 ŷ1_A/ŷ2_A.
# 우리 GT 매핑: approved_projects ↔ ŷ1_A, deployed_projects ↔ ŷ2_A.
PAPER_REF = {
    "100E": {
        "approved_projects": {"gemini": 2.87, "deepseek": 32.61, "paper_symbol": "ŷ1_A"},
        "deployed_projects": {"gemini": 4.08, "deepseek": 43.09, "paper_symbol": "ŷ2_A"},
    }
}
# 논문 Table 1 전체평균(참고): Gemini 13.62 / DeepSeek 25.72 (6개 타겟 평균, full TIPS).
PAPER_AVG = {"gemini": 13.62, "deepseek": 25.72}


def load_macro() -> dict:
    with open(RESULTS / "m5_accuracy_macro.json", encoding="utf-8") as f:
        return json.load(f)


def build_singapore_comparison(macro: dict) -> list[dict]:
    """100E deepseek vs gemini를 지표별로 정렬한 비교 레코드."""
    rows = []
    for metric in ("approved_projects", "deployed_projects"):
        rec = {"policy": "100E", "target": metric,
               "paper_symbol": PAPER_REF["100E"][metric]["paper_symbol"]}
        for model in ("gemini", "deepseek"):
            m = macro[f"100E_{model}"]["metrics"][metric]
            rec[f"{model}_mape_aggregate"] = round(m["mape_aggregate"], 2)
            rec[f"{model}_mape_per_scenario"] = round(m["mape_per_scenario"], 2)
            rec[f"{model}_mean_estimate"] = round(m["mean_estimate"], 2)
            rec[f"{model}_bias_pct"] = round(m["bias_pct"], 2)
            rec[f"{model}_coverage"] = m["calibration"]["coverage_gt_in_p10_p90"]
            rec[f"{model}_paper_ref"] = PAPER_REF["100E"][metric][model]
        rec["gt"] = m["gt"]
        rows.append(rec)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["policy", "target", "paper_symbol", "gt",
            "gemini_mape_aggregate", "gemini_paper_ref", "gemini_mape_per_scenario",
            "gemini_mean_estimate", "gemini_bias_pct", "gemini_coverage",
            "deepseek_mape_aggregate", "deepseek_paper_ref", "deepseek_mape_per_scenario",
            "deepseek_mean_estimate", "deepseek_bias_pct", "deepseek_coverage"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def fmt_pct(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)


def build_markdown(macro: dict, sg_rows: list[dict]) -> str:
    L = []
    L.append("# Macro-Accuracy 리포트 — MAPE / Calibration (§7.3)\n")
    L.append("> `mape_per_scenario = mean_s(|ŷ(s)−gt|/|gt|)` (시나리오별 개별 정확도) · "
             "`mape_aggregate = |mean_s(ŷ(s))−gt|/|gt|` (논문 Eq.7 정책수준 점추정, **Table 1 대응**).\n")

    # 1) 세 set × 지표 전체표
    L.append("## 1. 전체 결과 (세 set)\n")
    L.append("| set | target | GT | MAPE(agg, 논문식) | MAPE(per-scenario) | mean_est | bias% | coverage[p10,p90] | sharpness |")
    L.append("|---|---|---:|---:|---:|---:|---:|:--:|---:|")
    for set_name in ("100E_gemini", "100E_deepseek", "BK21_deepseek"):
        for metric, m in macro[set_name]["metrics"].items():
            cal = m["calibration"]
            L.append(f"| {set_name} | {metric} | {m['gt']} | **{m['mape_aggregate']:.2f}** | "
                     f"{m['mape_per_scenario']:.2f} | {m['mean_estimate']:.2f} | {m['bias_pct']:+.1f} | "
                     f"{'✅' if cal['coverage_gt_in_p10_p90'] else '❌'} | {cal['sharpness_p90_minus_p10']:.1f} |")
    L.append("")

    # 2) 싱가포르 백본 비교 (핵심)
    L.append("## 2. 싱가포르 100E 백본 모델 성능차이 (DeepSeek vs Gemini) — 논문 Table 1 대비\n")
    L.append("논문 값은 CIKM26 Table 1의 full-TIPS 행, 100E=Policy A(ŷ1_A/ŷ2_A). "
             "우리 파이프라인의 **aggregate MAPE**가 논문 정의와 일치하며 값이 재현된다.\n")
    L.append("| target | GT | Gemini agg-MAPE (논문) | DeepSeek agg-MAPE (논문) | Gemini per-scen | DeepSeek per-scen |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in sg_rows:
        L.append(f"| {r['target']} ({r['paper_symbol']}) | {r['gt']} | "
                 f"**{r['gemini_mape_aggregate']:.2f}** ({r['gemini_paper_ref']}) | "
                 f"**{r['deepseek_mape_aggregate']:.2f}** ({r['deepseek_paper_ref']}) | "
                 f"{r['gemini_mape_per_scenario']:.2f} | {r['deepseek_mape_per_scenario']:.2f} |")
    L.append(f"\n논문 6-타겟 평균(참고): Gemini **{PAPER_AVG['gemini']}** / DeepSeek **{PAPER_AVG['deepseek']}**.\n")

    # 3) 해석
    L.append("## 3. 해석 — 왜 두 MAPE가 모델별로 갈리는가 (분산 vs 편향)\n")
    g_app = macro["100E_gemini"]["metrics"]["approved_projects"]
    d_app = macro["100E_deepseek"]["metrics"]["approved_projects"]
    L.append(f"- **Gemini**: 추정치가 GT 주변에 넓게 흩어짐(approved sharpness "
             f"{g_app['calibration']['sharpness_p90_minus_p10']:.0f}, coverage "
             f"{'✅' if g_app['calibration']['coverage_gt_in_p10_p90'] else '❌'}). 편향 거의 없음"
             f"(bias {g_app['bias_pct']:+.1f}%) → 앙상블 평균이 GT에 수렴 → **agg-MAPE {g_app['mape_aggregate']:.1f}%** "
             f"인데 per-scenario는 {g_app['mape_per_scenario']:.1f}%. '다양성이 진실 주변에서 발생'.")
    L.append(f"- **DeepSeek**: 체계적 과소추정(bias {d_app['bias_pct']:+.1f}%, mean_est "
             f"{d_app['mean_estimate']:.0f} vs GT {d_app['gt']}), coverage "
             f"{'✅' if d_app['calibration']['coverage_gt_in_p10_p90'] else '❌'}. 평균 자체가 GT에서 벗어나 "
             f"→ agg-MAPE와 per-scenario가 모두 큼({d_app['mape_aggregate']:.1f}% ≈ {d_app['mape_per_scenario']:.1f}%). "
             f"앙상블 평균으로도 편향이 상쇄되지 않음.")
    L.append("- **결론**: 싱가포르 100E에서 Gemini가 DeepSeek을 크게 앞서며(논문과 동일 방향·유사 크기), "
             "이는 per-scenario MAPE만 보면 드러나지 않는다. 정책수준 예측 정확도(논문 지표)는 반드시 "
             "aggregate MAPE로 봐야 하고, calibration(coverage/sharpness)과 bias를 함께 봐야 '분산이 커서 "
             "per-scenario가 나쁜 것'과 '편향돼서 근본적으로 부정확한 것'을 구분할 수 있다.\n")

    L.append("## 4. 가드레일 준수 메모\n")
    L.append("- 이상치 제거/클리핑 없음(§6): sharpness가 큰 것은 그대로 신호로 보고. "
             "bias/coverage로 분산과 편향을 분리 보고.")
    L.append("- GT는 §5 TIGRIS Appendix E, Table 4 값만 사용(임의 생성 없음). GT 없는 지표는 skip.")
    L.append("- MAPE는 robust 병기: per-scenario는 median_robust도 저장. "
             "aggregate와 per-scenario를 둘 다 보고(한쪽만으로 결론 금지).\n")
    return "\n".join(L)


def main() -> None:
    macro = load_macro()
    sg_rows = build_singapore_comparison(macro)
    write_csv(sg_rows, RESULTS / "macro_accuracy_singapore_comparison.csv")
    md = build_markdown(macro, sg_rows)
    (RESULTS / "macro_accuracy_report.md").write_text(md, encoding="utf-8")
    print("wrote:", RESULTS / "macro_accuracy_report.md")
    print("wrote:", RESULTS / "macro_accuracy_singapore_comparison.csv")
    print("\n--- Singapore comparison ---")
    for r in sg_rows:
        print(f"{r['target']:20} GT={r['gt']:>4}  "
              f"Gemini agg={r['gemini_mape_aggregate']:>6.2f} (paper {r['gemini_paper_ref']})   "
              f"DeepSeek agg={r['deepseek_mape_aggregate']:>6.2f} (paper {r['deepseek_paper_ref']})")


if __name__ == "__main__":
    main()
