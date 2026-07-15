"""PPT용 시각화 — 특히 싱가포르(100E) 백본 모델 성능차이 중심.

출력: results/figures/
  fig1_singapore_agg_mape.png      : 100E aggregate MAPE by model + 논문 Table1 참조
  fig2_scenario_dispersion.png     : 100E approved 시나리오별 추정 분포(분산 vs 편향)
  fig3_per_vs_agg_mape.png         : per-scenario vs aggregate MAPE (전체 set)
  fig4_adr_axis_level.png          : ADR axis×level 프레임워크 + 진행상황 표
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 한국어 폰트 등록 (NanumGothic)
_KFONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
_KFONT_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
if Path(_KFONT).exists():
    font_manager.fontManager.addfont(_KFONT)
    if Path(_KFONT_BOLD).exists():
        font_manager.fontManager.addfont(_KFONT_BOLD)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_KFONT).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from load_scenarios import load_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# 팔레트(dataviz 기본, CVD-safe): Gemini=blue slot1, DeepSeek=orange slot6
C_GEMINI = "#2a78d6"
C_DEEPSEEK = "#eb6834"
C_GT = "#d03b3b"       # status/critical red — GT 기준선
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

PAPER_REF = {
    ("approved_projects"): {"gemini": 2.87, "deepseek": 32.61},
    ("deployed_projects"): {"gemini": 4.08, "deepseek": 43.09},
}


def load_macro():
    return json.load(open(RESULTS / "m5_accuracy_macro.json", encoding="utf-8"))


def scenario_estimates(policy_id, model_id, phase, var):
    scen = [json.loads(l) for l in open(REPO_ROOT / "data/ir/scenarios.jsonl", encoding="utf-8")]
    out = []
    for s in scen:
        if s["policy_id"] != policy_id or s["model_id"] != model_id:
            continue
        for cc in s["cross_checks"]:
            if cc["phase"] == phase and cc["variable"] == var and cc.get("aggregated_value") is not None:
                out.append(float(cc["aggregated_value"]))
                break
    return np.array(out)


# ---------- Fig 1: Singapore aggregate MAPE, ours vs paper ----------
def fig1(macro):
    targets = ["approved_projects", "deployed_projects"]
    labels = ["approved\n$\\hat{y}_1^{A}$", "deployed\n$\\hat{y}_2^{A}$"]
    g_ours = [macro["100E_gemini"]["metrics"][t]["mape_aggregate"] for t in targets]
    d_ours = [macro["100E_deepseek"]["metrics"][t]["mape_aggregate"] for t in targets]
    g_pap = [PAPER_REF[t]["gemini"] for t in targets]
    d_pap = [PAPER_REF[t]["deepseek"] for t in targets]

    x = np.arange(len(targets)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    b1 = ax.bar(x - 1.5 * w, g_ours, w, color=C_GEMINI, label="Gemini (ours)")
    ax.bar(x - 0.5 * w, g_pap, w, color=C_GEMINI, alpha=0.35, hatch="//",
           edgecolor=C_GEMINI, label="Gemini (TIPS paper)")
    b3 = ax.bar(x + 0.5 * w, d_ours, w, color=C_DEEPSEEK, label="DeepSeek (ours)")
    ax.bar(x + 1.5 * w, d_pap, w, color=C_DEEPSEEK, alpha=0.35, hatch="//",
           edgecolor=C_DEEPSEEK, label="DeepSeek (TIPS paper)")

    for bars in (b1, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    for xi, v in zip(x - 0.5 * w, g_pap):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=MUTED)
    for xi, v in zip(x + 1.5 * w, d_pap):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=MUTED)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Aggregate MAPE (%)  — 낮을수록 정확")
    ax.set_title("싱가포르 100E: 백본 모델별 정책수준 예측 정확도 (논문 Eq.7 / Table 1 재현)",
                 fontsize=12.5, color=INK, pad=12)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_ylim(0, max(d_pap) * 1.25)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig1_singapore_agg_mape.png", dpi=200)
    plt.close(fig)


# ---------- Fig 2: scenario dispersion (variance vs bias) ----------
def fig2(macro):
    var = "number_of_100e_projects_approved_in_2023"; gt = 115
    g = scenario_estimates("100E", "gemini", "Outcomes", var)
    d = scenario_estimates("100E", "deepseek", "Outcomes", var)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    rng = np.random.default_rng(7)
    for i, (vals, c, name) in enumerate([(g, C_GEMINI, "Gemini"), (d, C_DEEPSEEK, "DeepSeek")]):
        jit = rng.uniform(-0.11, 0.11, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, s=26, color=c, alpha=0.55,
                   edgecolor="white", linewidth=0.5, zorder=3)
        mean = vals.mean()
        ax.hlines(mean, i - 0.24, i + 0.24, color=c, lw=2.5, zorder=4)
        ax.text(i + 0.28, mean, f"mean={mean:.0f}\n(agg-MAPE {macro['100E_'+name.lower()]['metrics']['approved_projects']['mape_aggregate']:.1f}%)",
                va="center", fontsize=9, color=c)
    ax.axhline(gt, color=C_GT, lw=1.8, ls="--", zorder=2)
    ax.text(1.45, gt, f"GT = {gt}", color=C_GT, va="bottom", ha="right", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Gemini", "DeepSeek"])
    ax.set_ylabel("시나리오별 approved-projects 추정치")
    ax.set_title("왜 두 모델이 갈리나: 분산(Gemini, GT 주변 폭넓음) vs 편향(DeepSeek, 체계적 과소추정)",
                 fontsize=11.8, color=INK, pad=12)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.set_xlim(-0.5, 1.9)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig2_scenario_dispersion.png", dpi=200)
    plt.close(fig)


# ---------- Fig 3: per-scenario vs aggregate MAPE (all sets) ----------
def fig3(macro):
    rows = []
    for set_name in ["100E_gemini", "100E_deepseek", "BK21_deepseek"]:
        for t, m in macro[set_name]["metrics"].items():
            rows.append((f"{set_name}\n{t.replace('_projects','').replace('_rate','')}",
                         m["mape_per_scenario"], m["mape_aggregate"]))
    labels = [r[0] for r in rows]
    per = [r[1] for r in rows]; agg = [r[2] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - w / 2, per, w, color=MUTED, label="MAPE per-scenario:  mean( |est-gt| / gt )")
    ax.bar(x + w / 2, agg, w, color=C_GEMINI, label="MAPE aggregate (논문 Eq.7):  |mean(est)-gt| / gt")
    for xi, v in zip(x - w / 2, per):
        ax.text(xi, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    for xi, v in zip(x + w / 2, agg):
        ax.text(xi, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=C_GEMINI)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("두 MAPE 정의의 괴리 = 앙상블 평균의 오차 상쇄 여력 (편향된 set은 괴리가 사라짐)",
                 fontsize=11.8, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig3_per_vs_agg_mape.png", dpi=200)
    plt.close(fig)


# ---------- Fig 4: ADR axis×level framework + status table ----------
def fig4():
    header = ["Axis", "Level", "측정 방식 (본 파이프라인)", "산출물 / 결과", "진행"]
    rows = [
        ["Accuracy", "micro", "규칙위반률(이진)·grounding(TF-IDF cos)·fabrication(정규식 대조)",
         "m2_set_results.json", "완료·미검증"],
        ["Accuracy", "meso", "anchoring β(OLS,수렴제외)·convergence·responsiveness·coherence",
         "m4_set_results.json", "완료"],
        ["Accuracy", "macro", "MAPE(per-scenario+aggregate 논문Eq.7)·calibration·bias",
         "m5_accuracy_macro.json", "완료(보강)"],
        ["Diversity", "micro", "집단별 예측분산(z)·distinct-n·self-BLEU",
         "m5_diversity.json", "완료"],
        ["Diversity", "macro", "outcome range/IQR·bin entropy·trajectory clustering",
         "m5_diversity.json", "완료"],
        ["Reflection", "process", "KG/ToC 구조화 과정 — 유저스터디(코드 아님)",
         "— (user study)", "설계중"],
        ["Reflection", "outcome", "pathway 해석·decision-support 신뢰 — 유저스터디",
         "— (user study)", "설계중"],
    ]
    axis_color = {"Accuracy": "#2a78d6", "Diversity": "#1baf7a", "Reflection": "#4a3aa7"}
    status_color = {"완료": "#0ca30c", "완료(보강)": "#0ca30c", "완료·미검증": "#eda100",
                    "설계중": "#52514e"}

    fig, ax = plt.subplots(figsize=(13, 5.2)); ax.axis("off")
    ncol = len(header); nrow = len(rows) + 1
    widths = [0.09, 0.07, 0.44, 0.24, 0.16]
    xpos = np.concatenate([[0], np.cumsum(widths)])
    yh = 1.0 / nrow
    for j, h in enumerate(header):
        ax.add_patch(plt.Rectangle((xpos[j], 1 - yh), widths[j], yh, color="#f0efec"))
        ax.text(xpos[j] + 0.008, 1 - yh / 2, h, va="center", ha="left", fontsize=10.5, fontweight="bold", color=INK)
    for i, r in enumerate(rows):
        y = 1 - (i + 2) * yh
        for j, cell in enumerate(r):
            ax.text(xpos[j] + 0.008, y + yh / 2, cell, va="center", ha="left",
                    fontsize=8.6, color=INK, wrap=True)
        ax.add_patch(plt.Rectangle((xpos[0], y), widths[0], yh, color=axis_color[r[0]], alpha=0.14))
        ax.text(xpos[0] + 0.008, y + yh / 2, r[0], va="center", ha="left", fontsize=9,
                fontweight="bold", color=axis_color[r[0]])
        sc = status_color.get(r[4], MUTED)
        ax.text(xpos[4] + 0.008, y + yh / 2, r[4], va="center", ha="left", fontsize=8.6, color=sc, fontweight="bold")
        ax.hlines(y, 0, 1, color=GRID, lw=0.8)
    for j in range(ncol + 1):
        ax.vlines(xpos[j], 0, 1, color=GRID, lw=0.8)
    ax.hlines(1, 0, 1, color=MUTED, lw=1.2)
    ax.set_title("ADR (Accuracy·Diversity·Reflection) × Level — 측정 방식과 진행 상황",
                 fontsize=13, color=INK, pad=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig4_adr_axis_level.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    macro = load_macro()
    fig1(macro); fig2(macro); fig3(macro); fig4()
    for p in sorted(FIGDIR.glob("*.png")):
        print("wrote:", p)


if __name__ == "__main__":
    main()
