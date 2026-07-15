"""§7.2 Accuracy/meso: anchoring β · convergence_rate · responsiveness · cross-phase coherence.
전부 numeric/로그 기반(rule/lexical 아님). 입력은 IR(scenarios/utterances)만 받는다(§12).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))

PHASE_ORDER = ["Inputs", "Activities", "Outputs", "Outcomes", "Impact"]
IDENTICAL_REL_TOL = 0.001  # "값 전부 동일"(수렴) 판정 허용 오차


def _all_identical(values: list[float]) -> bool:
    lo, hi = min(values), max(values)
    scale = max(abs(lo), abs(hi), 1.0)
    return (hi - lo) <= IDENTICAL_REL_TOL * scale


def _collect_cells(utterances: list[dict]) -> dict[tuple[str, str, str], dict[str, dict[str, float]]]:
    """(scenario_uid, phase, target) -> {persona_name: {"initial":v, "revised":v}}"""
    by_scenario_phase: dict[tuple[str, str], dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for u in utterances:
        if u["round"] not in ("initial", "revised"):
            continue
        key = (u["scenario_uid"], u["phase"])
        by_scenario_phase[key][u["persona_name"]][u["round"]] = u["prediction_values"]

    cells: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for (scenario_uid, phase), persona_data in by_scenario_phase.items():
        personas = list(persona_data.keys())
        if not personas:
            continue
        common_targets = set(persona_data[personas[0]].get("initial", {}).keys())
        for p in personas:
            common_targets &= set(persona_data[p].get("initial", {}).keys())
            common_targets &= set(persona_data[p].get("revised", {}).keys())
        for target in common_targets:
            per_persona = {}
            ok = True
            for p in personas:
                iv = persona_data[p]["initial"].get(target)
                rv = persona_data[p]["revised"].get(target)
                if not isinstance(iv, (int, float)) or not isinstance(rv, (int, float)):
                    ok = False
                    break
                per_persona[p] = {"initial": float(iv), "revised": float(rv)}
            if ok and len(per_persona) >= 3:
                cells[(scenario_uid, phase, target)] = per_persona
    return cells


def compute_anchoring_beta(cells: dict) -> dict:
    betas = []
    excluded_converged = 0
    excluded_degenerate = 0
    for key, persona_vals in cells.items():
        personas = list(persona_vals.keys())
        initial = [persona_vals[p]["initial"] for p in personas]
        revised = [persona_vals[p]["revised"] for p in personas]

        if _all_identical(revised):
            # §7.2: 수렴 시리즈(revised 전부 동일)는 β=1 인공물을 만들어서 제외.
            # (전원이 동일값으로 수렴하면 Δ_i=V-initial_i, peer_signal_i도 -initial_i에 거의 비례해
            # 실제 앵커링 메커니즘과 무관하게 β≈1이 기계적으로 나옴)
            excluded_converged += 1
            continue

        n = len(personas)
        peer_signal = []
        for i in range(n):
            others = [initial[j] for j in range(n) if j != i]
            peer_signal.append(sum(others) / len(others) - initial[i])

        if _all_identical(peer_signal):
            # initial 값이 전원 동일 -> peer_signal 분산 0 -> 기울기 추정 불가(수학적으로 다른 이유의 제외)
            excluded_degenerate += 1
            continue

        delta = [revised[i] - initial[i] for i in range(n)]
        X = sm.add_constant(np.array(peer_signal))
        y = np.array(delta)
        try:
            model = sm.OLS(y, X).fit()
            beta = float(model.params[1])
        except Exception:
            excluded_degenerate += 1
            continue
        betas.append({"key": key, "beta": beta, "n": n})

    return {
        "n_cells_total": len(cells),
        "n_used": len(betas),
        "n_excluded_converged": excluded_converged,
        "n_excluded_degenerate": excluded_degenerate,
        "mean_beta": float(np.mean([b["beta"] for b in betas])) if betas else None,
        "median_beta": float(np.median([b["beta"] for b in betas])) if betas else None,
        "betas": betas,
    }


def compute_convergence_rate(cells: dict) -> dict:
    converged = 0
    for key, persona_vals in cells.items():
        revised = [v["revised"] for v in persona_vals.values()]
        if _all_identical(revised):
            converged += 1
    n = len(cells)
    return {"n_cells": n, "n_converged": converged, "rate": (converged / n) if n else None}


def compute_responsiveness(cells: dict) -> dict:
    moved = 0
    total = 0
    for key, persona_vals in cells.items():
        for v in persona_vals.values():
            total += 1
            if not _all_identical([v["initial"], v["revised"]]):
                moved += 1
    return {"n": total, "n_moved": moved, "moved_ratio": (moved / total) if total else None}


def compute_cross_phase_coherence(scenarios: list[dict]) -> dict:
    """연속 phase_summary 간 TF-IDF 코사인 + 유의어 carry-over 비율(§7.2)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    all_docs = []
    doc_index: dict[tuple[str, str], int] = {}
    for s in scenarios:
        for phase, text in s["phase_summaries"].items():
            doc_index[(s["scenario_uid"], phase)] = len(all_docs)
            all_docs.append(text)

    if not all_docs:
        return {"n_scenarios": 0, "mean_cosine": None, "mean_word_overlap": None}

    vectorizer = TfidfVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(all_docs)

    import re

    def sig_words(t: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z]{4,}", t.lower()))

    cosines = []
    overlaps = []
    for s in scenarios:
        phases_present = [p for p in PHASE_ORDER if p in s["phase_summaries"]]
        for a, b in zip(phases_present, phases_present[1:]):
            ia, ib = doc_index[(s["scenario_uid"], a)], doc_index[(s["scenario_uid"], b)]
            cos = float(cosine_similarity(matrix[ia], matrix[ib])[0][0])
            cosines.append(cos)
            wa, wb = sig_words(s["phase_summaries"][a]), sig_words(s["phase_summaries"][b])
            if wa or wb:
                overlaps.append(len(wa & wb) / len(wa | wb))

    return {
        "n_scenarios": len(scenarios),
        "n_pairs": len(cosines),
        "mean_cosine": float(np.mean(cosines)) if cosines else None,
        "mean_word_overlap_jaccard": float(np.mean(overlaps)) if overlaps else None,
    }


def compute_set_meso_metrics(utterances: list[dict], scenarios: list[dict]) -> dict:
    by_set_utts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in utterances:
        by_set_utts[(u["policy_id"], u["model_id"])].append(u)
    by_set_scenarios: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in scenarios:
        by_set_scenarios[(s["policy_id"], s["model_id"])].append(s)

    results = {}
    for (policy_id, model_id), utts in by_set_utts.items():
        set_name = f"{policy_id}_{model_id}"
        cells = _collect_cells(utts)
        anchoring = compute_anchoring_beta(cells)
        convergence = compute_convergence_rate(cells)
        responsiveness = compute_responsiveness(cells)
        coherence = compute_cross_phase_coherence(by_set_scenarios[(policy_id, model_id)])
        results[set_name] = {
            "policy_id": policy_id,
            "model_id": model_id,
            "anchoring_beta": anchoring,
            "convergence_rate": convergence,
            "responsiveness": responsiveness,
            "cross_phase_coherence": coherence,
        }
    return results
