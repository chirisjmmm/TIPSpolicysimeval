"""§7.4 Diversity: micro(집단별 분산+distinct-n+self-BLEU) + macro(outcome range+bin
coverage/entropy+trajectory clustering). 입력은 IR만 받는다(§12).

TODO(plausible_diversity_filter): M2 gold(§8) 확정 전이라 지금은 자리만 두고 필터링하지 않는다.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from load_scenarios import load_config  # noqa: E402

PHASE_ORDER = ["Inputs", "Activities", "Outputs", "Outcomes", "Impact"]
OUTCOME_PHASES = ["Outcomes", "Impact"]
N_BINS_DEFAULT = 5  # 도메인 구간이 policies.yaml에 아직 없어 데이터 기반 등폭 5구간을 기본값으로 씀(unvalidated)


def plausible_diversity_filter(utterances: list[dict], m2_gold_results: dict | None = None) -> list[dict]:
    """TODO: M2 gold(§8, human-validated fabrication/grounding) 확정되면
    grounded==True and fabricated==False(또는 human-검증판)인 발화만 남기도록 구현.
    지금은 자리만 만들어두고 그대로 통과시킨다(passthrough, 필터링 없음).
    """
    # TODO(M2 gold 확정 후 적용): plausible-diversity 게이트
    return utterances


def _role_type(kg_role: str) -> str:
    return kg_role.split(":", 1)[0]


def _collect_cells(utterances: list[dict]) -> dict[tuple[str, str, str], dict[str, dict]]:
    """(scenario_uid, phase, target) -> {persona_name: {"value":v, "kg_role":..., "text":...}} (revised round만
    — initial과 revised가 이 코퍼스에서 100% 동일해서(§ 확인됨) revised 하나만 쓴다. 둘 다 쓰면
    동일 텍스트가 두 번 들어가 distinct-n/self-BLEU가 인위적으로 낮아진다."""
    by_scenario_phase: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in utterances:
        if u["round"] == "revised":
            by_scenario_phase[(u["scenario_uid"], u["phase"])].append(u)

    cells: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for (scenario_uid, phase), utts in by_scenario_phase.items():
        targets: set[str] = set()
        for u in utts:
            targets |= set(k for k, v in u["prediction_values"].items() if isinstance(v, (int, float)))
        for target in targets:
            for u in utts:
                if target in u["prediction_values"] and isinstance(u["prediction_values"][target], (int, float)):
                    cells[(scenario_uid, phase, target)][u["persona_name"]] = {
                        "value": float(u["prediction_values"][target]),
                        "kg_role": u["kg_role"],
                    }
    return cells


# ---------------------------------------------------------------------------
# micro
# ---------------------------------------------------------------------------
def compute_group_variance(cells: dict) -> dict:
    """집단(kg_role type)별 예측값을 z-정규화 후 집단-평균 간 분산(between-group variance)."""
    between_vars = []
    for key, persona_vals in cells.items():
        values = np.array([v["value"] for v in persona_vals.values()])
        if len(values) < 3 or np.std(values) == 0:
            continue
        z = (values - values.mean()) / values.std()
        by_group: dict[str, list[float]] = defaultdict(list)
        for z_val, v in zip(z, persona_vals.values()):
            by_group[_role_type(v["kg_role"])].append(z_val)
        if len(by_group) < 2:
            continue
        group_means = [np.mean(vs) for vs in by_group.values()]
        between_vars.append(float(np.var(group_means)))
    return {
        "n_cells": len(between_vars),
        "mean_between_group_variance": float(np.mean(between_vars)) if between_vars else None,
    }


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[a-zA-Z가-힣]{2,}", (text or "").lower())


def compute_distinct_n(texts: list[str], n: int) -> dict:
    all_ngrams = []
    for t in texts:
        tokens = _tokenize(t)
        all_ngrams.extend(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    if not all_ngrams:
        return {"distinct_n": None, "n_ngrams": 0}
    return {"distinct_n": len(set(all_ngrams)) / len(all_ngrams), "n_ngrams": len(all_ngrams)}


def compute_self_bleu(texts_by_group: list[list[str]], max_groups: int | None = 200) -> dict:
    """(scenario,phase) 그룹별 페르소나 5명 발화끼리 self-BLEU. 높을수록 다양성↓(§12 가드레일 방향 주의)."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoothing = SmoothingFunction().method1
    scores = []
    groups = texts_by_group if max_groups is None else texts_by_group[:max_groups]
    for texts in groups:
        tokenized = [_tokenize(t) for t in texts if t]
        if len(tokenized) < 2:
            continue
        for i, hyp in enumerate(tokenized):
            refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
            if not hyp or not refs:
                continue
            score = sentence_bleu(refs, hyp, weights=(0.5, 0.5), smoothing_function=smoothing)
            scores.append(score)
    return {"self_bleu": float(np.mean(scores)) if scores else None, "n_pairs": len(scores)}


def compute_micro_diversity(utterances: list[dict], cells: dict) -> dict:
    group_var = compute_group_variance(cells)

    revised_texts = [u["text"] for u in utterances if u["round"] == "revised" and u["text"]]
    distinct_1 = compute_distinct_n(revised_texts, 1)
    distinct_2 = compute_distinct_n(revised_texts, 2)

    by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for u in utterances:
        if u["round"] == "revised":
            by_group[(u["scenario_uid"], u["phase"])].append(u["text"])
    self_bleu = compute_self_bleu(list(by_group.values()))

    return {
        "group_variance": group_var,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "self_bleu": self_bleu,
    }


# ---------------------------------------------------------------------------
# macro
# ---------------------------------------------------------------------------
def _scenario_estimate(scenario: dict, phase: str, var: str, utterances_by_sp: dict) -> float | None:
    for cc in scenario["cross_checks"]:
        if cc["phase"] == phase and cc["variable"] == var and cc.get("aggregated_value") is not None:
            return float(cc["aggregated_value"])
    utts = utterances_by_sp.get((scenario["scenario_uid"], phase), [])
    vals = [
        u["prediction_values"][var] for u in utts
        if u["round"] == "revised" and var in u["prediction_values"] and isinstance(u["prediction_values"][var], (int, float))
    ]
    return float(np.mean(vals)) if vals else None


def compute_outcome_range_and_bins(scenarios: list[dict], utterances: list[dict], n_bins: int = N_BINS_DEFAULT) -> dict:
    utterances_by_sp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in utterances:
        utterances_by_sp[(u["scenario_uid"], u["phase"])].append(u)

    by_phase_var: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in scenarios:
        for phase in OUTCOME_PHASES:
            targets: set[str] = set()
            for u in utterances_by_sp.get((s["scenario_uid"], phase), []):
                targets |= set(k for k, v in u["prediction_values"].items() if isinstance(v, (int, float)))
            for var in targets:
                est = _scenario_estimate(s, phase, var, utterances_by_sp)
                if est is not None:
                    by_phase_var[(phase, var)].append(est)

    results = {}
    for (phase, var), values in by_phase_var.items():
        arr = np.array(values)
        if len(arr) < 2:
            continue
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        lo, hi = float(arr.min()), float(arr.max())

        if hi > lo:
            bin_edges = np.linspace(lo, hi, n_bins + 1)
            bin_idx = np.clip(np.digitize(arr, bin_edges[1:-1]), 0, n_bins - 1)
            counts = np.bincount(bin_idx, minlength=n_bins)
            probs = counts / counts.sum()
            nonzero = probs[probs > 0]
            entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(n_bins))
            coverage = float(np.sum(counts > 0) / n_bins)
        else:
            entropy, coverage = 0.0, 1.0 / n_bins

        results[f"{phase}:{var}"] = {
            "n": len(arr), "min": lo, "max": hi, "range": hi - lo,
            "iqr": q3 - q1, "q1": q1, "q3": q3,
            "bin_coverage": coverage, "bin_entropy_normalized": entropy, "n_bins": n_bins,
        }
    return results


def compute_trajectory_clustering(scenarios: list[dict], utterances: list[dict], k_range: range = range(2, 6)) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    utterances_by_sp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in utterances:
        utterances_by_sp[(u["scenario_uid"], u["phase"])].append(u)

    all_phase_targets: set[tuple[str, str]] = set()
    for s in scenarios:
        for phase in PHASE_ORDER:
            for u in utterances_by_sp.get((s["scenario_uid"], phase), []):
                for k, v in u["prediction_values"].items():
                    if isinstance(v, (int, float)):
                        all_phase_targets.add((phase, k))
    phase_targets = sorted(all_phase_targets)

    raw_matrix = []
    for s in scenarios:
        row = []
        for phase, var in phase_targets:
            est = _scenario_estimate(s, phase, var, utterances_by_sp)
            row.append(est if est is not None else np.nan)
        raw_matrix.append(row)
    raw_matrix = np.array(raw_matrix)

    col_mean = np.nanmean(raw_matrix, axis=0)
    col_std = np.nanstd(raw_matrix, axis=0)
    col_std[col_std == 0] = 1.0
    inds = np.where(np.isnan(raw_matrix))
    raw_matrix[inds] = np.take(col_mean, inds[1])
    z_matrix = (raw_matrix - col_mean) / col_std

    best_k, best_score, best_labels = None, -1.0, None
    for k in k_range:
        if k >= len(scenarios):
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(z_matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(z_matrix, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        return {"n_scenarios": len(scenarios), "n_features": z_matrix.shape[1], "best_k": None, "note": "군집 실패(데이터 부족/퇴화)"}

    sizes = np.bincount(best_labels).tolist()
    return {
        "n_scenarios": len(scenarios), "n_features": z_matrix.shape[1],
        "best_k": best_k, "silhouette_score": float(best_score), "cluster_sizes": sizes,
    }


def compute_macro_diversity(scenarios: list[dict], utterances: list[dict]) -> dict:
    return {
        "outcome_range_and_bins": compute_outcome_range_and_bins(scenarios, utterances),
        "trajectory_clustering": compute_trajectory_clustering(scenarios, utterances),
    }


def compute_set_diversity_metrics(
    utterances: list[dict], scenarios: list[dict], m2_gold_results: dict | None = None
) -> dict:
    utterances = plausible_diversity_filter(utterances, m2_gold_results)  # TODO: passthrough for now

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
        set_scenarios = by_set_scenarios[(policy_id, model_id)]
        results[set_name] = {
            "policy_id": policy_id,
            "model_id": model_id,
            "micro": compute_micro_diversity(utts, cells),
            "macro": compute_macro_diversity(set_scenarios, utts),
        }
    return results
