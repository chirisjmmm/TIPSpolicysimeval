"""§8 human gold 라벨링 전 표본 풀 추출. 채점/라벨링 로직 없음 — 표만 뽑아서 CSV로 저장한다.
라벨은 사람이 직접 채운다(gold_* 컬럼은 전부 빈 채로 저장).
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metrics"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from accuracy_micro import AUTHORITY_CLAIM_RE  # noqa: E402
from load_scenarios import REPO_ROOT  # noqa: E402

IR_DIR = REPO_ROOT / "data" / "ir"
RESULTS_DIR = REPO_ROOT / "results"
GOLD_DIR = REPO_ROOT / "data" / "gold"
SEED = 20260716


def load_utterances() -> dict[str, dict]:
    path = IR_DIR / "utterances.jsonl"
    return {(u := json.loads(line))["utterance_id"]: u for line in open(path, encoding="utf-8")}


def load_labels() -> dict[str, dict]:
    path = RESULTS_DIR / "m2_per_utterance_labels.csv"
    return {r["utterance_id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def largest_remainder_alloc(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    """group별 크기 비례 배분(최대잔여법) — 그룹 크기가 달라져도 비율 유지."""
    grand_total = sum(group_sizes.values())
    raw = {k: total * v / grand_total for k, v in group_sizes.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(alloc.values())
    order = sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)
    for k in order[:remainder]:
        alloc[k] += 1
    return alloc


# ---------------------------------------------------------------------------
# 1) random_gold_pool: (policy,model) 비율 유지, 완전 무작위 300건
# ---------------------------------------------------------------------------
def build_random_gold_pool(utterances: dict[str, dict], labels: dict[str, dict], n_total: int = 300) -> list[dict]:
    by_set: dict[tuple[str, str], list[str]] = {}
    for uid, u in utterances.items():
        by_set.setdefault((u["policy_id"], u["model_id"]), []).append(uid)

    group_sizes = {f"{p}_{m}": len(uids) for (p, m), uids in by_set.items()}
    alloc = largest_remainder_alloc(group_sizes, n_total)

    rng = random.Random(SEED)
    rows = []
    for (policy_id, model_id), uids in by_set.items():
        set_name = f"{policy_id}_{model_id}"
        k = alloc[set_name]
        sample_uids = rng.sample(uids, k)
        for uid in sample_uids:
            u = utterances[uid]
            r = labels[uid]
            rows.append(
                {
                    "utterance_id": uid,
                    "policy_id": policy_id,
                    "model_id": model_id,
                    "scenario_uid": u["scenario_uid"],
                    "phase": u["phase"],
                    "round": u["round"],
                    "persona_name": u["persona_name"],
                    "kg_role": u["kg_role"],
                    "deontic_status": u["deontic_status"],
                    "machine_violation_applicable": r["violation_applicable"],
                    "machine_violation": r["violation"],
                    "machine_violation_reason": r["violation_reason"],
                    "machine_grounded": r["grounded"],
                    "machine_max_cosine": r["max_cosine"],
                    "machine_fabrication_applicable": r["fabrication_applicable"],
                    "machine_fabricated": r["fabricated"],
                    "machine_derived_estimate": r["derived_estimate"],
                    "full_text": u["text"],
                    "gold_violation": "",
                    "gold_grounded": "",
                    "gold_fabrication": "",
                }
            )
    rows.sort(key=lambda r: (r["policy_id"], r["model_id"], r["scenario_uid"]))
    print(f"[random_gold_pool] 배분: {alloc} (합계 {sum(alloc.values())})")
    return rows


RANDOM_POOL_COLUMNS = [
    "utterance_id", "policy_id", "model_id", "scenario_uid", "phase", "round", "persona_name",
    "kg_role", "deontic_status",
    "machine_violation_applicable", "machine_violation", "machine_violation_reason",
    "machine_grounded", "machine_max_cosine",
    "machine_fabrication_applicable", "machine_fabricated", "machine_derived_estimate",
    "full_text",
    "gold_violation", "gold_grounded", "gold_fabrication",
]


# ---------------------------------------------------------------------------
# 2) boundary_gold_pool: 진단용 조각 6개 묶음, 조각당 해당 축 gold_label 하나만
# ---------------------------------------------------------------------------
BOUNDARY_COLUMNS = [
    "pool_component", "axis", "utterance_id", "policy_id", "model_id", "scenario_uid",
    "kg_role", "phase", "round", "machine_note", "full_text", "gold_label",
]


def _machine_note_violation(r: dict) -> str:
    return f"violation={r['violation']}; reason={r['violation_reason']}"


def _machine_note_grounded(r: dict) -> str:
    return f"grounded={r['grounded']}; max_cosine={r['max_cosine']}"


def _machine_note_fabrication(r: dict) -> str:
    return (f"fabricated={r['fabricated']}; derived_estimate={r['derived_estimate']}; "
            f"unsupported={r['unsupported_tokens']}; derived_tokens={r['derived_estimate_tokens']}")


def component_violation_diff_existing() -> list[dict]:
    path = GOLD_DIR / "qa_m2_violation_diff.csv"
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        rows.append(
            {
                "pool_component": "violation_diff_existing18",
                "axis": "violation",
                "utterance_id": r["utterance_id"],
                "policy_id": "BK21",
                "model_id": "deepseek",
                "scenario_uid": r["scenario_uid"],
                "kg_role": r["kg_role"],
                "phase": "",
                "round": "",
                "machine_note": f"status={r['status']}; pre={r['pre_reason']}; post={r['post_reason']}",
                "full_text": r["full_text"],
                "gold_label": "",
            }
        )
    return rows


def component_cross_role_authority_extra(
    utterances: dict[str, dict], labels: dict[str, dict], used_uids: set[str], n: int = 30
) -> list[dict]:
    candidates = []
    for uid, u in utterances.items():
        if uid in used_uids:
            continue
        if u["deontic_status"] != "applicable":
            continue
        if AUTHORITY_CLAIM_RE.search(u["text"] or ""):
            candidates.append(uid)

    rng = random.Random(SEED)
    sample = rng.sample(candidates, min(n, len(candidates)))
    print(f"[boundary: cross_role_authority_claim_extra] 후보 {len(candidates)}건 중 {len(sample)}건 샘플")
    rows = []
    for uid in sample:
        u = utterances[uid]
        r = labels[uid]
        rows.append(
            {
                "pool_component": "cross_role_authority_claim_extra30",
                "axis": "violation",
                "utterance_id": uid,
                "policy_id": u["policy_id"],
                "model_id": u["model_id"],
                "scenario_uid": u["scenario_uid"],
                "kg_role": u["kg_role"],
                "phase": u["phase"],
                "round": u["round"],
                "machine_note": _machine_note_violation(r),
                "full_text": u["text"],
                "gold_label": "",
            }
        )
    used_uids.update(sample)
    return rows


def component_grounding_boundary_existing() -> list[dict]:
    path = GOLD_DIR / "qa_m2_grounding_boundary.csv"
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        rows.append(
            {
                "pool_component": "grounding_boundary_existing40",
                "axis": "grounded",
                "utterance_id": r["utterance_id"],
                "policy_id": r["policy_id"],
                "model_id": r["model_id"],
                "scenario_uid": r["scenario_uid"],
                "kg_role": r["kg_role"],
                "phase": "",
                "round": "",
                "machine_note": f"grounded={r['grounded']}; max_cosine={r['max_cosine']}; threshold={r['threshold']}",
                "full_text": r["full_text"],
                "gold_label": r.get("gold_grounded", ""),
            }
        )
    return rows


ARITH_CUE_RE = re.compile(
    r"%|(\bper\b)|(\btotal\b)|(\bsum\b)|(\baverage\b)|(\bdivid(e|ed|ing)\b)|(\bmultipl(y|ied|ying)\b)"
    r"|(\d\s*[\*x×]\s*\d)|(\d\s*/\s*\d)|(\bproportion\w*\b)|(\bshare\b)|(≈)|(\bso about\b)",
    re.IGNORECASE,
)


def _has_nearby_arith_cue(text: str, unsupported_tokens: str, window: int = 60) -> bool:
    for tok in (t.strip() for t in unsupported_tokens.split(";") if t.strip()):
        core = re.match(r"[\d,.]+", tok)
        search_str = core.group(0) if core else tok
        idx = text.find(search_str)
        if idx < 0:
            continue
        w = text[max(0, idx - window): idx + len(search_str) + window]
        if ARITH_CUE_RE.search(w):
            return True
    return False


def component_fabrication_slices(
    utterances: dict[str, dict], labels: dict[str, dict], used_uids: set[str]
) -> list[dict]:
    rng = random.Random(SEED)
    rows = []

    # (a) derived_estimate=True 거의 전수
    derived_uids = [uid for uid, r in labels.items() if r["derived_estimate"] == "True" and uid not in used_uids]
    print(f"[boundary: fabrication derived_estimate_all] {len(derived_uids)}건 전부 포함")
    for uid in derived_uids:
        u = utterances[uid]
        r = labels[uid]
        rows.append(
            {
                "pool_component": "fabrication_derived_estimate_all",
                "axis": "fabrication",
                "utterance_id": uid,
                "policy_id": u["policy_id"],
                "model_id": u["model_id"],
                "scenario_uid": u["scenario_uid"],
                "kg_role": u["kg_role"],
                "phase": u["phase"],
                "round": u["round"],
                "machine_note": _machine_note_fabrication(r),
                "full_text": u["text"],
                "gold_label": "",
            }
        )
    used_uids.update(derived_uids)

    # (b) fabricated=True 잔여 무작위 40 (1-hop 의심군은 아래서 따로 뽑으므로 여기서 먼저 제외해두고 뽑음)
    fabricated_uids_pool = [uid for uid, r in labels.items() if r["fabricated"] == "True" and uid not in used_uids]
    onehop_suspect_all = [
        uid for uid in fabricated_uids_pool
        if _has_nearby_arith_cue(utterances[uid]["text"] or "", labels[uid]["unsupported_tokens"])
    ]
    residual_pool = [uid for uid in fabricated_uids_pool if uid not in onehop_suspect_all]
    residual_sample = rng.sample(residual_pool, min(40, len(residual_pool)))
    print(f"[boundary: fabrication_residual] 후보 {len(residual_pool)}건 중 {len(residual_sample)}건 샘플")
    for uid in residual_sample:
        u = utterances[uid]
        r = labels[uid]
        rows.append(
            {
                "pool_component": "fabrication_residual40",
                "axis": "fabrication",
                "utterance_id": uid,
                "policy_id": u["policy_id"],
                "model_id": u["model_id"],
                "scenario_uid": u["scenario_uid"],
                "kg_role": u["kg_role"],
                "phase": u["phase"],
                "round": u["round"],
                "machine_note": _machine_note_fabrication(r),
                "full_text": u["text"],
                "gold_label": "",
            }
        )
    used_uids.update(residual_sample)

    # (c) 1-hop 제약 의심 20건: fabricated=True & derived_estimate=False인데 토큰 근처(±60자)에 산술 단서
    onehop_sample = rng.sample(onehop_suspect_all, min(20, len(onehop_suspect_all)))
    print(f"[boundary: fabrication_1hop_suspect] 후보 {len(onehop_suspect_all)}건 중 {len(onehop_sample)}건 샘플")
    for uid in onehop_sample:
        u = utterances[uid]
        r = labels[uid]
        rows.append(
            {
                "pool_component": "fabrication_1hop_suspect20",
                "axis": "fabrication",
                "utterance_id": uid,
                "policy_id": u["policy_id"],
                "model_id": u["model_id"],
                "scenario_uid": u["scenario_uid"],
                "kg_role": u["kg_role"],
                "phase": u["phase"],
                "round": u["round"],
                "machine_note": _machine_note_fabrication(r),
                "full_text": u["text"],
                "gold_label": "",
            }
        )
    used_uids.update(onehop_sample)

    return rows


def build_boundary_gold_pool(utterances: dict[str, dict], labels: dict[str, dict]) -> list[dict]:
    used_uids: set[str] = set()

    violation_diff_rows = component_violation_diff_existing()
    used_uids.update(r["utterance_id"] for r in violation_diff_rows)

    cross_role_rows = component_cross_role_authority_extra(utterances, labels, used_uids)

    grounding_rows = component_grounding_boundary_existing()
    used_uids.update(r["utterance_id"] for r in grounding_rows)

    fabrication_rows = component_fabrication_slices(utterances, labels, used_uids)

    return violation_diff_rows + cross_role_rows + grounding_rows + fabrication_rows


def main() -> None:
    utterances = load_utterances()
    labels = load_labels()

    print("=== 1) random_gold_pool (300건, policy×model 비율 유지, 완전 무작위) ===")
    random_rows = build_random_gold_pool(utterances, labels)
    write_csv(random_rows, GOLD_DIR / "random_gold_pool.csv", RANDOM_POOL_COLUMNS)
    print(f"  저장: {GOLD_DIR / 'random_gold_pool.csv'} ({len(random_rows)}행)")

    print("\n=== 2) boundary_gold_pool (진단 표본 묶음) ===")
    boundary_rows = build_boundary_gold_pool(utterances, labels)
    write_csv(boundary_rows, GOLD_DIR / "boundary_gold_pool.csv", BOUNDARY_COLUMNS)
    from collections import Counter
    comp_counts = Counter(r["pool_component"] for r in boundary_rows)
    print(f"  구성: {dict(comp_counts)}")
    print(f"  저장: {GOLD_DIR / 'boundary_gold_pool.csv'} ({len(boundary_rows)}행)")

    print(f"\nrandom seed = {SEED}")


if __name__ == "__main__":
    main()
