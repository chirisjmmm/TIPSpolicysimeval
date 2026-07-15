"""§8 신뢰성 검증: violation 축 gold vs machine 대조, Krippendorff α, 혼동행렬,
initial/revised 판정 불일치 점검. 채점 로직은 여기서 끝 — M2 rule 자체는 건드리지 않는다.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import krippendorff
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "data" / "gold"
ALPHA_THRESHOLD = 0.667


def _parse_bool(s: str) -> int | None:
    s = (s or "").strip().upper()
    if s in ("TRUE", "1", "YES", "Y"):
        return 1
    if s in ("FALSE", "0", "NO", "N"):
        return 0
    return None  # 공란/미라벨링


def load_violation_gold(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compute_alpha(rows: list[dict]) -> dict:
    machine_codes = []
    gold_codes = []
    used_rows = []
    for r in rows:
        m = _parse_bool(r["machine_violation"])
        g = _parse_bool(r["gold_violation"])
        if m is None or g is None:
            continue
        machine_codes.append(m)
        gold_codes.append(g)
        used_rows.append(r)

    reliability_data = np.array([machine_codes, gold_codes], dtype=float)
    alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="nominal")
    return {"alpha": float(alpha), "n_units": len(used_rows), "used_rows": used_rows,
            "machine_codes": machine_codes, "gold_codes": gold_codes}


def confusion_matrix(machine_codes: list[int], gold_codes: list[int]) -> dict:
    tp = fp = fn = tn = 0
    for m, g in zip(machine_codes, gold_codes):
        if m == 1 and g == 1:
            tp += 1
        elif m == 1 and g == 0:
            fp += 1
        elif m == 0 and g == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def check_initial_revised_consistency(rows: list[dict]) -> dict:
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        key = (r["scenario_uid"], r["phase"], r["persona_name"])
        by_key[key][r["round"]] = r

    mismatches = []
    n_pairs_checked = 0
    for key, rounds in by_key.items():
        if "initial" not in rounds or "revised" not in rounds:
            continue
        n_pairs_checked += 1
        g_init = _parse_bool(rounds["initial"]["gold_violation"])
        g_rev = _parse_bool(rounds["revised"]["gold_violation"])
        if g_init is None or g_rev is None:
            continue
        if g_init != g_rev:
            mismatches.append(
                {
                    "scenario_uid": key[0], "phase": key[1], "persona_name": key[2],
                    "gold_initial": rounds["initial"]["gold_violation"],
                    "gold_revised": rounds["revised"]["gold_violation"],
                    "utterance_id_initial": rounds["initial"]["utterance_id"],
                    "utterance_id_revised": rounds["revised"]["utterance_id"],
                }
            )
    return {"n_pairs_checked": n_pairs_checked, "n_mismatches": len(mismatches), "mismatches": mismatches}


def main() -> None:
    path = GOLD_DIR / "violation_gold_pool.csv"
    rows = load_violation_gold(path)

    n_labeled = sum(1 for r in rows if _parse_bool(r["gold_violation"]) is not None)
    print(f"violation_gold_pool.csv: 총 {len(rows)}행, gold_violation 라벨링됨 {n_labeled}행")

    result = compute_alpha(rows)
    alpha = result["alpha"]
    print(f"\nKrippendorff alpha (violation, nominal, n={result['n_units']}) = {alpha:.4f}")

    cm = confusion_matrix(result["machine_codes"], result["gold_codes"])
    print("\n혼동행렬 (machine vs gold):")
    print(f"  TP(둘다 위반)={cm['tp']}  FP(machine만 위반=오탐)={cm['fp']}  "
          f"FN(gold만 위반=누락)={cm['fn']}  TN(둘다 무위반)={cm['tn']}")
    prec = f"{cm['precision']:.3f}" if cm["precision"] is not None else "N/A(분모 0)"
    rec = f"{cm['recall']:.3f}" if cm["recall"] is not None else "N/A(분모 0)"
    print(f"  precision={prec}  recall={rec}")

    print("\ninitial/revised 판정 불일치 점검:")
    ir = check_initial_revised_consistency(result["used_rows"])
    print(f"  initial/revised 쌍 확인 가능: {ir['n_pairs_checked']}쌍, 불일치: {ir['n_mismatches']}건")
    for m in ir["mismatches"]:
        print(f"    {m['scenario_uid']} {m['phase']} {m['persona_name']}: "
              f"initial={m['gold_initial']} revised={m['gold_revised']}")

    print(f"\n{'='*70}")
    if alpha >= ALPHA_THRESHOLD:
        print(f"alpha={alpha:.4f} >= {ALPHA_THRESHOLD} -> violation_rate: VALIDATED로 표시 가능")
    else:
        print(f"alpha={alpha:.4f} < {ALPHA_THRESHOLD} -> violation_rate: UNVALIDATED 유지")
        print("\n오탐(machine=True, gold=False) 사례:")
        for r in result["used_rows"]:
            if _parse_bool(r["machine_violation"]) == 1 and _parse_bool(r["gold_violation"]) == 0:
                print(f"  [{r['utterance_id']}] kg_role={r['kg_role']}")
                print(f"    reason={r['machine_violation_reason']}")
        print("\n누락(machine=False, gold=True) 사례:")
        found_fn = False
        for r in result["used_rows"]:
            if _parse_bool(r["machine_violation"]) == 0 and _parse_bool(r["gold_violation"]) == 1:
                found_fn = True
                print(f"  [{r['utterance_id']}] kg_role={r['kg_role']}")
                print(f"    text(앞부분)={r['full_text'][:200]}")
        if not found_fn:
            print("  (없음)")


if __name__ == "__main__":
    main()
