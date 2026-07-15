"""grounding 임계값을 human gold로 재검토하기 위한 준비 스크립트(§8 전 단계).

data/gold/qa_m2_grounding_boundary.csv에 사람이 "gold_grounded"(True/False) 컬럼을 채워 넣은 뒤
이 스크립트를 실행하면, policy별로 여러 후보 임계값의 precision/recall/F1을 계산해 표로 보여준다.
아직 gold_grounded가 비어 있으면(=라벨링 전) 그 사실만 알리고 종료한다 — 라벨링 전에 미리 점수를
계산하거나 임계값을 확정하지 않는다.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))
from load_scenarios import REPO_ROOT  # noqa: E402

GOLD_PATH = REPO_ROOT / "data" / "gold" / "qa_m2_grounding_boundary.csv"

TRUE_STRINGS = {"true", "1", "y", "yes", "grounded", "o"}
FALSE_STRINGS = {"false", "0", "n", "no", "not_grounded", "x"}


def _parse_bool(s: str) -> bool | None:
    s = (s or "").strip().lower()
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    return None  # 비어있거나 인식 못하는 값 = 아직 라벨링 안 됨


def load_labeled_rows(path: Path = GOLD_PATH) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gold = _parse_bool(r.get("gold_grounded", ""))
            if gold is None:
                continue
            rows.append(
                {
                    "policy_id": r["policy_id"],
                    "max_cosine": float(r["max_cosine"]),
                    "gold_grounded": gold,
                }
            )
    return rows


def precision_recall_f1(rows: list[dict], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        pred = r["max_cosine"] >= threshold
        gold = r["gold_grounded"]
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) else None
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def sweep_thresholds(rows: list[dict], step: float = 0.02) -> list[dict]:
    if not rows:
        return []
    cosines = sorted(r["max_cosine"] for r in rows)
    lo, hi = cosines[0], cosines[-1]
    thresholds = []
    t = lo
    while t <= hi + 1e-9:
        thresholds.append(round(t, 4))
        t += step
    return [precision_recall_f1(rows, t) for t in thresholds]


def main() -> None:
    if not GOLD_PATH.exists():
        print(f"{GOLD_PATH} 없음 — 먼저 qa_m2_sample.py로 생성해야 함")
        return

    all_rows = list(csv.DictReader(open(GOLD_PATH, encoding="utf-8-sig")))
    n_total = len(all_rows)
    labeled = load_labeled_rows()
    n_labeled = len(labeled)

    print(f"gold_grounded 라벨링 현황: {n_labeled}/{n_total}행")
    if n_labeled == 0:
        print("아직 라벨링된 행이 없다. qa_m2_grounding_boundary.csv의 gold_grounded 컬럼을 채운 뒤 다시 실행할 것.")
        print("허용 값: true/false, 1/0, y/n, yes/no (대소문자 무관)")
        return

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for r in labeled:
        by_policy[r["policy_id"]].append(r)

    for policy_id, rows in by_policy.items():
        print(f"\n=== {policy_id} (라벨링된 {len(rows)}행) ===")
        results = sweep_thresholds(rows)
        best = max((r for r in results if r["f1"] is not None), key=lambda r: r["f1"], default=None)
        print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'tp':>5}{'fp':>5}{'fn':>5}{'tn':>5}")
        for r in results:
            p = f"{r['precision']:.3f}" if r["precision"] is not None else "N/A"
            rec = f"{r['recall']:.3f}" if r["recall"] is not None else "N/A"
            f1 = f"{r['f1']:.3f}" if r["f1"] is not None else "N/A"
            marker = "  <- best F1" if best and r["threshold"] == best["threshold"] else ""
            print(f"{r['threshold']:>10.3f} {p:>10} {rec:>10} {f1:>10} "
                  f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}{r['tn']:>5}{marker}")
        if best:
            print(f"  -> F1 최대 임계값: {best['threshold']:.3f} (precision={best['precision']:.3f}, recall={best['recall']:.3f})")
        print("  (표본이 median ±0.05 경계 근처에서만 뽑혔으므로 이 구간 바깥의 precision/recall은 추정 불가 — 참고용)")


if __name__ == "__main__":
    main()
