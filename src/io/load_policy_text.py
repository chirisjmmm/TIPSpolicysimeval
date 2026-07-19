"""§7.1 grounding/fabrication/deontic 준비: {policy}_policy_graph.json 파싱.

파싱 로직만 담당한다(metric 계산 없음, §12 가드레일). KG의 top-level "edges"만 신뢰하고
"validation_warnings"(source_text 근거가 약해 감사용으로만 남긴 항목, unsupported_source_evidence)는
정식 규범/그라운딩 소스로 쓰지 않는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

POLICY_GRAPH_PATHS = {
    "BK21": "data/kor_BK21/BK21_policy_graph.json",
    "100E": "data/sing_100E/100E_policy_graph.json",
}

DEONTIC_VALUES = {"must", "can", "cannot"}


@dataclass
class PolicyKG:
    policy_id: str
    nodes: list[dict]
    edges: list[dict]
    # source 노드 type(=stakeholder_type/kg_role 접두어) -> 그 type이 source인 deontic edge 목록
    norms_by_role_type: dict[str, list[dict]] = field(default_factory=dict)
    # role type -> 그 role만 갖는 decision_authority류 권한 서술(짧은 구) 목록
    authority_phrases_by_role_type: dict[str, list[str]] = field(default_factory=dict)
    # 그라운딩/날조 검사용 텍스트 조각(노드 summary+source_spans, 엣지 fact+source_text). 중복 제거.
    # pooled 형태 — lexical(TF-IDF) baseline 및 fabrication 수치 인덱스용으로 유지.
    grounding_corpus: list[str] = field(default_factory=list)
    # semantic grounding(§7.1)용 "규범 항목" 단위 리스트. pooled grounding_corpus와 내용은 겹치지만
    # 이쪽은 각 항목이 어디서 왔는지(kind/ref_id)를 보존해 "어느 규범에 근거했는가" 추적이 가능하다.
    # kind: node_summary | source_span | edge_fact | deontic_norm(must/can/cannot 엣지의 fact/source_text
    # — 가장 중요한 규범이라 edge_fact와 구분해서 표시). (text, kind) 기준 중복만 제거.
    norm_units: list[dict] = field(default_factory=list)


def _node_type(node_id_or_type: str) -> str:
    """'Type:Name' 형식에서 Type만 추출. kg_role 필드에도 그대로 쓸 수 있게 콜론 기준으로 자른다."""
    return node_id_or_type.split(":", 1)[0]


def load_policy_kg(policy_id: str, repo_root: Path | None = None) -> PolicyKG:
    repo_root = repo_root or REPO_ROOT
    path = repo_root / POLICY_GRAPH_PATHS[policy_id]
    raw = json.loads(path.read_text(encoding="utf-8"))

    nodes = raw["nodes"]
    edges = raw["edges"]  # validation_warnings는 의도적으로 제외(§ 상단 docstring)

    norms_by_role_type: dict[str, list[dict]] = {}
    grounding_corpus: list[str] = []
    norm_units: list[dict] = []
    seen_text: set[str] = set()
    seen_unit: set[tuple[str, str]] = set()

    def _add_text(text: str | None) -> None:
        if text and text not in seen_text:
            seen_text.add(text)
            grounding_corpus.append(text)

    def _add_unit(text: str | None, kind: str, ref_id: str) -> None:
        if not text:
            return
        key = (text, kind)
        if key in seen_unit:
            return
        seen_unit.add(key)
        norm_units.append({"text": text, "kind": kind, "ref_id": ref_id})

    authority_phrases_by_role_type: dict[str, list[str]] = {}
    for node in nodes:
        ntype = node.get("type", _node_type(node["id"]))
        summary = node.get("summary")
        _add_text(summary)
        _add_unit(summary, "node_summary", node["id"])
        for i, span in enumerate(node.get("source_spans", []) or []):
            span_text = span.get("text")
            _add_text(span_text)
            _add_unit(span_text, "source_span", f"{node['id']}#span{i}")
        decision_authority = (node.get("attributes") or {}).get("decision_authority")
        if decision_authority:
            authority_phrases_by_role_type.setdefault(ntype, []).append(decision_authority)

    for edge in edges:
        fact = edge.get("fact")
        source_text = edge.get("source_text")
        _add_text(fact)
        _add_text(source_text)
        deontic = edge.get("deontic")
        # must/can/cannot 엣지의 fact/source_text는 "deontic_norm"으로 별도 표시(가장 중요한 규범).
        unit_kind = "deontic_norm" if deontic in DEONTIC_VALUES else "edge_fact"
        _add_unit(fact, unit_kind, edge["id"])
        _add_unit(source_text, unit_kind, f"{edge['id']}#source_text")
        if deontic in DEONTIC_VALUES:
            source_type = _node_type(edge["source"])
            norms_by_role_type.setdefault(source_type, []).append(
                {
                    "edge_id": edge["id"],
                    "type": edge["type"],
                    "deontic": deontic,
                    "source": edge["source"],
                    "target": edge["target"],
                    "fact": edge.get("fact"),
                    "source_text": edge.get("source_text"),
                }
            )

    return PolicyKG(
        policy_id=policy_id,
        nodes=nodes,
        edges=edges,
        norms_by_role_type=norms_by_role_type,
        authority_phrases_by_role_type=authority_phrases_by_role_type,
        grounding_corpus=grounding_corpus,
        norm_units=norm_units,
    )


def load_all_policy_kgs(repo_root: Path | None = None) -> dict[str, PolicyKG]:
    return {policy_id: load_policy_kg(policy_id, repo_root) for policy_id in POLICY_GRAPH_PATHS}
