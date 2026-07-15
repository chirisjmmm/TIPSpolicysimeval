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
    grounding_corpus: list[str] = field(default_factory=list)


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
    seen_text: set[str] = set()

    def _add_text(text: str | None) -> None:
        if text and text not in seen_text:
            seen_text.add(text)
            grounding_corpus.append(text)

    authority_phrases_by_role_type: dict[str, list[str]] = {}
    for node in nodes:
        ntype = node.get("type", _node_type(node["id"]))
        _add_text(node.get("summary"))
        for span in node.get("source_spans", []) or []:
            _add_text(span.get("text"))
        decision_authority = (node.get("attributes") or {}).get("decision_authority")
        if decision_authority:
            authority_phrases_by_role_type.setdefault(ntype, []).append(decision_authority)

    for edge in edges:
        _add_text(edge.get("fact"))
        _add_text(edge.get("source_text"))
        deontic = edge.get("deontic")
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
    )


def load_all_policy_kgs(repo_root: Path | None = None) -> dict[str, PolicyKG]:
    return {policy_id: load_policy_kg(policy_id, repo_root) for policy_id in POLICY_GRAPH_PATHS}
