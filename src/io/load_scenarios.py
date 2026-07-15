"""§3, §4 IR 파서: 원본 scenario_*.json -> 공통 IR(Scenario, Utterance).

파싱 로직만 담당한다(metric 계산 없음, §12 가드레일). raw 필드는 그대로 보존하고
(scenario_uid, kg_role, flag_policyrole, turn_index 등) 파생 필드만 덧붙인다.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_ORDER = ["Inputs", "Activities", "Outputs", "Outcomes", "Impact"]
PASS_FIELDS = ["forward_pass", "backward_pass", "fwd_bwd_fwd_pass", "fwd_bwd_fwd_bwd_pass"]
ROUNDS = ["initial_posts", "revised_posts", "refined_posts"]
SCENARIO_FILE_RE = re.compile(r"^scenario_(\d+)\.json$")


class ParseError(AssertionError):
    """M1 수용기준 위반 시 raise. 가이드 §3: 실패 시 assert로 중단."""


def load_config(config_path: Path | None = None) -> dict:
    config_path = config_path or (REPO_ROOT / "config" / "policies.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _natural_scenario_files(dir_path: Path) -> list[tuple[int, Path]]:
    """폴더 내 scenario_*.json만, 자연 정렬(숫자 기준)로 반환. 메타 파일/__MACOSX 제외."""
    out = []
    for f in dir_path.iterdir():
        if "__MACOSX" in str(f):
            continue
        m = SCENARIO_FILE_RE.match(f.name)
        if m:
            out.append((int(m.group(1)), f))
    out.sort(key=lambda t: t[0])
    return out


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _build_keyword_pattern(keywords_cfg: dict) -> re.Pattern:
    all_kw = keywords_cfg.get("ko", []) + keywords_cfg.get("en", [])
    escaped = sorted((re.escape(kw) for kw in all_kw), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def _flag_policyrole(participant: dict, keywords_cfg: dict) -> bool:
    """§3 flag_policyrole 휴리스틱. 단어 경계 매칭 이유:
    - "internal"의 "intern" 부분일치 오탐 방지.
    - 복수형 "students"(제3자 대상, 예: "students with dyslexia")는 단수 "student" 단어 경계
      매칭에서 자동 제외되어, 사실상 페르소나 자신을 가리키는 단수형 서술만 남는다.
    """
    if participant["stakeholder_type"] != "PolicyRole":
        return False
    pattern = _build_keyword_pattern(keywords_cfg)
    fields = [
        participant.get("occupation"),
        participant.get("education_level"),
        participant.get("professional_persona"),
    ]
    return any(f and pattern.search(f) for f in fields)


def resolve_participant_role(participant: dict, policy_id: str, config: dict) -> dict:
    """참가자 1명의 kg_role/flag_policyrole/mislabel_source를 계산한다.
    kg_role은 policies.yaml role_map에 등록된 실제 KG 노드 인덱스 키("Type:Name")이며,
    entity_name에서 역산한 키와 실제로 일치하는지 매 참가자마다 assert한다(M0 가정 검증).
    """
    role_map_for_policy = config["role_map"][policy_id]
    keywords_cfg = config["policyrole_student_keywords"]
    merge_target = config["bk21_policyrole_merge_target"]

    st = participant["stakeholder_type"]
    if st not in role_map_for_policy:
        raise ParseError(
            f"unmapped stakeholder_type {st!r} for policy {policy_id} "
            f"(policies.yaml role_map.{policy_id}에 등록 필요)"
        )
    expected_kg_key = role_map_for_policy[st]

    entity_name = participant.get("entity_name", "") or ""
    name_part, sep, suffix = entity_name.rpartition(" | ")
    if not sep or not suffix.startswith("KG-Gen-"):
        raise ParseError(f"entity_name 형식 불일치(참가자 {participant.get('name')!r}): {entity_name!r}")
    actual_kg_key = f"{st}:{name_part}"
    if actual_kg_key != expected_kg_key:
        raise ParseError(
            f"entity_name 기반 KG 키 {actual_kg_key!r} != role_map[{policy_id}][{st}]={expected_kg_key!r} "
            f"(M0 가정과 실제 데이터가 어긋남, policies.yaml role_map 갱신 필요)"
        )

    flag = _flag_policyrole(participant, keywords_cfg)
    kg_role = expected_kg_key
    mislabel_source = None
    if policy_id == "BK21" and flag:
        kg_role = role_map_for_policy[merge_target]
        mislabel_source = "policyrole_merged"

    no_institutional_position = config.get("kg_role_no_institutional_position", [])
    deontic_status = "no_institutional_position_in_kg" if kg_role in no_institutional_position else "applicable"

    return {
        "stakeholder_type": st,
        "kg_role": kg_role,
        "flag_policyrole": flag,
        "mislabel_source": mislabel_source,
        "deontic_status": deontic_status,
    }


@dataclass
class Scenario:
    scenario_uid: str
    source_id: str
    policy_id: str
    model_id: str
    global_index: int
    src_path: str
    participants: list[dict]
    cross_checks: list[dict]
    scenario_impact: dict
    scenario_confidence: dict
    phase_summaries: dict[str, str]  # phase -> phase_summary 원문(§7.2 cross-phase coherence용)


@dataclass
class Utterance:
    utterance_id: str
    scenario_uid: str
    policy_id: str
    model_id: str
    pass_name: str
    direction: str
    phase: str
    round: str
    turn_index: int
    persona_name: str
    stakeholder_type: str
    kg_role: str
    flag_policyrole: bool
    mislabel_source: str | None
    deontic_status: str
    text: str
    evidence: list
    judgment: Any
    prediction_values: dict


def _round_key_to_label(round_key: str) -> str:
    return round_key.replace("_posts", "")  # initial_posts -> initial


def parse_scenario_file(
    path: Path,
    policy_id: str,
    model_id: str,
    global_index: int,
    config: dict,
) -> tuple[Scenario, list[Utterance]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenario_uid = f"{policy_id}_{model_id}_{global_index:02d}"

    participants = raw["participants"]
    if len(participants) != 5:
        raise ParseError(f"{path}: participants != 5 (got {len(participants)})")

    persona_role: dict[str, dict] = {}
    for p in participants:
        try:
            persona_role[p["name"]] = resolve_participant_role(p, policy_id, config)
        except ParseError as e:
            raise ParseError(f"{path}: {e}") from e

    forward_pass = raw.get("forward_pass")
    if forward_pass is None:
        raise ParseError(f"{path}: forward_pass is null")
    phases_found = [ph["phase"] for ph in forward_pass]
    if phases_found != PHASE_ORDER:
        raise ParseError(f"{path}: phase order {phases_found} != {PHASE_ORDER}")

    utterances: list[Utterance] = []
    phase_summaries: dict[str, str] = {}

    for pass_field in PASS_FIELDS:
        pass_data = raw.get(pass_field)
        if pass_data is None:
            continue  # §3: non-null pass만 순회
        pass_name = pass_field.replace("_pass", "")
        for phase_dict in pass_data:
            phase = phase_dict["phase"]
            direction = phase_dict["direction"]
            posting_order = phase_dict["posting_order"]
            if pass_name == "forward" and phase_dict.get("phase_summary"):
                phase_summaries[phase] = phase_dict["phase_summary"]

            for round_key in ROUNDS:
                posts = phase_dict.get(round_key)
                if posts is None:
                    continue  # refined_posts는 없을 수 있음(§3)
                if round_key in ("initial_posts", "revised_posts") and len(posts) != 5:
                    raise ParseError(
                        f"{path}: phase={phase} {round_key} has {len(posts)} posts (expected 5)"
                    )
                post_personas = {p["persona_name"] for p in posts}
                if set(posting_order) != post_personas:
                    raise ParseError(
                        f"{path}: phase={phase} {round_key} persona set {post_personas} "
                        f"!= posting_order set {set(posting_order)}"
                    )

                round_label = _round_key_to_label(round_key)
                for post in posts:
                    persona_name = post["persona_name"]
                    turn_index = posting_order.index(persona_name)
                    role_info = persona_role.get(persona_name)
                    if role_info is None:
                        raise ParseError(
                            f"{path}: persona_name {persona_name!r} in posting_order/post "
                            f"not found in participants"
                        )
                    utterances.append(
                        Utterance(
                            utterance_id=f"{scenario_uid}_{phase}_{round_label}_{turn_index}",
                            scenario_uid=scenario_uid,
                            policy_id=policy_id,
                            model_id=model_id,
                            pass_name=pass_name,
                            direction=direction,
                            phase=phase,
                            round=round_label,
                            turn_index=turn_index,
                            persona_name=persona_name,
                            stakeholder_type=post["stakeholder_type"],
                            kg_role=role_info["kg_role"],
                            flag_policyrole=role_info["flag_policyrole"],
                            mislabel_source=role_info["mislabel_source"],
                            deontic_status=role_info["deontic_status"],
                            text=post["narrative"],
                            evidence=post.get("evidence", []),
                            judgment=post.get("judgment"),
                            prediction_values=post.get("prediction_values", {}),
                        )
                    )

    scenario = Scenario(
        scenario_uid=scenario_uid,
        source_id=raw["scenario_id"],
        policy_id=policy_id,
        model_id=model_id,
        global_index=global_index,
        src_path=str(path),
        participants=participants,
        cross_checks=raw.get("cross_checks", []),
        scenario_impact=raw.get("scenario_impact", {}),
        scenario_confidence=raw.get("scenario_confidence", {}),
        phase_summaries=phase_summaries,
    )
    return scenario, utterances


def load_set(set_name: str, policy_id: str, model_id: str, config: dict) -> tuple[list[Scenario], list[Utterance]]:
    """set_name: config['raw_data_roots']의 키. 폴더 리스트 순서 = §4 folder_rank 순서."""
    dirs = [REPO_ROOT / d for d in config["raw_data_roots"][set_name]]

    all_files_by_global_index: dict[int, Path] = {}
    if len(dirs) == 1:
        for local_idx, path in _natural_scenario_files(dirs[0]):
            all_files_by_global_index[local_idx] = path
    else:
        # §4: gemini 5폴더 x 10파일(이름 1~10 중복) -> global_index = (folder_rank-1)*10 + local_index
        for folder_rank, d in enumerate(dirs, start=1):
            local_files = _natural_scenario_files(d)
            if len(local_files) != 10:
                raise ParseError(f"{set_name}: folder {d} has {len(local_files)} scenario files (expected 10)")
            for local_idx, path in local_files:
                if not (1 <= local_idx <= 10):
                    raise ParseError(f"{set_name}: unexpected local index {local_idx} in {d}")
                global_index = (folder_rank - 1) * 10 + local_idx
                all_files_by_global_index[global_index] = path

    # 중복 방지 (a): 전역 인덱스 유일성은 dict 구성 자체로 보장되나, 폴더 수 mismatch로 인한
    # 인덱스 누락/충돌을 아래 assert로 명시적으로 잡는다.
    if len(dirs) > 1:
        expected = set(range(1, 51))
        got = set(all_files_by_global_index.keys())
        if got != expected:
            raise ParseError(f"{set_name}: global_index set {sorted(got)} != 1..50")

    # 중복 방지 (b): 파일 내용 md5로 물리적 중복 탐지
    hashes: dict[str, list[int]] = {}
    for gi, path in all_files_by_global_index.items():
        h = _md5(path)
        hashes.setdefault(h, []).append(gi)
    dup_hashes = {h: idxs for h, idxs in hashes.items() if len(idxs) > 1}
    if dup_hashes:
        raise ParseError(f"{set_name}: 내용 중복 파일 발견(md5 충돌): {dup_hashes}")

    scenarios: list[Scenario] = []
    utterances: list[Utterance] = []
    for gi in sorted(all_files_by_global_index):
        path = all_files_by_global_index[gi]
        scenario, utts = parse_scenario_file(path, policy_id, model_id, gi, config)
        scenarios.append(scenario)
        utterances.extend(utts)

    # 중복 방지 (c): uid 유일성 + 50개 확인
    uids = [s.scenario_uid for s in scenarios]
    if len(uids) != 50:
        raise ParseError(f"{set_name}: scenario count {len(uids)} != 50")
    if len(set(uids)) != 50:
        raise ParseError(f"{set_name}: scenario_uid 중복 발견")

    return scenarios, utterances


SET_SPECS = [
    ("BK21_deepseek", "BK21", "deepseek"),
    ("100E_deepseek", "100E", "deepseek"),
    ("100E_gemini", "100E", "gemini"),
]


def load_all(config: dict | None = None) -> tuple[list[Scenario], list[Utterance]]:
    config = config or load_config()
    all_scenarios: list[Scenario] = []
    all_utterances: list[Utterance] = []
    for set_name, policy_id, model_id in SET_SPECS:
        scenarios, utterances = load_set(set_name, policy_id, model_id, config)
        all_scenarios.extend(scenarios)
        all_utterances.extend(utterances)

    all_uids = [s.scenario_uid for s in all_scenarios]
    if len(all_uids) != 150:
        raise ParseError(f"전체 scenario 수 {len(all_uids)} != 150 (50x3)")
    if len(set(all_uids)) != 150:
        raise ParseError("전체 scenario_uid 중복 발견 (set 간 uid 충돌)")

    return all_scenarios, all_utterances
