import atexit
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# =========================
# 配置区（常改）
# =========================
# 当前阶段目录：大规模实验/phase3
BASE_DIR = Path(__file__).resolve().parent.parent
# 项目根目录：大规模实验
PROJECT_ROOT = BASE_DIR.parent

# 运行模式：class=类概念轮；property=属性轮
RUN_MODE: Literal["class", "property"] = "property"
INPUT_VERSION = "V3"
OUTPUT_VERSION = "V4"
# S3-03 输出目录：本步骤所有产物统一落盘到此目录
OUTPUT_DIR = BASE_DIR / "outputs_phase3_03"
if RUN_MODE == "class":
    # Class round 输入：class validated 决策
    INPUT_VALIDATED = BASE_DIR / "outputs_phase3_02" / "validated_class_decisions.json"
    # Class round 基座：原始本体 TTL
    INPUT_TTL = PROJECT_ROOT / "shared_resources" / f"TrafficKG_{INPUT_VERSION}.ttl"
else:
    # Property round 输入：property validated 决策
    INPUT_VALIDATED = BASE_DIR / "outputs_phase3_02" / "validated_property_decisions.json"
    # Property round 基座：class round 合并后的预览 TTL
    INPUT_TTL = BASE_DIR / "outputs_phase3_04" / f"TrafficKG_{OUTPUT_VERSION}_class_merged_preview.ttl"

# 输出1：增量变更集（供 S3-04 读取）
OUTPUT_CHANGESET = OUTPUT_DIR / f"{RUN_MODE}_changeset.json"
# 输出2：预览 TTL（便于人工核查）
OUTPUT_PREVIEW_TTL = OUTPUT_DIR / f"{RUN_MODE}_changeset.ttl"
# 输出3：构建失败明细
OUTPUT_ERRORS = OUTPUT_DIR / f"{RUN_MODE}_changeset_errors.json"
# 输出4：运行摘要（参数、计数、耗时）
OUTPUT_MANIFEST = OUTPUT_DIR / f"{RUN_MODE}_changeset_manifest.json"
OUTPUT_SUMMARY = OUTPUT_DIR / f"{RUN_MODE}_changeset_summary.json"
# 是否将控制台输出同步写入日志文件
ENABLE_FILE_LOG = True
# 统一日志目录：保存本脚本控制台输出
LOG_DIR = PROJECT_ROOT / "logs"

# 小样本参数：起始索引（默认从第 0 条开始）
START_INDEX = 0
# 小样本参数：最大处理条数，0 表示处理全部
MAX_ITEMS = 0
# 单条记录 schema 校验失败后的最大重试次数
MAX_RETRIES = 2
# 重试指数退避基准秒数
RETRY_BASE_SECONDS = 1.0

_LOG_FILE_HANDLE = None
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None


class TeeStream:
    """将控制台输出同时写入原始流与日志文件。"""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            try:
                if getattr(stream, "closed", False):
                    continue
                stream.write(data)
            except ValueError:
                continue
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                if getattr(stream, "closed", False):
                    continue
                stream.flush()
            except ValueError:
                continue


def setup_console_log(script_name: str) -> Path | None:
    """启用控制台双写：控制台照常显示，同时保存到日志文件。"""
    global _LOG_FILE_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    if not ENABLE_FILE_LOG:
        return None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{script_name}_{timestamp}.log"
    _LOG_FILE_HANDLE = open(log_path, "w", encoding="utf-8")

    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _LOG_FILE_HANDLE)
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _LOG_FILE_HANDLE)

    def _close_log_file() -> None:
        global _LOG_FILE_HANDLE
        if _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        if _ORIGINAL_STDERR is not None:
            sys.stderr = _ORIGINAL_STDERR
        if _LOG_FILE_HANDLE and not _LOG_FILE_HANDLE.closed:
            try:
                _LOG_FILE_HANDLE.flush()
            except ValueError:
                pass
            _LOG_FILE_HANDLE.close()
            _LOG_FILE_HANDLE = None

    atexit.register(_close_log_file)
    return log_path

# =========================
# 枚举与模型
# =========================
ElementType = Literal["Class", "ObjectProperty", "DatatypeProperty"]
EffectiveAction = Literal["INSERT", "SKIP", "PENDING"]
ChangeType = Literal["ADD_CLASS", "ADD_OBJECT_PROPERTY", "ADD_DATATYPE_PROPERTY"]


class ReviewValidationIssue(BaseModel):
    """S3-02 业务校验问题对象（原样透传到 S3-03）。"""

    model_config = ConfigDict(extra="forbid")
    code: str
    field: str
    level: Literal["error", "warning"]
    message: str


class ValidatedDecision(BaseModel):
    """S3-03 输入记录：来源于 S3-02 的 validated 决策。"""

    model_config = ConfigDict(extra="forbid")
    review_id: str
    cluster_id: str
    element_type: ElementType
    concept_name: str
    agent3_confidence_level: str = ""
    expert_action: str
    is_valid: bool
    issues: List[ReviewValidationIssue] = Field(default_factory=list)
    effective_action: EffectiveAction
    effective_name_zh: Optional[str] = None
    effective_name_en: Optional[str] = None
    effective_comment_zh: Optional[str] = None
    effective_parent_class: Optional[str] = None
    effective_domain: Optional[str] = None
    effective_range: Optional[str] = None
    effective_xsd_range: Optional[str] = None
    effective_super_property: Optional[str] = None
    reviewer: str = ""
    review_time: str = ""
    expert_comment: str = ""
    source_hash: str


class TripleItem(BaseModel):
    """变更集中最小三元组单元。"""

    model_config = ConfigDict(extra="forbid")
    s: str
    p: str
    o: str
    o_type: Literal["iri", "literal"]
    datatype: Optional[str] = None


class DeltaChange(BaseModel):
    """单条增量变更对象（S3-03 输出、S3-04 输入）。"""

    model_config = ConfigDict(extra="forbid")
    change_id: str
    review_id: str
    cluster_id: str
    element_type: ElementType
    change_type: ChangeType
    subject_iri: str
    triples: List[TripleItem]
    provenance_reviewer: str = ""
    provenance_time: str = ""
    provenance_comment: str = ""
    source_hash: str


class DeltaBuildError(BaseModel):
    """S3-03 构建阶段的错误记录。"""

    model_config = ConfigDict(extra="forbid")
    phase: Literal["build_delta_changeset"]
    index: int
    review_id: Optional[str] = None
    error_type: str
    error: str


# =========================
# 工具函数
# =========================
def write_json(path: Path, data: Any) -> None:
    """写 UTF-8 JSON 文件，并自动创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """写 UTF-8 文本文件，并自动创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    """读取 JSON 数组文件；若不是数组则抛异常。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Input must be JSON array: {path}")
    return [x for x in payload if isinstance(x, dict)]


def read_text(path: Path) -> str:
    """读取 UTF-8 文本文件。"""

    return path.read_text(encoding="utf-8")


def clean_optional(text: Optional[str]) -> Optional[str]:
    """把空字符串归一为 None，便于后续统一校验。"""

    if text is None:
        return None
    t = str(text).strip()
    return t if t else None


def extract_ttl_class_anchors(ttl_text: str) -> set[str]:
    """从 TTL 文本中提取已存在的 owl:Class 本地名集合（不含冒号）。"""

    pattern = re.compile(r"^\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+rdf:type\s+owl:Class\b", re.MULTILINE)
    return {m.group("name") for m in pattern.finditer(ttl_text)}


def extract_default_prefix_iri(ttl_text: str) -> str:
    """提取 TTL 默认前缀(:)的 IRI；若缺失则回退到默认值。"""

    m = re.search(r"^\s*@prefix\s+:\s*<([^>]+)>\s*\.\s*$", ttl_text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return "http://www.semanticweb.org/traffickg#"


def normalize_anchor_name(raw: Optional[str]) -> str:
    """把 :ClassName / <IRI> / 全 IRI 归一为本地名。"""

    if raw is None:
        return ""
    text = str(raw).strip().strip("`'\"")
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.lstrip(":").strip()


def local_name_exists_in_ttl(local_name: str, ttl_text: str) -> bool:
    """判断本地名是否已在基座 TTL 中出现，避免重复新增实体。"""

    if not local_name:
        return False
    escaped = re.escape(local_name)
    return bool(re.search(rf"(^|\s):{escaped}(\s|;|,|\.)", ttl_text))


def looks_like_class_iri_local_name(value: Optional[str]) -> bool:
    """校验 Class 的 name_en 是否可作为 IRI 本地名（PascalCase + 字母数字）。"""

    if value is None:
        return False
    t = value.strip()
    if not t:
        return False
    return bool(re.match(r"^[A-Z][A-Za-z0-9]*$", t))


def looks_like_property_iri_local_name(value: Optional[str]) -> bool:
    """校验属性的 name_en 是否可作为 IRI 本地名（lowerCamelCase + 字母数字）。"""

    if value is None:
        return False
    t = value.strip()
    if not t:
        return False
    return bool(re.match(r"^[a-z][A-Za-z0-9]*$", t))


def build_subject_local_name(item: ValidatedDecision) -> str:
    """构建新增主体本地名：优先合法 name_en，否则回退到稳定兜底名。"""

    if item.element_type == "Class" and looks_like_class_iri_local_name(item.effective_name_en):
        return str(item.effective_name_en).strip()
    if item.element_type in {"ObjectProperty", "DatatypeProperty"} and looks_like_property_iri_local_name(item.effective_name_en):
        return str(item.effective_name_en).strip()
    prefix = {"Class": "C", "ObjectProperty": "OP", "DatatypeProperty": "DP"}[item.element_type]
    safe_cluster = re.sub(r"[^A-Za-z0-9_]", "_", item.cluster_id)
    return f"S3_{prefix}_{safe_cluster}"


def to_ttl_object(triple: TripleItem) -> str:
    """将三元组对象序列化为 TTL 可写形式。"""

    if triple.o_type == "iri":
        return triple.o
    escaped = triple.o.replace('"', '\\"')
    if triple.datatype == "@zh":
        return f'"{escaped}"@zh'
    if triple.datatype:
        return f'"{escaped}"^^{triple.datatype}'
    return f'"{escaped}"'


def build_change(item: ValidatedDecision, ttl_text: str, class_anchors: set[str], index: int) -> DeltaChange:
    """将一条 INSERT 决策转换成 DeltaChange，并做最小安全校验。"""

    subject_local = build_subject_local_name(item)
    subject_iri = f":{subject_local}"

    label_zh = (item.effective_name_zh or item.concept_name or "").strip()
    if not label_zh:
        raise ValueError("Missing effective_name_zh for rdfs:label@zh")
    # comment 优先使用有效注释；为空则回退为中文标签
    comment_zh = clean_optional(item.effective_comment_zh) or label_zh

    if local_name_exists_in_ttl(subject_local, ttl_text):
        raise ValueError(f"Subject already exists in base TTL: {subject_iri}")

    triples: List[TripleItem] = []
    if item.element_type == "Class":
        parent = clean_optional(item.effective_parent_class)
        if not parent:
            raise ValueError("Class INSERT requires effective_parent_class")
        parent_local = normalize_anchor_name(parent)
        if parent_local not in class_anchors:
            raise ValueError(f"Class parent anchor not found in TTL: {parent}")

        triples.extend(
            [
                TripleItem(s=subject_iri, p="rdf:type", o="owl:Class", o_type="iri"),
                TripleItem(s=subject_iri, p="rdfs:subClassOf", o=f":{parent_local}", o_type="iri"),
                TripleItem(s=subject_iri, p="rdfs:label", o=label_zh, o_type="literal", datatype="@zh"),
                TripleItem(s=subject_iri, p="rdfs:comment", o=comment_zh, o_type="literal", datatype="@zh"),
            ]
        )
        change_type: ChangeType = "ADD_CLASS"

    elif item.element_type == "ObjectProperty":
        domain = clean_optional(item.effective_domain)
        rng = clean_optional(item.effective_range)
        super_property = clean_optional(item.effective_super_property)
        if not domain or not rng:
            raise ValueError("ObjectProperty INSERT requires effective_domain and effective_range")
        domain_local = normalize_anchor_name(domain)
        range_local = normalize_anchor_name(rng)
        if domain_local not in class_anchors:
            raise ValueError(f"ObjectProperty domain anchor not found in TTL: {domain}")
        if range_local not in class_anchors:
            raise ValueError(f"ObjectProperty range anchor not found in TTL: {rng}")

        triples.extend(
            [
                TripleItem(s=subject_iri, p="rdf:type", o="owl:ObjectProperty", o_type="iri"),
                *(
                    [TripleItem(s=subject_iri, p="rdfs:subPropertyOf", o=f":{normalize_anchor_name(super_property)}", o_type="iri")]
                    if super_property
                    else []
                ),
                TripleItem(s=subject_iri, p="rdfs:domain", o=f":{domain_local}", o_type="iri"),
                TripleItem(s=subject_iri, p="rdfs:range", o=f":{range_local}", o_type="iri"),
                TripleItem(s=subject_iri, p="rdfs:label", o=label_zh, o_type="literal", datatype="@zh"),
                TripleItem(s=subject_iri, p="rdfs:comment", o=comment_zh, o_type="literal", datatype="@zh"),
            ]
        )
        change_type = "ADD_OBJECT_PROPERTY"

    else:
        domain = clean_optional(item.effective_domain)
        xsd_range = clean_optional(item.effective_xsd_range)
        if not domain or not xsd_range:
            raise ValueError("DatatypeProperty INSERT requires effective_domain and effective_xsd_range")
        domain_local = normalize_anchor_name(domain)
        if domain_local not in class_anchors:
            raise ValueError(f"DatatypeProperty domain anchor not found in TTL: {domain}")

        triples.extend(
            [
                TripleItem(s=subject_iri, p="rdf:type", o="owl:DatatypeProperty", o_type="iri"),
                TripleItem(s=subject_iri, p="rdfs:domain", o=f":{domain_local}", o_type="iri"),
                TripleItem(s=subject_iri, p="rdfs:range", o=xsd_range, o_type="iri"),
                TripleItem(s=subject_iri, p="rdfs:label", o=label_zh, o_type="literal", datatype="@zh"),
                TripleItem(s=subject_iri, p="rdfs:comment", o=comment_zh, o_type="literal", datatype="@zh"),
            ]
        )
        change_type = "ADD_DATATYPE_PROPERTY"

    return DeltaChange(
        change_id=f"S3D_{index:06d}",
        review_id=item.review_id,
        cluster_id=item.cluster_id,
        element_type=item.element_type,
        change_type=change_type,
        subject_iri=subject_iri,
        triples=triples,
        provenance_reviewer=item.reviewer,
        provenance_time=item.review_time,
        provenance_comment=item.expert_comment,
        source_hash=item.source_hash,
    )


def build_preview_ttl(changes: List[DeltaChange], default_prefix_iri: str) -> str:
    """把变更集渲染成预览 TTL（按一行一三元组，便于快速核对）。"""

    lines: List[str] = [
        f"@prefix : <{default_prefix_iri}> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "# Auto-generated by S3-03 (preview only)",
        f"# generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    # 去重输出，避免同一三元组重复写入
    seen: set[str] = set()
    for ch in changes:
        lines.append(f"# {ch.change_id} | {ch.change_type} | {ch.review_id}")
        for t in ch.triples:
            stmt = f"{t.s} {t.p} {to_ttl_object(t)} ."
            if stmt in seen:
                continue
            seen.add(stmt)
            lines.append(stmt)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_changeset_summary(
    changes: List[DeltaChange],
    errors: List[DeltaBuildError],
    insert_candidate_total: int,
) -> Dict[str, Any]:
    change_type_counts: Dict[str, int] = {}
    error_code_counts: Dict[str, int] = {}

    for change in changes:
        change_type_counts[change.change_type] = change_type_counts.get(change.change_type, 0) + 1

    for err in errors:
        error_code_counts[err.error_type] = error_code_counts.get(err.error_type, 0) + 1

    ratio = round((len(changes) / insert_candidate_total), 6) if insert_candidate_total > 0 else 0.0
    return {
        "change_type_counts": change_type_counts,
        "error_code_counts": error_code_counts,
        "insert_candidate_total": insert_candidate_total,
        "changeset_total": len(changes),
        "error_total": len(errors),
        "insert_candidate_to_changeset_ratio": ratio,
    }


def main() -> None:
    """S3-03 主流程：validated -> changeset + preview ttl + 错误与摘要。"""

    started = time.time()
    log_path = setup_console_log(f"phase3_03_build_delta_changeset_{RUN_MODE}")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"[运行启动] S3-03 构建本体增量变更集 | mode={RUN_MODE}")
    print(f"[输入信息] validated输入={INPUT_VALIDATED}")
    print(f"[输入信息] 本体TTL={INPUT_TTL}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print(f"[参数配置] start_index={START_INDEX} | max_items={MAX_ITEMS}")
    print("=" * 88)

    if not INPUT_VALIDATED.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_VALIDATED}")
    if not INPUT_TTL.exists():
        raise FileNotFoundError(f"Missing base ttl file: {INPUT_TTL}")

    raw_all = load_json_array(INPUT_VALIDATED)
    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    raw_selected = raw_all[START_INDEX:end_index]

    ttl_text = read_text(INPUT_TTL)
    default_prefix_iri = extract_default_prefix_iri(ttl_text)
    class_anchors = extract_ttl_class_anchors(ttl_text)

    errors: List[DeltaBuildError] = []
    changes: List[DeltaChange] = []
    selected_insert = 0

    for i, raw in enumerate(raw_selected, start=1):
        global_index = START_INDEX + i - 1
        print(f"[开始处理] [{i}/{len(raw_selected)}] index={global_index}")

        item: Optional[ValidatedDecision] = None
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                item = ValidatedDecision.model_validate(raw)
                break
            except ValidationError as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    sleep_sec = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    print(f"  - [Schema校验失败] {sleep_sec:.1f}s 后重试")
                    time.sleep(sleep_sec)
            except Exception as e:  # noqa: BLE001
                last_error = f"{type(e).__name__}: {e}"
                if attempt < MAX_RETRIES:
                    sleep_sec = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    print(f"  - [异常错误] {sleep_sec:.1f}s 后重试")
                    time.sleep(sleep_sec)

        if item is None:
            errors.append(
                DeltaBuildError(
                    phase="build_delta_changeset",
                    index=global_index,
                    review_id=raw.get("review_id") if isinstance(raw, dict) else None,
                    error_type="schema_validation_error",
                    error=last_error,
                )
            )
            continue

        # 只对 INSERT 且 is_valid=True 的记录生成变更
        if item.effective_action != "INSERT":
            continue
        if not item.is_valid:
            errors.append(
                DeltaBuildError(
                    phase="build_delta_changeset",
                    index=global_index,
                    review_id=item.review_id,
                    error_type="invalid_record",
                    error="effective_action=INSERT but is_valid=false",
                )
            )
            continue

        selected_insert += 1
        try:
            change = build_change(item, ttl_text, class_anchors, len(changes) + 1)
            changes.append(change)
        except Exception as e:  # noqa: BLE001
            errors.append(
                DeltaBuildError(
                    phase="build_delta_changeset",
                    index=global_index,
                    review_id=item.review_id,
                    error_type="build_change_failed",
                    error=f"{type(e).__name__}: {e}",
                )
            )

    write_json(OUTPUT_CHANGESET, [c.model_dump() for c in changes])
    write_text(OUTPUT_PREVIEW_TTL, build_preview_ttl(changes, default_prefix_iri))
    write_json(OUTPUT_ERRORS, [e.model_dump() for e in errors])
    summary = build_changeset_summary(changes, errors, selected_insert)
    write_json(OUTPUT_SUMMARY, summary)

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_mode": RUN_MODE,
        "input_validated": str(INPUT_VALIDATED),
        "input_base_ttl": str(INPUT_TTL),
        "output_changeset": str(OUTPUT_CHANGESET),
        "output_preview_ttl": str(OUTPUT_PREVIEW_TTL),
        "output_errors": str(OUTPUT_ERRORS),
        "summary_file": str(OUTPUT_SUMMARY),
        "run_range": {
            "start_index": START_INDEX,
            "max_items": MAX_ITEMS,
            "selected_total": len(raw_selected),
        },
        "counts": {
            "input_total": len(raw_all),
            "insert_candidate_total": selected_insert,
            "changeset_total": len(changes),
            "error_total": len(errors),
            "class_anchor_total": len(class_anchors),
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(OUTPUT_MANIFEST, manifest)

    print("\n" + "=" * 88)
    print("[运行结束] S3-03 本体增量变更集构建完成")
    print(f"[运行摘要] INSERT候选={selected_insert} | 成功生成={len(changes)} | 错误条数={len(errors)}")
    print(f"[性能统计] 总耗时={manifest['elapsed_sec']}s")
    print(f"[关键输出] {OUTPUT_CHANGESET} -> 增量变更集（供 S3-04 读取）")
    print(f"[关键输出] {OUTPUT_PREVIEW_TTL} -> 预览 TTL（便于人工核查）")
    print(f"[关键输出] {OUTPUT_ERRORS} -> 构建错误明细")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要")
    print("=" * 88)


if __name__ == "__main__":
    main()
