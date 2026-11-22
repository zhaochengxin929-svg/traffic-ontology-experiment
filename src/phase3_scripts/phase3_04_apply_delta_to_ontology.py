import atexit
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, ValidationError

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
# S3-04 输出目录：本步骤所有产物统一写入此目录
OUTPUT_DIR = BASE_DIR / "outputs_phase3_04"
if RUN_MODE == "class":
    # Class round 输入：class changeset
    INPUT_CHANGESET = BASE_DIR / "outputs_phase3_03" / "class_changeset.json"
    # Class round 基座：原始本体 TTL
    INPUT_TTL = PROJECT_ROOT / "shared_resources" / f"TrafficKG_{INPUT_VERSION}.ttl"
    # 输出1：类概念合并后的预览本体
    OUTPUT_MERGED_TTL = OUTPUT_DIR / f"TrafficKG_{OUTPUT_VERSION}_class_merged_preview.ttl"
else:
    # Property round 输入：property changeset
    INPUT_CHANGESET = BASE_DIR / "outputs_phase3_03" / "property_changeset.json"
    # Property round 基座：class round 合并预览 TTL
    INPUT_TTL = OUTPUT_DIR / f"TrafficKG_{OUTPUT_VERSION}_class_merged_preview.ttl"
    # 输出1：最终合并预览本体
    OUTPUT_MERGED_TTL = OUTPUT_DIR / f"TrafficKG_{OUTPUT_VERSION}_merged_preview.ttl"
# 输出2：成功应用记录（含落位层级与策略）
OUTPUT_APPLIED = OUTPUT_DIR / f"{RUN_MODE}_applied_changes.json"
# 输出3：跳过与冲突明细
OUTPUT_SKIPPED = OUTPUT_DIR / f"{RUN_MODE}_skipped_or_conflicts.json"
# 输出4：运行摘要（参数、计数、耗时）
OUTPUT_MANIFEST = OUTPUT_DIR / f"{RUN_MODE}_merge_manifest.json"
# 输出5：逐条审计日志（jsonl）
OUTPUT_AUDIT_LOG = OUTPUT_DIR / f"{RUN_MODE}_merge_audit_log.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / f"{RUN_MODE}_merge_summary.json"
# 是否将控制台输出同步写入日志文件
ENABLE_FILE_LOG = True
# 统一日志目录：保存本脚本控制台输出
LOG_DIR = PROJECT_ROOT / "logs"

# 安全开关：False=只输出预览，不覆盖主本体；True=写回 INPUT_TTL（会先备份）
OVERWRITE_BASE = False

# 小样本参数：起始索引（默认从第 0 条）
START_INDEX = 0
# 小样本参数：最大处理条数，0 表示处理全部
MAX_ITEMS = 0
# schema 校验失败后的最大重试次数
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
LayerName = Literal["L1", "L2", "L3", "L4", "UNKNOWN"]
PlacementStrategy = Literal["by_parent", "by_domain", "fallback_layer_end", "fallback_tail"]
ElementType = Literal["Class", "ObjectProperty", "DatatypeProperty"]
ChangeType = Literal["ADD_CLASS", "ADD_OBJECT_PROPERTY", "ADD_DATATYPE_PROPERTY"]


class TripleItem(BaseModel):
    """变更集中的最小三元组对象。"""

    model_config = ConfigDict(extra="forbid")
    s: str
    p: str
    o: str
    o_type: Literal["iri", "literal"]
    datatype: Optional[str] = None


class DeltaChange(BaseModel):
    """S3-04 输入变更对象（来源于 S3-03 changeset）。"""

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


class ApplyRecord(BaseModel):
    """成功应用记录，用于输出 applied_changes.json。"""

    model_config = ConfigDict(extra="forbid")
    change_id: str
    review_id: str
    subject_iri: str
    change_type: ChangeType
    triple_count: int
    resolved_layer: LayerName
    placement_strategy: PlacementStrategy
    placement_warning: str = ""
    applied_at: str


class SkippedOrConflictRecord(BaseModel):
    """跳过/冲突/结构错误记录，用于输出 skipped_or_conflicts.json。"""

    model_config = ConfigDict(extra="forbid")
    change_id: Optional[str] = None
    review_id: Optional[str] = None
    subject_iri: Optional[str] = None
    status: Literal["schema_error", "skipped", "conflict"]
    reason: str


class AuditLogRecord(BaseModel):
    """逐条审计日志记录（jsonl）。"""

    model_config = ConfigDict(extra="forbid")
    timestamp: str
    index: int
    change_id: Optional[str] = None
    review_id: Optional[str] = None
    status: Literal["schema_error", "skipped", "conflict", "applied"]
    detail: str


class PreparedAppliedItem(BaseModel):
    """准备写入阶段对象：包含落位索引和落位策略。"""

    model_config = ConfigDict(extra="forbid")
    change: DeltaChange
    insert_index: int
    resolved_layer: LayerName
    placement_strategy: PlacementStrategy
    placement_warning: str = ""


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


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """将多条记录追加写入 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    """读取 JSON 数组文件（兼容 UTF-8 BOM）。"""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Input must be JSON array: {path}")
    return [x for x in payload if isinstance(x, dict)]


def read_text(path: Path) -> str:
    """读取 UTF-8 文本文件。"""

    return path.read_text(encoding="utf-8")


def to_ttl_object(triple: TripleItem) -> str:
    """把 TripleItem 的 object 部分渲染为 TTL 语法对象。"""

    if triple.o_type == "iri":
        return triple.o
    escaped = triple.o.replace('"', '\\"')
    if triple.datatype == "@zh":
        return f'"{escaped}"@zh'
    if triple.datatype:
        return f'"{escaped}"^^{triple.datatype}'
    return f'"{escaped}"'


def triple_signature(triple: TripleItem) -> str:
    """生成三元组去重签名（s/p/o 序列化）。"""

    return f"{triple.s} {triple.p} {to_ttl_object(triple)}"


def local_name_from_prefixed(subject_iri: str) -> str:
    """从 :LocalName 提取本地名 LocalName。"""

    return subject_iri.lstrip(":").strip()


def normalize_anchor_name(raw: Optional[str]) -> str:
    """将 :Class / <IRI> / 全 IRI 统一归一为本地名。"""

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


def extract_ttl_class_anchors(ttl_text: str) -> set[str]:
    """提取基座 TTL 中已有的 Class 本地名集合。"""

    pattern = re.compile(r"^\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+rdf:type\s+owl:Class\b", re.MULTILINE)
    return {m.group("name") for m in pattern.finditer(ttl_text)}


def extract_ttl_object_property_anchors(ttl_text: str) -> set[str]:
    pattern = re.compile(r"^\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+rdf:type\s+owl:ObjectProperty\b", re.MULTILINE)
    return {m.group("name") for m in pattern.finditer(ttl_text)}


def extract_existing_local_names(ttl_text: str) -> set[str]:
    """提取 TTL 中已出现过的主体本地名，用于新增冲突检查。"""

    pattern = re.compile(
        r"(^|\s):([A-Za-z_][A-Za-z0-9_]*)\s+"
        r"(rdf:type|rdfs:subClassOf|rdfs:subPropertyOf|rdfs:domain|rdfs:range|rdfs:label|rdfs:comment)\b"
    )
    return {m.group(2) for m in pattern.finditer(ttl_text)}


def extract_default_prefix_iri(ttl_text: str) -> str:
    """提取默认前缀(:)的 IRI，供 manifest 记录。"""

    m = re.search(r"^\s*@prefix\s+:\s*<([^>]+)>\s*\.\s*$", ttl_text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return "http://www.semanticweb.org/traffickg#"


def extract_layer_ranges(lines: List[str]) -> Dict[str, Tuple[int, int]]:
    """解析 L1~L4 在文件中的行区间 [start, end)。"""

    layer_markers: List[Tuple[str, int]] = []
    pattern = re.compile(r"^\s*#\s*L([1-4])\s*:")
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            layer_markers.append((f"L{m.group(1)}", i))

    layer_ranges: Dict[str, Tuple[int, int]] = {}
    for idx, (layer, start) in enumerate(layer_markers):
        end = layer_markers[idx + 1][1] if idx + 1 < len(layer_markers) else len(lines)
        layer_ranges[layer] = (start, end)
    return layer_ranges


def extract_entity_layer_map(lines: List[str]) -> Dict[str, str]:
    """建立实体本地名 -> 层级(L1~L4)映射，用于后续自动落位。"""

    current_layer: Optional[str] = None
    layer_pattern = re.compile(r"^\s*#\s*L([1-4])\s*:")
    entity_pattern = re.compile(
        r"^\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+rdf:type\s+"
        r"(owl:Class|owl:ObjectProperty|owl:DatatypeProperty)\b"
    )

    mapping: Dict[str, str] = {}
    for line in lines:
        lm = layer_pattern.match(line)
        if lm:
            current_layer = f"L{lm.group(1)}"
            continue

        em = entity_pattern.match(line)
        if em and current_layer:
            mapping[em.group("name")] = current_layer

    return mapping


def ordered_triples(change: DeltaChange) -> List[TripleItem]:
    """按约定谓词顺序重排三元组，提升可读性并保证输出稳定。"""

    order_map: Dict[ChangeType, List[str]] = {
        "ADD_CLASS": ["rdf:type", "rdfs:subClassOf", "rdfs:label", "rdfs:comment"],
        "ADD_OBJECT_PROPERTY": ["rdf:type", "rdfs:subPropertyOf", "rdfs:domain", "rdfs:range", "rdfs:label", "rdfs:comment"],
        "ADD_DATATYPE_PROPERTY": ["rdf:type", "rdfs:domain", "rdfs:range", "rdfs:label", "rdfs:comment"],
    }

    wanted = order_map[change.change_type]
    by_pred: Dict[str, TripleItem] = {}
    extras: List[TripleItem] = []

    for t in change.triples:
        if t.p in by_pred:
            continue
        if t.p in wanted:
            by_pred[t.p] = t
        else:
            extras.append(t)

    out = [by_pred[p] for p in wanted if p in by_pred]
    out.extend(extras)
    return out


def render_change_block(change: DeltaChange) -> str:
    """把单条变更渲染为“分号换行”风格 TTL 块。"""

    triples = ordered_triples(change)
    if not triples:
        return ""

    first = triples[0]
    lines: List[str] = [f"{change.subject_iri} {first.p} {to_ttl_object(first)} ;"]

    for t in triples[1:-1]:
        lines.append(f"    {t.p} {to_ttl_object(t)} ;")

    if len(triples) > 1:
        last = triples[-1]
        lines.append(f"    {last.p} {to_ttl_object(last)} .")
    else:
        lines[0] = lines[0][:-1] + "."

    return "\n".join(lines)


def validate_change_against_anchors(
    change: DeltaChange,
    class_anchors: set[str],
    object_property_anchors: set[str],
) -> Optional[str]:
    """校验变更引用锚点是否存在；返回 None 表示通过。"""

    pred_to_obj = {t.p: t.o for t in change.triples}

    if change.change_type == "ADD_CLASS":
        parent = pred_to_obj.get("rdfs:subClassOf")
        if not parent:
            return "missing rdfs:subClassOf for ADD_CLASS"
        if normalize_anchor_name(parent) not in class_anchors:
            return f"parent class anchor not found: {parent}"

    elif change.change_type == "ADD_OBJECT_PROPERTY":
        super_property = pred_to_obj.get("rdfs:subPropertyOf")
        domain = pred_to_obj.get("rdfs:domain")
        rng = pred_to_obj.get("rdfs:range")
        if not domain or not rng:
            return "missing rdfs:domain/rdfs:range for ADD_OBJECT_PROPERTY"
        if super_property and normalize_anchor_name(super_property) not in object_property_anchors:
            return f"super property anchor not found: {super_property}"
        if normalize_anchor_name(domain) not in class_anchors:
            return f"domain anchor not found: {domain}"
        if normalize_anchor_name(rng) not in class_anchors:
            return f"range anchor not found: {rng}"

    else:
        domain = pred_to_obj.get("rdfs:domain")
        rng = pred_to_obj.get("rdfs:range")
        if not domain or not rng:
            return "missing rdfs:domain/rdfs:range for ADD_DATATYPE_PROPERTY"
        if normalize_anchor_name(domain) not in class_anchors:
            return f"domain anchor not found: {domain}"

    return None


def resolve_layer_by_rule(change: DeltaChange, entity_layer_map: Dict[str, str]) -> Tuple[LayerName, PlacementStrategy, str]:
    """按规则解析目标层级：Class 走父类，Property 走 domain。"""

    pred_to_obj = {t.p: t.o for t in change.triples}

    if change.change_type == "ADD_CLASS":
        parent_local = normalize_anchor_name(pred_to_obj.get("rdfs:subClassOf"))
        if parent_local in entity_layer_map:
            return entity_layer_map[parent_local], "by_parent", ""
        return "UNKNOWN", "fallback_tail", f"cannot resolve parent layer: {pred_to_obj.get('rdfs:subClassOf', '')}"

    domain_local = normalize_anchor_name(pred_to_obj.get("rdfs:domain"))
    if domain_local in entity_layer_map:
        return entity_layer_map[domain_local], "by_domain", ""
    return "UNKNOWN", "fallback_tail", f"cannot resolve domain layer: {pred_to_obj.get('rdfs:domain', '')}"


def find_insert_index_in_layer(lines: List[str], layer_start: int, layer_end: int, change_type: ChangeType) -> Tuple[int, str]:
    """在目标层中找到插入位置：优先对应子区块尾部，找不到则回退层末。"""

    class_start: Optional[int] = None
    obj_start: Optional[int] = None
    data_start: Optional[int] = None

    for i in range(layer_start, layer_end):
        text = lines[i].strip()
        if not text.startswith("#"):
            continue
        if "ObjectProperty" in text:
            obj_start = i if obj_start is None else obj_start
        elif "DatatypeProperty" in text:
            data_start = i if data_start is None else data_start
        elif "Class" in text:
            class_start = i if class_start is None else class_start

    if change_type == "ADD_CLASS":
        target_start = class_start
    elif change_type == "ADD_OBJECT_PROPERTY":
        target_start = obj_start
    else:
        target_start = data_start

    if target_start is None:
        return layer_end, "target section marker not found in layer"

    candidates = [x for x in [class_start, obj_start, data_start] if x is not None and x > target_start]
    section_end = min(candidates) if candidates else layer_end

    insert_idx = section_end
    while insert_idx > target_start + 1 and lines[insert_idx - 1].strip() == "":
        insert_idx -= 1

    return insert_idx, ""


def build_merged_ttl(base_ttl_text: str, prepared_items: List[PreparedAppliedItem]) -> str:
    """将所有已解析落位的变更插入原 TTL，输出合并文本。"""

    if not prepared_items:
        return base_ttl_text if base_ttl_text.endswith("\n") else (base_ttl_text + "\n")

    lines = base_ttl_text.splitlines()
    inserts: Dict[int, List[str]] = {}

    for item in prepared_items:
        block = render_change_block(item.change)
        if not block:
            continue
        payload = [""] + block.splitlines() + [""]
        inserts.setdefault(item.insert_index, []).extend(payload)

    offset = 0
    for idx in sorted(inserts.keys()):
        payload = inserts[idx]
        lines[idx + offset: idx + offset] = payload
        offset += len(payload)

    return "\n".join(lines).rstrip() + "\n"


def build_merge_summary(
    applied_records: List[ApplyRecord],
    skipped_records: List[SkippedOrConflictRecord],
) -> Dict[str, Any]:
    resolved_layer_counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "UNKNOWN": 0}
    placement_strategy_counts: Dict[str, int] = {}
    skipped_total = 0
    conflict_total = 0

    for rec in applied_records:
        resolved_layer_counts[rec.resolved_layer] = resolved_layer_counts.get(rec.resolved_layer, 0) + 1
        placement_strategy_counts[rec.placement_strategy] = placement_strategy_counts.get(rec.placement_strategy, 0) + 1

    for rec in skipped_records:
        if rec.status == "skipped":
            skipped_total += 1
        elif rec.status == "conflict":
            conflict_total += 1

    return {
        "applied_total": len(applied_records),
        "skipped_total": skipped_total,
        "conflict_total": conflict_total,
        "resolved_layer_counts": resolved_layer_counts,
        "placement_strategy_counts": placement_strategy_counts,
    }


def main() -> None:
    """S3-04 主流程：读取 changeset -> 校验 -> 分层落位 -> 输出合并与审计文件。"""

    started = time.time()
    log_path = setup_console_log(f"phase3_04_apply_delta_to_ontology_{RUN_MODE}")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_AUDIT_LOG.exists():
        OUTPUT_AUDIT_LOG.unlink()

    print("=" * 88)
    print(f"[运行启动] S3-04 按层级合并增量到本体 | mode={RUN_MODE}")
    print(f"[输入信息] changeset输入={INPUT_CHANGESET}")
    print(f"[输入信息] 本体TTL={INPUT_TTL}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print(f"[参数配置] overwrite_base={OVERWRITE_BASE} | start_index={START_INDEX} | max_items={MAX_ITEMS}")
    print("=" * 88)

    if not INPUT_CHANGESET.exists():
        raise FileNotFoundError(f"Missing input changeset: {INPUT_CHANGESET}")
    if not INPUT_TTL.exists():
        raise FileNotFoundError(f"Missing base ttl: {INPUT_TTL}")

    raw_all = load_json_array(INPUT_CHANGESET)
    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    raw_selected = raw_all[START_INDEX:end_index]

    base_ttl_text = read_text(INPUT_TTL)
    base_lines = base_ttl_text.splitlines()
    base_prefix_iri = extract_default_prefix_iri(base_ttl_text)

    layer_ranges = extract_layer_ranges(base_lines)
    entity_layer_map = extract_entity_layer_map(base_lines)
    existing_local_names = extract_existing_local_names(base_ttl_text)
    class_anchors = extract_ttl_class_anchors(base_ttl_text)
    object_property_anchors = extract_ttl_object_property_anchors(base_ttl_text)

    prepared_items: List[PreparedAppliedItem] = []
    applied_records: List[ApplyRecord] = []
    skipped_records: List[SkippedOrConflictRecord] = []

    for i, raw in enumerate(raw_selected, start=1):
        global_index = START_INDEX + i - 1
        print(f"[开始处理] [{i}/{len(raw_selected)}] index={global_index}")

        row_logs: List[AuditLogRecord] = []
        change: Optional[DeltaChange] = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                change = DeltaChange.model_validate(raw)
                break
            except ValidationError as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    sleep_sec = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    print(f"  - [Schema校验失败] {sleep_sec:.1f}s 后重试")
                    time.sleep(sleep_sec)

        if change is None:
            skipped_records.append(
                SkippedOrConflictRecord(
                    change_id=raw.get("change_id") if isinstance(raw, dict) else None,
                    review_id=raw.get("review_id") if isinstance(raw, dict) else None,
                    subject_iri=raw.get("subject_iri") if isinstance(raw, dict) else None,
                    status="schema_error",
                    reason=last_error or "unknown schema validation error",
                )
            )
            row_logs.append(
                AuditLogRecord(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    index=global_index,
                    change_id=raw.get("change_id") if isinstance(raw, dict) else None,
                    review_id=raw.get("review_id") if isinstance(raw, dict) else None,
                    status="schema_error",
                    detail=last_error or "unknown schema validation error",
                )
            )
            append_jsonl(OUTPUT_AUDIT_LOG, [x.model_dump() for x in row_logs])
            continue

        subject_local = local_name_from_prefixed(change.subject_iri)
        if subject_local in existing_local_names:
            msg = f"subject already exists in base ttl: {change.subject_iri}"
            skipped_records.append(
                SkippedOrConflictRecord(
                    change_id=change.change_id,
                    review_id=change.review_id,
                    subject_iri=change.subject_iri,
                    status="conflict",
                    reason=msg,
                )
            )
            row_logs.append(
                AuditLogRecord(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    index=global_index,
                    change_id=change.change_id,
                    review_id=change.review_id,
                    status="conflict",
                    detail=msg,
                )
            )
            append_jsonl(OUTPUT_AUDIT_LOG, [x.model_dump() for x in row_logs])
            continue

        anchor_error = validate_change_against_anchors(change, class_anchors, object_property_anchors)
        if anchor_error:
            skipped_records.append(
                SkippedOrConflictRecord(
                    change_id=change.change_id,
                    review_id=change.review_id,
                    subject_iri=change.subject_iri,
                    status="skipped",
                    reason=anchor_error,
                )
            )
            row_logs.append(
                AuditLogRecord(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    index=global_index,
                    change_id=change.change_id,
                    review_id=change.review_id,
                    status="skipped",
                    detail=anchor_error,
                )
            )
            append_jsonl(OUTPUT_AUDIT_LOG, [x.model_dump() for x in row_logs])
            continue

        # 单变更内去重
        seen_signatures: set[str] = set()
        dedup_triples: List[TripleItem] = []
        for t in change.triples:
            sig = triple_signature(t)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            dedup_triples.append(t)
        change = change.model_copy(update={"triples": dedup_triples})

        resolved_layer, strategy, layer_warn = resolve_layer_by_rule(change, entity_layer_map)

        placement_warning = layer_warn
        insert_index = len(base_lines)
        placement_strategy: PlacementStrategy = strategy

        if resolved_layer != "UNKNOWN" and resolved_layer in layer_ranges:
            start, end = layer_ranges[resolved_layer]
            section_insert_index, section_warn = find_insert_index_in_layer(base_lines, start, end, change.change_type)
            insert_index = section_insert_index
            if section_warn:
                placement_strategy = "fallback_layer_end"
                placement_warning = (placement_warning + " | " + section_warn).strip(" |")
        else:
            placement_strategy = "fallback_tail"

        prepared_items.append(
            PreparedAppliedItem(
                change=change,
                insert_index=insert_index,
                resolved_layer=resolved_layer,
                placement_strategy=placement_strategy,
                placement_warning=placement_warning,
            )
        )

        applied_records.append(
            ApplyRecord(
                change_id=change.change_id,
                review_id=change.review_id,
                subject_iri=change.subject_iri,
                change_type=change.change_type,
                triple_count=len(change.triples),
                resolved_layer=resolved_layer,
                placement_strategy=placement_strategy,
                placement_warning=placement_warning,
                applied_at=datetime.now().isoformat(timespec="seconds"),
            )
        )

        row_logs.append(
            AuditLogRecord(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                index=global_index,
                change_id=change.change_id,
                review_id=change.review_id,
                status="applied",
                detail=(
                    f"applied with {len(change.triples)} triples; "
                    f"layer={resolved_layer}; strategy={placement_strategy}; "
                    f"warning={placement_warning or 'none'}"
                ),
            )
        )
        append_jsonl(OUTPUT_AUDIT_LOG, [x.model_dump() for x in row_logs])

        # 更新运行时映射：后续变更可引用本轮新增类
        existing_local_names.add(subject_local)
        if change.change_type == "ADD_CLASS":
            class_anchors.add(subject_local)
            if resolved_layer != "UNKNOWN":
                entity_layer_map[subject_local] = resolved_layer

    merged_ttl = build_merged_ttl(base_ttl_text, prepared_items)
    write_text(OUTPUT_MERGED_TTL, merged_ttl)
    write_json(OUTPUT_APPLIED, [x.model_dump() for x in applied_records])
    write_json(OUTPUT_SKIPPED, [x.model_dump() for x in skipped_records])
    summary = build_merge_summary(applied_records, skipped_records)
    write_json(OUTPUT_SUMMARY, summary)

    backup_path = None
    if OVERWRITE_BASE:
        backup_path = OUTPUT_DIR / f"TrafficKG_{INPUT_VERSION}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.ttl"
        write_text(backup_path, base_ttl_text)
        write_text(INPUT_TTL, merged_ttl)

    layer_counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "UNKNOWN": 0}
    strategy_counts = {
        "by_parent": 0,
        "by_domain": 0,
        "fallback_layer_end": 0,
        "fallback_tail": 0,
    }
    for rec in applied_records:
        layer_counts[rec.resolved_layer] += 1
        strategy_counts[rec.placement_strategy] += 1

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_mode": RUN_MODE,
        "input_changeset": str(INPUT_CHANGESET),
        "input_base_ttl": str(INPUT_TTL),
        "base_prefix_iri": base_prefix_iri,
        "output_merged_ttl": str(OUTPUT_MERGED_TTL),
        "output_applied": str(OUTPUT_APPLIED),
        "output_skipped": str(OUTPUT_SKIPPED),
        "output_manifest": str(OUTPUT_MANIFEST),
        "output_audit_log": str(OUTPUT_AUDIT_LOG),
        "summary_file": str(OUTPUT_SUMMARY),
        "overwrite_base": OVERWRITE_BASE,
        "backup_path": str(backup_path) if backup_path else "",
        "run_range": {
            "start_index": START_INDEX,
            "max_items": MAX_ITEMS,
            "selected_total": len(raw_selected),
        },
        "counts": {
            "changeset_total": len(raw_all),
            "applied_total": len(applied_records),
            "skipped_or_conflict_total": len(skipped_records),
            "applied_by_layer": layer_counts,
            "applied_by_strategy": strategy_counts,
        },
        "layer_ranges_detected": {k: {"start": v[0], "end": v[1]} for k, v in layer_ranges.items()},
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(OUTPUT_MANIFEST, manifest)

    print("\n" + "=" * 88)
    print("[运行结束] S3-04 本体增量合并完成")
    print(
        f"[运行摘要] selected={len(raw_selected)} | applied={len(applied_records)} | "
        f"skipped/conflict={len(skipped_records)}"
    )
    print(f"[性能统计] 总耗时={manifest['elapsed_sec']}s")
    print(f"[关键输出] {OUTPUT_MERGED_TTL} -> 分层落位后的本体预览")
    print(f"[关键输出] {OUTPUT_APPLIED} -> 应用成功列表（含 resolved_layer/placement_strategy）")
    print(f"[关键输出] {OUTPUT_SKIPPED} -> 跳过/冲突明细")
    print(f"[关键输出] {OUTPUT_AUDIT_LOG} -> 逐条审计日志")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要")
    if backup_path:
        print(f"[关键输出] {backup_path} -> 主本体备份")
    print("=" * 88)


if __name__ == "__main__":
    main()
