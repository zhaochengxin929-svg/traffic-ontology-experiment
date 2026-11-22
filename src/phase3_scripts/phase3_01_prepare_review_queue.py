import atexit
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from openpyxl import Workbook
from openpyxl.styles import PatternFill
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


# =========================
# 配置区（常改）
# =========================
# 当前阶段目录：大规模实验/phase3
BASE_DIR = Path(__file__).resolve().parent.parent
# 项目根目录：大规模实验
PROJECT_ROOT = BASE_DIR.parent

# 运行模式：class=类概念轮；property=属性轮
RUN_MODE: Literal["class", "property"] = "property"
# S3-01 输出目录：本步骤所有文件统一放这里
OUTPUT_DIR = BASE_DIR / "outputs_phase3_01"
if RUN_MODE == "class":
    # Class round 输入：phase2_04a 的类写入判定结果
    INPUT_DECISIONS = PROJECT_ROOT / "phase2" / "outputs_phase2_04a" / "agent3_class_positioning_decisions.json"
else:
    # Property round 输入：当前联调用测试版属性写入判定结果
    INPUT_DECISIONS = PROJECT_ROOT / "phase2" / "outputs_phase2_04c" / "agent3_property_positioning_decisions.json"
# 审核主表（JSON）：阶段三后续脚本的权威输入，包含候选挂载与专家回填字段
OUTPUT_REVIEW_QUEUE_JSON = OUTPUT_DIR / f"agent3_{RUN_MODE}_review_flat.json"
# 审核台表格（CSV）：给专家在 Excel 中查看与填写
OUTPUT_REVIEW_QUEUE_CSV = OUTPUT_DIR / f"agent3_{RUN_MODE}_review_flat.csv"
# 审核台表格（XLSX）：支持设置单元格底色等样式
OUTPUT_REVIEW_QUEUE_XLSX = OUTPUT_DIR / f"agent3_{RUN_MODE}_review_flat.xlsx"
# 专家回填模板（JSON）：用于程序化填写/回传审核结果
OUTPUT_REVIEWED_TEMPLATE_JSON = OUTPUT_DIR / f"agent3_{RUN_MODE}_review_template.json"
# 预处理错误明细：记录输入解析/校验失败条目
OUTPUT_ERRORS = OUTPUT_DIR / f"agent3_{RUN_MODE}_review_prep_errors.json"
# 运行清单：记录本次参数、计数统计、耗时与输入输出路径
OUTPUT_MANIFEST = OUTPUT_DIR / f"agent3_{RUN_MODE}_review_prep_manifest.json"
# 是否将控制台输出同步写入日志文件
ENABLE_FILE_LOG = True
# 统一日志目录：保存本脚本控制台输出
LOG_DIR = PROJECT_ROOT / "logs"

# 小样本/断点参数：
START_INDEX = 0
MAX_ITEMS = 0

# 重试参数：单条解析失败时指数退避
MAX_RETRIES = 2
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
# Enums
# =========================
ElementType = Literal["Class", "ObjectProperty", "DatatypeProperty"]
Agent3Decision = Literal["POSITIONED", "UNSURE", "REJECTED"]
CheckLevel = Literal["error", "warning", "info"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
ConfidenceSource = Literal["local", "fallback"]


# =========================
# Models
# =========================
class TopologyCheck(BaseModel):
    """承接 Agent3 输出的单条拓扑规则检查结果。"""
    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(min_length=1)
    passed: bool
    level: CheckLevel
    message: str = Field(min_length=1)


class EvidenceRef(BaseModel):
    """审核台中展示的证据引用。"""
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    section: str
    chunk_id: str
    evidence_text: str = ""


class Agent3DecisionRecord(BaseModel):
    """S3-01 输入记录：对应 Agent3 的单条定位结果。"""
    model_config = ConfigDict(extra="allow")
    cluster_id: str
    element_type: ElementType
    name_raw: str = ""
    name_zh: Optional[str] = None
    name_en: Optional[str] = None
    decision: Agent3Decision
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: Optional[Literal["HIGH", "LOW"]] = None
    confidence_source: ConfidenceSource
    # 候选定义文本（若 Agent3 顶层有输出）
    definition: Optional[str] = None
    merged_definitions: List[str] = Field(default_factory=list)
    # Agent3 输入快照（常含 merged_definitions，阶段三用于回填 rdfs:comment）
    input_snapshot: Dict[str, Any] = Field(default_factory=dict)
    parent_class: Optional[str] = None
    suggested_domain: Optional[str] = None
    suggested_range: Optional[str] = None
    suggested_xsd_range: Optional[str] = None
    candidate_super_property: Optional[str] = None
    reasoning: str = Field(min_length=1)
    topology_checks: List[TopologyCheck] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def fill_names(self) -> "Agent3DecisionRecord":
        self.name_raw = (self.name_raw or "").strip()
        self.name_zh = (self.name_zh or "").strip() or None
        self.name_en = (self.name_en or "").strip() or None

        if not (self.name_raw or self.name_zh or self.name_en):
            raise ValueError("Missing concept name: require one of name_raw/name_zh/name_en.")
        if not self.name_raw:
            self.name_raw = (self.name_zh or self.name_en or "").strip()
        if not self.name_zh and self.name_raw:
            self.name_zh = self.name_raw
        return self


class ReviewQueueItem(BaseModel):
    """S3-01 输出的审核队列标准记录。"""
    model_config = ConfigDict(extra="forbid")
    review_id: str
    cluster_id: str
    element_type: ElementType
    concept_name: str
    name_zh: str = ""
    name_en: str = ""
    agent3_decision: Agent3Decision
    agent3_confidence: float = Field(ge=0.0, le=1.0)
    agent3_confidence_level: str = ""
    candidate_parent_class: Optional[str] = None
    candidate_domain: Optional[str] = None
    candidate_range: Optional[str] = None
    candidate_xsd_range: Optional[str] = None
    candidate_super_property: Optional[str] = None
    candidate_comment_zh: str = ""
    reasoning: str = Field(min_length=1)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    topology_checks: List[TopologyCheck] = Field(default_factory=list)
    risk_level: RiskLevel
    expert_action: Literal[""] = ""
    final_name_zh: Optional[str] = None
    final_name_en: Optional[str] = None
    final_comment_zh: Optional[str] = None
    final_parent_class: Optional[str] = None
    final_domain: Optional[str] = None
    final_range: Optional[str] = None
    final_xsd_range: Optional[str] = None
    final_super_property: Optional[str] = None
    expert_comment: str = ""
    reviewer: str = ""
    review_time: str = ""
    source_hash: str


class ReviewPrepError(BaseModel):
    """S3-01 预处理阶段的错误记录。"""
    model_config = ConfigDict(extra="forbid")
    phase: Literal["prepare_review_queue"]
    index: int
    cluster_id: Optional[str] = None
    error_type: str
    error: str


# =========================
# Helpers
# =========================
def write_json(path: Path, data: Any) -> None:
    """写 UTF-8 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    """读取 JSON 数组文件；若不是数组则抛错。"""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Input must be JSON array: {path}")
    return [x for x in payload if isinstance(x, dict)]


def display_name(rec: Agent3DecisionRecord) -> str:
    """返回审核台优先展示名称。"""
    return (rec.name_zh or rec.name_raw or rec.name_en or "").strip()


def to_short_reasoning(text: str, limit: int = 240) -> str:
    """压缩推理文本长度，便于审核台展示。"""
    clean = " ".join(str(text).split()).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def extract_candidate_comment_zh(rec: Agent3DecisionRecord) -> str:
    """提取候选中文定义：优先 merged_definitions，再回退 definition。"""
    candidates: List[str] = []

    # 1) 顶层 merged_definitions
    for x in rec.merged_definitions:
        t = str(x).strip()
        if t:
            candidates.append(t)

    # 2) input_snapshot.merged_definitions / input_snapshot.definition
    snap = rec.input_snapshot if isinstance(rec.input_snapshot, dict) else {}
    snap_md = snap.get("merged_definitions")
    if isinstance(snap_md, list):
        for x in snap_md:
            t = str(x).strip()
            if t:
                candidates.append(t)
    snap_def = snap.get("definition")
    if snap_def is not None:
        t = str(snap_def).strip()
        if t:
            candidates.append(t)

    # 3) 顶层 definition
    if rec.definition:
        t = str(rec.definition).strip()
        if t:
            candidates.append(t)

    # 保序去重
    seen = set()
    uniq: List[str] = []
    for t in candidates:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)

    return "；".join(uniq)


def extract_single_comment_zh(rec: Agent3DecisionRecord) -> str:
    """只保留 input_snapshot.merged_definitions 中的第一条有效定义。"""
    snap = rec.input_snapshot if isinstance(rec.input_snapshot, dict) else {}
    snap_md = snap.get("merged_definitions")
    if not isinstance(snap_md, list):
        return ""

    seen = set()
    for x in snap_md:
        t = str(x).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        return t
    return ""


def derive_risk_level(checks: List[TopologyCheck]) -> RiskLevel:
    """根据 topology_checks 失败级别映射风险等级。"""
    has_failed_error = any((not c.passed) and c.level == "error" for c in checks)
    if has_failed_error:
        return "HIGH"
    has_failed_warning = any((not c.passed) and c.level == "warning" for c in checks)
    if has_failed_warning:
        return "MEDIUM"
    return "LOW"


def build_source_hash(rec: Agent3DecisionRecord) -> str:
    """对关键输入快照做 SHA256，支持后续可追溯与防篡改检查。"""
    canonical_payload = {
        "cluster_id": rec.cluster_id,
        "element_type": rec.element_type,
        "name_raw": rec.name_raw,
        "name_zh": rec.name_zh,
        "name_en": rec.name_en,
        "decision": rec.decision,
        "confidence": rec.confidence,
        "parent_class": rec.parent_class,
        "suggested_domain": rec.suggested_domain,
        "suggested_range": rec.suggested_range,
        "suggested_xsd_range": rec.suggested_xsd_range,
        "candidate_super_property": rec.candidate_super_property,
        "definition": rec.definition,
        "merged_definitions": rec.merged_definitions,
        "input_snapshot_merged_definitions": rec.input_snapshot.get("merged_definitions", []) if isinstance(rec.input_snapshot, dict) else [],
        "reasoning": rec.reasoning,
        "topology_checks": [c.model_dump() for c in rec.topology_checks],
        "evidence_refs": [e.model_dump() for e in rec.evidence_refs],
    }
    blob = json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_review_item(review_id: str, rec: Agent3DecisionRecord) -> ReviewQueueItem:
    """将 Agent3 一条决策转换为审核队列标准记录。"""
    candidate_parent = rec.parent_class if rec.element_type == "Class" else None
    candidate_domain = rec.suggested_domain if rec.element_type in {"ObjectProperty", "DatatypeProperty"} else None
    candidate_range = rec.suggested_range if rec.element_type == "ObjectProperty" else None
    candidate_xsd = rec.suggested_xsd_range if rec.element_type == "DatatypeProperty" else None
    candidate_super_property = rec.candidate_super_property if rec.element_type == "ObjectProperty" else None

    name_zh = (rec.name_zh or rec.name_raw or "").strip()
    name_en = (rec.name_en or "").strip()
    concept_name = display_name(rec)
    candidate_comment_zh = extract_single_comment_zh(rec)

    return ReviewQueueItem(
        review_id=review_id,
        cluster_id=rec.cluster_id,
        element_type=rec.element_type,
        concept_name=concept_name,
        name_zh=name_zh,
        name_en=name_en,
        agent3_decision=rec.decision,
        agent3_confidence=rec.confidence,
        agent3_confidence_level=rec.confidence_level or "",
        candidate_parent_class=candidate_parent,
        candidate_domain=candidate_domain,
        candidate_range=candidate_range,
        candidate_xsd_range=candidate_xsd,
        candidate_super_property=candidate_super_property,
        candidate_comment_zh=candidate_comment_zh,
        reasoning=to_short_reasoning(rec.reasoning),
        evidence_refs=rec.evidence_refs,
        topology_checks=rec.topology_checks,
        risk_level=derive_risk_level(rec.topology_checks),
        source_hash=build_source_hash(rec),
    )


def write_csv(path: Path, rows: List[ReviewQueueItem]) -> None:
    """输出给专家阅读/回填的 CSV（Excel 友好）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "review_id",
        "cluster_id",
        "element_type",
        "concept_name",
        "name_zh",
        "name_en",
        "判定是否新增",
        "判定代码说明",
        "agent3_decision",
        "agent3_confidence",
        "agent3_confidence_level",
        "candidate_parent_class",
        "candidate_domain",
        "candidate_range",
        "candidate_xsd_range",
        "candidate_super_property",
        "candidate_comment_zh",
        "reasoning",
        "risk_level",
        "evidence_preview",
        "expert_action_code",
        "final_name_zh",
        "final_name_en",
        "final_comment_zh",
        "final_parent_class",
        "final_domain",
        "final_range",
        "final_xsd_range",
        "final_super_property",
        "expert_comment",
        "reviewer",
        "review_time",
        "source_hash",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            ev = row.evidence_refs[0] if row.evidence_refs else None
            evidence_preview = ""
            if ev:
                evidence_preview = f"{ev.paper_id} | {ev.section} | {ev.chunk_id} | {ev.evidence_text[:120]}"
            writer.writerow(
                {
                    "review_id": row.review_id,
                    "cluster_id": row.cluster_id,
                    "element_type": row.element_type,
                    "concept_name": row.concept_name,
                    "name_zh": row.name_zh,
                    "name_en": row.name_en,
                    "判定是否新增": "是（KEEP_NEW候选）",
                    "判定代码说明": "1=INSERT,2=SKIP,3=MODIFY,4=DEFER",
                    "agent3_decision": row.agent3_decision,
                    "agent3_confidence": row.agent3_confidence,
                    "agent3_confidence_level": row.agent3_confidence_level,
                    "candidate_parent_class": row.candidate_parent_class or "",
                    "candidate_domain": row.candidate_domain or "",
                    "candidate_range": row.candidate_range or "",
                    "candidate_xsd_range": row.candidate_xsd_range or "",
                    "candidate_super_property": row.candidate_super_property or "",
                    "candidate_comment_zh": row.candidate_comment_zh,
                    "reasoning": row.reasoning,
                    "risk_level": row.risk_level,
                    "evidence_preview": evidence_preview,
                    "expert_action_code": "",
                    "final_name_zh": row.final_name_zh or "",
                    "final_name_en": row.final_name_en or "",
                    "final_comment_zh": row.final_comment_zh or "",
                    "final_parent_class": row.final_parent_class or "",
                    "final_domain": row.final_domain or "",
                    "final_range": row.final_range or "",
                    "final_xsd_range": row.final_xsd_range or "",
                    "final_super_property": row.final_super_property or "",
                    "expert_comment": row.expert_comment,
                    "reviewer": row.reviewer,
                    "review_time": row.review_time,
                    "source_hash": row.source_hash,
                }
            )


def write_xlsx(path: Path, rows: List[ReviewQueueItem]) -> None:
    """输出 XLSX，并将 expert_action_code（专家填写列）整列设置为红底。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "review_queue"

    # 表头第一行加入填写说明，便于专家直接参考
    legend_text = "填写说明：expert_action_code 请填数字代码（1=INSERT写入, 2=SKIP不写入, 3=MODIFY修改后写入, 4=DEFER暂缓）"
    ws.append([legend_text])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=33)

    headers = [
        "review_id",
        "cluster_id",
        "element_type",
        "concept_name",
        "name_zh",
        "name_en",
        "判定是否新增",
        "判定代码说明",
        "agent3_decision",
        "agent3_confidence",
        "agent3_confidence_level",
        "candidate_parent_class",
        "candidate_domain",
        "candidate_range",
        "candidate_xsd_range",
        "candidate_super_property",
        "candidate_comment_zh",
        "reasoning",
        "risk_level",
        "evidence_preview",
        "expert_action_code",
        "final_name_zh",
        "final_name_en",
        "final_comment_zh",
        "final_parent_class",
        "final_domain",
        "final_range",
        "final_xsd_range",
        "final_super_property",
        "expert_comment",
        "reviewer",
        "review_time",
        "source_hash",
    ]
    ws.append(headers)

    for row in rows:
        ev = row.evidence_refs[0] if row.evidence_refs else None
        evidence_preview = ""
        if ev:
            evidence_preview = f"{ev.paper_id} | {ev.section} | {ev.chunk_id} | {ev.evidence_text[:120]}"
        ws.append(
            [
                row.review_id,
                row.cluster_id,
                row.element_type,
                row.concept_name,
                row.name_zh,
                row.name_en,
                "是（KEEP_NEW候选）",
                "1=INSERT,2=SKIP,3=MODIFY,4=DEFER",
                row.agent3_decision,
                row.agent3_confidence,
                row.agent3_confidence_level,
                row.candidate_parent_class or "",
                row.candidate_domain or "",
                row.candidate_range or "",
                row.candidate_xsd_range or "",
                row.candidate_super_property or "",
                row.candidate_comment_zh,
                row.reasoning,
                row.risk_level,
                evidence_preview,
                "",
                row.final_name_zh or "",
                row.final_name_en or "",
                row.final_comment_zh or "",
                row.final_parent_class or "",
                row.final_domain or "",
                row.final_range or "",
                row.final_xsd_range or "",
                row.final_super_property or "",
                row.expert_comment,
                row.reviewer,
                row.review_time,
                row.source_hash,
            ]
        )

    # expert_action_code 是第 21 列（U 列），整列浅红标注（含表头）
    red_fill = PatternFill(fill_type="solid", fgColor="FDE2E1")
    target_col_idx = 21
    for r in range(1, ws.max_row + 1):
        ws.cell(row=r, column=target_col_idx).fill = red_fill

    wb.save(path)


def main() -> None:
    """S3-01 主流程：读取 Agent3 决策 -> 生成审核队列 JSON/CSV -> 写运行清单。"""
    started = time.time()
    log_path = setup_console_log(f"phase3_01_prepare_review_queue_{RUN_MODE}")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"[运行启动] S3-01 生成专家审核队列 | mode={RUN_MODE}")
    print(f"[输入信息] 决策输入={INPUT_DECISIONS}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print(f"[参数配置] start_index={START_INDEX} | max_items={MAX_ITEMS}")
    print("=" * 88)

    if not INPUT_DECISIONS.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_DECISIONS}")

    raw_all = load_json_array(INPUT_DECISIONS)
    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    raw_selected = raw_all[START_INDEX:end_index]

    errors: List[ReviewPrepError] = []
    queue_items: List[ReviewQueueItem] = []

    for i, raw in enumerate(raw_selected, start=1):
        global_index = START_INDEX + i - 1
        print(f"[开始处理] [{i}/{len(raw_selected)}] index={global_index}")
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                rec = Agent3DecisionRecord.model_validate(raw)
                review_id = f"S3R_{len(queue_items) + 1:06d}"
                queue_items.append(build_review_item(review_id, rec))
                break
            except ValidationError as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    sleep_sec = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    print(f"  - [校验失败] {sleep_sec:.1f}s 后重试")
                    time.sleep(sleep_sec)
            except Exception as e:  # noqa: BLE001
                last_error = f"{type(e).__name__}: {e}"
                if attempt < MAX_RETRIES:
                    sleep_sec = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    print(f"  - [异常错误] {sleep_sec:.1f}s 后重试")
                    time.sleep(sleep_sec)

        if last_error and (len(queue_items) < i):
            cluster_id = raw.get("cluster_id") if isinstance(raw, dict) else None
            errors.append(
                ReviewPrepError(
                    phase="prepare_review_queue",
                    index=global_index,
                    cluster_id=cluster_id,
                    error_type="validation_error",
                    error=last_error,
                )
            )

    write_json(OUTPUT_REVIEW_QUEUE_JSON, [x.model_dump() for x in queue_items])
    write_json(OUTPUT_REVIEWED_TEMPLATE_JSON, [x.model_dump() for x in queue_items])
    write_csv(OUTPUT_REVIEW_QUEUE_CSV, queue_items)
    write_xlsx(OUTPUT_REVIEW_QUEUE_XLSX, queue_items)
    write_json(OUTPUT_ERRORS, [x.model_dump() for x in errors])

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_mode": RUN_MODE,
        "input_file": str(INPUT_DECISIONS),
        "output_review_queue_json": str(OUTPUT_REVIEW_QUEUE_JSON),
        "output_review_queue_csv": str(OUTPUT_REVIEW_QUEUE_CSV),
        "output_review_queue_xlsx": str(OUTPUT_REVIEW_QUEUE_XLSX),
        "output_reviewed_template_json": str(OUTPUT_REVIEWED_TEMPLATE_JSON),
        "output_errors": str(OUTPUT_ERRORS),
        "run_range": {
            "start_index": START_INDEX,
            "max_items": MAX_ITEMS,
            "selected_total": len(raw_selected),
        },
        "counts": {
            "input_total": len(raw_all),
            "queue_total": len(queue_items),
            "error_total": len(errors),
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(OUTPUT_MANIFEST, manifest)

    print("\n" + "=" * 88)
    print("[运行结束] S3-01 专家审核队列生成完成")
    print(f"[运行摘要] 审核队列条数={len(queue_items)} | 错误条数={len(errors)}")
    print(f"[性能统计] 总耗时={manifest['elapsed_sec']}s")
    print(f"[关键输出] {OUTPUT_REVIEW_QUEUE_JSON} -> 审核主表（JSON，后续流程权威输入）")
    print(f"[关键输出] {OUTPUT_REVIEW_QUEUE_XLSX} -> 审核台表格（XLSX，expert_action_code 为浅红填写列）")
    print(f"[关键输出] {OUTPUT_REVIEWED_TEMPLATE_JSON} -> 专家回填模板（JSON）")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要（参数、统计、耗时）")
    print(f"[关键输出] {OUTPUT_ERRORS} -> 预处理阶段错误明细")
    print("=" * 88)


if __name__ == "__main__":
    main()
