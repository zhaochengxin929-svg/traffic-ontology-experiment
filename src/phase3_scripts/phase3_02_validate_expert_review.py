import atexit
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from openpyxl import load_workbook
from pydantic import BaseModel, ConfigDict, Field, ValidationError


# =========================
# 配置区（常改）
# =========================
# 当前阶段目录：大规模实验/phase3
BASE_DIR = Path(__file__).resolve().parent.parent
# 运行模式：class=类概念轮；property=属性轮
RUN_MODE: Literal["class", "property"] = "property"
# S3-01 输出目录（S3-02 的默认输入来源）
INPUT_DIR = BASE_DIR / "outputs_phase3_01"
# S3-02 输出目录（与 S3-01 物理解耦）
OUTPUT_DIR = BASE_DIR / "outputs_phase3_02"

# 专家回填文件：优先读取已填写的 xlsx；若不存在则回退到 reviewed.json；再回退到模板（用于快速联调）
INPUT_REVIEWED_JSON = INPUT_DIR / f"agent3_{RUN_MODE}_review_reviewed.json"
INPUT_REVIEWED_XLSX = INPUT_DIR / f"agent3_{RUN_MODE}_review_flat.xlsx"
FALLBACK_TEMPLATE_JSON = INPUT_DIR / f"agent3_{RUN_MODE}_review_template.json"

# S3-02 输出文件
# 校验后决策主文件：包含 is_valid、issues、effective_action、effective_* 锚点
OUTPUT_VALIDATED = OUTPUT_DIR / f"validated_{RUN_MODE}_decisions.json"
# Schema 级错误文件：记录无法通过 Pydantic 结构校验的记录
OUTPUT_ERRORS = OUTPUT_DIR / f"validated_{RUN_MODE}_errors.json"
# 运行清单：记录本次参数、计数统计、耗时与输入输出路径
OUTPUT_MANIFEST = OUTPUT_DIR / f"validated_{RUN_MODE}_manifest.json"
OUTPUT_SUMMARY = OUTPUT_DIR / f"validated_{RUN_MODE}_summary.json"
# 是否将控制台输出同步写入日志文件
ENABLE_FILE_LOG = True
# 统一日志目录：保存本脚本控制台输出
LOG_DIR = BASE_DIR.parent / "logs"

# 小样本/断点参数：默认先串行跑 20 条
START_INDEX = 0
MAX_ITEMS = 0
# 重试参数：单条 schema 校验异常时重试
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

# DatatypeProperty 可接受的 xsd 数据类型白名单
XSD_ALLOWED = {
    "xsd:string",
    "xsd:float",
    "xsd:double",
    "xsd:decimal",
    "xsd:int",
    "xsd:integer",
    "xsd:boolean",
    "xsd:dateTime",
    "xsd:date",
}

ACTION_CODE_MAP: Dict[str, str] = {
    "1": "INSERT",
    "2": "SKIP",
    "3": "MODIFY",
    "4": "DEFER",
}


# =========================
# Enums
# =========================
ElementType = Literal["Class", "ObjectProperty", "DatatypeProperty"]
Agent3Decision = Literal["POSITIONED", "UNSURE", "REJECTED"]
CheckLevel = Literal["error", "warning", "info"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
FinalAction = Literal["INSERT", "SKIP", "MODIFY", "DEFER"]


# =========================
# Models
# =========================
class TopologyCheck(BaseModel):
    """承接审核表中的拓扑规则检查结果。"""
    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(min_length=1)
    passed: bool
    level: CheckLevel
    message: str = Field(min_length=1)


class EvidenceRef(BaseModel):
    """承接审核表中的证据引用。"""
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    section: str
    chunk_id: str
    evidence_text: str = ""


class ExpertReviewedItem(BaseModel):
    """S3-02 输入记录：专家回填后的完整审核项。"""
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
    reasoning: str
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    topology_checks: List[TopologyCheck] = Field(default_factory=list)
    risk_level: RiskLevel
    expert_action: FinalAction
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


class ReviewValidationIssue(BaseModel):
    """单条业务规则校验问题。"""
    model_config = ConfigDict(extra="forbid")
    code: str
    field: str
    level: Literal["error", "warning"]
    message: str


class ValidatedDecision(BaseModel):
    """S3-02 输出记录：供 S3-03 直接消费的 validated 决策。"""
    model_config = ConfigDict(extra="forbid")
    review_id: str
    cluster_id: str
    element_type: ElementType
    concept_name: str
    agent3_confidence_level: str = ""
    expert_action: FinalAction
    is_valid: bool
    issues: List[ReviewValidationIssue] = Field(default_factory=list)
    effective_action: Literal["INSERT", "SKIP", "PENDING"]
    effective_name_zh: Optional[str] = None
    effective_name_en: Optional[str] = None
    effective_comment_zh: Optional[str] = None
    effective_parent_class: Optional[str] = None
    effective_domain: Optional[str] = None
    effective_range: Optional[str] = None
    effective_xsd_range: Optional[str] = None
    effective_super_property: Optional[str] = None
    reviewer: str
    review_time: str
    expert_comment: str
    source_hash: str


class ReviewValidationError(BaseModel):
    """S3-02 结构校验失败记录。"""
    model_config = ConfigDict(extra="forbid")
    phase: Literal["validate_expert_review"]
    index: int
    review_id: Optional[str] = None
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
    """读取 JSON 数组文件；若格式不符则抛错。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Input must be JSON array: {path}")
    return [x for x in payload if isinstance(x, dict)]


def normalize_action_code(value: Any) -> str:
    """将数字判定代码转换为标准动作字符串。"""
    if value is None:
        return ""
    text = str(value).strip()
    if text in ACTION_CODE_MAP:
        return ACTION_CODE_MAP[text]
    upper_text = text.upper()
    if upper_text in {"INSERT", "SKIP", "MODIFY", "DEFER"}:
        return upper_text
    return text


def preprocess_raw_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """预处理原始记录：支持 expert_action_code(1/2/3/4) 输入。"""
    data = dict(raw)
    action = normalize_action_code(data.get("expert_action"))
    if not action:
        action = normalize_action_code(data.get("expert_action_code"))
    data["expert_action"] = action
    # 命名字段兼容回填
    data["name_zh"] = str(data.get("name_zh") or data.get("concept_name") or "").strip()
    data["name_en"] = str(data.get("name_en") or "").strip()
    data["candidate_super_property"] = str(data.get("candidate_super_property") or "").strip() or None
    data["candidate_comment_zh"] = str(data.get("candidate_comment_zh") or "").strip()
    data["final_comment_zh"] = str(data.get("final_comment_zh") or "").strip() or None
    data["final_super_property"] = str(data.get("final_super_property") or "").strip() or None
    return data


def load_reviewed_from_xlsx(path: Path) -> List[Dict[str, Any]]:
    """读取审核 XLSX，并转换为 reviewed.json 等价结构。"""
    wb = load_workbook(path, data_only=True)
    ws = wb.active

    header_row_idx = None
    headers: List[str] = []
    for ridx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        values = [str(x).strip() if x is not None else "" for x in row]
        if "review_id" in values:
            header_row_idx = ridx
            headers = values
            break

    if header_row_idx is None:
        raise ValueError(f"Cannot find header row with 'review_id' in xlsx: {path}")

    rows: List[Dict[str, Any]] = []
    for row in ws.iter_rows(min_row=header_row_idx + 1, values_only=True):
        record = {headers[i]: row[i] if i < len(row) else None for i in range(len(headers))}
        if not str(record.get("review_id") or "").strip():
            continue

        converted = {
            "review_id": str(record.get("review_id") or "").strip(),
            "cluster_id": str(record.get("cluster_id") or "").strip(),
            "element_type": str(record.get("element_type") or "").strip(),
            "concept_name": str(record.get("concept_name") or "").strip(),
            "name_zh": str(record.get("name_zh") or record.get("concept_name") or "").strip(),
            "name_en": str(record.get("name_en") or "").strip(),
            "agent3_decision": str(record.get("agent3_decision") or "").strip(),
            "agent3_confidence": float(record.get("agent3_confidence") or 0.0),
            "agent3_confidence_level": str(record.get("agent3_confidence_level") or "").strip(),
            "candidate_parent_class": str(record.get("candidate_parent_class") or "").strip() or None,
            "candidate_domain": str(record.get("candidate_domain") or "").strip() or None,
            "candidate_range": str(record.get("candidate_range") or "").strip() or None,
            "candidate_xsd_range": str(record.get("candidate_xsd_range") or "").strip() or None,
            "candidate_super_property": str(record.get("candidate_super_property") or "").strip() or None,
            "candidate_comment_zh": str(record.get("candidate_comment_zh") or "").strip(),
            "reasoning": str(record.get("reasoning") or "").strip(),
            "risk_level": str(record.get("risk_level") or "LOW").strip(),
            "expert_action": normalize_action_code(record.get("expert_action_code")),
            "final_name_zh": str(record.get("final_name_zh") or "").strip() or None,
            "final_name_en": str(record.get("final_name_en") or "").strip() or None,
            "final_comment_zh": str(record.get("final_comment_zh") or "").strip() or None,
            "final_parent_class": str(record.get("final_parent_class") or "").strip() or None,
            "final_domain": str(record.get("final_domain") or "").strip() or None,
            "final_range": str(record.get("final_range") or "").strip() or None,
            "final_xsd_range": str(record.get("final_xsd_range") or "").strip() or None,
            "final_super_property": str(record.get("final_super_property") or "").strip() or None,
            "expert_comment": str(record.get("expert_comment") or "").strip(),
            "reviewer": str(record.get("reviewer") or "").strip(),
            "review_time": str(record.get("review_time") or "").strip(),
            "source_hash": str(record.get("source_hash") or "").strip(),
            "evidence_refs": [],
            "topology_checks": [],
        }
        rows.append(converted)

    return rows


def clean_optional(text: Optional[str]) -> Optional[str]:
    """把空字符串标准化为 None，便于后续统一校验。"""
    if text is None:
        return None
    t = str(text).strip()
    return t if t else None


def looks_like_class_anchor(value: Optional[str]) -> bool:
    """校验类锚点格式是否符合 :ClassName。"""
    if value is None:
        return False
    t = value.strip()
    if not t:
        return False
    return bool(re.match(r"^:[A-Za-z_][A-Za-z0-9_]*$", t))


def looks_like_property_anchor(value: Optional[str]) -> bool:
    """校验属性锚点格式是否符合 :lowerCamelCase。"""
    if value is None:
        return False
    t = value.strip()
    if not t:
        return False
    return bool(re.match(r"^:[a-z][A-Za-z0-9_]*$", t))


def looks_like_xsd(value: Optional[str]) -> bool:
    """校验 xsd 类型是否在白名单中。"""
    if value is None:
        return False
    return value.strip() in XSD_ALLOWED


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


def validate_iso8601_like(value: str) -> bool:
    """校验时间字符串是否为 ISO8601 可解析格式。"""
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def make_issue(code: str, field: str, level: Literal["error", "warning"], message: str) -> ReviewValidationIssue:
    """构造统一的校验问题对象。"""
    return ReviewValidationIssue(code=code, field=field, level=level, message=message)


def validate_business_rules(item: ExpertReviewedItem) -> List[ReviewValidationIssue]:
    """执行业务规则校验，重点包含 MODIFY 的完整锚点约束。"""
    issues: List[ReviewValidationIssue] = []

    if item.review_time.strip() and (not validate_iso8601_like(item.review_time)):
        issues.append(make_issue("E_REVIEW_TIME_FORMAT", "review_time", "error", "review_time must be ISO8601 format"))

    candidate_super = clean_optional(item.candidate_super_property)
    final_super = clean_optional(item.final_super_property)
    if item.element_type != "ObjectProperty":
        if candidate_super:
            issues.append(make_issue("E_SUPER_PROPERTY_NON_OP", "candidate_super_property", "error", "candidate_super_property is only allowed for ObjectProperty"))
        if final_super:
            issues.append(make_issue("E_SUPER_PROPERTY_NON_OP", "final_super_property", "error", "final_super_property is only allowed for ObjectProperty"))
    else:
        if candidate_super and not looks_like_property_anchor(candidate_super):
            issues.append(make_issue("E_CANDIDATE_SUPER_PROPERTY_FORMAT", "candidate_super_property", "error", "candidate_super_property must match :propertyName"))
        if final_super and not looks_like_property_anchor(final_super):
            issues.append(make_issue("E_FINAL_SUPER_PROPERTY_FORMAT", "final_super_property", "error", "final_super_property must match :propertyName"))

    # 命名规则：入库候选必须具备中文名与合法英文 IRI 本地名
    if item.expert_action == "MODIFY":
        effective_name_zh = clean_optional(item.final_name_zh) or clean_optional(item.name_zh)
        effective_name_en = clean_optional(item.final_name_en) or clean_optional(item.name_en)
    else:
        effective_name_zh = clean_optional(item.name_zh)
        effective_name_en = clean_optional(item.name_en)
    if item.expert_action in {"INSERT", "MODIFY"}:
        if not effective_name_zh:
            issues.append(make_issue("E_NAME_ZH_REQUIRED", "name_zh", "error", "INSERT candidate requires non-empty name_zh"))
        if not effective_name_en:
            issues.append(make_issue("E_NAME_EN_REQUIRED", "name_en", "error", "INSERT candidate requires non-empty name_en"))
        elif item.element_type == "Class" and (not looks_like_class_iri_local_name(effective_name_en)):
            issues.append(make_issue("E_NAME_EN_INVALID_IRI", "name_en", "error", "Class name_en must match IRI local name pattern: ^[A-Z][A-Za-z0-9]*$"))
        elif item.element_type in {"ObjectProperty", "DatatypeProperty"} and (not looks_like_property_iri_local_name(effective_name_en)):
            issues.append(make_issue("E_NAME_EN_INVALID_IRI", "name_en", "error", "Property name_en must match IRI local name pattern: ^[a-z][A-Za-z0-9]*$"))

    if item.expert_action == "MODIFY":
        if item.element_type == "Class":
            if not clean_optional(item.final_parent_class):
                issues.append(make_issue("E_MODIFY_CLASS_PARENT_REQUIRED", "final_parent_class", "error", "MODIFY-Class requires final_parent_class"))
            elif not looks_like_class_anchor(item.final_parent_class):
                issues.append(make_issue("E_MODIFY_CLASS_PARENT_FORMAT", "final_parent_class", "error", "final_parent_class must match :ClassName"))

        if item.element_type == "ObjectProperty":
            if not clean_optional(item.final_domain):
                issues.append(make_issue("E_MODIFY_OP_DOMAIN_REQUIRED", "final_domain", "error", "MODIFY-ObjectProperty requires final_domain"))
            if not clean_optional(item.final_range):
                issues.append(make_issue("E_MODIFY_OP_RANGE_REQUIRED", "final_range", "error", "MODIFY-ObjectProperty requires final_range"))
            if clean_optional(item.final_domain) and not looks_like_class_anchor(item.final_domain):
                issues.append(make_issue("E_MODIFY_OP_DOMAIN_FORMAT", "final_domain", "error", "final_domain must match :ClassName"))
            if clean_optional(item.final_range) and not looks_like_class_anchor(item.final_range):
                issues.append(make_issue("E_MODIFY_OP_RANGE_FORMAT", "final_range", "error", "final_range must match :ClassName"))

        if item.element_type == "DatatypeProperty":
            if not clean_optional(item.final_domain):
                issues.append(make_issue("E_MODIFY_DP_DOMAIN_REQUIRED", "final_domain", "error", "MODIFY-DatatypeProperty requires final_domain"))
            if not clean_optional(item.final_xsd_range):
                issues.append(make_issue("E_MODIFY_DP_XSD_REQUIRED", "final_xsd_range", "error", "MODIFY-DatatypeProperty requires final_xsd_range"))
            if clean_optional(item.final_domain) and not looks_like_class_anchor(item.final_domain):
                issues.append(make_issue("E_MODIFY_DP_DOMAIN_FORMAT", "final_domain", "error", "final_domain must match :ClassName"))
            if clean_optional(item.final_xsd_range) and not looks_like_xsd(item.final_xsd_range):
                issues.append(make_issue("E_MODIFY_DP_XSD_INVALID", "final_xsd_range", "error", "final_xsd_range is not in allowed xsd set"))

    return issues


def build_effective_fields(item: ExpertReviewedItem) -> Dict[str, Optional[str]]:
    """生成后续入库使用的 effective_* 字段（MODIFY 优先 final_*）。"""
    if item.expert_action == "MODIFY":
        effective_super_property = clean_optional(item.final_super_property) or clean_optional(item.candidate_super_property)
        return {
            "effective_name_zh": clean_optional(item.final_name_zh) or clean_optional(item.name_zh),
            "effective_name_en": clean_optional(item.final_name_en) or clean_optional(item.name_en),
            "effective_comment_zh": clean_optional(item.final_comment_zh) or clean_optional(item.candidate_comment_zh),
            "effective_parent_class": clean_optional(item.final_parent_class),
            "effective_domain": clean_optional(item.final_domain),
            "effective_range": clean_optional(item.final_range),
            "effective_xsd_range": clean_optional(item.final_xsd_range),
            "effective_super_property": effective_super_property if item.element_type == "ObjectProperty" else None,
        }

    return {
        "effective_name_zh": clean_optional(item.name_zh) or clean_optional(item.concept_name),
        "effective_name_en": clean_optional(item.name_en),
        "effective_comment_zh": clean_optional(item.candidate_comment_zh),
        "effective_parent_class": clean_optional(item.candidate_parent_class),
        "effective_domain": clean_optional(item.candidate_domain),
        "effective_range": clean_optional(item.candidate_range),
        "effective_xsd_range": clean_optional(item.candidate_xsd_range),
        "effective_super_property": clean_optional(item.candidate_super_property) if item.element_type == "ObjectProperty" else None,
    }


def build_validated_summary(validated: List[ValidatedDecision], errors: List[ReviewValidationError]) -> Dict[str, Any]:
    expert_action_to_effective_action: Dict[str, Dict[str, int]] = {}
    issue_code_counts: Dict[str, int] = {}
    invalid_business_issue_code_counts: Dict[str, int] = {}
    pending_issue_code_counts: Dict[str, int] = {}
    schema_error_type_counts: Dict[str, int] = {}

    for item in validated:
        expert_action_to_effective_action.setdefault(item.expert_action, {})
        bucket = expert_action_to_effective_action[item.expert_action]
        bucket[item.effective_action] = bucket.get(item.effective_action, 0) + 1

        issue_codes = [issue.code for issue in item.issues]
        for code in issue_codes:
            issue_code_counts[code] = issue_code_counts.get(code, 0) + 1
        if not item.is_valid:
            for code in issue_codes:
                invalid_business_issue_code_counts[code] = invalid_business_issue_code_counts.get(code, 0) + 1
        if item.effective_action == "PENDING":
            for code in issue_codes:
                pending_issue_code_counts[code] = pending_issue_code_counts.get(code, 0) + 1

    for err in errors:
        schema_error_type_counts[err.error_type] = schema_error_type_counts.get(err.error_type, 0) + 1

    return {
        "expert_action_to_effective_action": expert_action_to_effective_action,
        "issue_code_counts": issue_code_counts,
        "invalid_business_issue_code_counts": invalid_business_issue_code_counts,
        "pending_issue_code_counts": pending_issue_code_counts,
        "schema_error_type_counts": schema_error_type_counts,
    }


def main() -> None:
    """S3-02 主流程：读取专家回填 -> 规则校验 -> 生成 validated 结果。"""
    started = time.time()
    log_path = setup_console_log(f"phase3_02_validate_expert_review_{RUN_MODE}")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = None
    source_type = ""
    if INPUT_REVIEWED_XLSX.exists():
        source = INPUT_REVIEWED_XLSX
        source_type = "xlsx_reviewed"
    elif INPUT_REVIEWED_JSON.exists():
        source = INPUT_REVIEWED_JSON
        source_type = "json_reviewed"
    else:
        source = FALLBACK_TEMPLATE_JSON
        source_type = "json_template"

    print("=" * 88)
    print(f"[运行启动] S3-02 校验专家审核结果 | mode={RUN_MODE}")
    print(f"[输入信息] 审核输入={source}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print(f"[参数配置] start_index={START_INDEX} | max_items={MAX_ITEMS}")
    print("=" * 88)

    if not source.exists():
        raise FileNotFoundError(
            f"Missing input file. Expected one of: {INPUT_REVIEWED_JSON}, {INPUT_REVIEWED_XLSX}, {FALLBACK_TEMPLATE_JSON}"
        )

    if source_type == "xlsx_reviewed":
        try:
            raw_all = load_reviewed_from_xlsx(source)
            # 将转换结果固化为 reviewed.json，便于复查和后续复跑
            write_json(INPUT_REVIEWED_JSON, raw_all)
        except Exception as e:  # noqa: BLE001
            print(f"[警告] xlsx 读取失败，将回退到 reviewed.json/template: {type(e).__name__}: {e}")
            if INPUT_REVIEWED_JSON.exists():
                source = INPUT_REVIEWED_JSON
                source_type = "json_reviewed_fallback"
                raw_all = load_json_array(source)
            else:
                source = FALLBACK_TEMPLATE_JSON
                source_type = "json_template_fallback"
                raw_all = load_json_array(source)
    else:
        raw_all = load_json_array(source)
    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    raw_selected = raw_all[START_INDEX:end_index]

    validated: List[ValidatedDecision] = []
    errors: List[ReviewValidationError] = []

    for i, raw in enumerate(raw_selected, start=1):
        global_index = START_INDEX + i - 1
        print(f"[开始处理] [{i}/{len(raw_selected)}] index={global_index}")

        item: Optional[ExpertReviewedItem] = None
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                item = ExpertReviewedItem.model_validate(preprocess_raw_record(raw))
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
            review_id = raw.get("review_id") if isinstance(raw, dict) else None
            errors.append(
                ReviewValidationError(
                    phase="validate_expert_review",
                    index=global_index,
                    review_id=review_id,
                    error_type="schema_validation_error",
                    error=last_error,
                )
            )
            continue

        issues = validate_business_rules(item)
        has_error = any(x.level == "error" for x in issues)

        if item.expert_action in {"INSERT", "MODIFY"}:
            effective_action: Literal["INSERT", "SKIP", "PENDING"] = "INSERT"
        elif item.expert_action == "SKIP":
            effective_action = "SKIP"
        else:
            effective_action = "PENDING"

        if has_error:
            effective_action = "PENDING"

        ef = build_effective_fields(item)
        validated.append(
            ValidatedDecision(
                review_id=item.review_id,
                cluster_id=item.cluster_id,
                element_type=item.element_type,
                concept_name=item.concept_name,
                agent3_confidence_level=item.agent3_confidence_level,
                expert_action=item.expert_action,
                is_valid=not has_error,
                issues=issues,
                effective_action=effective_action,
                effective_name_zh=ef["effective_name_zh"],
                effective_name_en=ef["effective_name_en"],
                effective_comment_zh=ef["effective_comment_zh"],
                effective_parent_class=ef["effective_parent_class"],
                effective_domain=ef["effective_domain"],
                effective_range=ef["effective_range"],
                effective_xsd_range=ef["effective_xsd_range"],
                effective_super_property=ef["effective_super_property"],
                reviewer=item.reviewer,
                review_time=item.review_time,
                expert_comment=item.expert_comment,
                source_hash=item.source_hash,
            )
        )

    write_json(OUTPUT_VALIDATED, [x.model_dump() for x in validated])
    write_json(OUTPUT_ERRORS, [x.model_dump() for x in errors])
    summary = build_validated_summary(validated, errors)
    write_json(OUTPUT_SUMMARY, summary)

    count_insert = sum(1 for x in validated if x.effective_action == "INSERT")
    count_skip = sum(1 for x in validated if x.effective_action == "SKIP")
    count_pending = sum(1 for x in validated if x.effective_action == "PENDING")
    count_invalid = sum(1 for x in validated if not x.is_valid)

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_mode": RUN_MODE,
        "input_file": str(source),
        "input_type": source_type,
        "output_validated": str(OUTPUT_VALIDATED),
        "output_errors": str(OUTPUT_ERRORS),
        "summary_file": str(OUTPUT_SUMMARY),
        "run_range": {
            "start_index": START_INDEX,
            "max_items": MAX_ITEMS,
            "selected_total": len(raw_selected),
        },
        "counts": {
            "input_total": len(raw_all),
            "validated_total": len(validated),
            "schema_error_total": len(errors),
            "invalid_business_total": count_invalid,
            "insert_total": count_insert,
            "skip_total": count_skip,
            "pending_total": count_pending,
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(OUTPUT_MANIFEST, manifest)

    print("\n" + "=" * 88)
    print("[运行结束] S3-02 专家审核结果校验完成")
    print(
        f"[运行摘要] 校验记录={len(validated)} | schema错误={len(errors)} | "
        f"INSERT={count_insert} | SKIP={count_skip} | PENDING={count_pending}"
    )
    print(f"[性能统计] 总耗时={manifest['elapsed_sec']}s")
    print(f"[关键输出] {OUTPUT_VALIDATED} -> 校验后决策（含 is_valid、issues、effective_*）")
    print(f"[关键输出] {OUTPUT_ERRORS} -> Schema 级错误明细（无法进入业务规则校验的记录）")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要（参数、统计、耗时）")
    print("=" * 88)


if __name__ == "__main__":
    main()
