import atexit
import json
import re
import sys
import time
import statistics
from collections import Counter
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# ==========================================
# 配置区（常改）
# ==========================================
# Claude 兼容网关 API Key，可直接填写
API_KEY = ""

# 外部对齐使用的模型与网关配置
CLAUDE_MODEL = "claude-opus-4-6"
OPENAI_COMPAT_BASE_URL = "https://api2.aigcbest.top/v1"
API_TIMEOUT_SECONDS = 120

# 生成参数
TEMPERATURE = 0.1
MAX_TOKENS = 1024
TOP_P = None
DEFAULT_LLM_CONFIDENCE = 0.5

# 重试参数
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 2.0

# 外部对齐策略参数
USE_LLM_ARBITRATION = True
TOP_K = 3
LOW_SIM_THRESHOLD = 0.60
HIGH_SIM_THRESHOLD = 0.95

# 向量模型参数
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DEVICE = "cpu"
EMBED_BATCH_SIZE = 32
EMBED_NORMALIZE = True

# 小样本与断点续跑参数
MAX_ITEMS = 0
START_INDEX = 0

# 当前阶段目录：大规模实验/phase2
BASE_DIR = Path(__file__).resolve().parent.parent
# 项目根目录：大规模实验
PROJECT_ROOT = BASE_DIR.parent
RESOURCE_VERSION = "V3"
# 是否将控制台输出同时写入日志文件
ENABLE_FILE_LOG = True
# 日志目录：统一保存脚本控制台输出
LOG_DIR = PROJECT_ROOT / "logs"
# 输入文件：Agent2-内部融合输出的 canonical 概念
INPUT_CANONICAL_PATH = BASE_DIR / "outputs_phase2_02" / "agent2_internal_canonical.json"
# 输入文件：既有本体扁平词表
INPUT_VOCAB_PATH = PROJECT_ROOT / "shared_resources" / f"Agent2_Flattened_Vocabulary_{RESOURCE_VERSION}.json"

# 输出目录：Agent2-外部对齐全部产物
OUTPUT_DIR = BASE_DIR / "outputs_phase2_03"
# 每条候选概念的最终对齐决策
OUTPUT_DECISIONS = OUTPUT_DIR / "agent2_external_decisions.json"
# 判定为 KEEP_NEW 的新增概念
OUTPUT_KEEP_NEW = OUTPUT_DIR / "agent2_external_keep_new.json"
# 判定为 KEEP_NEW 的新增 Class
OUTPUT_KEEP_NEW_CLASSES = OUTPUT_DIR / "agent2_external_keep_new_classes.json"
# 判定为 KEEP_NEW 的新增属性（ObjectProperty / DatatypeProperty）
OUTPUT_KEEP_NEW_PROPERTIES = OUTPUT_DIR / "agent2_external_keep_new_properties.json"
# KEEP_NEW Class 的轻量索引文件，供 04b 本地查表使用
OUTPUT_KEEP_NEW_CLASS_INDEX = OUTPUT_DIR / "agent2_external_keep_new_class_index.json"
# 判定为 MERGE 的映射结果
OUTPUT_MERGED = OUTPUT_DIR / "agent2_external_merged.json"
# 运行错误与异常
OUTPUT_ERRORS = OUTPUT_DIR / "agent2_external_errors.json"
# 运行摘要与统计信息
OUTPUT_MANIFEST = OUTPUT_DIR / "agent2_external_manifest.json"
# 外部对齐统计摘要：供论文分析使用
OUTPUT_ALIGNMENT_STATS = OUTPUT_DIR / "agent2_external_alignment_stats.json"
# token 汇总报表：用于估算灰区仲裁的成本
OUTPUT_TOKEN_REPORT = OUTPUT_DIR / "agent2_external_token_report.json"
# 灰区仲裁原始大模型响应日志
OUTPUT_RAW_API_LOG = OUTPUT_DIR / "agent2_external_llm_raw_responses.jsonl"
# 类最终归属映射表：供属性对齐前继承 Class 的最终命名结果
OUTPUT_CLASS_RESOLUTION_MAP = OUTPUT_DIR / "class_resolution_map.json"
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

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


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

# 语义黑名单：当前留空，后续可补充互斥词对
LEXICAL_ANTONYM_BLACKLIST: List[Tuple[str, str]] = []


SYSTEM_PROMPT_TEMPLATE = """你需要承担城市交通态势本体要素对齐判定任务。

该任务用于判断一个待对齐候选要素是否已经被既有城市交通态势本体中的某个候选要素充分表达。你将收到一个待对齐候选要素，以及若干个来自既有本体扁平词表的候选目标要素。你需要判断二者是否具备语义等价性、粒度一致性和表达范围一致性，并据此给出合并或保留判断。

【任务目标】
请判断待对齐候选要素应当与既有本体候选要素合并，还是作为新增要素保留。

最终只能输出以下两类决策之一：
- MERGE：待对齐候选要素与某个既有本体候选要素语义等价，可以合并到该既有要素；
- KEEP_NEW：待对齐候选要素没有被候选目标充分覆盖，应作为新增要素保留。

【输入上下文】
你将收到两类输入：

1. 待对齐候选要素
该输入包含候选要素的类型、中文名称、英文名称、原始名称、定义、别名、融合后的定义信息以及前序步骤给出的结构建议。

2. 既有本体候选目标
该输入包含若干个来自既有本体扁平词表的候选目标要素。每个候选目标要素包含名称、类型、标签、注释和相似度信息。

你只能从提供的既有本体候选目标中选择合并对象，不得自行创造、改写或补全 target_old_name。

【链式处理步骤】
请按照以下顺序完成判断。每一步均需结合对应规则执行，后一步以前一步判断结果为基础。

Step 1：理解待对齐候选要素
- 读取内容：待对齐候选要素的 element_type、name_zh、name_en、name_raw、definition、aliases、merged_definitions 和 mapping_suggestion。
- 对应规则：【待对齐候选要素理解规则】。
- 处理目标：明确待对齐候选要素的核心语义、所属类型、适用对象、表达范围和结构位置。
- 结果作用：形成后续语义比较的判断基准。

Step 2：理解既有本体候选目标
- 读取内容：既有本体候选目标中的 old_name、old_type、label、comment 和 similarity。
- 对应规则：【既有本体候选目标理解规则】。
- 处理目标：逐一理解候选目标要素在既有本体中的语义含义、要素类型和表达范围。
- 结果作用：确定每个候选目标是否可能成为合并对象。

Step 3：判断语义等价性
- 读取内容：双方的名称、定义、别名、标签、注释和结构建议。
- 对应规则：【语义等价性判断规则】。
- 处理目标：判断二者是否表达同一交通概念、同一对象属性或同一数据属性。
- 结果作用：只有语义等价的候选目标才允许进入合并判断。

Step 4：判断粒度一致性与表达范围
- 读取内容：双方的要素类型、作用对象、空间尺度、业务范围和结构位置。
- 对应规则：【粒度一致性与表达范围判断规则】。
- 处理目标：判断二者是否处于同一抽象层级，是否存在上位—下位、整体—局部、相关但不同、范围更宽或范围更窄等情况。
- 结果作用：若二者粒度不一致或表达范围不一致，即使名称相似，也应判为 KEEP_NEW。

Step 5：判断既有要素是否充分覆盖候选要素
- 读取内容：待对齐候选要素的定义、融合定义、结构建议，以及既有本体候选目标的标签和注释。
- 对应规则：【覆盖关系判断规则】。
- 处理目标：判断既有候选目标是否已经充分表达待对齐候选要素的核心语义。
- 结果作用：只有既有要素能够充分表达待对齐候选要素时，才允许 MERGE；若只是部分覆盖、相关关联或上位概括，应判为 KEEP_NEW。

Step 6：生成最终对齐结果
- 读取内容：前述全部判断结果。
- 对应规则：【决策规则】和【输出要求】。
- 处理目标：在 MERGE 与 KEEP_NEW 之间给出最终决策，并输出目标既有要素和简要理由。
- 结果作用：形成严格 JSON 格式的本体要素对齐判定结果。

【待对齐候选要素理解规则】
1. element_type 是重要约束。Class、ObjectProperty 和 DatatypeProperty 之间原则上不得跨类型合并。
2. 判断候选要素语义时，应综合名称、定义、别名、融合定义和结构建议，不得只依据名称相似作出合并。
3. 对 Class，应重点理解其表示的交通对象、交通状态、交通指标、交通事件、拥堵过程、运行等级或其他业务概念。
4. 对 ObjectProperty，应重点理解其定义域、值域以及表示的对象—对象关系。
5. 对 DatatypeProperty，应重点理解其依附对象、字面值属性含义和取值类型。
6. mapping_suggestion 仅作为结构理解线索，不单独决定合并结果。

【既有本体候选目标理解规则】
1. 只能从既有本体候选目标中选择合并对象，不得自行生成、翻译、改写或补全 target_old_name。
2. old_type 必须与待对齐候选要素的 element_type 一致。类型不一致时默认不能 MERGE。
3. label 和 comment 用于理解既有本体要素的语义范围。
4. similarity 只表示向量相似度参考，不等同于语义等价。相似度较高仍需进行语义、粒度和表达范围判断。
5. 若多个候选目标相似，应选择语义最等价、粒度最一致、覆盖最充分的一个；若无法确定，应 KEEP_NEW。

【语义等价性判断规则】
只有满足下列条件时，才可以认为二者具备合并基础：

1. 二者表达同一概念、同一关系或同一属性；
2. 二者可以被视为同义词、近义改写、翻译对应或命名差异；
3. 二者定义中的核心对象、核心含义和业务作用一致；
4. 二者在本体中承担的语义功能一致。

以下情况不得视为语义等价：

1. 名称相似但核心含义不同；
2. 一个表示状态，另一个表示指标、事件、过程或方法；
3. 一个表示对象，另一个表示对象属性或数据属性；
4. 一个表示交通业务概念，另一个表示研究任务、模型方法、实验结果或计算过程；
5. 二者只是存在因果、影响、表征、组成、派生或相关关系。

【粒度一致性与表达范围判断规则】
MERGE 要求二者具有相同或高度一致的语义粒度。以下情况应判为 KEEP_NEW：

1. 上位类与下位类之间不得合并；
2. 总体现象与具体子现象之间不得合并；
3. 通用指标与特定空间尺度指标之间不得合并，除非二者定义明确等价；
4. 路段、交叉口、区域、路网等不同空间尺度的概念不得仅因名称相似而合并；
5. 状态、指标、事件、过程、等级等不同业务范畴不得互相合并；
6. 属性的定义域、值域或适用对象不同，不得合并；
7. 数据属性的取值含义或取值类型不同，不得合并。

如果既有候选目标只是待对齐候选要素的上位概括、下位细化、相关概念或邻近概念，应判为 KEEP_NEW。

【覆盖关系判断规则】
1. 只有既有候选目标能够充分表达待对齐候选要素的核心语义时，才允许 MERGE。
2. 若待对齐候选要素包含既有候选目标没有表达的新语义、新空间尺度、新业务对象或新结构关系，应判为 KEEP_NEW。
3. 若既有候选目标只是部分覆盖待对齐候选要素，应判为 KEEP_NEW。
4. 若二者只是一个可用于表征另一个，或一个会影响另一个，不构成合并关系。
5. 若二者只是同属于某一上位类别，不构成合并关系。
6. 若判断存在不确定性，应优先 KEEP_NEW，避免错误合并导致既有本体语义被压缩或混淆。

【决策规则】
1. 只有在语义等价、粒度一致、表达范围一致且既有候选目标能够充分覆盖待对齐候选要素时，才输出 MERGE。
2. 若输出 MERGE，target_old_name 必须从既有本体候选目标中的 old_name 原样复制，不得改写、翻译、补全或创造。
3. 若输出 MERGE，应选择最匹配的一个既有本体候选要素作为 target_old_name。
4. 若没有任何候选目标满足合并条件，输出 KEEP_NEW。
5. 若存在不确定性、候选目标之间难以区分，或只能判断为相关但不同，输出 KEEP_NEW。
6. 不要为了减少新增要素而强行合并。
7. 不要仅凭向量相似度高、名称相似或同属一个大类就输出 MERGE。

【输出要求】
1. 只能输出严格 JSON，不允许输出 Markdown、代码块或解释性废话。
2. JSON 顶层必须包含以下三个字段：
   - decision
   - reason
   - target_old_name
3. decision 只能是 "MERGE" 或 "KEEP_NEW"。
4. reason 必须用中文简要说明判断依据，重点说明语义等价性、粒度一致性和覆盖关系。
5. 若 decision = "MERGE"，target_old_name 必须一字不差地复制自既有本体候选目标中的 old_name。
6. 若 decision = "KEEP_NEW"，target_old_name 必须为 null。
7. 不得新增任何其他字段。
8. 不得输出推理过程，只输出最终 JSON。

【输出格式】
{
  "decision": "MERGE|KEEP_NEW",
  "reason": "中文简要理由",
  "target_old_name": "候选目标old_name或null"
}
"""

USER_PROMPT_TEMPLATE = """Please arbitrate this case and return JSON only.

new_concept:
{new_concept_json}

candidate_old_concepts:
{candidate_json}
"""


# ==========================================
# Pydantic 数据模型
# ==========================================
ElementType = Literal["Class", "ObjectProperty", "DatatypeProperty"]
InternalStatus = Literal["merged", "singleton"]
Decision = Literal["MERGE", "KEEP_NEW"]


class SourceInfo(BaseModel):
    """证据来源定位信息。"""
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    section: str
    chunk_id: str


class MappingSuggestion(BaseModel):
    """承接上一步内部融合后的映射建议字段。"""
    model_config = ConfigDict(extra="allow")
    suggested_parent_class: Optional[str] = None
    suggested_domain: Optional[str] = None
    suggested_range: Optional[str] = None
    suggested_xsd_range: Optional[str] = None
    candidate_super_property: Optional[str] = None
    reason: str = ""


def normalize_name_en_by_type(element_type: Any, value: Any) -> str:
    """按元素类型对 name_en 做轻量命名风格归一化。"""
    text = str(value or "").strip()
    while text.startswith(":"):
        text = text[1:].strip()
    if not text:
        return ""
    element_type_text = str(element_type or "").strip()
    if element_type_text == "Class":
        return text[0].upper() + text[1:]
    if element_type_text in {"ObjectProperty", "DatatypeProperty"}:
        return text[0].lower() + text[1:]
    return text


class CanonicalElement(BaseModel):
    """内部融合后的 canonical 概念主体。"""
    model_config = ConfigDict(extra="allow")

    name_raw: str = ""
    name_zh: str = ""
    name_en: str = ""
    element_type: ElementType
    definition: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    source: SourceInfo
    mapping_suggestion: MappingSuggestion = Field(default_factory=MappingSuggestion)
    is_instance: Optional[bool] = None

    @model_validator(mode="after")
    def fill_name_triplet(self) -> "CanonicalElement":
        self.name_raw = (self.name_raw or "").strip()
        if self.name_zh and self.name_zh.strip():
            self.name_zh = self.name_zh.strip()
        elif re.search(r"[\u4e00-\u9fff]", self.name_raw):
            self.name_zh = self.name_raw
        else:
            self.name_zh = ""

        self.name_en = normalize_name_en_by_type(self.element_type, self.name_en)

        if not (self.name_raw or self.name_zh or self.name_en):
            raise ValueError("名称字段缺失：至少需要 name_raw/name_zh/name_en 之一。")
        if not self.name_raw:
            self.name_raw = self.name_zh or self.name_en
        if not self.name_zh and re.search(r"[\u4e00-\u9fff]", self.name_raw):
            self.name_zh = self.name_raw
        return self


class MergedEvidence(BaseModel):
    """内部融合后保留的证据片段。"""
    model_config = ConfigDict(extra="allow")
    name_zh: str = ""
    name_en: str = ""
    evidence_text: str
    source: SourceInfo

    @model_validator(mode="after")
    def fill_names(self) -> "MergedEvidence":
        self.name_zh = (self.name_zh or "").strip()
        self.name_en = (self.name_en or "").strip()
        return self


class InternalSimilarity(BaseModel):
    """内部融合阶段输出的簇内相似度统计。"""
    model_config = ConfigDict(extra="forbid")
    max_pair_similarity: float
    avg_pair_similarity: float
    threshold: float


class CanonicalInputRecord(BaseModel):
    """Agent2-Step2 的主输入记录。"""
    model_config = ConfigDict(extra="forbid")
    cluster_id: str
    element_type: ElementType
    member_count: int
    internal_status: InternalStatus
    internal_similarity: InternalSimilarity
    canonical_element: CanonicalElement
    aliases: List[str] = Field(default_factory=list)
    merged_definitions: List[str] = Field(default_factory=list)
    merged_evidence: List[MergedEvidence] = Field(default_factory=list)


class FlattenedVocabItem(BaseModel):
    """既有本体扁平词表中的单个条目。"""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    type: ElementType
    label: str = ""
    comment: str = ""


class CandidateScore(BaseModel):
    """新概念与旧词表候选项之间的相似度评分。"""
    model_config = ConfigDict(extra="forbid")
    rank: int
    old_name: str
    old_type: ElementType
    similarity: float
    blocked_by_blacklist: bool
    blacklist_reason: str = ""
    label: str = ""
    comment: str = ""


class ArbitrationResponse(BaseModel):
    """灰区候选经 LLM 仲裁后的结构化决策。"""
    model_config = ConfigDict(extra="forbid")
    decision: Decision
    reason: str = Field(min_length=1)
    target_old_name: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_target(self) -> "ArbitrationResponse":
        if self.decision == "MERGE" and not (self.target_old_name and self.target_old_name.strip()):
            raise ValueError("MERGE requires target_old_name.")
        if self.decision == "KEEP_NEW" and self.target_old_name:
            raise ValueError("KEEP_NEW must not include target_old_name.")
        return self


# ==========================================
# 工具函数
# ==========================================
def write_json(path: Path, data: Any) -> None:
    """将对象写入 UTF-8 JSON 文件，并自动创建父目录。"""
    # 将对象写入 JSON 文件（UTF-8，带缩进）
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    """向 JSONL 日志文件追加一条记录。"""
    # 以 JSONL 方式追加写入一条日志
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def normalize_for_match(value: Optional[str]) -> str:
    """文本标准化：去空白、大小写折叠，便于稳健匹配。"""
    # 文本标准化：去空白、大小写折叠，便于关键词匹配
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value)).casefold().strip()


def summarize_similarity(values: List[float]) -> Dict[str, float]:
    """汇总相似度列表的均值/中位数/最小值/最大值。"""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": round(sum(values) / len(values), 6),
        "median": round(float(statistics.median(values)), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def non_unsure_text(value: Any) -> bool:
    """判断字段是否为有效非 UNSURE 文本。"""
    text = str(value or "").strip()
    return bool(text) and normalize_for_match(text) not in {"unsure", ":unsure"}


def to_jsonable(value: Any) -> Any:
    """将复杂对象递归转换为可 JSON 序列化结构。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def extract_usage_tokens(usage: Any) -> Tuple[int, int]:
    """从 usage 对象中提取输入/输出 token，兼容不同字段命名。"""
    payload = to_jsonable(usage)
    if not isinstance(payload, dict):
        return 0, 0

    def to_non_negative_int(v: Any) -> int:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return 0
        return iv if iv > 0 else 0

    input_tokens = to_non_negative_int(payload.get("prompt_tokens"))
    output_tokens = to_non_negative_int(payload.get("completion_tokens"))
    if input_tokens == 0:
        input_tokens = to_non_negative_int(payload.get("input_tokens"))
    if output_tokens == 0:
        output_tokens = to_non_negative_int(payload.get("output_tokens"))
    return input_tokens, output_tokens


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    """读取 JSON 数组文件，并只保留对象项。"""
    # 读取 JSON 数组文件，并只保留对象项
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Input file must be a JSON array: {path}")
    return [x for x in payload if isinstance(x, dict)]


def preferred_name_zh(ce: CanonicalElement) -> str:
    """优先返回中文展示名，不足时回退到原名或英文名。"""
    return (ce.name_zh or ce.name_raw or ce.name_en).strip()


def preferred_name_en(ce: CanonicalElement) -> str:
    """优先返回英文 IRI 名。"""
    return (ce.name_en or "").strip()


def display_name(ce: CanonicalElement) -> str:
    """返回用于日志和匹配的优先展示名。"""
    return preferred_name_zh(ce) or preferred_name_en(ce)


def ensure_anchor_name(value: Optional[str]) -> Optional[str]:
    """将类名归一为 :ClassName 形式；空值返回 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if text.startswith(":") else f":{text}"


def class_lookup_keys(rec: CanonicalInputRecord) -> List[str]:
    """收集类候选可用于匹配属性依赖的名称键。"""
    ce = rec.canonical_element
    raw_values = [preferred_name_zh(ce), preferred_name_en(ce), ce.name_raw, *rec.aliases]
    keys: set[str] = set()
    for value in raw_values:
        text = str(value or "").strip()
        if not text:
            continue
        keys.add(normalize_for_match(text))
        anchor_text = ensure_anchor_name(text)
        if anchor_text:
            keys.add(normalize_for_match(anchor_text))
            keys.add(normalize_for_match(anchor_text[1:]))
    return sorted(k for k in keys if k)


def ensure_sentence_transformers():
    """延迟导入向量库，避免脚本启动阶段硬失败。"""
    # 延迟导入向量库，避免脚本启动阶段硬失败
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("sentence-transformers is required. Install: pip install -U sentence-transformers")
    return SentenceTransformer


def cosine_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray, normalized: bool) -> np.ndarray:
    """计算新概念与旧词表之间的相似度矩阵。"""
    if embeddings_a.size == 0 or embeddings_b.size == 0:
        return np.zeros((embeddings_a.shape[0], embeddings_b.shape[0]), dtype=float)
    if normalized:
        sim = embeddings_a @ embeddings_b.T
    else:
        norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)
        denom = np.clip(norms_a @ norms_b.T, a_min=1e-12, a_max=None)
        sim = (embeddings_a @ embeddings_b.T) / denom
    return np.clip(sim, -1.0, 1.0)


def build_new_embedding_text(rec: CanonicalInputRecord) -> str:
    """组装“新概念”向量化文本。"""
    # 组装“新概念”向量化文本
    ce = rec.canonical_element
    merged_defs = " | ".join(rec.merged_definitions[:3])
    aliases = " | ".join(rec.aliases[:5])
    return (
        f"type: {rec.element_type}\n"
        f"name_zh: {preferred_name_zh(ce)}\n"
        f"name_en: {preferred_name_en(ce)}\n"
        f"name_raw: {ce.name_raw}\n"
        f"definition: {ce.definition}\n"
        f"merged_definitions: {merged_defs}\n"
        f"aliases: {aliases}"
    )


def build_old_embedding_text(item: FlattenedVocabItem) -> str:
    """组装“旧词表概念”向量化文本。"""
    # 组装“旧词表概念”向量化文本
    return (
        f"type: {item.type}\n"
        f"name: {item.name}\n"
        f"label: {item.label}\n"
        f"comment: {item.comment}"
    )


def check_blacklist_block(rec: CanonicalInputRecord, old_item: FlattenedVocabItem) -> Tuple[bool, str]:
    """检查是否命中语义黑名单，命中则禁止合并。"""
    # 若命中黑名单词对，则该候选不参与合并
    if not LEXICAL_ANTONYM_BLACKLIST:
        return False, ""

    new_text = normalize_for_match(
        f"{display_name(rec.canonical_element)} {preferred_name_en(rec.canonical_element)} "
        f"{rec.canonical_element.definition} {' '.join(rec.aliases)}"
    )
    old_text = normalize_for_match(f"{old_item.name} {old_item.label} {old_item.comment}")

    for a, b in LEXICAL_ANTONYM_BLACKLIST:
        ka = normalize_for_match(a)
        kb = normalize_for_match(b)
        if (ka in new_text and kb in old_text) or (kb in new_text and ka in old_text):
            return True, f"blacklist_hit:{a}<->{b}"
    return False, ""


def _extract_text_from_content(raw_content: Any) -> str:
    """兼容网关返回格式，统一提取文本正文。"""
    # 兼容网关返回格式，抽取文本正文
    def get_field(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        parts: List[str] = []
        for item in raw_content:
            part = get_field(item, "text")
            if part:
                parts.append(str(part).strip())
        return "\n".join([p for p in parts if p]).strip()
    if raw_content is None:
        return ""
    return str(raw_content).strip()


def extract_json_string(text: str) -> str:
    """从模型文本中截取首个 JSON 对象。"""
    # 清理包裹标记并截取首个 JSON 对象
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    if clean.startswith("{") and clean.endswith("}"):
        return clean
    first = clean.find("{")
    last = clean.rfind("}")
    if first != -1 and last != -1 and last > first:
        return clean[first:last + 1]
    return clean


def _extract_string_field_with_delimiters(text: str, key: str, next_keys: List[str]) -> Optional[str]:
    """在非严格 JSON 文本里按边界提取目标字符串字段。"""
    # 在“可能不合法的 JSON 文本”里提取字符串字段，按后续字段名定位结束边界
    start_match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text, flags=re.IGNORECASE)
    if not start_match:
        return None
    start = start_match.end()
    suffix = text[start:]

    end_positions: List[int] = []
    for next_key in next_keys:
        end_match = re.search(
            rf'"\s*,\s*"{re.escape(next_key)}"\s*:',
            suffix,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if end_match:
            end_positions.append(start + end_match.start())

    tail_match = re.search(r'"\s*}', suffix, flags=re.DOTALL)
    if tail_match:
        end_positions.append(start + tail_match.start())

    if not end_positions:
        return None

    end = min(end_positions)
    value = text[start:end]
    return value.replace('\\"', '"').replace("\\n", "\n").strip()


def recover_arbitration_payload_from_malformed_json(text: str) -> Optional[Dict[str, Any]]:
    """当仲裁 JSON 解析失败时，兜底恢复关键字段。"""
    # 当 json.loads 失败时，尝试从文本里兜底提取关键字段
    decision_match = re.search(r'"decision"\s*:\s*"(MERGE|KEEP_NEW)"', text, flags=re.IGNORECASE)
    if not decision_match:
        return None

    decision = decision_match.group(1).upper()

    reason = _extract_string_field_with_delimiters(
        text,
        "reason",
        ["target_old_name", "confidence", "decision"],
    )
    if not reason:
        reason = _extract_string_field_with_delimiters(
            text,
            "reasoning",
            ["target_old_name", "confidence", "decision"],
        )

    target_old_name: Optional[str] = None
    target_match = re.search(
        r'"target_old_name"\s*:\s*(null|"[^"]*")',
        text,
        flags=re.IGNORECASE,
    )
    if target_match:
        raw_target = target_match.group(1).strip()
        if raw_target.lower() != "null":
            target_old_name = raw_target.strip('"')

    confidence: float = float(DEFAULT_LLM_CONFIDENCE)
    confidence_match = re.search(
        r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        text,
        flags=re.IGNORECASE,
    )
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
        except (TypeError, ValueError):
            confidence = float(DEFAULT_LLM_CONFIDENCE)

    return {
        "decision": decision,
        "reason": reason or "",
        "target_old_name": target_old_name,
        "confidence": confidence,
    }


def normalize_arbitration_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """归一化仲裁字段差异，并补齐默认值。"""
    # 兼容网关/模型返回的字段差异，并补齐必填键
    reason = (
        payload.get("reason")
        or payload.get("reasoning")
        or payload.get("rationale")
        or payload.get("explanation")
        or ""
    )
    reason = str(reason).strip()
    if not reason:
        reason = "No explicit reason provided by model."

    confidence_raw = payload.get("confidence", DEFAULT_LLM_CONFIDENCE)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = float(DEFAULT_LLM_CONFIDENCE)
    confidence = max(0.0, min(1.0, confidence))

    decision = str(payload.get("decision", "")).strip().upper()
    target_old_name = payload.get("target_old_name")
    if decision == "KEEP_NEW":
        target_old_name = None

    return {
        "decision": decision,
        "reason": reason,
        "target_old_name": target_old_name,
        "confidence": confidence,
    }


def call_claude_arbitration(
    client: OpenAI,
    rec: CanonicalInputRecord,
    candidate_payload: List[Dict[str, Any]],
    token_usage_counter: Dict[str, int],
) -> ArbitrationResponse:
    """调用 LLM 对灰区候选执行外部对齐仲裁。"""
    # 构造灰区仲裁输入
    new_payload = {
        "cluster_id": rec.cluster_id,
        "element_type": rec.element_type,
        "name_zh": preferred_name_zh(rec.canonical_element),
        "name_en": preferred_name_en(rec.canonical_element),
        "name_raw": rec.canonical_element.name_raw,
        "definition": rec.canonical_element.definition,
        "aliases": rec.aliases,
        "merged_definitions": rec.merged_definitions[:5],
        "mapping_suggestion": rec.canonical_element.mapping_suggestion.model_dump(),
    }

    user_prompt = USER_PROMPT_TEMPLATE.format(
        new_concept_json=json.dumps(new_payload, ensure_ascii=False, indent=2),
        candidate_json=json.dumps(candidate_payload, ensure_ascii=False, indent=2),
    )

    # 按 Agent1 的模式构造请求参数
    request_kwargs: Dict[str, Any] = {
        "model": CLAUDE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if TOP_P is not None:
        request_kwargs["top_p"] = TOP_P

    # 调用大模型
    response = client.chat.completions.create(**request_kwargs)
    usage_obj = getattr(response, "usage", None)
    input_tokens, output_tokens = extract_usage_tokens(usage_obj)
    token_usage_counter["input_tokens"] += input_tokens
    token_usage_counter["output_tokens"] += output_tokens
    choices = response.choices or []
    message = choices[0].message if choices else None
    raw_content = _extract_text_from_content(message.content if message else "")
    json_text = extract_json_string(raw_content)

    # 记录原始响应，方便实验复盘
    append_jsonl(
        OUTPUT_RAW_API_LOG,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cluster_id": rec.cluster_id,
            "new_name_zh": preferred_name_zh(rec.canonical_element),
            "new_name_en": preferred_name_en(rec.canonical_element),
            "candidate_old_names": [c.get("old_name") for c in candidate_payload],
            "raw_content": raw_content,
            "usage": to_jsonable(usage_obj),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": choices[0].finish_reason if choices else "",
        },
    )

    try:
        payload = json.loads(json_text)
        if not isinstance(payload, dict):
            raise ValueError("LLM JSON payload must be an object.")
    except json.JSONDecodeError:
        recovered = recover_arbitration_payload_from_malformed_json(json_text)
        if recovered is None:
            raise
        payload = recovered

    normalized_payload = normalize_arbitration_payload(payload)
    result = ArbitrationResponse.model_validate(normalized_payload)
    candidate_names = {c.get("old_name", "") for c in candidate_payload}
    if result.decision == "MERGE" and result.target_old_name not in candidate_names:
        raise ValueError("LLM returned target_old_name outside provided candidates.")
    return result


def build_class_resolution_map(
    class_records: List[CanonicalInputRecord],
    class_decisions: List[Dict[str, Any]],
    existing_class_names: List[str],
) -> Dict[str, Any]:
    """根据 Class 对齐结果生成‘一个类最终对应谁’的映射表。"""
    rec_by_cluster = {rec.cluster_id: rec for rec in class_records}
    lookup: Dict[str, Dict[str, Any]] = {}
    entries: List[Dict[str, Any]] = []

    for decision in class_decisions:
        cluster_id = decision["cluster_id"]
        rec = rec_by_cluster.get(cluster_id)
        if rec is None:
            continue

        if decision["decision"] == "MERGE":
            canonical_anchor_name = ensure_anchor_name(decision.get("target_old_name"))
            source_decision = "MERGE"
        else:
            canonical_anchor_name = ensure_anchor_name(preferred_name_en(rec.canonical_element))
            source_decision = "KEEP_NEW"

        if not canonical_anchor_name:
            continue

        keys = class_lookup_keys(rec)
        entry = {
            "cluster_id": cluster_id,
            "name_zh": preferred_name_zh(rec.canonical_element),
            "name_en": preferred_name_en(rec.canonical_element),
            "aliases": rec.aliases,
            "source_decision": source_decision,
            "canonical_anchor_name": canonical_anchor_name,
            "lookup_keys": keys,
        }
        entries.append(entry)

        for key in keys:
            lookup[key] = {
                "canonical_anchor_name": canonical_anchor_name,
                "source_decision": source_decision,
                "source_cluster_id": cluster_id,
            }

    standard_class_keys: set[str] = set()
    for name in existing_class_names:
        anchor_name = ensure_anchor_name(name)
        if not anchor_name:
            continue
        standard_class_keys.add(normalize_for_match(anchor_name))
        standard_class_keys.add(normalize_for_match(anchor_name[1:]))
    for entry in entries:
        anchor_name = entry["canonical_anchor_name"]
        standard_class_keys.add(normalize_for_match(anchor_name))
        standard_class_keys.add(normalize_for_match(anchor_name[1:]))

    return {
        "entries": entries,
        "lookup": lookup,
        "standard_class_keys": sorted(k for k in standard_class_keys if k),
    }


def resolve_class_reference(
    value: Optional[str],
    resolution_lookup: Dict[str, Dict[str, Any]],
    standard_class_keys: set[str],
) -> Tuple[Optional[str], bool]:
    """解析属性中的类引用；返回（重写后的值，是否未解析）。"""
    text = str(value or "").strip()
    if not text:
        return None, False

    key = normalize_for_match(text)
    anchor_name = ensure_anchor_name(text)
    anchor_key = normalize_for_match(anchor_name)

    if key in resolution_lookup:
        return resolution_lookup[key]["canonical_anchor_name"], False
    if anchor_key in resolution_lookup:
        return resolution_lookup[anchor_key]["canonical_anchor_name"], False

    if key in standard_class_keys or anchor_key in standard_class_keys:
        return anchor_name, False

    return text, True


def rewrite_property_dependencies(
    rec: CanonicalInputRecord,
    resolution_lookup: Dict[str, Dict[str, Any]],
    standard_class_keys: set[str],
) -> Tuple[CanonicalInputRecord, bool]:
    """在属性进入外部对齐前，先用类最终结果重写 domain/range。"""
    if rec.element_type not in {"ObjectProperty", "DatatypeProperty"}:
        return rec, False

    new_rec = rec.model_copy(deep=True)
    ms = new_rec.canonical_element.mapping_suggestion
    dependency_unresolved = False

    new_domain, domain_unresolved = resolve_class_reference(ms.suggested_domain, resolution_lookup, standard_class_keys)
    ms.suggested_domain = new_domain
    dependency_unresolved = dependency_unresolved or domain_unresolved

    if rec.element_type == "ObjectProperty":
        new_range, range_unresolved = resolve_class_reference(ms.suggested_range, resolution_lookup, standard_class_keys)
        ms.suggested_range = new_range
        dependency_unresolved = dependency_unresolved or range_unresolved

    return new_rec, dependency_unresolved


def split_keep_new_records(
    keep_new: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """将 KEEP_NEW 结果拆分为 Class 与属性两组稳定文件。"""
    keep_new_classes: List[Dict[str, Any]] = []
    keep_new_properties: List[Dict[str, Any]] = []

    for record in keep_new:
        element_type = str(record.get("element_type", "")).strip()
        if element_type == "Class":
            keep_new_classes.append(record)
        elif element_type in {"ObjectProperty", "DatatypeProperty"}:
            keep_new_properties.append(record)

    return keep_new_classes, keep_new_properties


def build_keep_new_class_index(keep_new_classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构建 04b 使用的轻量 Class 索引，不承载决策语义。"""
    index_rows: List[Dict[str, Any]] = []

    for record in keep_new_classes:
        canonical_record = record.get("canonical_record") or {}
        canonical_element = canonical_record.get("canonical_element") or {}

        name_zh = str(canonical_element.get("name_zh") or "").strip()
        name_en = normalize_name_en_by_type("Class", canonical_element.get("name_en"))
        name_raw = str(canonical_element.get("name_raw") or "").strip()
        aliases = canonical_record.get("aliases") or []
        aliases = [str(alias).strip() for alias in aliases if str(alias).strip()]

        canonical_anchor_name = ensure_anchor_name(name_en or name_zh or name_raw)
        normalized_class_name = normalize_for_match(canonical_anchor_name or name_en or name_zh or name_raw)

        index_row = {
            "cluster_id": record.get("cluster_id", ""),
            "name_zh": name_zh,
            "name_en": name_en,
            "normalized_class_name": normalized_class_name,
            "aliases": aliases,
        }
        if canonical_anchor_name:
            index_row["canonical_anchor_name"] = canonical_anchor_name
        index_rows.append(index_row)

    return index_rows


def write_split_keep_new_outputs(
    keep_new: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """将 KEEP_NEW 拆分写盘，并返回拆分后的结果集合。"""
    keep_new_classes, keep_new_properties = split_keep_new_records(keep_new)
    keep_new_class_index = build_keep_new_class_index(keep_new_classes)

    write_json(OUTPUT_KEEP_NEW_CLASSES, keep_new_classes)
    write_json(OUTPUT_KEEP_NEW_PROPERTIES, keep_new_properties)
    write_json(OUTPUT_KEEP_NEW_CLASS_INDEX, keep_new_class_index)
    return keep_new_classes, keep_new_properties, keep_new_class_index


def build_external_alignment_stats(
    decisions: List[Dict[str, Any]],
    keep_new: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """生成外部对齐统计摘要，不影响 keep_new/merged 主记录结构。"""
    element_types = ["Class", "ObjectProperty", "DatatypeProperty"]

    per_type_decision_counts: Dict[str, Dict[str, int]] = {}
    best_similarity_by_type: Dict[str, List[float]] = {tp: [] for tp in element_types}
    best_similarity_all: List[float] = []
    for tp in element_types:
        tp_decisions = [rec for rec in decisions if rec.get("element_type") == tp]
        per_type_decision_counts[tp] = {
            "KEEP_NEW": sum(1 for rec in tp_decisions if rec.get("decision") == "KEEP_NEW"),
            "MERGE": sum(1 for rec in tp_decisions if rec.get("decision") == "MERGE"),
        }
        for rec in tp_decisions:
            sim = rec.get("best_similarity")
            if isinstance(sim, (int, float)):
                best_similarity_by_type[tp].append(float(sim))
                best_similarity_all.append(float(sim))

    keep_new_classes = [rec for rec in keep_new if rec.get("element_type") == "Class"]
    keep_new_object_properties = [rec for rec in keep_new if rec.get("element_type") == "ObjectProperty"]
    keep_new_datatype_properties = [rec for rec in keep_new if rec.get("element_type") == "DatatypeProperty"]

    class_unsure_parent_total = 0
    for rec in keep_new_classes:
        mapping = (((rec.get("canonical_record") or {}).get("canonical_element") or {}).get("mapping_suggestion") or {})
        if normalize_for_match(mapping.get("suggested_parent_class")) in {"unsure", ":unsure"}:
            class_unsure_parent_total += 1

    object_property_dependency_unresolved_total = sum(
        1 for rec in keep_new_object_properties if bool(rec.get("dependency_unresolved"))
    )
    datatype_property_dependency_unresolved_total = sum(
        1 for rec in keep_new_datatype_properties if bool(rec.get("dependency_unresolved"))
    )

    def object_property_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
        domain_missing_total = 0
        range_missing_total = 0
        both_present_total = 0
        for rec in records:
            mapping = (((rec.get("canonical_record") or {}).get("canonical_element") or {}).get("mapping_suggestion") or {})
            domain_ok = non_unsure_text(mapping.get("suggested_domain"))
            range_ok = non_unsure_text(mapping.get("suggested_range"))
            if not domain_ok:
                domain_missing_total += 1
            if not range_ok:
                range_missing_total += 1
            if domain_ok and range_ok:
                both_present_total += 1
        return {
            "domain_missing_total": domain_missing_total,
            "range_missing_total": range_missing_total,
            "domain_range_both_present_total": both_present_total,
        }

    def datatype_property_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
        domain_missing_total = 0
        xsd_range_missing_total = 0
        both_present_total = 0
        for rec in records:
            mapping = (((rec.get("canonical_record") or {}).get("canonical_element") or {}).get("mapping_suggestion") or {})
            domain_ok = non_unsure_text(mapping.get("suggested_domain"))
            xsd_ok = non_unsure_text(mapping.get("suggested_xsd_range"))
            if not domain_ok:
                domain_missing_total += 1
            if not xsd_ok:
                xsd_range_missing_total += 1
            if domain_ok and xsd_ok:
                both_present_total += 1
        return {
            "domain_missing_total": domain_missing_total,
            "xsd_range_missing_total": xsd_range_missing_total,
            "domain_xsd_range_both_present_total": both_present_total,
        }

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "per_type_decision_counts": per_type_decision_counts,
        "external_similarity_stats": {
            "overall": summarize_similarity(best_similarity_all),
            "per_type": {
                tp: summarize_similarity(best_similarity_by_type[tp]) for tp in element_types
            },
        },
        "uncertainty_stats": {
            "unsure_parent_total": class_unsure_parent_total,
            "class_unsure_parent_total": class_unsure_parent_total,
            "object_property_dependency_unresolved_total": object_property_dependency_unresolved_total,
            "datatype_property_dependency_unresolved_total": datatype_property_dependency_unresolved_total,
            "dependency_unresolved_total": (
                object_property_dependency_unresolved_total + datatype_property_dependency_unresolved_total
            ),
        },
        "property_structure_stats": {
            "ObjectProperty": object_property_counts(keep_new_object_properties),
            "DatatypeProperty": datatype_property_counts(keep_new_datatype_properties),
        },
    }


def align_records_for_type(
    tp: str,
    rec_bucket: List[CanonicalInputRecord],
    dependency_flags: Dict[str, bool],
    old_bucket: List[FlattenedVocabItem],
    old_emb: Optional[np.ndarray],
    model: Any,
    client: Optional[OpenAI],
    token_usage_counter: Dict[str, int],
    total_work: int,
    progress_start: int,
) -> Dict[str, Any]:
    """对单一 element_type 执行现有外部对齐逻辑。"""
    decisions: List[Dict[str, Any]] = []
    keep_new: List[Dict[str, Any]] = []
    merged: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    route_counter: Counter = Counter()
    decision_counter: Counter = Counter()
    llm_stats: Counter = Counter()
    llm_elapsed_total = 0.0
    llm_elapsed_count = 0

    print(f"\n[类型处理] {tp} | new={len(rec_bucket)} | old={len(old_bucket)}")

    new_texts = [build_new_embedding_text(r) for r in rec_bucket]
    new_emb = model.encode(
        new_texts,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=EMBED_NORMALIZE,
        show_progress_bar=False,
    )
    sim_matrix = cosine_similarity_matrix(new_emb, old_emb, EMBED_NORMALIZE) if old_emb is not None else np.zeros((len(rec_bucket), 0), dtype=float)

    for local_idx, rec in enumerate(rec_bucket):
        progress = progress_start + local_idx + 1
        ce = rec.canonical_element
        print(f"[{progress}/{total_work}] {rec.cluster_id} | {tp} | {display_name(ce)}")

        candidate_scores: List[CandidateScore] = []
        best_candidate: Optional[CandidateScore] = None

        if not old_bucket:
            route = "no_old_vocab_for_type"
            decision = "KEEP_NEW"
            final_reason = "No existing vocabulary of same type."
            best_sim = None
            final_target = None
        else:
            sims = sim_matrix[local_idx]
            sorted_idx = np.argsort(-sims)
            top_idx = sorted_idx[: min(TOP_K, len(sorted_idx))]

            for rank, pos in enumerate(top_idx, start=1):
                old_item = old_bucket[int(pos)]
                blocked, blocked_reason = check_blacklist_block(rec, old_item)
                candidate_scores.append(
                    CandidateScore(
                        rank=rank,
                        old_name=old_item.name,
                        old_type=old_item.type,
                        similarity=round(float(sims[int(pos)]), 6),
                        blocked_by_blacklist=blocked,
                        blacklist_reason=blocked_reason,
                        label=old_item.label,
                        comment=old_item.comment,
                    )
                )

            for pos in sorted_idx:
                old_item = old_bucket[int(pos)]
                blocked, blocked_reason = check_blacklist_block(rec, old_item)
                if blocked:
                    continue
                best_candidate = CandidateScore(
                    rank=1,
                    old_name=old_item.name,
                    old_type=old_item.type,
                    similarity=round(float(sims[int(pos)]), 6),
                    blocked_by_blacklist=False,
                    blacklist_reason=blocked_reason,
                    label=old_item.label,
                    comment=old_item.comment,
                )
                break

            if best_candidate is None:
                route = "all_candidates_blacklisted"
                decision = "KEEP_NEW"
                final_reason = "All candidates blocked by blacklist."
                best_sim = None
                final_target = None
            else:
                best_sim = float(best_candidate.similarity)
                final_target = best_candidate.old_name

                if best_sim < LOW_SIM_THRESHOLD:
                    route = "auto_low_similarity"
                    decision = "KEEP_NEW"
                    final_reason = f"Best similarity {best_sim:.6f} < {LOW_SIM_THRESHOLD:.2f}."
                    final_target = None
                elif best_sim > HIGH_SIM_THRESHOLD:
                    route = "auto_high_similarity"
                    decision = "MERGE"
                    final_reason = f"Best similarity {best_sim:.6f} > {HIGH_SIM_THRESHOLD:.2f}."
                else:
                    route = "llm_gray_zone"
                    llm_stats["gray_zone_total"] += 1

                    if not USE_LLM_ARBITRATION:
                        decision = "KEEP_NEW"
                        final_reason = "LLM arbitration disabled; conservative keep."
                        final_target = None
                        route = "llm_disabled_fallback"
                    else:
                        llm_stats["llm_call_total"] += 1
                        last_error = ""
                        llm_result: Optional[ArbitrationResponse] = None
                        candidate_payload = [c.model_dump() for c in candidate_scores if not c.blocked_by_blacklist]
                        if not candidate_payload and best_candidate is not None:
                            candidate_payload = [best_candidate.model_dump()]

                        arbitration_started = time.time()
                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                assert client is not None
                                llm_result = call_claude_arbitration(client, rec, candidate_payload, token_usage_counter)
                                break
                            except Exception as e:
                                last_error = f"{type(e).__name__}: {e}"
                                if attempt < MAX_RETRIES:
                                    time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
                        arbitration_elapsed = round(time.time() - arbitration_started, 3)
                        llm_elapsed_total += arbitration_elapsed
                        llm_elapsed_count += 1

                        if llm_result is None:
                            decision = "KEEP_NEW"
                            final_reason = f"LLM arbitration failed; conservative keep. last_error={last_error}"
                            final_target = None
                            route = "llm_error_fallback"
                            errors.append(
                                {
                                    "phase": "llm_arbitration",
                                    "cluster_id": rec.cluster_id,
                                    "error_type": "llm_failed",
                                    "error": last_error,
                                    "new_name_zh": preferred_name_zh(ce),
                                    "new_name_en": preferred_name_en(ce),
                                    "best_candidate": best_candidate.model_dump() if best_candidate else None,
                                }
                            )
                        else:
                            decision = llm_result.decision
                            final_reason = f"{llm_result.reason} (confidence={llm_result.confidence:.2f})"
                            if decision == "MERGE":
                                final_target = llm_result.target_old_name
                                llm_stats["llm_merge_total"] += 1
                            else:
                                final_target = None
                                llm_stats["llm_keep_total"] += 1

        route_counter[route] += 1
        decision_counter[decision] += 1
        name_zh = preferred_name_zh(ce)
        name_en = preferred_name_en(ce)
        name_raw = (ce.name_raw or "").strip()
        dependency_unresolved = dependency_flags.get(rec.cluster_id, False)
        candidate_super_property = (
            ce.mapping_suggestion.candidate_super_property if rec.element_type == "ObjectProperty" else None
        )

        decision_record = {
            "cluster_id": rec.cluster_id,
            "element_type": rec.element_type,
            "name_raw": name_raw,
            "name_zh": name_zh,
            "name_en": name_en,
            "internal_status": rec.internal_status,
            "decision_route": route,
            "decision": decision,
            "target_old_name": final_target,
            "best_similarity": best_sim,
            "reason": final_reason,
            "top_k_candidates": [c.model_dump() for c in candidate_scores],
        }
        if rec.element_type in {"ObjectProperty", "DatatypeProperty"}:
            decision_record["dependency_unresolved"] = dependency_unresolved
        if rec.element_type == "ObjectProperty":
            decision_record["candidate_super_property"] = None if decision == "MERGE" else candidate_super_property
        decisions.append(decision_record)

        if decision == "MERGE":
            merged_record = {
                "cluster_id": rec.cluster_id,
                "element_type": rec.element_type,
                "name_raw": name_raw,
                "name_zh": name_zh,
                "name_en": name_en,
                "target_old_name": final_target,
                "best_similarity": best_sim,
                "decision_route": route,
                "reason": final_reason,
                "aliases": rec.aliases,
            }
            if rec.element_type in {"ObjectProperty", "DatatypeProperty"}:
                merged_record["dependency_unresolved"] = dependency_unresolved
            if rec.element_type == "ObjectProperty":
                merged_record["candidate_super_property"] = None
            merged.append(merged_record)
        else:
            keep_record = {
                "cluster_id": rec.cluster_id,
                "element_type": rec.element_type,
                "canonical_record": rec.model_dump(),
                "external_alignment": {
                    "decision_route": route,
                    "decision": decision,
                    "target_old_name": final_target,
                    "best_similarity": best_sim,
                    "reason": final_reason,
                },
            }
            if rec.element_type in {"ObjectProperty", "DatatypeProperty"}:
                keep_record["dependency_unresolved"] = dependency_unresolved
            if rec.element_type == "ObjectProperty":
                keep_record["candidate_super_property"] = candidate_super_property
            keep_new.append(keep_record)

    return {
        "decisions": decisions,
        "keep_new": keep_new,
        "merged": merged,
        "errors": errors,
        "route_counter": route_counter,
        "decision_counter": decision_counter,
        "llm_stats": llm_stats,
        "llm_elapsed_total": llm_elapsed_total,
        "llm_elapsed_count": llm_elapsed_count,
        "processed_count": len(rec_bucket),
    }


def main() -> None:
    """主流程：读取 canonical 和词表，完成外部对齐并写出结果。"""
    # 运行入口：加载输入、做外部对齐、输出多份结果文件
    started = time.time()
    log_path = setup_console_log("phase2_03_external_alignment")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    required = [INPUT_CANONICAL_PATH, INPUT_VOCAB_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取输入数据
    raw_canonical = load_json_list(INPUT_CANONICAL_PATH)
    raw_vocab = load_json_list(INPUT_VOCAB_PATH)

    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    selected_raw = raw_canonical[START_INDEX:end_index]

    print("=" * 80)
    print("[运行启动] Agent2 Step2 外部全局对齐启动")
    print(f"[输入信息] canonical输入={INPUT_CANONICAL_PATH}")
    print(f"[输入信息] 词表输入={INPUT_VOCAB_PATH}")
    print(f"[输入信息] 本次处理={len(selected_raw)} (START_INDEX={START_INDEX}, MAX_ITEMS={MAX_ITEMS})")
    print(f"[参数配置] 阈值: low<{LOW_SIM_THRESHOLD}, high>{HIGH_SIM_THRESHOLD}")
    print(f"[参数配置] top_k={TOP_K} | 启用LLM仲裁={USE_LLM_ARBITRATION}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print("=" * 80)

    if USE_LLM_ARBITRATION and not API_KEY:
        raise RuntimeError("Missing API Key. Please fill API_KEY in the script config before external alignment.")

    canonical_records: List[CanonicalInputRecord] = []
    vocab_items: List[FlattenedVocabItem] = []
    errors: List[Dict[str, Any]] = []

    # 校验 Step1 产物结构
    for idx, raw in enumerate(selected_raw):
        try:
            rec = CanonicalInputRecord.model_validate(raw)
            canonical_records.append(rec)
        except ValidationError as e:
            errors.append(
                {
                    "phase": "validate_canonical",
                    "index": START_INDEX + idx,
                    "error_type": "validation_error",
                    "error": str(e),
                    "raw": raw,
                }
            )

    # 校验旧词表结构
    for idx, raw in enumerate(raw_vocab):
        try:
            vocab_items.append(FlattenedVocabItem.model_validate(raw))
        except ValidationError as e:
            errors.append(
                {
                    "phase": "validate_vocab",
                    "index": idx,
                    "error_type": "validation_error",
                    "error": str(e),
                    "raw": raw,
                }
            )

    write_json(OUTPUT_ERRORS, errors)
    print(
        f"[校验结果] canonical有效={len(canonical_records)} | canonical无效={len(selected_raw) - len(canonical_records)} "
        f"| vocab有效={len(vocab_items)} | vocab无效={len(raw_vocab) - len(vocab_items)}"
    )

    if not canonical_records:
        write_json(OUTPUT_DECISIONS, [])
        write_json(OUTPUT_KEEP_NEW, [])
        write_json(OUTPUT_MERGED, [])
        write_json(OUTPUT_KEEP_NEW_CLASSES, [])
        write_json(OUTPUT_KEEP_NEW_PROPERTIES, [])
        write_json(OUTPUT_KEEP_NEW_CLASS_INDEX, [])
        write_json(OUTPUT_ALIGNMENT_STATS, {})
        write_json(
            OUTPUT_CLASS_RESOLUTION_MAP,
            {"entries": [], "lookup": {}, "standard_class_keys": []},
        )
        manifest = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "no_valid_canonical_records",
            "input_canonical_file": str(INPUT_CANONICAL_PATH),
            "input_vocab_file": str(INPUT_VOCAB_PATH),
            "alignment_stats_file": str(OUTPUT_ALIGNMENT_STATS),
            "selected_total": len(selected_raw),
            "valid_canonical_total": 0,
            "valid_vocab_total": len(vocab_items),
            "invalid_total": len(errors),
            "elapsed_sec": round(time.time() - started, 3),
        }
        write_json(OUTPUT_MANIFEST, manifest)
        print("[运行结束] 没有可处理的有效 canonical 记录。")
        return

    SentenceTransformer = ensure_sentence_transformers()
    model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    print(f"[向量模型] model={EMBED_MODEL_NAME} | device={EMBED_DEVICE} | batch_size={EMBED_BATCH_SIZE}")

    vocab_by_type: Dict[str, List[FlattenedVocabItem]] = defaultdict(list)
    for item in vocab_items:
        vocab_by_type[item.type].append(item)

    canonical_by_type: Dict[str, List[CanonicalInputRecord]] = defaultdict(list)
    for rec in canonical_records:
        canonical_by_type[rec.element_type].append(rec)

    # 按类型预编码旧词表文本，避免在循环内重复编码
    old_emb_by_type: Dict[str, np.ndarray] = {}
    for tp, bucket in vocab_by_type.items():
        if not bucket:
            continue
        texts = [build_old_embedding_text(x) for x in bucket]
        old_emb_by_type[tp] = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=EMBED_NORMALIZE,
            show_progress_bar=False,
        )

    decisions: List[Dict[str, Any]] = []
    keep_new: List[Dict[str, Any]] = []
    merged: List[Dict[str, Any]] = []

    route_counter: Counter = Counter()
    decision_counter: Counter = Counter()
    llm_stats: Counter = Counter()
    token_usage_counter: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    llm_elapsed_total = 0.0
    llm_elapsed_count = 0

    client: Optional[OpenAI] = None
    if USE_LLM_ARBITRATION:
        client = OpenAI(
            api_key=API_KEY,
            base_url=OPENAI_COMPAT_BASE_URL,
            timeout=API_TIMEOUT_SECONDS,
        )

    total_work = len(canonical_records)
    progress = 0
    class_old_bucket = vocab_by_type.get("Class", [])
    class_old_emb = old_emb_by_type.get("Class")
    # 兜底：即使本批次没有新 Class，也先装入旧词表中的标准类名，
    # 避免属性仅引用旧本体已有类时被误判为 dependency_unresolved。
    class_resolution_map_payload = build_class_resolution_map(
        [],
        [],
        [item.name for item in class_old_bucket],
    )

    # 第一阶段：先处理 Class，对齐结果将作为属性 domain/range 的统一来源
    class_bucket = canonical_by_type.get("Class", [])
    if class_bucket:
        class_result = align_records_for_type(
            "Class",
            class_bucket,
            {},
            class_old_bucket,
            class_old_emb,
            model,
            client,
            token_usage_counter,
            total_work,
            progress,
        )
        progress += class_result["processed_count"]
        decisions.extend(class_result["decisions"])
        keep_new.extend(class_result["keep_new"])
        merged.extend(class_result["merged"])
        errors.extend(class_result["errors"])
        route_counter.update(class_result["route_counter"])
        decision_counter.update(class_result["decision_counter"])
        llm_stats.update(class_result["llm_stats"])
        llm_elapsed_total += class_result["llm_elapsed_total"]
        llm_elapsed_count += class_result["llm_elapsed_count"]

        class_resolution_map_payload = build_class_resolution_map(
            class_bucket,
            class_result["decisions"],
            [item.name for item in class_old_bucket],
        )
        write_json(OUTPUT_DECISIONS, decisions)
        write_json(OUTPUT_KEEP_NEW, keep_new)
        write_json(OUTPUT_MERGED, merged)
        write_json(OUTPUT_ERRORS, errors)
        write_split_keep_new_outputs(keep_new)

    write_json(OUTPUT_CLASS_RESOLUTION_MAP, class_resolution_map_payload)

    resolution_lookup = class_resolution_map_payload.get("lookup", {})
    standard_class_keys = set(class_resolution_map_payload.get("standard_class_keys", []))

    # 第二阶段：属性先继承 Class 最终命名结果，再继续走现有外部对齐
    for tp in ["ObjectProperty", "DatatypeProperty"]:
        rec_bucket = canonical_by_type.get(tp, [])
        if not rec_bucket:
            continue
        old_bucket = vocab_by_type.get(tp, [])
        old_emb = old_emb_by_type.get(tp)

        rewritten_bucket: List[CanonicalInputRecord] = []
        dependency_flags: Dict[str, bool] = {}
        unresolved_count = 0
        for rec in rec_bucket:
            new_rec, dependency_unresolved = rewrite_property_dependencies(rec, resolution_lookup, standard_class_keys)
            rewritten_bucket.append(new_rec)
            dependency_flags[new_rec.cluster_id] = dependency_unresolved
            if dependency_unresolved:
                unresolved_count += 1

        print(f"[依赖归一化] {tp} | dependency_unresolved={unresolved_count}/{len(rewritten_bucket)}")

        attr_result = align_records_for_type(
            tp,
            rewritten_bucket,
            dependency_flags,
            old_bucket,
            old_emb,
            model,
            client,
            token_usage_counter,
            total_work,
            progress,
        )
        progress += attr_result["processed_count"]
        decisions.extend(attr_result["decisions"])
        keep_new.extend(attr_result["keep_new"])
        merged.extend(attr_result["merged"])
        errors.extend(attr_result["errors"])
        route_counter.update(attr_result["route_counter"])
        decision_counter.update(attr_result["decision_counter"])
        llm_stats.update(attr_result["llm_stats"])
        llm_elapsed_total += attr_result["llm_elapsed_total"]
        llm_elapsed_count += attr_result["llm_elapsed_count"]

        write_json(OUTPUT_DECISIONS, decisions)
        write_json(OUTPUT_KEEP_NEW, keep_new)
        write_json(OUTPUT_MERGED, merged)
        write_json(OUTPUT_ERRORS, errors)
        write_split_keep_new_outputs(keep_new)

    elapsed = round(time.time() - started, 3)
    keep_new_classes, keep_new_properties, keep_new_class_index = write_split_keep_new_outputs(keep_new)
    alignment_stats = build_external_alignment_stats(decisions, keep_new)
    llm_call_total = int(llm_stats.get("llm_call_total", 0))
    token_report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gray_zone_total": int(llm_stats.get("gray_zone_total", 0)),
        "llm_call_total": llm_call_total,
        "llm_merge_total": int(llm_stats.get("llm_merge_total", 0)),
        "llm_keep_total": int(llm_stats.get("llm_keep_total", 0)),
        "input_tokens": int(token_usage_counter.get("input_tokens", 0)),
        "output_tokens": int(token_usage_counter.get("output_tokens", 0)),
        "avg_input_tokens_per_llm_call": round(token_usage_counter.get("input_tokens", 0) / llm_call_total, 3) if llm_call_total > 0 else 0.0,
        "avg_output_tokens_per_llm_call": round(token_usage_counter.get("output_tokens", 0) / llm_call_total, 3) if llm_call_total > 0 else 0.0,
        "llm_elapsed_total_sec": round(llm_elapsed_total, 3),
        "avg_llm_elapsed_sec": round(llm_elapsed_total / llm_elapsed_count, 3) if llm_elapsed_count > 0 else 0.0,
    }
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_canonical_file": str(INPUT_CANONICAL_PATH),
        "input_vocab_file": str(INPUT_VOCAB_PATH),
        "output_dir": str(OUTPUT_DIR),
        "alignment": {
            "top_k": TOP_K,
            "low_sim_threshold": LOW_SIM_THRESHOLD,
            "high_sim_threshold": HIGH_SIM_THRESHOLD,
            "blacklist_size": len(LEXICAL_ANTONYM_BLACKLIST),
        },
        "embedding": {
            "model_name": EMBED_MODEL_NAME,
            "device": EMBED_DEVICE,
            "batch_size": EMBED_BATCH_SIZE,
            "normalize_embeddings": EMBED_NORMALIZE,
        },
        "llm": {
            "enabled": USE_LLM_ARBITRATION,
            "model": CLAUDE_MODEL,
            "base_url": OPENAI_COMPAT_BASE_URL,
            "api_timeout_seconds": API_TIMEOUT_SECONDS,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
        "run_range": {
            "start_index": START_INDEX,
            "max_items": MAX_ITEMS,
            "selected_total": len(selected_raw),
        },
        "counts": {
            "input_canonical_total": len(raw_canonical),
            "selected_canonical_total": len(selected_raw),
            "valid_canonical_total": len(canonical_records),
            "valid_vocab_total": len(vocab_items),
            "invalid_total": len(errors),
            "decision_total": len(decisions),
            "keep_new_total": len(keep_new),
            "keep_new_class_total": len(keep_new_classes),
            "keep_new_property_total": len(keep_new_properties),
            "keep_new_class_index_total": len(keep_new_class_index),
            "merged_total": len(merged),
        },
        "route_counts": dict(route_counter),
        "decision_counts": dict(decision_counter),
        "alignment_stats_file": str(OUTPUT_ALIGNMENT_STATS),
        "class_resolution_map_file": str(OUTPUT_CLASS_RESOLUTION_MAP),
        "class_resolution_entry_total": len(class_resolution_map_payload.get("entries", [])),
        "keep_new_classes_file": str(OUTPUT_KEEP_NEW_CLASSES),
        "keep_new_properties_file": str(OUTPUT_KEEP_NEW_PROPERTIES),
        "keep_new_class_index_file": str(OUTPUT_KEEP_NEW_CLASS_INDEX),
        "llm_stats": dict(llm_stats),
        "token_usage": token_report,
        "elapsed_sec": elapsed,
    }
    write_json(OUTPUT_MANIFEST, manifest)
    write_json(OUTPUT_TOKEN_REPORT, token_report)
    write_json(OUTPUT_ALIGNMENT_STATS, alignment_stats)

    print("\n" + "=" * 80)
    print("[运行结束] Agent2 Step2 外部全局对齐完成")
    print(
        f"[运行摘要] 决策总数={len(decisions)} | KEEP_NEW={len(keep_new)} | MERGE={len(merged)} | 错误={len(errors)}"
    )
    print(f"[性能统计] 总耗时={elapsed}s")
    print(
        f"[性能统计] LLM调用={token_report['llm_call_total']} | 输入token={token_report['input_tokens']} | "
        f"输出token={token_report['output_tokens']} | 平均LLM耗时={token_report['avg_llm_elapsed_sec']}s"
    )
    print(f"[关键输出] {OUTPUT_DECISIONS} -> 每条候选概念的最终对齐决策")
    print(f"[关键输出] {OUTPUT_KEEP_NEW} -> 判定为 KEEP_NEW 的新增概念集合")
    print(f"[关键输出] {OUTPUT_KEEP_NEW_CLASSES} -> KEEP_NEW 中的 Class 子集")
    print(f"[关键输出] {OUTPUT_KEEP_NEW_PROPERTIES} -> KEEP_NEW 中的属性子集")
    print(f"[关键输出] {OUTPUT_KEEP_NEW_CLASS_INDEX} -> 04b 使用的轻量 Class 索引")
    print(f"[关键输出] {OUTPUT_MERGED} -> 判定为 MERGE 的映射结果集合")
    print(f"[关键输出] {OUTPUT_ALIGNMENT_STATS} -> 外部对齐统计摘要")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要")
    print(f"[关键输出] {OUTPUT_TOKEN_REPORT} -> 灰区仲裁 token 汇总")
    print(f"[关键输出] {OUTPUT_CLASS_RESOLUTION_MAP} -> Class 最终归属映射表")
    print("=" * 80)


if __name__ == "__main__":
    main()
