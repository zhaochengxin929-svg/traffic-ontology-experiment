import atexit
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


# ==========================================
# 配置区（常改）
# ==========================================
# Claude 兼容网关 API Key，可直接填写
API_KEY = ""  # Claude 兼容接口密钥，可直接填写

# 本体定位使用的模型与 Native Messages 网关配置
CLAUDE_MODEL = "claude-opus-4-6"  # 本体定位阶段使用的模型名称
CLAUDE_NATIVE_BASE_URL = "https://api2.aigcbest.top"  # Claude Native Messages 网关地址
MESSAGES_PATH = "/v1/messages"  # Claude Native Messages 路径
ANTHROPIC_VERSION = "2023-06-01"  # Claude Native 接口版本
ANTHROPIC_BETA = ""  # 可选 beta header，默认留空
API_TIMEOUT_SECONDS = 120  # 单次 API 请求超时时间，单位秒

# 生成参数
TEMPERATURE = 0.1  # 生成温度，越低越稳定；本阶段建议保持低温
MAX_TOKENS = 2000  # 单条定位请求允许模型返回的最大 token 数
TOP_P = None  # nucleus sampling 参数；通常保持 None，避免与 temperature 叠加扰动
ENABLE_PROMPT_CACHE = True  # 是否启用 prompt cache
USE_PERSISTENT_SESSION = True  # 是否复用 requests.Session 连接
# 重试参数
MAX_RETRIES = 3  # 单条记录最多重试次数
RETRY_BASE_SECONDS = 2.0  # 指数退避的基础等待秒数

# 小样本与断点续跑参数
MAX_ITEMS = 0  # 本次最多处理多少条 KEEP_NEW 记录；设为 0 表示从 START_INDEX 起全量处理
START_INDEX = 0  # 本次处理起始下标，用于小样本测试或断点续跑

# 当前阶段目录：大规模实验/phase2
BASE_DIR = Path(__file__).resolve().parent.parent  # 当前阶段根目录：大规模实验/phase2
# 项目根目录：大规模实验
PROJECT_ROOT = BASE_DIR.parent  # 大规模实验总目录
RESOURCE_VERSION = "V3"
ONTOLOGY_TTL_FILENAME = f"TrafficKG_{RESOURCE_VERSION}.ttl"
# 是否将控制台输出同时写入日志文件
ENABLE_FILE_LOG = True  # 是否将控制台输出同步写入日志文件
# 日志目录：统一保存脚本控制台输出
LOG_DIR = PROJECT_ROOT / "logs"  # 统一日志目录
# 输入文件：Agent2-外部对齐输出的 KEEP_NEW 结果
INPUT_KEEP_NEW_PATH = BASE_DIR / "outputs_phase2_03" / "agent2_external_keep_new_classes.json"  # Agent2 外部对齐后保留为新 Class 的输入文件
# 输入文件：本体基座 TTL
ONTOLOGY_TTL_PATH = PROJECT_ROOT / "shared_resources" / ONTOLOGY_TTL_FILENAME  # 本体基座 TTL，用于抽取类锚点并约束定位

# 输出目录：Agent3 的全部产物
OUTPUT_DIR = BASE_DIR / "outputs_phase2_04a"  # Agent3 04a（Class 写入判定）全部输出的根目录
# 原生定位决策结果
OUTPUT_DECISIONS = OUTPUT_DIR / "agent3_class_positioning_decisions.json"  # 原生类定位决策结果
# 供阶段三审核使用的扁平结果
OUTPUT_REVIEW_FLAT = OUTPUT_DIR / "agent3_class_positioning_review_flat.json"  # 供后续审核使用的类扁平结果
# 错误明细
OUTPUT_ERRORS = OUTPUT_DIR / "agent3_class_positioning_errors.json"  # 输入校验失败、LLM 失败等错误明细
# 运行摘要与统计信息
OUTPUT_MANIFEST = OUTPUT_DIR / "agent3_class_positioning_manifest.json"  # 本次运行摘要、参数与统计信息
OUTPUT_STATS = OUTPUT_DIR / "agent3_class_positioning_stats.json"
# 原始大模型响应日志
OUTPUT_RAW_API_LOG = OUTPUT_DIR / "agent3_class_llm_raw_responses.jsonl"  # 原始大模型响应日志，便于追溯与排错
# token 统计结果
OUTPUT_TOKEN_REPORT = OUTPUT_DIR / "agent3_class_token_report.json"  # 输入/输出 token 汇总结果

# ==========================================
# 提示词区
# ==========================================
# System Prompt：注入角色、硬约束与全量 TTL 上下文。
SYSTEM_PROMPT_TEMPLATE = """你需要承担城市交通态势本体类要素更新判定任务。

该任务用于判断一个待纳入类要素是否适合进入城市交通态势本体，并在既有本体结构约束下给出父类归属建议。你需要综合候选要素的名称、定义、证据文本、前序结构建议和完整本体结构，判断该候选是否具备独立建模价值，以及它在本体类层级中的合理位置。

本任务只处理 Class 类型候选，不处理 ObjectProperty 和 DatatypeProperty。

【任务目标】
请对单个待纳入类要素给出更新判定结果。

最终只能输出以下三类决策之一：
- POSITIONED：该候选适合作为类要素纳入本体，并且可以给出明确父类位置；
- UNSURE：该候选可能具有本体建模价值，但当前既有本体结构下无法安全确定父类位置，需要人工复核；
- REJECTED：该候选不适合作为类要素纳入本体。

当 decision = POSITIONED 时，必须给出 parent_class，并给出判定把握等级 confidence_level。
当 decision = UNSURE 或 REJECTED 时，parent_class 和 confidence_level 均为 null。

【输入上下文】
你将收到以下信息：
1. 待纳入类要素：包括名称、定义、别名、证据文本、来源信息、融合定义和前序结构建议；
2. 既有本体类锚点：来自当前版本城市交通态势本体的已有类；
3. 完整本体上下文：用于理解既有本体的类层级、语义组织方式和结构约束。

前序结构建议可以作为参考，但不是最终判定结果。你需要在完整本体结构下重新判断其可纳入性和父类归属。

【链式处理步骤】
请按照以下顺序完成判断。每一步均需结合对应规则执行，后一步以前一步判断结果为基础。

Step 1：理解既有本体结构
- 读取内容：完整本体上下文和既有本体类锚点。
- 对应规则：【本体结构理解规则】。
- 处理目标：理解当前本体中空间、时间、状态、指标、事件、过程等类的组织方式。
- 结果作用：为后续判断提供父类选择范围和结构约束。

Step 2：理解待纳入类要素语义
- 读取内容：待纳入类要素的名称、定义、别名、融合定义、证据文本和前序结构建议。
- 对应规则：【候选类要素理解规则】。
- 处理目标：判断该候选表达的是交通对象、交通状态、交通指标、拥堵现象、交通事件、演化过程、运行等级，还是其他内容。
- 结果作用：明确该候选是否具有独立类建模价值。

Step 3：判断是否适合作为类要素纳入本体
- 读取内容：Step 2 的语义理解结果。
- 对应规则：【类要素可纳入性判断规则】和【REJECTED 判定规则】。
- 处理目标：判断该候选是否应作为独立 Class 保留。
- 结果作用：若候选不具备独立建模价值，输出 REJECTED；若具备建模价值，继续判断父类位置。

Step 4：判断父类归属
- 读取内容：待纳入类要素语义、前序结构建议、完整本体类层级和可用类锚点。
- 对应规则：【父类归属判断规则】。
- 处理目标：在既有本体中选择语义最接近、层级最稳定的父类。
- 结果作用：若可以确定安全父类，输出 POSITIONED；若无法确定安全父类，输出 UNSURE。

Step 5：判断结果把握等级
- 读取内容：父类归属判断结果、残余不确定性和证据支撑情况。
- 对应规则：【判定把握等级规则】。
- 处理目标：对 POSITIONED 结果给出 HIGH 或 LOW。
- 结果作用：为后续人工复核区分一般通过项和重点关注项。

Step 6：生成结构化输出
- 读取内容：前述全部判断结果。
- 对应规则：【输出要求】。
- 处理目标：输出严格 JSON 格式结果。
- 结果作用：形成可供后续形式化构造和人工复核使用的类要素更新判定结果。

【本体结构理解规则】
1. 城市交通态势本体中的状态实体主要承担结构连接与中转作用，用于连接空间、时间以及具体指标、事件和过程等业务内容。
2. 交通业务语义通常由具体指标、拥堵或运行现象、交通事件、演化过程、运行等级等类承载。
3. 对带有“状态色彩”的候选类，不应仅根据字面名称将其归入状态类。若其核心语义是可观测、可判别或可度量的交通运行内容，应优先判断其是否属于具体指标、运行现象、事件或过程等业务类。
4. 交通指标类应按空间对象和业务含义进行细分。路段尺度指标优先归入路段指标类，交叉口尺度指标优先归入交叉口指标类，区域或路网尺度指标优先归入区域或路网指标类。
5. 动态交通指标、交通状态或运行评价类概念不得直接挂到 RoadSegment、Intersection、Road 等微观空间实体下。微观空间实体是指标依附或发生的对象，不是动态指标本身的父类。

【候选类要素理解规则】
1. 若候选要素具有稳定语义边界，可以被定义、分类、对齐和纳入本体类层级，则具备类要素判断基础。
2. 若候选要素表示交通对象、交通状态、交通指标、拥堵现象、交通事件、演化过程、运行等级或空间对象，且具有独立交通业务语义，可以继续判断其父类归属。
3. 若候选要素只是具体地点、具体时刻、具体事件个体、具体数值、实验对象或数据样本，不应作为类要素纳入。
4. 若候选要素本质上是研究任务、算法过程、模型方法、系统平台、实验步骤或分析流程，不应作为类要素纳入。
5. 若候选要素更适合作为已有实体的静态属性、配置属性或说明性属性，不应作为 Class 纳入。

【类要素可纳入性判断规则】
1. 候选要素若承载相对独立的交通业务语义，并且能在本体类层级中找到合理位置，应优先输出 POSITIONED。
2. 候选要素若具有一定建模价值，但既有本体中缺少可安全挂载的父类，或其语义跨越多个类家族且无法确定默认位置，应输出 UNSURE。
3. 候选要素若不具备独立建模价值，或明显不应作为 Class 进入本体，应输出 REJECTED。
4. 不要因为候选名称与前序结构建议存在轻微差异就直接 UNSURE。只要能在既有本体中找到安全默认父类，应给出明确判定。
5. 不要为了减少人工复核而强行 POSITIONED。若父类位置会明显误导后续本体结构，应输出 UNSURE 或 REJECTED。

【父类归属判断规则】
1. parent_class 必须来自既有本体类锚点，不得自行创造不存在的父类。
2. 若前序结构建议与候选定义、证据文本和既有本体层级一致，可以优先采纳。
3. 若前序结构建议不合理，应说明核心原因，并给出更合适的父类。
4. 若候选属于某个已有类家族，应选择语义最接近、层级最稳定的父类，不要仅因名称不完全一致而返回 UNSURE。
5. 若存在多个相近父类，应比较其空间尺度、业务对象和抽象层级，优先选择最具体且不破坏本体结构的父类。
6. 若没有任何安全父类位置，输出 UNSURE。
7. 若候选应被排除，不得通过给出宽泛父类来强行纳入。

【REJECTED 判定规则】
满足以下任一情况时，输出 REJECTED：
1. 当前候选明显不是类要素；
2. 当前候选本质上是实例、数值、地点、时间、样本或具体事件个体；
3. 当前候选本质上是算法、模型、方法、系统、实验过程或研究任务；
4. 当前候选只是已有实体的静态属性、配置属性或说明性属性；
5. 当前候选若作为 Class 纳入会破坏既有本体的基本分类逻辑。

【UNSURE 判定规则】
满足以下情况之一时，输出 UNSURE：
1. 候选可能具有类建模价值，但既有本体中没有安全父类位置；
2. 候选语义跨越多个类家族，任何单一路径都会造成明显误导；
3. 候选定义、证据和前序结构建议之间存在实质冲突，无法通过现有本体结构安全解决；
4. 候选所需的上位结构在当前本体中缺失，需要人工判断是否新增上位类。

【判定把握等级规则】
confidence_level 只在 decision = POSITIONED 时输出。

- HIGH：
  1. 父类位置明确；
  2. 候选定义、证据文本、前序结构建议和既有本体层级基本一致；
  3. 不存在明显残余冲突；
  4. 该结果可作为一般复核项进入后续处理。

- LOW：
  1. 可以给出安全父类位置；
  2. 仍存在一定语义边界、父类竞争、尺度差异或证据不足问题；
  3. 该结果应作为重点复核项进入后续处理。

不要因为存在轻微犹豫就将所有 POSITIONED 结果标为 LOW。若规则已经能够支持明确默认父类，应输出 HIGH。

【reasoning 写作要求】
1. reasoning 使用中文，写成 2—3 句短句。
2. 第一句直接说明最终建议，例如“建议将该类挂载到 :XXX 下”或“该候选不适合作为类要素纳入本体”。
3. 第二句说明最核心的语义理由，重点说明其业务语义、空间尺度或父类归属依据。
4. 第三句可补充保留意见或人工复核点。没有明显风险时可省略。
5. 不要复述完整定义，不要展开长篇推理，不要只罗列冲突而不给结论。

【输出要求】
1. 只能输出严格 JSON，不输出 Markdown、代码块或额外解释。
2. JSON 顶层必须包含以下字段：
   - decision
   - confidence_level
   - parent_class
   - suggested_domain
   - suggested_range
   - suggested_xsd_range
   - reasoning
3. decision 只能是 "POSITIONED"、"UNSURE" 或 "REJECTED"。
4. 当 decision = "POSITIONED" 时：
   - confidence_level 必须是 "HIGH" 或 "LOW"；
   - parent_class 必须是一个既有本体类锚点；
   - suggested_domain、suggested_range、suggested_xsd_range 必须为 null。
5. 当 decision = "UNSURE" 或 "REJECTED" 时：
   - confidence_level、parent_class、suggested_domain、suggested_range、suggested_xsd_range 均为 null。
6. 不得新增任何其他字段。

【输出格式】
{{
  "decision": "POSITIONED|UNSURE|REJECTED",
  "confidence_level": "HIGH|LOW|null",
  "parent_class": ":ClassName|null",
  "suggested_domain": null,
  "suggested_range": null,
  "suggested_xsd_range": null,
  "reasoning": "中文简要理由"
}}

【可用本体类锚点】
{class_anchor_hint}

【完整本体上下文（{ontology_ttl_filename}）】
{traffickg_ttl_text}
"""

# User Prompt：每次仅传入单条候选概念，便于串行调试。
USER_PROMPT_TEMPLATE = """请按 System 规则处理以下候选概念，并仅输出 JSON。

candidate_concept:
{candidate_json}
"""

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

# 微观空间实体类集合：用于检查动态指标是否越级挂载
MICRO_SPATIAL_CLASSES = {"RoadSegment", "Intersection", "Road"}
# 指标语义关键词：用于触发局部拓扑规则
INDICATOR_KEYWORDS = [
    "指标",
    "评价",
    "态势",
    "拥堵",
    "速度",
    "流量",
    "密度",
    "延误",
    "系数",
    "指数",
    "indicator",
    "congestion",
    "speed",
    "flow",
    "density",
    "delay",
    "ratio",
    "index",
]
# DatatypeProperty 可接受的 xsd 类型白名单
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
# LLM 结构化输出的 JSON Schema（第一道约束）。
POSITIONING_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "confidence_level",
        "reasoning",
        "parent_class",
        "suggested_domain",
        "suggested_range",
        "suggested_xsd_range",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["POSITIONED", "UNSURE", "REJECTED"],
        },
        "confidence_level": {
            "type": ["string", "null"],
            "enum": ["HIGH", "LOW", None],
        },
        "reasoning": {"type": "string", "minLength": 1},
        "parent_class": {"type": ["string", "null"]},
        "suggested_domain": {"type": ["string", "null"]},
        "suggested_range": {"type": ["string", "null"]},
        "suggested_xsd_range": {"type": ["string", "null"]},
    },
}


# ==========================================
# Pydantic 输入模型
# ==========================================
ElementType = Literal["Class", "ObjectProperty", "DatatypeProperty"]
InternalStatus = Literal["merged", "singleton"]
Decision = Literal["POSITIONED", "UNSURE", "REJECTED"]
ConfidenceLevel = Literal["HIGH", "LOW"]
CheckLevel = Literal["error", "warning", "info"]
ConfidenceSource = Literal["local", "fallback"]


class SourceInfo(BaseModel):
    """证据来源定位信息。"""
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    section: str
    chunk_id: str


class MappingSuggestion(BaseModel):
    """承接前序阶段的锚点建议字段。"""
    model_config = ConfigDict(extra="allow")
    suggested_parent_class: Optional[str] = None
    suggested_domain: Optional[str] = None
    suggested_range: Optional[str] = None
    suggested_xsd_range: Optional[str] = None
    reason: str = ""


class CanonicalElement(BaseModel):
    """待定位的 canonical 概念主体。"""
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

        self.name_en = (self.name_en or "").strip()

        if not (self.name_raw or self.name_zh or self.name_en):
            raise ValueError("名称字段缺失：至少需要 name_raw/name_zh/name_en 之一。")
        if not self.name_raw:
            self.name_raw = self.name_zh or self.name_en
        if not self.name_zh and re.search(r"[\u4e00-\u9fff]", self.name_raw):
            self.name_zh = self.name_raw
        return self


class MergedEvidence(BaseModel):
    """待定位概念保留的证据片段。"""
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
    """承接内部融合结果的标准输入记录。"""
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


class ExternalAlignment(BaseModel):
    """外部对齐阶段输出的决策摘要。"""
    model_config = ConfigDict(extra="forbid")
    decision_route: str
    decision: str
    target_old_name: Optional[str] = None
    best_similarity: Optional[float] = None
    reason: str = ""


class KeepNewRecord(BaseModel):
    """进入 Agent3 本体定位阶段的 KEEP_NEW 记录。"""
    model_config = ConfigDict(extra="forbid")
    cluster_id: str
    element_type: ElementType
    canonical_record: CanonicalInputRecord
    external_alignment: ExternalAlignment


# ==========================================
# Pydantic 输出模型
# ==========================================
class TopologyCheck(BaseModel):
    """单条本地拓扑规则检查结果。"""
    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(min_length=1)
    passed: bool
    level: CheckLevel
    message: str = Field(min_length=1)


class EvidenceRef(BaseModel):
    """写入定位结果的证据引用。"""
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    section: str
    chunk_id: str
    evidence_text: str = ""


class PositioningResponse(BaseModel):
    """Agent3 最终落盘的定位结果。"""
    model_config = ConfigDict(extra="forbid")
    decision: Decision
    confidence_level: Optional[ConfidenceLevel] = None
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_source: ConfidenceSource
    reasoning: str = Field(min_length=1)
    parent_class: Optional[str] = None
    suggested_domain: Optional[str] = None
    suggested_range: Optional[str] = None
    suggested_xsd_range: Optional[str] = None
    topology_checks: List[TopologyCheck] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)


class LLMPositioningResponse(BaseModel):
    """LLM 原始定位响应的结构化中间模型。"""
    model_config = ConfigDict(extra="forbid")
    decision: Decision
    confidence_level: Optional[ConfidenceLevel] = None
    reasoning: str = Field(min_length=1)
    parent_class: Optional[str] = None
    suggested_domain: Optional[str] = None
    suggested_range: Optional[str] = None
    suggested_xsd_range: Optional[str] = None


# ==========================================
# 工具函数
# ==========================================
def write_json(path: Path, data: Any) -> None:
    """将对象写入 UTF-8 JSON 文件（自动创建目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    """向 JSONL 日志文件追加一行记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def build_class_positioning_stats(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    decision_counts = {"POSITIONED": 0, "UNSURE": 0, "REJECTED": 0}
    confidence_level_counts = {"HIGH": 0, "LOW": 0, "null_total": 0}
    parent_class_present_total = 0
    parent_class_unsure_total = 0
    distinct_parent_classes = set()
    structural_failed_total = 0
    topology_failed_total = 0
    risk_flag_total = 0

    for record in decisions:
        decision = str(record.get("decision", "")).strip().upper()
        if decision in decision_counts:
            decision_counts[decision] += 1

        confidence_level = record.get("confidence_level")
        if confidence_level in {"HIGH", "LOW"}:
            confidence_level_counts[str(confidence_level)] += 1
        else:
            confidence_level_counts["null_total"] += 1

        parent_class = str(record.get("parent_class") or "").strip()
        if parent_class:
            parent_class_present_total += 1
            if parent_class.upper() in {"UNSURE", ":UNSURE"}:
                parent_class_unsure_total += 1
            else:
                distinct_parent_classes.add(parent_class)

        topology_checks = record.get("topology_checks") or []
        failed_checks = [check for check in topology_checks if not bool(check.get("passed", False))]
        if failed_checks:
            risk_flag_total += 1
        for check in failed_checks:
            rule_id = str(check.get("rule_id", "")).strip().upper()
            if rule_id.startswith("R_TOPO_"):
                topology_failed_total += 1
            else:
                structural_failed_total += 1

    return {
        "decision_counts": decision_counts,
        "confidence_level_counts": confidence_level_counts,
        "parent_class_stats": {
            "parent_class_present_total": parent_class_present_total,
            "parent_class_unsure_total": parent_class_unsure_total,
            "distinct_parent_class_total": len(distinct_parent_classes),
        },
        "local_rule_stats": {
            "structural_failed_total": structural_failed_total,
            "topology_failed_total": topology_failed_total,
            "risk_flag_total": risk_flag_total,
        },
    }


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


def extract_usage_metrics(usage: Any) -> Dict[str, int]:
    """从 usage 对象中提取输入/输出与缓存 token。"""
    payload = to_jsonable(usage)
    if not isinstance(payload, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_input_for_threshold": 0,
        }

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

    cache_creation = to_non_negative_int(payload.get("cache_creation_input_tokens"))
    cache_read = to_non_negative_int(payload.get("cache_read_input_tokens"))
    cache_creation_obj = payload.get("cache_creation")
    if isinstance(cache_creation_obj, dict) and cache_creation == 0:
        cache_creation = to_non_negative_int(
            cache_creation_obj.get("ephemeral_5m_input_tokens")
            or cache_creation_obj.get("ephemeral_1h_input_tokens")
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "total_input_for_threshold": input_tokens + cache_creation + cache_read,
    }


def normalize_for_match(value: Optional[str]) -> str:
    """文本标准化：去空白 + 小写折叠，用于稳健匹配。"""
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value)).casefold().strip()


def clean_optional_text(value: Optional[str]) -> Optional[str]:
    """清理可空文本：空串归一为 None。"""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_anchor_name(raw: Optional[str]) -> str:
    """归一化锚点名：移除前缀/IRI 包装，仅保留类名主体。"""
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = text.strip("`'\"")
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = text.lstrip(":").strip()
    return text


def is_anchor_present(anchor_raw: Optional[str], anchor_norm_set: set[str]) -> bool:
    """判断给定锚点是否命中 TTL 类锚点集合。"""
    norm = normalize_for_match(normalize_anchor_name(anchor_raw))
    return bool(norm and norm in anchor_norm_set)


def contains_indicator_keywords(*texts: str) -> bool:
    """通过关键词快速判断概念是否偏“指标/态势”语义。"""
    joined = " ".join([t for t in texts if t]).lower()
    return any(k in joined for k in INDICATOR_KEYWORDS)


def extract_ttl_class_anchors(ttl_text: str) -> List[str]:
    """从 TTL 文本中提取 owl:Class 的本地类名。"""
    pattern = re.compile(r"^\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+rdf:type\s+owl:Class\b", re.MULTILINE)
    names = sorted({m.group("name") for m in pattern.finditer(ttl_text)})
    return names


def read_text(path: Path) -> str:
    """读取 UTF-8 文本文件。"""
    return path.read_text(encoding="utf-8")


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    """读取 JSON 数组文件，并保留其中的对象项。"""
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
    """返回用于日志和规则判断的优先展示名。"""
    return preferred_name_zh(ce) or preferred_name_en(ce)


def build_system_blocks(ttl_text: str, class_anchors: List[str]) -> List[Dict[str, Any]]:
    """构建 Native Messages system blocks：固定规则与 TTL 分块缓存。"""
    anchor_hint = ", ".join(f":{x}" for x in class_anchors[:300])
    rules_text = SYSTEM_PROMPT_TEMPLATE.format(
        class_anchor_hint=anchor_hint,
        ontology_ttl_filename=ONTOLOGY_TTL_FILENAME,
        traffickg_ttl_text="",
    ).strip()
    cache_control = {"type": "ephemeral"} if ENABLE_PROMPT_CACHE else None
    rules_block: Dict[str, Any] = {
        "type": "text",
        "text": rules_text,
    }
    ttl_block: Dict[str, Any] = {
        "type": "text",
        "text": ttl_text,
    }
    if cache_control:
        rules_block["cache_control"] = cache_control
        ttl_block["cache_control"] = cache_control
    return [rules_block, ttl_block]


def _extract_text_from_content(raw_content: Any) -> str:
    """兼容不同网关返回结构，统一提取 message 文本内容。"""
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
    """从模型文本中提取首个 JSON 对象（容忍 ```json 包裹）。"""
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


def recover_positioning_payload_from_malformed_json(text: str) -> Optional[Dict[str, Any]]:
    """当 JSON 解析失败时，尽力从文本中兜底恢复关键字段。"""
    decision_match = re.search(
        r'"decision"\s*:\s*"(POSITIONED|UNSURE|REJECTED)"',
        text,
        flags=re.IGNORECASE,
    )
    if not decision_match:
        return None

    decision = decision_match.group(1).upper()
    confidence_match = re.search(
        r'"confidence_level"\s*:\s*(null|"HIGH"|"LOW"|"high"|"low")',
        text,
        flags=re.IGNORECASE,
    )
    if confidence_match:
        raw_confidence = confidence_match.group(1).strip()
        if raw_confidence.lower() == "null":
            confidence_level = None
        else:
            confidence_level = raw_confidence.strip('"').upper()
    else:
        confidence_level = None
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, flags=re.IGNORECASE | re.DOTALL)
    reasoning = reason_match.group(1).replace('\\"', '"').strip() if reason_match else "No reasoning recovered."

    def extract_optional(key: str) -> Optional[str]:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*(null|"[^"]*")', text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        raw = m.group(1).strip()
        if raw.lower() == "null":
            return None
        return raw.strip('"').strip()

    return {
        "decision": decision,
        "confidence_level": confidence_level,
        "reasoning": reasoning if reasoning else "No reasoning recovered.",
        "parent_class": extract_optional("parent_class"),
        "suggested_domain": extract_optional("suggested_domain"),
        "suggested_range": extract_optional("suggested_range"),
        "suggested_xsd_range": extract_optional("suggested_xsd_range"),
    }


def normalize_positioning_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """归一化模型字段差异并补齐默认值，便于 Pydantic 校验。"""
    decision = str(payload.get("decision", "")).strip().upper()
    confidence_level_raw = payload.get("confidence_level")
    if confidence_level_raw is None:
        confidence_level = None
    else:
        confidence_level = str(confidence_level_raw).strip().upper() or None
    reasoning = (
        payload.get("reasoning")
        or payload.get("reason")
        or payload.get("rationale")
        or payload.get("explanation")
        or ""
    )
    reasoning = str(reasoning).strip() or "No explicit reasoning provided."

    return {
        "decision": decision,
        "confidence_level": confidence_level,
        "reasoning": reasoning,
        "parent_class": clean_optional_text(payload.get("parent_class")),
        "suggested_domain": clean_optional_text(payload.get("suggested_domain")),
        "suggested_range": clean_optional_text(payload.get("suggested_range")),
        "suggested_xsd_range": clean_optional_text(payload.get("suggested_xsd_range")),
    }


def to_short_reasoning(text: str) -> str:
    """将理由压缩为短句，默认不超过 300 字符。"""
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if not compact:
        compact = "证据不足，按规则作出保守定位结论。"
    if len(compact) > 300:
        compact = compact[:297].rstrip() + "..."
    return compact


def build_local_evidence_refs(rec: KeepNewRecord, max_items: int = 3) -> List[EvidenceRef]:
    """从输入记录构建本地证据引用列表。"""
    evidence: List[EvidenceRef] = []
    for item in rec.canonical_record.merged_evidence[:max_items]:
        evidence.append(
            EvidenceRef(
                paper_id=item.source.paper_id,
                section=item.source.section,
                chunk_id=item.source.chunk_id,
                evidence_text=item.evidence_text,
            )
        )
    if evidence:
        return evidence

    ce = rec.canonical_record.canonical_element
    return [
        EvidenceRef(
            paper_id=ce.source.paper_id,
            section=ce.source.section,
            chunk_id=ce.source.chunk_id,
            evidence_text=ce.evidence_text,
        )
    ]


def compute_local_confidence(
    decision: Decision,
    checks: List[TopologyCheck],
    llm_failed: bool,
    reasoning: str,
    evidence_count: int,
) -> Tuple[float, ConfidenceSource]:
    """基于本地规则ID计算可解释置信度。"""
    if llm_failed:
        return 0.10, "fallback"

    base_map: Dict[str, float] = {
        "POSITIONED": 0.78,
        "UNSURE": 0.46,
        "REJECTED": 0.62,
    }
    score = base_map.get(str(decision), 0.46)

    rule_weight: Dict[str, Tuple[float, float]] = {
        "R_CLASS_PARENT_REQUIRED": (0.04, -0.30),
        "R_OP_DOMAIN_RANGE_REQUIRED": (0.04, -0.30),
        "R_DP_XSD_REQUIRED_VALID": (0.04, -0.32),
        "R_DYNAMIC_INDICATOR_NO_DIRECT_MICRO_SPATIAL": (0.05, -0.35),
        "R_CLASS_PARENT_ANCHOR_EXISTS": (0.02, -0.15),
        "R_OP_DOMAIN_ANCHOR_EXISTS": (0.015, -0.10),
        "R_OP_RANGE_ANCHOR_EXISTS": (0.015, -0.10),
        "R_DP_DOMAIN_ANCHOR_EXISTS": (0.015, -0.10),
    }

    for c in checks:
        pair = rule_weight.get(c.rule_id)
        if pair is None:
            continue
        score += pair[0] if c.passed else pair[1]

    if evidence_count >= 1:
        score += 0.03
    if evidence_count >= 2:
        score += 0.02
    if evidence_count >= 3:
        score += 0.01

    if "[LOCAL_RULE_BLOCK" in reasoning:
        score = min(score, 0.35)
    if "[LOCAL_RULE_DEGRADE" in reasoning:
        score = min(score, 0.55)

    score = max(0.0, min(1.0, score))
    return round(score, 2), "local"


def call_claude_positioning(
    session: requests.Session,
    system_blocks: List[Dict[str, Any]],
    rec: KeepNewRecord,
    token_usage_counter: Dict[str, int],
) -> Tuple[LLMPositioningResponse, Dict[str, Any]]:
    """调用 LLM 完成本体定位，并返回结构化响应对象。"""
    url = f"{CLAUDE_NATIVE_BASE_URL.rstrip('/')}{MESSAGES_PATH}"
    ce = rec.canonical_record.canonical_element
    # 组装传给 LLM 的候选概念上下文（含 Agent1/2 的关键线索）。
    candidate_payload = {
        "cluster_id": rec.cluster_id,
        "element_type": rec.element_type,
        "name_raw": ce.name_raw,
        "name_zh": preferred_name_zh(ce),
        "name_en": preferred_name_en(ce),
        "definition": ce.definition,
        "aliases": rec.canonical_record.aliases,
        "merged_definitions": rec.canonical_record.merged_definitions[:5],
        "mapping_suggestion": ce.mapping_suggestion.model_dump(),
        "external_alignment": rec.external_alignment.model_dump(),
        "evidence_refs": [x.model_dump() for x in rec.canonical_record.merged_evidence[:1]],
    }
    user_prompt = USER_PROMPT_TEMPLATE.format(
        candidate_json=json.dumps(candidate_payload, ensure_ascii=False, indent=2)
    )

    payload: Dict[str, Any] = {
        "model": CLAUDE_MODEL,
        "system": system_blocks,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    }
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if TOP_P is not None:
        payload["top_p"] = TOP_P

    # 发送请求并记录原始返回，便于实验复盘与问题定位。
    response = session.post(url, json=payload, timeout=API_TIMEOUT_SECONDS)
    raw_response_text = response.text
    response.raise_for_status()
    response_payload = json.loads(raw_response_text)

    usage_obj = response_payload.get("usage", {})
    usage_metrics = extract_usage_metrics(usage_obj)
    token_usage_counter["input_tokens"] += usage_metrics["input_tokens"]
    token_usage_counter["output_tokens"] += usage_metrics["output_tokens"]
    token_usage_counter["cache_creation_input_tokens"] += usage_metrics["cache_creation_input_tokens"]
    token_usage_counter["cache_read_input_tokens"] += usage_metrics["cache_read_input_tokens"]

    raw_content = _extract_text_from_content(response_payload.get("content", []))
    json_text = extract_json_string(raw_content)
    stop_reason = str(response_payload.get("stop_reason", "") or "")
    response_shape = (
        "anthropic_message"
        if response_payload.get("type") == "message" and isinstance(response_payload.get("content"), list)
        else "unknown"
    )

    append_jsonl(
        OUTPUT_RAW_API_LOG,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cluster_id": rec.cluster_id,
            "name_zh": preferred_name_zh(ce),
            "name_en": preferred_name_en(ce),
            "raw_content": raw_content,
            "response_shape": response_shape,
            "raw_response": response_payload,
            "usage": to_jsonable(usage_obj),
            **usage_metrics,
            "stop_reason": stop_reason,
        },
    )

    try:
        payload = json.loads(json_text)
        if not isinstance(payload, dict):
            raise ValueError("LLM JSON payload must be an object.")
    except json.JSONDecodeError:
        # 极端情况下尝试从非严格 JSON 文本中恢复关键字段。
        recovered = recover_positioning_payload_from_malformed_json(json_text)
        if recovered is None:
            raise
        payload = recovered

    normalized_payload = normalize_positioning_payload(payload)
    normalized_payload["reasoning"] = to_short_reasoning(normalized_payload.get("reasoning", ""))
    response_meta = {
        "usage_metrics": usage_metrics,
        "stop_reason": stop_reason,
        "response_shape": response_shape,
    }
    return LLMPositioningResponse.model_validate(normalized_payload), response_meta


def fallback_unsure_response(rec: KeepNewRecord, last_error: str) -> LLMPositioningResponse:
    """LLM 失败时的保守兜底：统一降级为 UNSURE。"""
    _ = rec, last_error
    return LLMPositioningResponse(
        decision="UNSURE",
        confidence_level=None,
        reasoning="LLM 调用失败，采用保守策略 UNSURE，待人工复核。",
        parent_class=None,
        suggested_domain=None,
        suggested_range=None,
        suggested_xsd_range=None,
    )


def apply_local_rules(
    rec: KeepNewRecord,
    llm_resp: LLMPositioningResponse,
    anchor_norm_set: set[str],
    llm_failed: bool,
) -> Tuple[PositioningResponse, List[Dict[str, Any]], bool]:
    """执行本地二次规则校验，并按规则对决策进行降级/拦截。"""
    checks: List[TopologyCheck] = []
    ce = rec.canonical_record.canonical_element
    et = rec.element_type
    decision = llm_resp.decision
    structural_failed = False
    topology_failed = False
    anchor_failed = False
    missing_confidence_level = False

    def add_check(rule_id: str, passed: bool, level: CheckLevel, message: str) -> None:
        checks.append(TopologyCheck(rule_id=rule_id, passed=passed, level=level, message=message))

    if llm_failed:
        add_check(
            "R_LLM_CALL_FAILED",
            False,
            "warning",
            "LLM 调用失败，当前结果由本地保守策略生成。",
        )

    # 规则组1：结构完备性（按 element_type 检查必填字段）。
    if decision == "POSITIONED":
        has_confidence_level = llm_resp.confidence_level in {"HIGH", "LOW"}
        add_check(
            "R_POSITIONED_CONFIDENCE_LEVEL_REQUIRED",
            has_confidence_level,
            "error",
            "POSITIONED 在新版输出约束下必须提供 confidence_level=HIGH/LOW。"
            if not has_confidence_level
            else "POSITIONED 已提供合法 confidence_level。",
        )
        if not has_confidence_level:
            missing_confidence_level = True

        if et == "Class":
            ok = bool(clean_optional_text(llm_resp.parent_class))
            add_check(
                "R_CLASS_PARENT_REQUIRED",
                ok,
                "error",
                "Class 在 POSITIONED 情况下必须提供 parent_class。"
                if not ok
                else "Class parent_class 已提供。",
            )
            if not ok:
                structural_failed = True

        if et == "ObjectProperty":
            has_domain = bool(clean_optional_text(llm_resp.suggested_domain))
            has_range = bool(clean_optional_text(llm_resp.suggested_range))
            ok = has_domain and has_range
            add_check(
                "R_OP_DOMAIN_RANGE_REQUIRED",
                ok,
                "error",
                "ObjectProperty 在 POSITIONED 情况下必须提供 suggested_domain 与 suggested_range。"
                if not ok
                else "ObjectProperty 的 domain/range 已提供。",
            )
            if not ok:
                structural_failed = True

        if et == "DatatypeProperty":
            has_domain = bool(clean_optional_text(llm_resp.suggested_domain))
            xsd = clean_optional_text(llm_resp.suggested_xsd_range)
            has_xsd = bool(xsd)
            xsd_ok = bool(xsd in XSD_ALLOWED) if has_xsd else False
            ok = has_domain and has_xsd and xsd_ok
            msg = "DatatypeProperty 在 POSITIONED 情况下必须提供 domain 与合法 xsd range。"
            if has_domain and has_xsd and not xsd_ok:
                msg = f"xsd range 不在允许集合: {xsd}"
            if ok:
                msg = "DatatypeProperty 的 domain/xsd_range 均合法。"
            add_check("R_DP_XSD_REQUIRED_VALID", ok, "error", msg)
            if not ok:
                structural_failed = True

    # 规则组2：锚点存在性（挂载目标必须能在 TTL 类集合中命中）。
    if decision == "POSITIONED":
        if et == "Class":
            anchor_ok = is_anchor_present(llm_resp.parent_class, anchor_norm_set)
            add_check(
                "R_CLASS_PARENT_ANCHOR_EXISTS",
                anchor_ok,
                "warning",
                "parent_class 未在 TTL 类锚点集合中命中。"
                if not anchor_ok
                else "parent_class 命中 TTL 类锚点。",
            )
            if not anchor_ok:
                anchor_failed = True

        if et == "ObjectProperty":
            domain_ok = is_anchor_present(llm_resp.suggested_domain, anchor_norm_set)
            range_ok = is_anchor_present(llm_resp.suggested_range, anchor_norm_set)
            add_check(
                "R_OP_DOMAIN_ANCHOR_EXISTS",
                domain_ok,
                "warning",
                "suggested_domain 未在 TTL 类锚点集合中命中。"
                if not domain_ok
                else "suggested_domain 命中 TTL 类锚点。",
            )
            add_check(
                "R_OP_RANGE_ANCHOR_EXISTS",
                range_ok,
                "warning",
                "suggested_range 未在 TTL 类锚点集合中命中。"
                if not range_ok
                else "suggested_range 命中 TTL 类锚点。",
            )
            if not domain_ok or not range_ok:
                anchor_failed = True

        if et == "DatatypeProperty":
            domain_ok = is_anchor_present(llm_resp.suggested_domain, anchor_norm_set)
            add_check(
                "R_DP_DOMAIN_ANCHOR_EXISTS",
                domain_ok,
                "warning",
                "suggested_domain 未在 TTL 类锚点集合中命中。"
                if not domain_ok
                else "suggested_domain 命中 TTL 类锚点。",
            )
            if not domain_ok:
                anchor_failed = True

    # 规则组3：拓扑硬约束（动态指标不得越级直连微观空间实体）。
    semantic_text = " ".join(
        [
            display_name(ce),
            preferred_name_en(ce),
            ce.definition,
            llm_resp.reasoning,
            ce.mapping_suggestion.reason or "",
        ]
    )
    is_indicator_like = contains_indicator_keywords(semantic_text)

    if decision == "POSITIONED" and is_indicator_like:
        micro_norm = {normalize_for_match(x) for x in MICRO_SPATIAL_CLASSES}

        def is_micro(value: Optional[str]) -> bool:
            return normalize_for_match(normalize_anchor_name(value)) in micro_norm

        micro_hit = False
        if et == "Class" and is_micro(llm_resp.parent_class):
            micro_hit = True
        if et == "ObjectProperty" and (is_micro(llm_resp.suggested_domain) or is_micro(llm_resp.suggested_range)):
            micro_hit = True
        if et == "DatatypeProperty" and is_micro(llm_resp.suggested_domain):
            micro_hit = True

        add_check(
            "R_DYNAMIC_INDICATOR_NO_DIRECT_MICRO_SPATIAL",
            not micro_hit,
            "error",
            "动态指标疑似越级直连微观空间实体，违反拓扑硬约束。"
            if micro_hit
            else "未发现动态指标越级挂载到微观空间实体。",
        )
        if micro_hit:
            topology_failed = True

    # 最小规则清单覆盖检查：确保关键规则全部执行过，避免“漏检”。
    expected_rule_ids: List[str] = []
    if decision == "POSITIONED":
        if et == "Class":
            expected_rule_ids = ["R_CLASS_PARENT_REQUIRED", "R_CLASS_PARENT_ANCHOR_EXISTS"]
        elif et == "ObjectProperty":
            expected_rule_ids = [
                "R_OP_DOMAIN_RANGE_REQUIRED",
                "R_OP_DOMAIN_ANCHOR_EXISTS",
                "R_OP_RANGE_ANCHOR_EXISTS",
            ]
        elif et == "DatatypeProperty":
            expected_rule_ids = ["R_DP_XSD_REQUIRED_VALID", "R_DP_DOMAIN_ANCHOR_EXISTS"]
        if is_indicator_like:
            expected_rule_ids.append("R_DYNAMIC_INDICATOR_NO_DIRECT_MICRO_SPATIAL")

    if expected_rule_ids:
        existing_ids = {c.rule_id for c in checks}
        missing_ids = [rid for rid in expected_rule_ids if rid not in existing_ids]
        add_check(
            "R_MIN_RULESET_COVERAGE",
            len(missing_ids) == 0,
            "warning",
            "关键规则覆盖完整。"
            if not missing_ids
            else f"关键规则缺失: {', '.join(missing_ids)}",
        )
        if missing_ids:
            anchor_failed = True

    final_decision = decision
    final_reasoning = llm_resp.reasoning
    final_confidence_level = llm_resp.confidence_level

    # 规则后处理：硬错误降级为 REJECTED，锚点缺失降级为 UNSURE。
    if decision == "POSITIONED":
        if structural_failed or topology_failed:
            final_decision = "REJECTED"
            final_reasoning = f"{llm_resp.reasoning} [LOCAL_RULE_BLOCK: 结构或拓扑硬约束失败]"
            final_confidence_level = None
        elif anchor_failed:
            final_decision = "UNSURE"
            final_reasoning = f"{llm_resp.reasoning} [LOCAL_RULE_DEGRADE: 锚点未命中，降级为 UNSURE]"
            final_confidence_level = None
        else:
            if missing_confidence_level:
                final_confidence_level = "LOW"
            elif final_confidence_level == "HIGH":
                has_local_risk = any((not c.passed) for c in checks)
                if has_local_risk:
                    final_confidence_level = "LOW"

    if final_decision != "POSITIONED":
        final_confidence_level = None

    evidence_refs = build_local_evidence_refs(rec, max_items=3)
    confidence, confidence_source = compute_local_confidence(
        final_decision,
        checks,
        llm_failed=llm_failed,
        reasoning=final_reasoning,
        evidence_count=len(evidence_refs),
    )

    merged = PositioningResponse(
        decision=final_decision,
        confidence_level=final_confidence_level,
        confidence=confidence,
        confidence_source=confidence_source,
        reasoning=to_short_reasoning(final_reasoning),
        parent_class=clean_optional_text(llm_resp.parent_class),
        suggested_domain=clean_optional_text(llm_resp.suggested_domain),
        suggested_range=clean_optional_text(llm_resp.suggested_range),
        suggested_xsd_range=clean_optional_text(llm_resp.suggested_xsd_range),
        topology_checks=checks,
        evidence_refs=evidence_refs,
    )

    risk_flag = any(not c.passed for c in merged.topology_checks)
    return merged, [c.model_dump() for c in merged.topology_checks], risk_flag


def build_review_flat_row(
    rec: KeepNewRecord,
    final_resp: PositioningResponse,
    topology_risk_flag: bool,
) -> Dict[str, Any]:
    """将原生结果转换为阶段三审核台友好扁平行。"""
    first_ev = final_resp.evidence_refs[0] if final_resp.evidence_refs else None
    evidence_preview = (first_ev.evidence_text[:200] if first_ev else "").strip()
    evidence_source = (
        f"{first_ev.paper_id} | {first_ev.section} | {first_ev.chunk_id}" if first_ev else ""
    )

    parent_or_domain = ""
    range_or_xsd = ""
    if rec.element_type == "Class":
        parent_or_domain = final_resp.parent_class or ""
        range_or_xsd = ""
    elif rec.element_type == "ObjectProperty":
        parent_or_domain = final_resp.suggested_domain or ""
        range_or_xsd = final_resp.suggested_range or ""
    elif rec.element_type == "DatatypeProperty":
        parent_or_domain = final_resp.suggested_domain or ""
        range_or_xsd = final_resp.suggested_xsd_range or ""

    ce = rec.canonical_record.canonical_element
    name_zh = preferred_name_zh(ce)
    name_en = preferred_name_en(ce)
    name_raw = (ce.name_raw or "").strip()

    return {
        "cluster_id": rec.cluster_id,
        "concept_name": name_zh or name_en,
        "name_raw": name_raw,
        "name_zh": name_zh,
        "name_en": name_en,
        "type": rec.element_type,
        "parent_or_domain": parent_or_domain,
        "range_or_xsd": range_or_xsd,
        "decision": final_resp.decision,
        "confidence_level": final_resp.confidence_level or "",
        "confidence": final_resp.confidence,
        "confidence_source": final_resp.confidence_source,
        "reasoning": final_resp.reasoning,
        "evidence_text_preview": evidence_preview,
        "evidence_source": evidence_source,
        "topology_risk_flag": 1 if topology_risk_flag else 0,
        "expert_review": "",
    }


def ensure_paths() -> None:
    """运行前检查关键输入文件是否存在。"""
    required = [INPUT_KEEP_NEW_PATH, ONTOLOGY_TTL_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def main() -> None:
    """主流程：加载输入、串行定位、规则校验、双结果写盘。"""
    started = time.time()
    log_path = setup_console_log("phase2_04a_class_positioning")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    ensure_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_items = load_json_list(INPUT_KEEP_NEW_PATH)
    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    selected_raw = raw_items[START_INDEX:end_index]

    ttl_text = read_text(ONTOLOGY_TTL_PATH)
    class_anchors = extract_ttl_class_anchors(ttl_text)
    anchor_norm_set = {normalize_for_match(x) for x in class_anchors}
    system_blocks = build_system_blocks(ttl_text, class_anchors)

    print("=" * 84)
    print("[运行启动] Agent3 本体定位启动（拓扑挂载）")
    print(f"[输入信息] keep_new输入={INPUT_KEEP_NEW_PATH}")
    print(f"[输入信息] 本体TTL={ONTOLOGY_TTL_PATH}")
    print(f"[输入信息] 本次处理={len(selected_raw)} (START_INDEX={START_INDEX}, MAX_ITEMS={MAX_ITEMS})")
    print(f"[参数配置] class_anchor_count={len(class_anchors)} | model={CLAUDE_MODEL}")
    print(f"[调用方式] native_messages | prompt_cache={ENABLE_PROMPT_CACHE} | persistent_session={USE_PERSISTENT_SESSION}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print("=" * 84)

    validated: List[KeepNewRecord] = []
    errors: List[Dict[str, Any]] = []
    # 输入校验：确保承接 Agent2 的结构完全符合预期。
    for idx, raw in enumerate(selected_raw):
        try:
            item = KeepNewRecord.model_validate(raw)
            if item.element_type != "Class":
                errors.append(
                    {
                        "phase": "validate_input",
                        "index": START_INDEX + idx,
                        "error_type": "unexpected_element_type",
                        "error": f"phase2_04a only accepts Class, got {item.element_type}",
                        "raw": raw,
                    }
                )
                continue
            validated.append(item)
        except ValidationError as e:
            errors.append(
                {
                    "phase": "validate_input",
                    "index": START_INDEX + idx,
                    "error_type": "validation_error",
                    "error": str(e),
                    "raw": raw,
                }
            )
    write_json(OUTPUT_ERRORS, errors)
    print(f"[校验结果] 有效记录={len(validated)} | 无效记录={len(selected_raw) - len(validated)}")

    if not validated:
        write_json(OUTPUT_DECISIONS, [])
        write_json(OUTPUT_REVIEW_FLAT, [])
        write_json(OUTPUT_STATS, build_class_positioning_stats([]))
        manifest = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "no_valid_input",
            "input_keep_new_file": str(INPUT_KEEP_NEW_PATH),
            "ontology_ttl_file": str(ONTOLOGY_TTL_PATH),
            "positioning_scope": "Class",
            "selected_total": len(selected_raw),
            "valid_total": 0,
            "invalid_total": len(errors),
            "elapsed_sec": round(time.time() - started, 3),
        }
        manifest["class_stats_file"] = str(OUTPUT_STATS)
        write_json(OUTPUT_MANIFEST, manifest)
        print("[运行结束] 没有可处理的有效输入记录。")
        return

    if not API_KEY:
        raise RuntimeError("Missing API Key. Please fill API_KEY in the script config.")

    session = requests.Session() if USE_PERSISTENT_SESSION else requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
        }
    )
    if ANTHROPIC_BETA.strip():
        session.headers["anthropic-beta"] = ANTHROPIC_BETA.strip()

    decisions: List[Dict[str, Any]] = []
    review_flat: List[Dict[str, Any]] = []
    token_usage_counter: Dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    positioned_count = 0
    positioned_high_count = 0
    positioned_low_count = 0
    unsure_count = 0
    rejected_count = 0
    cache_hit_item_count = 0

    for i, rec in enumerate(validated, start=1):
        ce = rec.canonical_record.canonical_element
        print(f"\n[开始处理] [{i}/{len(validated)}] {rec.cluster_id} | {rec.element_type} | {display_name(ce)}")

        llm_resp: Optional[LLMPositioningResponse] = None
        llm_call_meta: Dict[str, Any] = {
            "usage_metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "total_input_for_threshold": 0,
            },
            "stop_reason": "",
            "response_shape": "not_called",
        }
        llm_failed = False
        last_error = ""
        started_one = time.time()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  - [模型调用] 第 {attempt}/{MAX_RETRIES} 次尝试...")
                before_cache_read = int(token_usage_counter.get("cache_read_input_tokens", 0))
                llm_resp, llm_call_meta = call_claude_positioning(session, system_blocks, rec, token_usage_counter)
                after_cache_read = int(token_usage_counter.get("cache_read_input_tokens", 0))
                used_cache_this_item = after_cache_read > before_cache_read
                usage_metrics = llm_call_meta.get("usage_metrics", {})
                print(
                    "  - [Token明细] "
                    f"input={int(usage_metrics.get('input_tokens', 0))} | "
                    f"output={int(usage_metrics.get('output_tokens', 0))} | "
                    f"cache_create={int(usage_metrics.get('cache_creation_input_tokens', 0))} | "
                    f"cache_read={int(usage_metrics.get('cache_read_input_tokens', 0))} | "
                    f"total_input_for_threshold={int(usage_metrics.get('total_input_for_threshold', 0))} | "
                    f"cache_hit={'Y' if used_cache_this_item else 'N'} | "
                    f"shape={llm_call_meta.get('response_shape', '-') or '-'} | "
                    f"stop_reason={llm_call_meta.get('stop_reason', '-') or '-'}"
                )
                break
            except Exception as e:  # noqa: BLE001
                last_error = f"{type(e).__name__}: {e}"
                print(f"  - [请求失败] {last_error}")
                if attempt < MAX_RETRIES:
                    sleep_sec = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    print(f"  - [重试等待] {sleep_sec:.1f}s 后重试...")
                    time.sleep(sleep_sec)

        if llm_resp is None:
            # LLM 失败时不终止全流程，记录错误并转为保守决策。
            llm_failed = True
            used_cache_this_item = False
            errors.append(
                {
                    "phase": "llm_call",
                    "cluster_id": rec.cluster_id,
                    "error_type": "llm_failed_after_retries",
                    "error": last_error,
                    "name_zh": preferred_name_zh(ce),
                    "name_en": preferred_name_en(ce),
                }
            )
            llm_resp = fallback_unsure_response(rec, last_error)

        final_resp, local_checks_dump, risk_flag = apply_local_rules(rec, llm_resp, anchor_norm_set, llm_failed)
        elapsed_one = round(time.time() - started_one, 3)
        if used_cache_this_item:
            cache_hit_item_count += 1

        if final_resp.decision == "POSITIONED":
            positioned_count += 1
            if final_resp.confidence_level == "HIGH":
                positioned_high_count += 1
            elif final_resp.confidence_level == "LOW":
                positioned_low_count += 1
        elif final_resp.decision == "UNSURE":
            unsure_count += 1
        else:
            rejected_count += 1

        decision_record = {
            "cluster_id": rec.cluster_id,
            "element_type": rec.element_type,
            "name_raw": (ce.name_raw or "").strip(),
            "name_zh": preferred_name_zh(ce),
            "name_en": preferred_name_en(ce),
            "decision": final_resp.decision,
            "confidence_level": final_resp.confidence_level,
            "confidence": final_resp.confidence,
            "confidence_source": final_resp.confidence_source,
            "parent_class": final_resp.parent_class,
            "suggested_domain": final_resp.suggested_domain,
            "suggested_range": final_resp.suggested_range,
            "suggested_xsd_range": final_resp.suggested_xsd_range,
            "reasoning": final_resp.reasoning,
            "topology_checks": local_checks_dump,
            "evidence_refs": [x.model_dump() for x in final_resp.evidence_refs],
            "input_snapshot": {
                "aliases": rec.canonical_record.aliases,
                "merged_definitions": rec.canonical_record.merged_definitions[:5],
                "mapping_suggestion": ce.mapping_suggestion.model_dump(),
                "external_alignment": rec.external_alignment.model_dump(),
            },
            "llm_call_meta": llm_call_meta,
            "used_cache_this_item": used_cache_this_item,
            "elapsed_sec": elapsed_one,
        }
        decisions.append(decision_record)

        review_flat.append(build_review_flat_row(rec, final_resp, risk_flag))

        print(
            f"  - [处理完成] decision={final_resp.decision} | confidence_level={final_resp.confidence_level or '-'} | "
            f"confidence={final_resp.confidence:.2f} | "
            f"risk_flag={1 if risk_flag else 0} | elapsed={elapsed_one}s"
        )

        # 增量写盘，保证中断后可观察、可恢复。
        write_json(OUTPUT_DECISIONS, decisions)
        write_json(OUTPUT_REVIEW_FLAT, review_flat)
        write_json(OUTPUT_ERRORS, errors)

    elapsed = round(time.time() - started, 3)
    token_report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_tokens": int(token_usage_counter.get("input_tokens", 0)),
        "output_tokens": int(token_usage_counter.get("output_tokens", 0)),
        "cache_creation_input_tokens": int(token_usage_counter.get("cache_creation_input_tokens", 0)),
        "cache_read_input_tokens": int(token_usage_counter.get("cache_read_input_tokens", 0)),
        "total_input_for_threshold": int(token_usage_counter.get("input_tokens", 0))
        + int(token_usage_counter.get("cache_creation_input_tokens", 0))
        + int(token_usage_counter.get("cache_read_input_tokens", 0)),
        "cache_hit_item_count": cache_hit_item_count,
        "cache_hit_ratio": round(cache_hit_item_count / len(decisions), 3) if decisions else 0.0,
    }
    write_json(OUTPUT_TOKEN_REPORT, token_report)
    stats = build_class_positioning_stats(decisions)
    write_json(OUTPUT_STATS, stats)

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_keep_new_file": str(INPUT_KEEP_NEW_PATH),
        "ontology_ttl_file": str(ONTOLOGY_TTL_PATH),
        "output_dir": str(OUTPUT_DIR),
        "positioning_scope": "Class",
        "model": {
            "name": CLAUDE_MODEL,
            "base_url": CLAUDE_NATIVE_BASE_URL,
            "messages_path": MESSAGES_PATH,
            "anthropic_version": ANTHROPIC_VERSION,
            "anthropic_beta": ANTHROPIC_BETA,
            "api_timeout_seconds": API_TIMEOUT_SECONDS,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_p": TOP_P,
            "enable_prompt_cache": ENABLE_PROMPT_CACHE,
            "use_persistent_session": USE_PERSISTENT_SESSION,
        },
        "retry_policy": {
            "max_retries": MAX_RETRIES,
            "retry_base_seconds": RETRY_BASE_SECONDS,
        },
        "run_range": {
            "start_index": START_INDEX,
            "max_items": MAX_ITEMS,
            "selected_total": len(selected_raw),
        },
        "counts": {
            "input_total": len(raw_items),
            "valid_total": len(validated),
            "invalid_total": len(selected_raw) - len(validated),
            "error_total": len(errors),
            "decision_total": len(decisions),
            "positioned_total": positioned_count,
            "positioned_high_total": positioned_high_count,
            "positioned_low_total": positioned_low_count,
            "unsure_total": unsure_count,
            "rejected_total": rejected_count,
            "review_flat_total": len(review_flat),
            "ttl_class_anchor_total": len(class_anchors),
        },
        "token_usage": token_report,
        "class_stats_file": str(OUTPUT_STATS),
        "elapsed_sec": elapsed,
    }
    write_json(OUTPUT_MANIFEST, manifest)

    print("\n" + "=" * 84)
    print("[运行结束] Agent3 本体定位完成")
    print(
        f"[运行摘要] 决策总数={len(decisions)} | positioned={positioned_count} | "
        f"unsure={unsure_count} | rejected={rejected_count} | errors={len(errors)}"
    )
    print(f"[运行摘要] positioned_high={positioned_high_count} | positioned_low={positioned_low_count}")
    print(f"[性能统计] 总耗时={elapsed}s")
    print(
        f"[性能统计] input_tokens={token_report['input_tokens']} | output_tokens={token_report['output_tokens']} | "
        f"cache_create={token_report['cache_creation_input_tokens']} | cache_read={token_report['cache_read_input_tokens']}"
    )
    print(f"[关键输出] {OUTPUT_DECISIONS} -> Agent3 原生定位结果")
    print(f"[关键输出] {OUTPUT_REVIEW_FLAT} -> 审核台友好扁平结果")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要")
    print(f"[关键输出] {OUTPUT_TOKEN_REPORT} -> 输入/输出 token 汇总")
    print(f"[关键输出] {OUTPUT_ERRORS} -> 错误明细")
    print("=" * 84)


if __name__ == "__main__":
    main()
