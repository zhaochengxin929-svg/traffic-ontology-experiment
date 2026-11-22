import atexit
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import requests
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


# ==========================================
# 配置区（常改）
# ==========================================
# Claude 兼容网关 API Key，可直接填写
API_KEY = ""
# 概念抽取使用的模型名称
CLAUDE_MODEL = "claude-opus-4-6"
# Claude Native Messages 网关地址
CLAUDE_NATIVE_BASE_URL = "https://api2.aigcbest.top"
# Claude Native Messages 路径
MESSAGES_PATH = "/v1/messages"
# Claude Native 接口版本
ANTHROPIC_VERSION = "2023-06-01"
# 可选 Beta Header；默认留空
ANTHROPIC_BETA = ""
# 单次请求超时时间（秒）
API_TIMEOUT_SECONDS = 120
# 生成温度，结构化抽取建议保持低温
TEMPERATURE = 0.1
# 单次最大输出 token
MAX_TOKENS = 4096
# top_p 采样参数；None 表示使用模型默认值
TOP_P = None
# 是否在 Native Messages 路径上启用 prompt cache
ENABLE_PROMPT_CACHE = True
# 是否使用 requests.Session 复用连接
USE_PERSISTENT_SESSION = True
# 是否启用扩展思考；当前试验链默认关闭，优先验证稳定 JSON 与缓存行为
THINKING_MODE = "off"
# adaptive 模式下的思考力度（当前试验链未启用，仅保留配置位）
THINKING_EFFORT = "medium"
# manual 模式下的思考预算（当前试验链未启用，仅保留配置位）
THINKING_BUDGET_TOKENS = 2000
# 单条样本最大重试次数
MAX_RETRIES = 3
# 指数退避基准秒数
RETRY_BASE_SECONDS = 2.0
# 本轮处理条数，试验链默认小样本验证；改为 0 表示处理全部
MAX_ITEMS = 0
# 起始索引，用于断点续跑
START_INDEX = 0
# 是否启用本地 JSON 修复
ENABLE_JSON_REPAIR = True
# 是否启用逐条元素抢救
ENABLE_ITEM_SALVAGE = True
# 逐条抢救后至少保留多少条，才视为该 chunk 成功
MIN_VALID_ITEMS_TO_ACCEPT = 1
# 当前脚本目录：大规模实验/phase2/scripts
SCRIPT_DIR = Path(__file__).resolve().parent
# 当前阶段目录：大规模实验/phase2
PHASE2_DIR = SCRIPT_DIR.parent
# 项目根目录：大规模实验
PROJECT_ROOT = PHASE2_DIR.parent
# 共享资源目录：统一存放 schema、词表、本体文件
SHARED_RESOURCES_DIR = PROJECT_ROOT / "shared_resources"
INPUT_ROUND_ID = "R4"
RESOURCE_VERSION = "V3"
# 是否将控制台输出同时写入日志文件
ENABLE_FILE_LOG = True
# 日志目录：统一保存脚本控制台输出
LOG_DIR = PROJECT_ROOT / "logs"
# Agent1 输入的 Mini Schema 文件
MINI_SCHEMA_PATH = SHARED_RESOURCES_DIR / f"Agent1_Mini_Schema_{RESOURCE_VERSION}.ttl"
# Agent1 输入语料：阶段一输出的理论语料 JSON
INPUT_JSON_PATH = PROJECT_ROOT / "phase1" / "outputs_clean" / f"cleaned_theory_corpus_{INPUT_ROUND_ID}.json"
# Agent1 输出目录
OUTPUT_DIR = PHASE2_DIR / "outputs_phase2_01"
# 完整抽取结果
OUTPUT_RAW_RESULTS = OUTPUT_DIR / "agent1_extractions.json"
# 通过 TBox 检查的概念结果
OUTPUT_PASSED_TBOX = OUTPUT_DIR / "agent1_passed_tbox.json"
# 被过滤的实例级结果
OUTPUT_BLOCKED_ABOX = OUTPUT_DIR / "agent1_filtered_abox.json"
# 运行错误与失败样本
OUTPUT_ERRORS = OUTPUT_DIR / "agent1_errors.json"
# 运行摘要与统计信息
OUTPUT_MANIFEST = OUTPUT_DIR / "agent1_manifest.json"
# token 汇总报表：用于小样本成本预估
OUTPUT_TOKEN_REPORT = OUTPUT_DIR / "agent1_token_report.json"
# 每个 chunk 的统计明细：用于观察单条 chunk 的耗时、token 与元素类型分布
OUTPUT_CHUNK_STATS = OUTPUT_DIR / "agent1_chunk_stats.json"
# 原始大模型响应日志（保留 native messages 原始返回，便于追溯缓存行为）
OUTPUT_RAW_API_LOG = OUTPUT_DIR / "agent1_llm_raw_responses.jsonl"
# JSON 修复与逐条抢救日志
OUTPUT_REPAIR_LOG = OUTPUT_DIR / "agent1_repair_log.jsonl"
# 逐条抢救时被丢弃的元素清单
OUTPUT_DROPPED_ITEMS = OUTPUT_DIR / "agent1_dropped_elements.json"
KNOWN_OBJECT_PROPERTY_NAMES: set[str] = set()
_LOG_FILE_HANDLE = None
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


class TeeStream:
    """将控制台输出同时写入原始流与日志文件。"""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except ValueError:
                continue
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
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
        sys.stdout = _ORIGINAL_STDOUT
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
# Claude Structured Outputs 的 JSON Schema（按 element_type 分型，禁止输出无关 null 字段）。
CLASS_ELEMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name_raw",
        "name_zh",
        "name_en",
        "element_type",
        "definition",
        "mapping_suggestion",
        "is_instance",
        "evidence_text",
    ],
    "properties": {
        "name_raw": {"type": "string", "minLength": 1},
        "name_zh": {"type": "string", "minLength": 1},
        "name_en": {"type": "string", "minLength": 1},
        "element_type": {"type": "string", "enum": ["Class"]},
        "definition": {"type": "string", "minLength": 1},
        "mapping_suggestion": {
            "type": "object",
            "additionalProperties": False,
            "required": ["suggested_parent_class", "reason"],
            "properties": {
                "suggested_parent_class": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
            },
        },
        "is_instance": {"type": "boolean"},
        "evidence_text": {"type": "string", "minLength": 1},
    },
}

OBJECT_PROPERTY_ELEMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name_raw",
        "name_zh",
        "name_en",
        "element_type",
        "definition",
        "mapping_suggestion",
        "is_instance",
        "evidence_text",
    ],
    "properties": {
        "name_raw": {"type": "string", "minLength": 1},
        "name_zh": {"type": "string", "minLength": 1},
        "name_en": {"type": "string", "minLength": 1},
        "element_type": {"type": "string", "enum": ["ObjectProperty"]},
        "definition": {"type": "string", "minLength": 1},
        "mapping_suggestion": {
            "type": "object",
            "additionalProperties": False,
            "required": ["suggested_domain", "suggested_range", "candidate_super_property", "reason"],
            "properties": {
                "suggested_domain": {"type": "string", "minLength": 1},
                "suggested_range": {"type": "string", "minLength": 1},
                "candidate_super_property": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ]
                },
                "reason": {"type": "string", "minLength": 1},
            },
        },
        "is_instance": {"type": "boolean"},
        "evidence_text": {"type": "string", "minLength": 1},
    },
}

DATATYPE_PROPERTY_ELEMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name_raw",
        "name_zh",
        "name_en",
        "element_type",
        "definition",
        "mapping_suggestion",
        "is_instance",
        "evidence_text",
    ],
    "properties": {
        "name_raw": {"type": "string", "minLength": 1},
        "name_zh": {"type": "string", "minLength": 1},
        "name_en": {"type": "string", "minLength": 1},
        "element_type": {"type": "string", "enum": ["DatatypeProperty"]},
        "definition": {"type": "string", "minLength": 1},
        "mapping_suggestion": {
            "type": "object",
            "additionalProperties": False,
            "required": ["suggested_domain", "suggested_xsd_range", "reason"],
            "properties": {
                "suggested_domain": {"type": "string", "minLength": 1},
                "suggested_xsd_range": {
                    "type": "string",
                    "enum": ["xsd:string", "xsd:float", "xsd:int", "xsd:boolean", "xsd:dateTime"],
                },
                "reason": {"type": "string", "minLength": 1},
            },
        },
        "is_instance": {"type": "boolean"},
        "evidence_text": {"type": "string", "minLength": 1},
    },
}

CLAUDE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["elements"],
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "oneOf": [
                    CLASS_ELEMENT_SCHEMA,
                    OBJECT_PROPERTY_ELEMENT_SCHEMA,
                    DATATYPE_PROPERTY_ELEMENT_SCHEMA,
                ]
            },
        }
    },
}


SYSTEM_PROMPT_TEMPLATE = """你需要承担城市交通态势本体要素抽取任务。

该任务面向城市交通态势本体的模式层扩展，从单条交通领域文献片段中识别具有本体建模价值的候选要素。候选要素包括三类：
- Class
- ObjectProperty
- DatatypeProperty

本任务只处理本体模式层内容，不进行实例层建模。

【输入上下文】
你将收到两类输入：

1) 城市交通态势本体简化结构（Mini Schema，Turtle文本）
2) 单条知识输入单元，包括 paper_id、section、chunk_id 和 text

本体简化结构用于提供既有本体的核心类层次、关键对象属性、数据属性及结构约束。知识输入单元用于提供当前待分析的交通领域文本内容。

【任务目标】
请在本体简化结构的约束下，从当前知识输入单元中识别具有本体建模价值的候选要素，并输出结构化结果。

候选要素必须满足以下基本条件：
1. 具有相对稳定的语义边界；
2. 与城市交通态势知识表达相关；
3. 能够作为类、对象属性或数据属性进入本体模式层；
4. 有当前知识输入单元中的原文依据支撑；
5. 能够给出合理的定义说明和初步结构建议。

【链式处理步骤】
请按照以下顺序完成判断。每一步均需结合对应规则模块执行，后一步以前一步的判断结果为基础。

Step 1：理解本体先验
- 读取内容：本体简化结构。
- 对应规则：【本体先验与建模意图】。
- 处理目标：理解既有本体中的核心父类、对象属性、数据属性及其适用范围。
- 结果作用：为候选识别、类型判定和结构建议提供本体结构约束。

Step 2：理解知识输入单元语义
- 读取内容：paper_id、section、chunk_id、text。
- 对应规则：【候选要素识别规则】。
- 处理目标：判断当前文本是否包含交通对象、交通状态、交通指标、拥堵过程、交通事件、空间关系、时间关系或指标关系等内容。
- 结果作用：确定当前文本中可能具有本体建模价值的语义范围。

Step 3：识别候选要素
- 读取内容：知识输入单元中的相关表述及原文证据。
- 对应规则：【候选要素识别规则】。
- 处理目标：从文本中识别可能作为 Class、ObjectProperty 或 DatatypeProperty 的候选内容。
- 结果作用：形成候选名称、候选定义、候选类型线索和原文依据。

Step 4：执行边界控制
- 读取内容：Step 3 形成的候选内容。
- 对应规则：【通用边界控制规则】和【实例与模式层边界规则】。
- 处理目标：排除具体实例、算法过程、模型方法、实验设置、研究任务、系统平台、纯技术流程等不应进入本体模式层的内容。
- 结果作用：保留具有本体建模价值的候选要素，并对实例性内容设置 is_instance 标记。

Step 5：完成类型专属判定与初步结构建议
- 读取内容：经过边界控制的候选要素、本体简化结构中的类层次和属性结构。
- 对应规则：【类型专属判定与初步结构建议】和【局部闭合与保守原则】。
- 处理目标：判断候选要素应作为 Class、ObjectProperty 还是 DatatypeProperty，并生成对应的初步结构建议。
- 结果作用：
  - Class 输出 suggested_parent_class；
  - ObjectProperty 输出 suggested_domain、suggested_range 和 candidate_super_property；
  - DatatypeProperty 输出 suggested_domain 和 suggested_xsd_range。

Step 6：执行输出检查
- 读取内容：全部候选要素及其结构化字段。
- 对应规则：【名称与引用格式】【字段定义】【输出要求】。
- 处理目标：检查名称、定义、原文依据、类型判定、结构建议、reason 和 JSON 格式是否符合要求。
- 结果作用：输出严格 JSON 格式的结构化候选要素结果，不输出额外解释性内容。

【本体先验与建模意图】
1. 本体简化结构是当前阶段本体要素抽取的结构先验。抽取候选要素时，应优先判断其是否属于本体简化结构中已有父类家族，而不是直接假设需要建立新的独立体系。
2. 当前城市交通态势本体中，状态实体主要承担结构连接与中转作用，用于连接空间、时间以及具体业务概念。真正承载交通业务语义的通常是具体指标、具体拥堵或运行现象、具体事件、演化过程、运行等级或其他可独立定义的业务概念。
3. 对同时带有“状态色彩”和“业务语义”的候选内容，不要只根据字面表述判断。只要其核心作用是表达具体交通运行指标、拥堵现象、运行事件、演化过程、运行等级或其他可独立建模的业务语义，应优先归入相应业务类，而不是优先归入状态类。
4. 对动态、时变、可在特定空间—时间条件下被观测、判别或度量的交通运行概念，应优先判断其是否适合作为 Class，尤其是交通指标、交通状态、拥堵现象和交通事件相关概念。
5. 对静态、稳定、配置型、固有属性，且不承载独立交通业务语义的内容，应优先判断其是否适合作为 DatatypeProperty。
6. 不要因为某个术语可以量化，就自动将其判为 DatatypeProperty。若它本身是动态交通指标、交通状态或交通事件概念，仍应优先作为 Class 候选。

【候选要素识别规则】
1. Class 候选通常对应可独立定义、分类、挂载父类、对齐和扩展的交通业务概念。
2. ObjectProperty 候选通常对应两个独立概念类之间稳定、抽象、可复用的语义关系。
3. DatatypeProperty 候选通常对应某个对象的字面值属性，不需要作为独立交通业务概念建模。
4. 只提取当前知识输入单元中最核心、最具代表性、最符合城市交通态势本体表达需求的候选要素。
5. 不要为了增加数量而抽取泛化词、边缘词、普通描述词或无法稳定挂接到本体结构中的表达。
6. 若当前文本只暗示某关系或属性，但无法明确其对象范围、语义含义或结构位置，优先不抽取。

【通用边界控制规则】
以下内容默认不作为本体要素抽取，除非文本明确将其定义为稳定的交通业务对象、交通运行概念、交通事件、交通指标、空间实体或可复用对象关系：

1. 研究任务、分析任务、预测任务、估计任务、识别任务；
2. 控制方法、优化算法、求解策略、建模方法；
3. 理论名称、模型名称、框架名称、系统名称、平台名称；
4. 算法流程、建模流程、实验流程、数据处理流程；
5. 系统架构、通信环境、控制器、设备、工具组件；
6. 纯实验设定、参数配置、实现细节、评价过程；
7. 特定数据集下的局部统计发现或条件性经验结论。

方法语境本身不构成排除条件。若某术语虽然出现在方法、模型或过程描述中，但其本身明确表示交通状态、交通指标、交通事件、空间实体或稳定对象关系，仍可作为候选要素进入类型判定。

【实例与模式层边界规则】
1. 本任务只做本体模式层抽取，不做实例层建模。
2. 具体路名、交叉口名、区域名、地点名、具体日期、具体时刻、具体事件个体、具体编号、具体数值等通常属于实例或属性值，不应作为普通模式层要素直接抽取。
3. 若某内容具有实例性，但模型仍识别其有记录价值，必须标记 is_instance=true。
4. 若某候选缺少稳定定义、缺少原文证据、语义边界不清，优先不抽取。
5. 与交通相关不等于可进入本体模式层。只有具备本体建模价值、语义边界稳定且有证据支撑的内容，才允许进入 elements。

【类型专属判定与初步结构建议】

一、Class 判定与父类建议

1. 正向判定规则
（1）若候选术语本身是需要被独立定义、分类、挂父类、对齐和扩展的交通概念，则优先抽为 Class。
（2）动态、时变、可在具体空间—时间条件下被观测、判别或度量的运行概念，优先抽为 Class。
（3）交通状态、交通指标、拥堵现象、拥堵演化阶段、交通事件、运行等级、空间对象等内容，若在文本中具有稳定语义边界，可以作为 Class 候选。
（4）若候选概念在不同空间尺度下对应不同具体语义，且当前文本能够判断其适用空间对象，应优先在名称中体现空间尺度，避免使用过于泛化的同名表述。

2. 父类建议规则
（1）mapping_suggestion.suggested_parent_class 必须优先来自本体简化结构中的已有类。
（2）若候选明显属于某个已有父类家族，应选择语义上最接近、层级上最稳定的已有父类。
（3）若候选是路段层级交通运行指标，优先建议挂载到 :SegmentIndicator 子类。
（4）若候选是交叉口层级交通运行指标，优先建议挂载到 :IntersectionIndicator 子类。
（5）若候选是区域或路网尺度交通运行指标，优先建议挂载到 :ZoneIndicator 子类。
（6）若候选是拥堵现象、运行事件、演化过程、运行等级等其他业务概念，应优先挂载到本体简化结构中语义最接近的已有业务父类。
（7）若存在多个接近父类，但它们属于同一家族或存在明显上下位关系，应优先选择最接近、最具体的已有父类。
（8）suggested_parent_class = "UNSURE" 不是默认安全出口。只有确实找不到合适父类家族时，才允许使用 "UNSURE"。
（9）若候选本质上属于交通状态、交通事件、交通指标或空间实体，且本体简化结构中已存在语义上足够接近的上位类，应优先选择最近的已有父类，不要仅因名称不完全一致而返回 "UNSURE"。

3. Class 类型防错规则
（1）不得把日期、时间、具体数值、ID、名称字符串等属性值或实例值当作 Class。
（2）不得把研究任务、理论名称、模型框架、算法流程、系统平台等方法性或技术性术语误判为 Class。
（3）若某术语只是论文中的分析过程或研究任务，且未被定义为稳定交通业务概念，不得作为 Class 输出。
（4）若 reason 已经说明该候选不适合作为 Class，则不得将其写入 elements。

二、ObjectProperty 判定与关系结构建议

1. 正向判定规则
（1）当候选内容能够稳定表达为“对象—对象”的二元关系，且该关系对城市交通态势本体表达具有建模价值时，可以抽取为 ObjectProperty。
（2）ObjectProperty 表达的是交通态势及其相关对象之间稳定、抽象、可复用的结构语义关系，不是论文中所有“影响”“相关”“导致”等表述。
（3）优先关注以下具有本体建模价值的关系类型：
  - 空间结构关系：表示交通对象之间的空间组织或拓扑联系，如包含、属于、位于、相邻、连通、上游、下游等；
  - 空间对象与状态或事件的承载关系：表示某状态或事件发生、作用或对应于某空间对象；
  - 状态、事件与过程之间的作用或演化关系：表示诱发、触发、传播、加剧、缓解、约束、接续等关系；
  - 指标与状态或事件之间的表征关系：表示某指标用于反映、表征、指示、评价或衡量某种交通状态或事件；
  - 指标与指标之间的构成关系：表示某指标由其他指标计算、聚合、派生、加权或组合得到。
（4）若某关系本质上由多个来源概念共同构成，应优先拆解为多条稳定的二元 ObjectProperty，不要把多个来源对象直接拼入同一个对象属性名称。

2. 定义域、值域和父属性建议规则
（1）ObjectProperty 用于连接对象与对象，因此 suggested_domain 和 suggested_range 必须都对应 Class，不能是 xsd 数据类型。
（2）若 suggested_domain 或 suggested_range 指向当前本体简化结构中不存在的新类，必须检查该新类是否能在当前知识输入单元中被明确识别、命名并给出稳定定义。
（3）若该新类在当前知识输入单元中可以被明确识别，应同步抽取该 Class。
（4）若该新类不能被明确识别、命名或定义，优先不抽取依赖该悬空类的 ObjectProperty。
（5）若对象属性名称已经具体到某些明确对象或指标，且当前知识输入单元内能稳定闭合，应尽量给出更具体的 suggested_domain 和 suggested_range；否则应保留在稳定父类层级。
（6）candidate_super_property 仅用于表示候选父属性引用。只有当前关系明显是本体简化结构中某个已有 ObjectProperty 的稳定下位特化时，才允许填写。
（7）若只是名称相似、语义接近或大致属于同一关系族，但不能确认其为稳定下位关系，则 candidate_super_property 必须返回 null。
（8）若填写 candidate_super_property，当前关系仍应作为独立候选关系输出，且 name_en 必须与父属性名不同，避免后续 IRI 冲突。
（9）candidate_super_property 只能引用当前本体简化结构中已存在的合法 ObjectProperty，不得臆造新的父属性。

3. ObjectProperty 类型防错规则
（1）若候选内容本质上是概念、指标、状态、事件、过程、模型、方法或系统，而不是关系表达，不得抽为 ObjectProperty。
（2）若关系只是特定实验、特定数据、特定模型条件下得到的经验性统计发现，且不具备稳定领域语义，不得抽为 ObjectProperty。
（3）若关系两端对象无法明确识别，或定义域和值域无法在当前知识输入单元内形成基本闭合，优先不抽取。
（4）不得把一般性动词或松散影响表述直接抽为 ObjectProperty。

三、DatatypeProperty 判定与属性结构建议

1. 正向判定规则
（1）若候选内容只是某个对象的字面值属性，且不需要作为独立交通业务概念建模，可以抽取为 DatatypeProperty。
（2）事件、状态或时间实体的具体时间值属性，如开始时间、结束时间、恢复时间、持续时间等，可以作为 DatatypeProperty 候选。
（3）空间实体或交通对象的静态、稳定、配置型属性，如长度、方向、编码、车道数、功能等级等，可以作为 DatatypeProperty 候选。
（4）若候选术语本质上只是已有实体的静态、稳定、非时变特征，不承载独立交通业务语义，也不需要在时空上下文中单独实例化，应优先判为 DatatypeProperty，而不是 Class。

2. 定义域和值类型建议规则
（1）DatatypeProperty 用于连接对象与数据值，因此 suggested_xsd_range 必须是合法 xsd 数据类型。
（2）suggested_xsd_range 只能从以下枚举中选择：
    xsd:string / xsd:float / xsd:int / xsd:boolean / xsd:dateTime
（3）若 DatatypeProperty 的 suggested_domain 指向当前本体简化结构中不存在的新类，必须检查该新类是否能在当前知识输入单元中被明确识别、命名并给出稳定定义。
（4）若该新类可以被明确识别，应同步抽取该 Class。
（5）若该新类不能被明确识别、命名或定义，优先不抽取依赖该悬空类的 DatatypeProperty。

3. DatatypeProperty 类型防错规则
（1）DatatypeProperty 不能连接到 Class，只能连接对象与字面值。
（2）不得把动态交通指标、交通状态、拥堵现象或交通事件名称误降为 DatatypeProperty。
（3）不得因为某术语可以量化，就自动判为 DatatypeProperty。若它本身是动态交通指标、交通状态或交通事件概念，仍应优先判为 Class。
（4）不得把需要独立定义、分类、对齐和挂载父类的交通业务概念降为普通数据属性。

【局部闭合与保守原则】
1. 本任务只要求当前知识输入单元内局部闭合，不要求跨知识输入单元推理补全。
2. 若候选属性依赖的新类在当前知识输入单元中无法明确识别，优先不抽取该属性。
3. 若候选要素的父类、定义域、值域或值类型无法稳定判断，应按规则保守处理。
4. 若已经能够明确说出一个或少数几个可接受的结构位置，且这些位置在本体简化结构中真实存在，应优先选择最合适的一个，不要轻易返回 "UNSURE"。
5. 若确实无法判断结构位置，但候选本身具有较高本体建模价值，可以保留候选并在相应结构字段中按规则使用 "UNSURE"。
6. 若候选本身不具备本体建模价值，不得为了保留结果而输出 "UNSURE"。

【名称与引用格式】
1. name_raw / name_zh / name_en 表示候选要素的名称。
2. suggested_parent_class / suggested_domain / suggested_range 表示指向某个类的引用。
3. candidate_super_property 表示指向某个已有对象属性的候选引用，仅用于 ObjectProperty。
4. name_en 必须是不带前导冒号的英文 local name，用于后续生成 IRI 本地名，并按 element_type 使用规范命名：
  - Class：使用 PascalCase，例如 QueueLength、CongestionEvolution；
  - ObjectProperty：使用 lowerCamelCase，例如 hasTrafficState、linkedToTrafficState；
  - DatatypeProperty：使用 lowerCamelCase，例如 eventStartTime、restorationTime。
5. 不允许把属性的 name_en 写成 PascalCase，也不允许把 Class 的 name_en 写成 lowerCamelCase。
6. suggested_parent_class / suggested_domain / suggested_range 若不是 "UNSURE"，必须使用带前导冒号的类引用形式，例如 :TrafficState、:RoadSegment。
7. candidate_super_property 若不是 null，必须使用带前导冒号的对象属性引用形式，例如 :computedFrom、:triggeredByEvent。
8. suggested_xsd_range 必须使用合法 xsd 数据类型枚举值，不允许输出 String、float、:String、:Float 等非规范写法。

【字段定义】
每个候选要素必须包含以下字段：
- name_raw: string
- name_zh: string
- name_en: string
- element_type: "Class" | "ObjectProperty" | "DatatypeProperty"
- definition: string
- mapping_suggestion: object
  - 若 element_type="Class":
    { "suggested_parent_class": string, "reason": string }
  - 若 element_type="ObjectProperty":
    { "suggested_domain": string, "suggested_range": string, "candidate_super_property": string|null, "reason": string }
  - 若 element_type="DatatypeProperty":
    { "suggested_domain": string, "suggested_xsd_range": "xsd:string|xsd:float|xsd:int|xsd:boolean|xsd:dateTime", "reason": string }
- is_instance: boolean
- evidence_text: string

【输出要求】
1. 只能输出严格 JSON，不允许输出 Markdown，不允许输出解释性废话，不允许输出代码块。
2. 输出 JSON 顶层只能包含 elements 字段。
3. elements 必须是数组。
4. 若无可抽取项，输出 {"elements":[]}。
5. 不要因为规则保守就随意输出空数组。只有在当前文本中确实不存在可稳定抽取的本体模式层候选要素时，才输出空数组。
6. mapping_suggestion 必须按 element_type 分型输出，只能包含该类型所需字段，禁止输出无关字段。
7. 若输出任何 element，则该 element 的 mapping_suggestion 必须完整包含该类型要求的全部字段，不得省略 reason。
8. 对 ObjectProperty，candidate_super_property 必须始终显式给出：要么是 null，要么是当前本体简化结构中已存在的合法父属性引用。
9. 不允许新增 schema 未声明字段。
10. 不允许出现“reason 中说明应忽略，但仍然把它写进 elements”的情况。
11. evidence_text 必须来自当前知识输入单元，不能编造。
12. definition 应概括候选要素的稳定语义，不得脱离原文证据臆造。

【输出格式】
仅输出一个 JSON 对象：

{
  "elements": [
    {
      "name_raw": "...",
      "name_zh": "...",
      "name_en": "...",
      "element_type": "Class|ObjectProperty|DatatypeProperty",
      "definition": "...",
      "mapping_suggestion": {...},
      "is_instance": false,
      "evidence_text": "..."
    }
  ]
}

若无可抽取项，输出：
{"elements":[]}

【本体简化结构】
{{AGENT1_MINI_SCHEMA_TTL_TEXT}}
"""

USER_PROMPT_TEMPLATE = """请按照System规则处理以下文献片段并仅输出JSON。

paper_id: {paper_id}
section: {section}
chunk_id: {chunk_id}
text:
{text}
"""


# ==========================================
# Pydantic 数据模型
# ==========================================
XsdRange = Literal["xsd:string", "xsd:float", "xsd:int", "xsd:boolean", "xsd:dateTime"]


class ClassMappingSuggestion(BaseModel):
    """Class 候选的挂载建议。"""
    model_config = ConfigDict(extra="forbid")

    suggested_parent_class: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ObjectPropertyMappingSuggestion(BaseModel):
    """ObjectProperty 候选的 domain/range 建议。"""
    model_config = ConfigDict(extra="forbid")

    suggested_domain: str = Field(min_length=1)
    suggested_range: str = Field(min_length=1)
    candidate_super_property: Optional[str] = None
    reason: str = Field(min_length=1)


class DatatypePropertyMappingSuggestion(BaseModel):
    """DatatypeProperty 候选的 domain/xsd range 建议。"""
    model_config = ConfigDict(extra="forbid")

    suggested_domain: str = Field(min_length=1)
    suggested_xsd_range: XsdRange
    reason: str = Field(min_length=1)


class BaseExtractedElement(BaseModel):
    """Agent1 抽取结果的公共字段基类。"""
    model_config = ConfigDict(extra="forbid")

    name_raw: str = Field(min_length=1)
    name_zh: str = Field(min_length=1)
    name_en: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    is_instance: bool
    evidence_text: str = Field(min_length=1)


class ClassElement(BaseExtractedElement):
    """Class 类型概念抽取结果。"""
    element_type: Literal["Class"]
    mapping_suggestion: ClassMappingSuggestion


class ObjectPropertyElement(BaseExtractedElement):
    """ObjectProperty 类型概念抽取结果。"""
    element_type: Literal["ObjectProperty"]
    mapping_suggestion: ObjectPropertyMappingSuggestion


class DatatypePropertyElement(BaseExtractedElement):
    """DatatypeProperty 类型概念抽取结果。"""
    element_type: Literal["DatatypeProperty"]
    mapping_suggestion: DatatypePropertyMappingSuggestion


ExtractedElement = Annotated[
    Union[ClassElement, ObjectPropertyElement, DatatypePropertyElement],
    Field(discriminator="element_type"),
]


class Agent1Response(BaseModel):
    """Agent1 单条语料的结构化总响应。"""
    model_config = ConfigDict(extra="forbid")
    elements: List[ExtractedElement] = Field(default_factory=list)


EXTRACTED_ELEMENT_ADAPTER = TypeAdapter(ExtractedElement)


def read_text(path: Path) -> str:
    """读取 UTF-8 文本文件。"""
    return path.read_text(encoding="utf-8")


def extract_object_property_names_from_ttl(ttl_text: str) -> set[str]:
    """从 TTL 文本中提取已声明的 ObjectProperty 本地名。"""
    names: set[str] = set()
    pattern = re.compile(
        r"(?m):([A-Za-z_][A-Za-z0-9_]*)\s+(?:a|rdf:type)\s+owl:ObjectProperty\b"
    )
    for match in pattern.finditer(ttl_text):
        names.add(match.group(1))
    return names


def write_json(path: Path, data: Any) -> None:
    """将对象写入 UTF-8 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    """向 JSONL 日志文件追加一行记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


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
    """从 native usage 对象中提取输入/输出与缓存 token。"""
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


def normalize_mapping_for_output(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """补齐 mapping_suggestion 字段全集，保持下游文件结构兼容。"""
    return {
        "suggested_parent_class": mapping.get("suggested_parent_class"),
        "suggested_domain": mapping.get("suggested_domain"),
        "suggested_range": mapping.get("suggested_range"),
        "suggested_xsd_range": mapping.get("suggested_xsd_range"),
        "candidate_super_property": mapping.get("candidate_super_property"),
        "reason": mapping.get("reason", ""),
    }


def normalize_iri_local_name(value: Any) -> str:
    """归一化 name_en：去空白并移除错误前导冒号。"""
    text = str(value or "").strip()
    while text.startswith(":"):
        text = text[1:].strip()
    return text


def normalize_name_en_by_type(element_type: Any, value: Any) -> str:
    """按元素类型对 name_en 做轻量命名风格归一化。"""
    text = normalize_iri_local_name(value)
    if not text:
        return ""
    element_type_text = str(element_type or "").strip()
    if element_type_text == "Class":
        return text[0].upper() + text[1:]
    if element_type_text in {"ObjectProperty", "DatatypeProperty"}:
        return text[0].lower() + text[1:]
    return text


def normalize_class_reference(value: Any) -> str:
    """归一化类引用字段：若非空且非 UNSURE，则统一为 :ClassName。"""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.upper() == "UNSURE":
        return "UNSURE"
    return text if text.startswith(":") else f":{text}"


def normalize_property_reference(value: Any) -> Optional[str]:
    """归一化属性引用字段：可为 null，否则统一为 :lowerCamelCase local name。"""
    text = str(value or "").strip()
    if not text:
        return None
    if text.upper() in {"UNSURE", "NONE", "N/A", "NA"}:
        return None
    if text.lower() == "null":
        return None
    while text.startswith(":"):
        text = text[1:].strip()
    if not text:
        return None
    normalized = text[0].lower() + text[1:]
    if KNOWN_OBJECT_PROPERTY_NAMES and normalized not in KNOWN_OBJECT_PROPERTY_NAMES:
        return None
    return f":{normalized}"


def normalize_xsd_range_value(value: Any) -> str:
    """归一化 xsd range；无法识别时保留原值供上层校验。"""
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    alias_map = {
        "string": "xsd:string",
        ":string": "xsd:string",
        "xsd:string": "xsd:string",
        "float": "xsd:float",
        ":float": "xsd:float",
        "xsd:float": "xsd:float",
        "int": "xsd:int",
        ":int": "xsd:int",
        "integer": "xsd:int",
        ":integer": "xsd:int",
        "xsd:int": "xsd:int",
        "xsd:integer": "xsd:int",
        "boolean": "xsd:boolean",
        ":boolean": "xsd:boolean",
        "bool": "xsd:boolean",
        ":bool": "xsd:boolean",
        "xsd:boolean": "xsd:boolean",
        "datetime": "xsd:dateTime",
        ":datetime": "xsd:dateTime",
        "dateTime": "xsd:dateTime",
        "xsd:datetime": "xsd:dateTime",
        "xsd:dateTime": "xsd:dateTime",
    }
    normalized = alias_map.get(lowered)
    if normalized is not None:
        return normalized
    return text


def normalize_element_for_output(element: Dict[str, Any]) -> Dict[str, Any]:
    """统一 Agent1 输出的名称字段与 mapping 字段口径。"""
    normalized = dict(element)
    normalized["name_raw"] = str(normalized.get("name_raw") or "").strip()
    normalized["name_zh"] = str(normalized.get("name_zh") or "").strip()
    normalized["name_en"] = normalize_name_en_by_type(
        normalized.get("element_type"),
        normalized.get("name_en"),
    )

    mapping = normalize_mapping_for_output(normalized.get("mapping_suggestion", {}))
    if normalized.get("element_type") == "Class":
        mapping["suggested_parent_class"] = normalize_class_reference(mapping.get("suggested_parent_class"))
    elif normalized.get("element_type") == "ObjectProperty":
        mapping["suggested_domain"] = normalize_class_reference(mapping.get("suggested_domain"))
        mapping["suggested_range"] = normalize_class_reference(mapping.get("suggested_range"))
        mapping["candidate_super_property"] = normalize_property_reference(mapping.get("candidate_super_property"))
    elif normalized.get("element_type") == "DatatypeProperty":
        mapping["suggested_domain"] = normalize_class_reference(mapping.get("suggested_domain"))
        mapping["suggested_xsd_range"] = normalize_xsd_range_value(mapping.get("suggested_xsd_range"))
        mapping["candidate_super_property"] = None

    normalized["mapping_suggestion"] = mapping
    return normalized


def sanitize_unescaped_quotes(text: str) -> str:
    """修复 JSON 字符串中常见的未转义英文双引号。"""
    repaired: List[str] = []
    in_string = False
    escaping = False
    length = len(text)
    i = 0

    while i < length:
        ch = text[i]
        if not in_string:
            repaired.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if escaping:
            repaired.append(ch)
            escaping = False
            i += 1
            continue

        if ch == "\\":
            repaired.append(ch)
            escaping = True
            i += 1
            continue

        if ch == '"':
            j = i + 1
            while j < length and text[j] in " \t\r\n":
                j += 1
            next_ch = text[j] if j < length else ""
            # 只有在可作为字符串闭合符时才保留原始双引号，否则转义为 \"
            if next_ch in {",", "}", "]", ":"} or not next_ch:
                repaired.append(ch)
                in_string = False
            else:
                repaired.append('\\"')
            i += 1
            continue

        repaired.append(ch)
        i += 1

    return "".join(repaired)


def extract_elements_array_text(json_text: str) -> Optional[str]:
    """从 JSON 文本中提取 elements 数组内部文本（不含方括号）。"""
    marker = '"elements"'
    marker_idx = json_text.find(marker)
    if marker_idx == -1:
        return None

    start_bracket = json_text.find("[", marker_idx)
    if start_bracket == -1:
        return None

    in_string = False
    escaping = False
    depth = 0
    end_bracket = -1

    for idx in range(start_bracket, len(json_text)):
        ch = json_text[idx]
        if in_string:
            if escaping:
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "[":
            depth += 1
            continue

        if ch == "]":
            depth -= 1
            if depth == 0:
                end_bracket = idx
                break

    if end_bracket == -1:
        return None
    return json_text[start_bracket + 1 : end_bracket]


def split_top_level_json_objects(array_text: str) -> List[str]:
    """按顶层大括号切分 elements 数组中的对象文本。"""
    objects: List[str] = []
    in_string = False
    escaping = False
    depth = 0
    start_idx = -1

    for idx, ch in enumerate(array_text):
        if in_string:
            if escaping:
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
            continue

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx != -1:
                    objects.append(array_text[start_idx : idx + 1])
                    start_idx = -1
            continue

    return objects


def salvage_elements_from_payload(payload: Any) -> Tuple[List[ExtractedElement], List[Dict[str, Any]]]:
    """当整包校验失败时，从已解析 payload 中逐条抢救可用元素。"""
    dropped: List[Dict[str, Any]] = []
    valid_elements: List[ExtractedElement] = []

    if not isinstance(payload, dict):
        dropped.append({"index": -1, "error": "payload 不是对象", "raw_snippet": str(payload)[:240]})
        return valid_elements, dropped

    raw_elements = payload.get("elements")
    if not isinstance(raw_elements, list):
        dropped.append({"index": -1, "error": "payload.elements 不是数组", "raw_snippet": str(raw_elements)[:240]})
        return valid_elements, dropped

    for i, item in enumerate(raw_elements):
        try:
            valid_elements.append(EXTRACTED_ELEMENT_ADAPTER.validate_python(item))
        except ValidationError as e:
            dropped.append(
                {
                    "index": i,
                    "error": f"ValidationError: {e}",
                    "raw_snippet": json.dumps(item, ensure_ascii=False)[:500] if isinstance(item, (dict, list)) else str(item)[:500],
                }
            )
        except Exception as e:  # noqa: BLE001
            dropped.append({"index": i, "error": f"{type(e).__name__}: {e}", "raw_snippet": str(item)[:500]})
    return valid_elements, dropped


def salvage_elements_from_json_text(json_text: str) -> Tuple[List[ExtractedElement], List[Dict[str, Any]]]:
    """当整包 JSON 无法解析时，按对象边界逐条解析并抢救。"""
    dropped: List[Dict[str, Any]] = []
    valid_elements: List[ExtractedElement] = []

    array_text = extract_elements_array_text(json_text)
    if array_text is None:
        dropped.append({"index": -1, "error": "无法定位 elements 数组", "raw_snippet": json_text[:500]})
        return valid_elements, dropped

    object_texts = split_top_level_json_objects(array_text)
    if not object_texts:
        dropped.append({"index": -1, "error": "elements 数组中未识别到对象", "raw_snippet": array_text[:500]})
        return valid_elements, dropped

    for i, object_text in enumerate(object_texts):
        candidate = object_text.strip()
        try:
            item_payload = json.loads(candidate)
        except Exception:
            try:
                item_payload = json.loads(sanitize_unescaped_quotes(candidate))
            except Exception as e:  # noqa: BLE001
                dropped.append({"index": i, "error": f"{type(e).__name__}: {e}", "raw_snippet": candidate[:500]})
                continue

        try:
            valid_elements.append(EXTRACTED_ELEMENT_ADAPTER.validate_python(item_payload))
        except ValidationError as e:
            dropped.append({"index": i, "error": f"ValidationError: {e}", "raw_snippet": candidate[:500]})
        except Exception as e:  # noqa: BLE001
            dropped.append({"index": i, "error": f"{type(e).__name__}: {e}", "raw_snippet": candidate[:500]})

    return valid_elements, dropped


def load_records(path: Path) -> List[Dict[str, Any]]:
    """读取阶段一输出语料，并仅保留对象型记录。"""
    payload = json.loads(read_text(path))
    if not isinstance(payload, list):
        raise ValueError("输入语料必须是 JSON 数组。")
    return [x for x in payload if isinstance(x, dict)]


def _extract_text_from_content(raw_content: Any) -> Tuple[str, str]:
    """兼容 OpenAI 兼容接口内容结构，提取正文与思考文本。"""
    def get_field(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    text_parts: List[str] = []
    reasoning_parts: List[str] = []

    if isinstance(raw_content, str):
        return raw_content.strip(), ""

    if isinstance(raw_content, list):
        for item in raw_content:
            item_type = (get_field(item, "type", "") or "").lower()
            if item_type in ("thinking", "reasoning"):
                part = get_field(item, "thinking") or get_field(item, "text") or ""
                if part:
                    reasoning_parts.append(str(part).strip())
                continue
            part = get_field(item, "text")
            if part:
                text_parts.append(str(part).strip())
        return "\n".join([p for p in text_parts if p]).strip(), "\n".join([p for p in reasoning_parts if p]).strip()

    if raw_content is None:
        return "", ""

    return str(raw_content).strip(), ""


def extract_json_string(text: str) -> str:
    """从模型文本中截取首个 JSON 对象，容忍代码块包裹。"""
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


def build_system_blocks(mini_schema_text: str) -> List[Dict[str, Any]]:
    """构造 Anthropic Native system blocks，并将固定规则与 Mini Schema 分块缓存。"""
    marker = "{{AGENT1_MINI_SCHEMA_TTL_TEXT}}"
    if marker in SYSTEM_PROMPT_TEMPLATE:
        rules_text = SYSTEM_PROMPT_TEMPLATE.replace(marker, "").strip()
    else:
        rules_text = f"{SYSTEM_PROMPT_TEMPLATE}\n\n【本体上下文（Mini Schema）】".strip()

    cache_control = {"type": "ephemeral"} if ENABLE_PROMPT_CACHE else None
    rules_block: Dict[str, Any] = {
        "type": "text",
        "text": rules_text,
    }
    schema_block: Dict[str, Any] = {
        "type": "text",
        "text": mini_schema_text,
    }
    if cache_control:
        rules_block["cache_control"] = cache_control
        schema_block["cache_control"] = cache_control
    return [rules_block, schema_block]


def build_user_prompt(record: Dict[str, Any], chunk_id: str) -> str:
    """把单条语料记录格式化为 User Prompt。"""
    return USER_PROMPT_TEMPLATE.format(
        paper_id=str(record.get("paper_id", "")).strip(),
        section=str(record.get("section", "")).strip(),
        chunk_id=chunk_id,
        text=str(record.get("content", "")).strip(),
    )


def call_claude(
    session: requests.Session,
    system_blocks: List[Dict[str, Any]],
    user_prompt: str,
    source_meta: Dict[str, str],
) -> Tuple[Agent1Response, Dict[str, Any]]:
    """调用 Claude Native Messages 完成概念抽取，并执行解析修复与逐条抢救。"""
    url = f"{CLAUDE_NATIVE_BASE_URL.rstrip('/')}{MESSAGES_PATH}"
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

    response = session.post(url, json=payload, timeout=API_TIMEOUT_SECONDS)
    raw_response_text = response.text
    response.raise_for_status()
    response_payload = json.loads(raw_response_text)

    usage_obj = response_payload.get("usage", {})
    usage_metrics = extract_usage_metrics(usage_obj)

    raw_output, reasoning_output = _extract_text_from_content(response_payload.get("content", []))
    stop_reason = str(response_payload.get("stop_reason", "") or "")
    response_shape = (
        "anthropic_message"
        if response_payload.get("type") == "message" and isinstance(response_payload.get("content"), list)
        else "unknown"
    )

    json_text = extract_json_string(raw_output)

    append_jsonl(
        OUTPUT_RAW_API_LOG,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": source_meta,
            "raw_content": raw_output,
            "raw_reasoning_content": reasoning_output,
            "response_shape": response_shape,
            "raw_response": response_payload,
            "usage": to_jsonable(usage_obj),
            **usage_metrics,
            "stop_reason": stop_reason,
        },
    )

    parse_meta: Dict[str, Any] = {
        "parse_mode": "direct",
        "repair_applied": False,
        "salvage_applied": False,
        "dropped_items": [],
        "errors": [],
        **usage_metrics,
        "response_shape": response_shape,
        "stop_reason": stop_reason,
    }

    def _write_repair_log(success: bool) -> None:
        append_jsonl(
            OUTPUT_REPAIR_LOG,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source": source_meta,
                "success": success,
                "parse_mode": parse_meta["parse_mode"],
                "repair_applied": parse_meta["repair_applied"],
                "salvage_applied": parse_meta["salvage_applied"],
                "dropped_count": len(parse_meta["dropped_items"]),
                "errors": parse_meta["errors"],
            },
        )

    # 第一层：直接解析整包 JSON + Pydantic 校验
    try:
        payload = json.loads(json_text)
        try:
            response_obj = Agent1Response.model_validate(payload)
            _write_repair_log(success=True)
            return response_obj, parse_meta
        except ValidationError as e:
            parse_meta["errors"].append(f"direct_validation_error: {e}")
            if ENABLE_ITEM_SALVAGE:
                valid_elements, dropped_items = salvage_elements_from_payload(payload)
                parse_meta["dropped_items"] = dropped_items
                if len(valid_elements) >= MIN_VALID_ITEMS_TO_ACCEPT:
                    parse_meta["parse_mode"] = "salvaged_from_payload"
                    parse_meta["salvage_applied"] = True
                    _write_repair_log(success=True)
                    return Agent1Response(elements=valid_elements), parse_meta
    except json.JSONDecodeError as e:
        parse_meta["errors"].append(f"direct_json_error: {e}")

    repaired_json_text = json_text

    # 第二层：本地 JSON 修复（主要修复未转义双引号）
    if ENABLE_JSON_REPAIR:
        candidate = sanitize_unescaped_quotes(json_text)
        if candidate != json_text:
            parse_meta["repair_applied"] = True
            repaired_json_text = candidate
            try:
                payload = json.loads(repaired_json_text)
                try:
                    response_obj = Agent1Response.model_validate(payload)
                    parse_meta["parse_mode"] = "repaired_json"
                    _write_repair_log(success=True)
                    return response_obj, parse_meta
                except ValidationError as e:
                    parse_meta["errors"].append(f"repaired_validation_error: {e}")
                    if ENABLE_ITEM_SALVAGE:
                        valid_elements, dropped_items = salvage_elements_from_payload(payload)
                        parse_meta["dropped_items"] = dropped_items
                        if len(valid_elements) >= MIN_VALID_ITEMS_TO_ACCEPT:
                            parse_meta["parse_mode"] = "salvaged_from_repaired_payload"
                            parse_meta["salvage_applied"] = True
                            _write_repair_log(success=True)
                            return Agent1Response(elements=valid_elements), parse_meta
            except json.JSONDecodeError as e:
                parse_meta["errors"].append(f"repaired_json_error: {e}")

    # 第三层：逐条对象抢救（坏一条丢一条）
    if ENABLE_ITEM_SALVAGE:
        valid_elements, dropped_items = salvage_elements_from_json_text(repaired_json_text)
        parse_meta["dropped_items"] = dropped_items
        if len(valid_elements) >= MIN_VALID_ITEMS_TO_ACCEPT:
            parse_meta["parse_mode"] = "salvaged_from_json_text"
            parse_meta["salvage_applied"] = True
            _write_repair_log(success=True)
            return Agent1Response(elements=valid_elements), parse_meta

    _write_repair_log(success=False)
    raise ValueError("LLM 返回内容无法解析为有效 Agent1Response；已尝试直接解析、JSON修复、逐条抢救。")


def ensure_paths() -> None:
    """运行前检查关键输入文件是否存在。"""
    required = [MINI_SCHEMA_PATH, INPUT_JSON_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"以下文件不存在: {missing}")


def init_element_type_counts() -> Dict[str, int]:
    """初始化三类本体元素计数字典。"""
    return {"Class": 0, "ObjectProperty": 0, "DatatypeProperty": 0}


def accumulate_element_type_counts(target: Dict[str, int], items: List[Dict[str, Any]]) -> None:
    """将一批元素的 element_type 计数累加到目标字典。"""
    for item in items:
        element_type = str(item.get("element_type", "")).strip()
        if element_type in target:
            target[element_type] += 1


def main() -> None:
    """主流程：加载语料、串行抽取、分流 TBox/ABox 并写出结果。"""
    global KNOWN_OBJECT_PROPERTY_NAMES
    started = time.time()
    log_path = setup_console_log("phase2_01_concept_extraction")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    if not API_KEY:
        raise RuntimeError("缺少 API Key，请在脚本配置区填写 API_KEY。")

    ensure_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mini_schema_text = read_text(MINI_SCHEMA_PATH)
    KNOWN_OBJECT_PROPERTY_NAMES = extract_object_property_names_from_ttl(mini_schema_text)
    system_blocks = build_system_blocks(mini_schema_text)
    records = load_records(INPUT_JSON_PATH)
    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    selected = records[START_INDEX:end_index]

    print("=" * 72)
    print("[运行启动] Agent1 概念抽取启动")
    print(f"[模型配置] model={CLAUDE_MODEL}")
    print(f"[Mini Schema] 已加载 ObjectProperty={len(KNOWN_OBJECT_PROPERTY_NAMES)}")
    print(f"[调用方式] native_messages | prompt_cache={ENABLE_PROMPT_CACHE} | persistent_session={USE_PERSISTENT_SESSION}")
    print(f"[输入信息] input_total={len(records)} | selected={len(selected)} | START_INDEX={START_INDEX} | MAX_ITEMS={MAX_ITEMS}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print("=" * 72)

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

    raw_results: List[Dict[str, Any]] = []
    passed_tbox: List[Dict[str, Any]] = []
    blocked_abox: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    dropped_elements: List[Dict[str, Any]] = []
    total_elements = 0
    total_passed = 0
    total_blocked = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_input_tokens = 0
    total_cache_read_input_tokens = 0
    total_elapsed_sec = 0.0
    success_chunk_count = 0
    cache_hit_chunk_count = 0
    chunk_stats: List[Dict[str, Any]] = []
    extracted_type_counts = init_element_type_counts()
    tbox_passed_type_counts = init_element_type_counts()

    # 初始化可追溯文件，避免本次运行无成功样本时沿用旧文件内容。
    write_json(OUTPUT_DROPPED_ITEMS, dropped_elements)
    write_json(OUTPUT_CHUNK_STATS, chunk_stats)

    for idx, record in enumerate(selected):
        global_idx = START_INDEX + idx
        input_chunk_id = str(record.get("chunk_id", "")).strip()
        chunk_id = input_chunk_id or f"chunk_{global_idx:05d}"
        source_meta = {
            "paper_id": str(record.get("paper_id", "")).strip(),
            "section": str(record.get("section", "")).strip(),
            "chunk_id": chunk_id,
        }

        print(f"\n[开始处理] [{idx + 1}/{len(selected)}] {chunk_id}")
        print(f"paper_id={source_meta['paper_id']}")
        print(f"section={source_meta['section']}")

        last_error = ""
        response_obj: Optional[Agent1Response] = None
        started_at = time.time()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[模型调用] 第 {attempt}/{MAX_RETRIES} 次尝试")
                user_prompt = build_user_prompt(record, chunk_id)
                response_obj, parse_meta = call_claude(session, system_blocks, user_prompt, source_meta)
                break
            except (ValidationError, json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = f"{type(e).__name__}: {e}"
                print(f"[结构化失败] {last_error}")
            except Exception as e:  # noqa: BLE001
                last_error = f"{type(e).__name__}: {e}"
                print(f"[请求失败] {last_error}")

            if attempt < MAX_RETRIES:
                sleep_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                print(f"[重试等待] {sleep_seconds:.1f}s 后重试")
                time.sleep(sleep_seconds)

        elapsed = round(time.time() - started_at, 3)

        if response_obj is None:
            print(f"[处理失败] 已记录错误 | elapsed={elapsed}s")
            errors.append({"source": source_meta, "error": last_error, "elapsed_sec": elapsed})
            write_json(OUTPUT_ERRORS, errors)
            continue

        dropped_items = parse_meta.get("dropped_items", []) if isinstance(parse_meta, dict) else []
        if dropped_items:
            for dropped in dropped_items:
                dropped_elements.append(
                    {
                        "source": source_meta,
                        "parse_mode": parse_meta.get("parse_mode", ""),
                        "repair_applied": bool(parse_meta.get("repair_applied", False)),
                        "salvage_applied": bool(parse_meta.get("salvage_applied", False)),
                        "item": dropped,
                    }
                )

        elements = [normalize_element_for_output(x.model_dump()) for x in response_obj.elements]
        for e in elements:
            e["source"] = source_meta

        blocked = [e for e in elements if e.get("is_instance") is True]
        passed = [e for e in elements if e.get("is_instance") is False]

        total_elements += len(elements)
        total_blocked += len(blocked)
        total_passed += len(passed)
        chunk_input_tokens = int(parse_meta.get("input_tokens", 0)) if isinstance(parse_meta, dict) else 0
        chunk_output_tokens = int(parse_meta.get("output_tokens", 0)) if isinstance(parse_meta, dict) else 0
        chunk_cache_creation_input_tokens = int(parse_meta.get("cache_creation_input_tokens", 0)) if isinstance(parse_meta, dict) else 0
        chunk_cache_read_input_tokens = int(parse_meta.get("cache_read_input_tokens", 0)) if isinstance(parse_meta, dict) else 0
        total_input_tokens += chunk_input_tokens
        total_output_tokens += chunk_output_tokens
        total_cache_creation_input_tokens += chunk_cache_creation_input_tokens
        total_cache_read_input_tokens += chunk_cache_read_input_tokens
        total_elapsed_sec += float(elapsed)
        success_chunk_count += 1
        if chunk_cache_read_input_tokens > 0:
            cache_hit_chunk_count += 1
        accumulate_element_type_counts(extracted_type_counts, elements)
        accumulate_element_type_counts(tbox_passed_type_counts, passed)

        chunk_type_counts = init_element_type_counts()
        chunk_passed_type_counts = init_element_type_counts()
        accumulate_element_type_counts(chunk_type_counts, elements)
        accumulate_element_type_counts(chunk_passed_type_counts, passed)

        raw_results.append(
            {
                "source": source_meta,
                "elapsed_sec": elapsed,
                "parse_mode": parse_meta.get("parse_mode", ""),
                "repair_applied": bool(parse_meta.get("repair_applied", False)),
                "salvage_applied": bool(parse_meta.get("salvage_applied", False)),
                "dropped_count": len(dropped_items),
                "input_tokens": chunk_input_tokens,
                "output_tokens": chunk_output_tokens,
                "cache_creation_input_tokens": chunk_cache_creation_input_tokens,
                "cache_read_input_tokens": chunk_cache_read_input_tokens,
                "total_tokens": chunk_input_tokens + chunk_output_tokens,
                "total_input_for_threshold": chunk_input_tokens + chunk_cache_creation_input_tokens + chunk_cache_read_input_tokens,
                "element_count": len(elements),
                "elements": elements,
            }
        )
        chunk_stats.append(
            {
                "source": source_meta,
                "elapsed_sec": elapsed,
                "input_tokens": chunk_input_tokens,
                "output_tokens": chunk_output_tokens,
                "cache_creation_input_tokens": chunk_cache_creation_input_tokens,
                "cache_read_input_tokens": chunk_cache_read_input_tokens,
                "total_input_for_threshold": chunk_input_tokens + chunk_cache_creation_input_tokens + chunk_cache_read_input_tokens,
                "total_tokens": chunk_input_tokens + chunk_output_tokens,
                "element_count": len(elements),
                "tbox_passed_count": len(passed),
                "abox_blocked_count": len(blocked),
                "element_type_counts": chunk_type_counts,
                "tbox_passed_type_counts": chunk_passed_type_counts,
                "parse_mode": parse_meta.get("parse_mode", ""),
                "repair_applied": bool(parse_meta.get("repair_applied", False)),
                "salvage_applied": bool(parse_meta.get("salvage_applied", False)),
                "dropped_count": len(dropped_items),
            }
        )
        blocked_abox.extend(blocked)
        passed_tbox.extend(passed)

        print(
            "[处理完成] "
            f"抽取={len(elements)} | "
            f"TBox放行={len(passed)} | "
            f"ABox拦截={len(blocked)} | "
            f"elapsed={elapsed}s"
        )
        print(
            "[chunk统计] "
            f"token={chunk_input_tokens + chunk_output_tokens} "
            f"(input={chunk_input_tokens}, output={chunk_output_tokens}, "
            f"cache_create={chunk_cache_creation_input_tokens}, cache_read={chunk_cache_read_input_tokens}) | "
            f"Class={chunk_passed_type_counts['Class']} | "
            f"ObjectProperty={chunk_passed_type_counts['ObjectProperty']} | "
            f"DatatypeProperty={chunk_passed_type_counts['DatatypeProperty']}"
        )
        if isinstance(parse_meta, dict):
            print(
                "[解析路径] "
                f"{parse_meta.get('parse_mode', 'unknown')} | "
                f"repair={parse_meta.get('repair_applied', False)} | "
                f"salvage={parse_meta.get('salvage_applied', False)} | "
                f"dropped={len(dropped_items)}"
            )

        write_json(OUTPUT_RAW_RESULTS, raw_results)
        write_json(OUTPUT_PASSED_TBOX, passed_tbox)
        write_json(OUTPUT_BLOCKED_ABOX, blocked_abox)
        write_json(OUTPUT_ERRORS, errors)
        write_json(OUTPUT_DROPPED_ITEMS, dropped_elements)
        write_json(OUTPUT_CHUNK_STATS, chunk_stats)

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": CLAUDE_MODEL,
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
        "thinking_mode": THINKING_MODE,
        "thinking_effort": THINKING_EFFORT,
        "thinking_budget_tokens": THINKING_BUDGET_TOKENS,
        "retry_policy": {
            "max_retries": MAX_RETRIES,
            "retry_base_seconds": RETRY_BASE_SECONDS,
        },
        "fallback_policy": {
            "enable_json_repair": ENABLE_JSON_REPAIR,
            "enable_item_salvage": ENABLE_ITEM_SALVAGE,
            "min_valid_items_to_accept": MIN_VALID_ITEMS_TO_ACCEPT,
        },
        "input_file": str(INPUT_JSON_PATH),
        "mini_schema_file": str(MINI_SCHEMA_PATH),
        "prompt_source": "embedded_in_code",
        "output_dir": str(OUTPUT_DIR),
        "processed_items": len(selected),
        "success_items": len(raw_results),
        "failed_items": len(errors),
        "total_elements": total_elements,
        "tbox_passed_elements": total_passed,
        "abox_blocked_elements": total_blocked,
        "extracted_type_counts": extracted_type_counts,
        "tbox_passed_type_counts": tbox_passed_type_counts,
        "dropped_elements_count": len(dropped_elements),
        "elapsed_sec": round(total_elapsed_sec, 3),
        "wall_elapsed_sec": round(time.time() - started, 3),
        "avg_elapsed_sec_per_success_chunk": round(total_elapsed_sec / success_chunk_count, 3) if success_chunk_count > 0 else 0.0,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_creation_input_tokens": total_cache_creation_input_tokens,
            "cache_read_input_tokens": total_cache_read_input_tokens,
            "total_input_for_threshold": total_input_tokens + total_cache_creation_input_tokens + total_cache_read_input_tokens,
            "cache_hit_chunk_count": cache_hit_chunk_count,
            "avg_input_tokens_per_success_chunk": round(total_input_tokens / success_chunk_count, 3) if success_chunk_count > 0 else 0.0,
            "avg_output_tokens_per_success_chunk": round(total_output_tokens / success_chunk_count, 3) if success_chunk_count > 0 else 0.0,
        },
    }
    token_report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "processed_items": len(selected),
        "success_items": success_chunk_count,
        "failed_items": len(errors),
        "extracted_type_counts": extracted_type_counts,
        "tbox_passed_type_counts": tbox_passed_type_counts,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cache_creation_input_tokens": total_cache_creation_input_tokens,
        "cache_read_input_tokens": total_cache_read_input_tokens,
        "total_input_for_threshold": total_input_tokens + total_cache_creation_input_tokens + total_cache_read_input_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "cache_hit_chunk_count": cache_hit_chunk_count,
        "cache_hit_ratio": round(cache_hit_chunk_count / success_chunk_count, 3) if success_chunk_count > 0 else 0.0,
        "avg_input_tokens_per_success_chunk": round(total_input_tokens / success_chunk_count, 3) if success_chunk_count > 0 else 0.0,
        "avg_output_tokens_per_success_chunk": round(total_output_tokens / success_chunk_count, 3) if success_chunk_count > 0 else 0.0,
        "avg_total_tokens_per_success_chunk": round((total_input_tokens + total_output_tokens) / success_chunk_count, 3) if success_chunk_count > 0 else 0.0,
        "total_elapsed_sec": round(total_elapsed_sec, 3),
        "wall_elapsed_sec": round(time.time() - started, 3),
        "avg_elapsed_sec_per_success_chunk": round(total_elapsed_sec / success_chunk_count, 3) if success_chunk_count > 0 else 0.0,
        "chunk_stats_file": str(OUTPUT_CHUNK_STATS),
    }
    write_json(OUTPUT_MANIFEST, manifest)
    write_json(OUTPUT_TOKEN_REPORT, token_report)

    print("\n" + "=" * 72)
    print("[运行结束] Agent1 概念抽取完成")
    print(f"[运行摘要] 成功样本={len(raw_results)} | 失败样本={len(errors)}")
    print(f"[运行摘要] 总抽取={total_elements} | TBox放行={total_passed} | ABox拦截={total_blocked}")
    print(
        "[类型统计] "
        f"Class={tbox_passed_type_counts['Class']} | "
        f"ObjectProperty={tbox_passed_type_counts['ObjectProperty']} | "
        f"DatatypeProperty={tbox_passed_type_counts['DatatypeProperty']}"
    )
    print(
        f"[性能统计] input_tokens={total_input_tokens} | output_tokens={total_output_tokens} | "
        f"cache_create={total_cache_creation_input_tokens} | cache_read={total_cache_read_input_tokens} | "
        f"total_tokens={total_input_tokens + total_output_tokens}"
    )
    print(
        f"[性能统计] avg_elapsed_per_chunk={token_report['avg_elapsed_sec_per_success_chunk']}s | "
        f"wall_elapsed={token_report['wall_elapsed_sec']}s"
    )
    print(f"[关键输出] {OUTPUT_PASSED_TBOX} -> Agent2 内部融合输入主文件")
    print(f"[关键输出] {OUTPUT_TOKEN_REPORT} -> token 汇总报告")
    print(f"[关键输出] {OUTPUT_CHUNK_STATS} -> 每个 chunk 的耗时与 token 统计")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要")
    print(f"[关键输出] {OUTPUT_ERRORS} -> 失败与错误记录")
    print("=" * 72)

if __name__ == "__main__":
    main()
