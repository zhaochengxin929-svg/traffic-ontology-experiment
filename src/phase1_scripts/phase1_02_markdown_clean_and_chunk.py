import atexit
import os
import json
import re
import sys
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ==========================================
# 配置区（常改）
# ==========================================
# 兼容网关 API Key，可直接填写
API_KEY = ""
# 标题仲裁与灰区判定使用的模型名称
CLAUDE_MODEL = "claude-opus-4-6"
# OpenAI 兼容网关地址
OPENAI_COMPAT_BASE_URL = "https://api2.aigcbest.top/v1"
# 单次请求超时时间（秒）
API_TIMEOUT_SECONDS = 45
# 并发仲裁线程数
MAX_CONCURRENT_REQUESTS = 10
# 起始文件索引：用于按文献批次分段处理
START_INDEX = 0
# 本轮最多处理多少篇 Markdown；0 表示处理全部
MAX_FILES = 0
# 扩展思考模式：off / adaptive / manual
THINKING_MODE = "adaptive"
# adaptive 模式下的思考力度
THINKING_EFFORT = "medium"
# manual 模式下的思考预算
THINKING_BUDGET_TOKENS = 2000
# 当前阶段目录：大规模实验/phase1
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 项目根目录：大规模实验
PROJECT_ROOT = os.path.dirname(BASE_DIR)
# 输入目录：OCR 阶段输出的 Markdown 文件
INPUT_MD_DIR = os.path.join(BASE_DIR, "outputs_ocr", "papers_md")
# 输出目录：清洗后的理论语料及辅助日志
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_clean")
# 是否将控制台输出同时写入日志文件
ENABLE_FILE_LOG = True
# 日志目录：统一保存脚本控制台输出
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
# 输出主文件：阶段一最终交给阶段二的理论语料 JSON
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "cleaned_theory_corpus.json")
# 标题筛选统计：用于分析多层章节筛选与语义降噪效果
OUTPUT_FILTER_STATS = os.path.join(OUTPUT_DIR, "phase1_filter_stats.json")
# chunk 分布统计：用于分析切块粒度、过短/过长分布与文献差异
OUTPUT_CHUNK_STATS = os.path.join(OUTPUT_DIR, "phase1_chunk_stats.json")
# 运行摘要：记录总耗时、总调用次数、每篇文献统计
OUTPUT_MANIFEST = os.path.join(OUTPUT_DIR, "phase1_clean_manifest.json")
# 标题判定缓存文件：用于复用历史仲裁结果
CACHE_FILE = os.path.join(OUTPUT_DIR, "title_decision_cache.json")
# 原始大模型响应日志：保留每次标题仲裁的原始返回
RAW_RESPONSE_LOG = os.path.join(OUTPUT_DIR, "llm_raw_responses.jsonl")
# 过短/过长 chunk 统计阈值（字符数）
SHORT_CHUNK_CHAR_THRESHOLD = 200
LONG_CHUNK_CHAR_THRESHOLD = 1800

_RAW_LOG_LOCK = threading.Lock()
_LOG_FILE_HANDLE = None
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr

# 1. 剔除词库：包含实验类、版面排版类、他人工作总结类噪音
DISCARD_WORDS_ZH = [
    # 版面尾部与非正文
    "致谢", "参考文献", "附录", "目录", "索引", "作者简介", "作者信息", "基金项目",
    "资助信息", "利益冲突", "声明", "发表论文情况", "科研成果", "攻读学位期间",
    # 文献综述 / 研究现状 / 展望类
    "综述", "文献综述", "研究综述", "现状", "研究现状", "应用现状", "进展", "研究进展",
    "国内外", "国内外研究", "国内外研究现状", "展望", "研究展望", "不足与展望",
    # 结构说明 / 论文组织类
    "技术路线", "研究路线", "章节", "章节安排", "章节设置", "章节结构",
    "论文结构", "文章结构",  "创新点",
    # 区域 / 对象概况类
    "概况", "研究区概况", "区域概况", "城市概况", "路网概况",
    # 数据 / 实现 / 参数 / 算法细节类
    "数据来源", "数据说明", "数据描述", "数据采集", "数据处理", "数据预处理",
    "数据集", "样本描述", "样本构成", "变量说明", "函数设计", "函数构造",
    "参数设置", "参数标定", "参数分析", "参数选取", "参数敏感性",
    "实验设置", "实验环境", "仿真设置", "仿真环境",
    "目标函数", "约束条件", "变量定义", "算法流程", "实现细节"
]
DISCARD_WORDS_EN = [
    # 非正文 / 尾部
    "acknowledgement", "acknowledgments", "references", "bibliography",
    "appendix","appendices", "contents", "index", "authorcontributions",
    "conflictofinterest", "conflictsofinterest", "funding",
    "supplementarymaterial", "supplementaryinformation",
    # 综述 / 现状 / 展望
    "literaturereview",  "relatedwork", "relatedstudies",
    "stateoftheart", "researchprogress", "futurework",
    # 结构说明
    "technicalroute", "researchroute", "paperorganization",
    "organizationofthepaper", "chapterarrangement", "chapterorganization",
    "researchcontent", "innovationpoints",
    # 概况类
    "studyarea", "areaoverview", "studyareaoverview", "overviewofthearea",
    # 数据 / 参数 / 实现细节
    "datadescription", "datacollection", "dataprocessing", "variables",
    "datapreprocessing", "dataset", "datasets", "parametersetting",
    "parametersettings", "parametercalibration", "parameteranalysis",
    "sensitivityanalysis", "experimentalsetup", "simulationsetup",
    "simulationenvironment", "implementationdetails", "algorithmimplementation",
    "objectivefunction", "constraints", "constraint", "variabledefinition"
]

# 2. 保留词库：包含业务核心论述及关键章节
KEEP_WORDS_ZH = ["摘要", "中文摘要", "英文摘要", "引言", "绪论", "前言",
                 "结论", "结语", "结束语", "总结", "小结", "讨论",]
KEEP_WORDS_EN = ["abstract", "introduction", "conclusion", "conclusions",
                 "discussion", "discussions",]


# ==========================================
# 核心功能模块 (Core Functions)
# ==========================================

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
                # 解释器退出阶段可能遇到已关闭流，这里静默跳过。
                pass
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                # 解释器退出阶段可能遇到已关闭流，这里静默跳过。
                pass

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_console_log(script_name: str) -> str | None:
    """启用控制台双写：控制台照常显示，同时保存到日志文件。"""
    global _LOG_FILE_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    if not ENABLE_FILE_LOG:
        return None

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")
    _LOG_FILE_HANDLE = open(log_path, "w", encoding="utf-8")

    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _LOG_FILE_HANDLE)
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _LOG_FILE_HANDLE)

    def _close_log_file() -> None:
        global _LOG_FILE_HANDLE
        # 先恢复原始 stdout/stderr，避免解释器退出阶段继续 flush 到已关闭日志文件。
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR
        if _LOG_FILE_HANDLE and not _LOG_FILE_HANDLE.closed:
            try:
                _LOG_FILE_HANDLE.flush()
            finally:
                _LOG_FILE_HANDLE.close()
        _LOG_FILE_HANDLE = None

    atexit.register(_close_log_file)
    return log_path

def load_cache(cache_path: str) -> dict:
    """读取本地缓存字典，复用历史判定结果，节约 API 资源。"""
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache_data: dict, cache_path: str) -> None:
    """将判定结果异步落盘保存为 JSON 格式缓存文件。"""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)


def append_raw_response_log(record: dict) -> None:
    """将单次模型原始返回以 JSONL 方式追加落盘。"""
    with _RAW_LOG_LOCK:
        with open(RAW_RESPONSE_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str, data: dict | list) -> None:
    """将对象写入 UTF-8 JSON 文件。"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def normalize_title(title: str) -> str:
    """
    标题前缀标准化器。
    用于剥离标题前面的数字、章节号或列表符，提取核心语义。
    """
    pattern = (
        r'^(第[一二三四五六七八九十百千万]+[章|节|篇|部分]\s*)|'
        r'(^[一二三四五六七八九十]+、\s*)|'
        r'(^[\(（][0-9a-zA-Z]+[\)）]\s*)|'
        r'(^[0-9a-zA-Z]+[\)）]\s*)|'
        r'(^[0-9]+(\.[0-9]+)*\.?\s*)'
    )
    clean_title = re.sub(pattern, '', title).strip()
    return clean_title if clean_title else title


def _extract_text_from_content(raw_content) -> tuple[str, str]:
    """尽可能从 OpenAI 兼容响应中提取文本与推理文本。"""
    def get_field(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    text_parts = []
    reasoning_parts = []

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


def _build_thinking_extra_body() -> dict | None:
    """根据配置构建 OpenAI 兼容层透传参数。"""
    mode = (THINKING_MODE or "off").strip().lower()
    if mode == "off":
        return None

    if mode == "adaptive":
        body = {"thinking": {"type": "adaptive"}}
        effort = (THINKING_EFFORT or "").strip().lower()
        if effort in {"low", "medium", "high", "max"}:
            body["output_config"] = {"effort": effort}
        return body

    if mode == "manual":
        budget = int(THINKING_BUDGET_TOKENS)
        if budget <= 0:
            return None
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    return None


def extract_usage_tokens(response) -> tuple[int, int, int]:
    """从 OpenAI 兼容响应中提取 prompt/completion/total token。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return prompt_tokens, completion_tokens, total_tokens


def single_llm_arbitration(title: str, client: OpenAI) -> tuple[str, float, int, int, int]:
    """
    单标题大语言模型仲裁器 (带计时追踪版)。
    返回一个元组：(判定结果, 耗时秒数, prompt_tokens, completion_tokens, total_tokens)
    """

    system_prompt = """
你需要承担城市交通态势本体知识输入筛选任务。

该任务用于判断单个论文章节标题是否应保留为后续本体要素抽取的知识输入。你的判断对象只有章节标题，不能看到章节正文。因此，判断必须基于标题语义、章节功能和标题所指向的内容类型，不得假设自己已经阅读正文。

【任务目标】
请判断当前章节标题是否可能包含与城市交通态势本体建模相关的知识内容。

这里的“相关知识内容”主要指能够为城市交通态势本体提供概念、属性、关系、层次、机制、状态、指标、事件、过程或结构化语义信息的内容。

最终只能输出以下两个词之一：
Keep
Discard

【输出含义】
Keep：该标题对应章节可能包含对城市交通态势本体建模有价值的知识内容，应保留进入后续处理。
Discard：该标题对应章节更可能属于方法实现、数据处理、实验设置、参数说明、结果展示、结构性内容或其他噪声内容，不宜进入后续处理。

【链式处理步骤】
请按照以下顺序在内部完成判断。不要输出推理过程，只输出最终结论。

Step 1：理解任务范围
- 读取内容：当前章节标题。
- 对应规则：【保留倾向规则】和【剔除倾向规则】。
- 处理目标：判断标题是否可能指向城市交通态势本体建模所需的领域知识。
- 结果作用：确定该标题是否具有进入后续知识输入构建过程的基本价值。

Step 2：判断标题的领域知识指向
- 读取内容：标题中的交通对象、运行状态、指标、事件、过程、时空关系、机制等语义线索。
- 对应规则：【保留倾向规则】。
- 处理目标：判断标题是否更可能讨论交通态势相关概念、状态、指标、事件、过程、关系或机理。
- 结果作用：若标题语义重心指向上述内容，倾向 Keep。

Step 3：判断标题的噪声内容指向
- 读取内容：标题中的数据、算法、模型、参数、实验、案例、结果、综述、章节结构等语义线索。
- 对应规则：【剔除倾向规则】。
- 处理目标：判断标题是否更可能讨论数据处理、算法实现、模型训练、参数设置、实验验证、结果比较、案例计算或论文结构。
- 结果作用：若标题语义重心指向上述内容，倾向 Discard。

Step 4：处理边界标题
- 读取内容：同时包含交通业务词和方法实验词的标题。
- 对应规则：【边界标题语义重心判断规则】。
- 处理目标：判断标题的语义重心更偏向交通领域知识，还是更偏向方法实现与实验过程。
- 结果作用：若重心偏向交通概念、状态、指标、机制、关系、过程或事件影响，判为 Keep；若重心偏向实现、计算、验证、对比、参数或具体结果展示，判为 Discard。

Step 5：执行最终检查
- 读取内容：前述判断结果。
- 对应规则：【输出要求】。
- 处理目标：确认最终输出只能是 Keep 或 Discard。
- 结果作用：输出唯一判定结果，不输出理由、解释、标点或其他文本。

【保留倾向规则】
如果标题更可能指向以下内容，倾向 Keep：

1. 交通对象、交通空间单元或交通系统组成；
2. 交通状态、运行状态、交通态势、运行水平、服务水平；
3. 交通指标、评价指标、运行特征、表征维度；
4. 拥堵现象、拥堵形成、拥堵传播、拥堵演化、拥堵消散；
5. 交通事件、事故、施工、管制、天气等外部扰动及其影响；
6. 时间变化、空间分布、时空关联、时空演化、传播关系；
7. 交通概念之间的分类、层次、组成、影响、耦合、约束、触发、聚合或派生关系；
8. 交通运行规律、形成机制、作用机理、状态划分、评价含义；
9. 能够为城市交通态势本体提供类、对象属性、数据属性或结构关系线索的内容。

【剔除倾向规则】
如果标题更可能指向以下内容，倾向 Discard：

1. 数据来源、数据采集、数据处理、数据预处理、样本说明、变量说明；
2. 参数设置、参数标定、参数分析、敏感性分析；
3. 算法流程、模型训练、方法实现、系统实现、平台设计；
4. 目标函数、约束条件、求解策略、优化算法、计算流程；
5. 实验设置、仿真环境、对比实验、消融实验、性能测试、精度评价；
6. 具体案例、具体区域、具体样本或具体数据集下的数值结果展示；
7. 文献综述、研究现状、相关工作、国内外进展；
8. 技术路线、章节安排、论文结构、研究内容、创新点；
9. 致谢、参考文献、附录、作者信息、基金项目、声明等非正文内容。

【边界标题语义重心判断规则】
1. 不要因为标题中出现“交通”“拥堵”“路网”“状态”“指标”“机制”“模型”“分析”等词就直接判为 Keep。
2. 不要因为标题中出现“实验”“案例”“结果”“验证”“应用”等词就直接判为 Discard。
3. 对同时包含交通业务词和方法实验词的标题，应判断其语义重心：
   - 若标题更可能讨论交通概念、状态、指标、机制、关系、过程或事件影响，判为 Keep；
   - 若标题更可能讨论方法实现、模型效果、实验验证、数值比较、参数设置、案例计算或算法流程，判为 Discard。
4. “模型”一词需要结合语义判断：
   - 若标题中的“模型”主要指交通运行机理、交通状态关系、拥堵传播机制或本体表达对象，倾向 Keep；
   - 若标题中的“模型”主要指预测模型、训练模型、算法模型或性能评估对象，倾向 Discard。
5. “结果”“分析”“讨论”类标题需要结合对象判断：
   - 若其主要分析交通状态、拥堵过程、时空变化、事件影响或指标含义，倾向 Keep；
   - 若其主要展示实验结果、仿真结果、模型精度、案例数值或对比结果，倾向 Discard。
6. “案例”“应用”类标题只有在明显主要呈现具体地区、具体样本或具体算例的计算结果时，才倾向 Discard；若其可能总结一般性的交通态势规律、拥堵传播机制、时空演化特征或指标关系，仍可判为 Keep。
7. 标题语义不足以判断时，优先保留为 Keep，避免误删可能具有本体建模价值的章节。

【输出要求】
1. 只输出 Keep 或 Discard。
2. 不输出推理过程。
3. 不输出理由。
4. 不输出标点、编号、Markdown 或其他任何文本。

    """


    user_content = f"待判定论文章节标题：{title}"

    start_time = time.time()  # 开始计时

    try:
        request_kwargs = {
            "model": CLAUDE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1,  # 温度
            "max_tokens": 1024,  # 限制输出
        }
        extra_body = _build_thinking_extra_body()
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(
            **request_kwargs
        )
        prompt_tokens, completion_tokens, total_tokens = extract_usage_tokens(response)

        end_time = time.time()  # 成功返回，停止计时
        elapsed_time = end_time - start_time

        choices = response.choices or []
        message = choices[0].message if choices else None
        raw_content = message.content if message else ""
        raw_output, reasoning_output = _extract_text_from_content(raw_content)
        if not reasoning_output and message is not None:
            reasoning_output = (
                getattr(message, "reasoning_content", "")
                or getattr(message, "reasoning", "")
                or ""
            )
            if not isinstance(reasoning_output, str):
                reasoning_output = str(reasoning_output)
            reasoning_output = reasoning_output.strip()
        finish_reason = choices[0].finish_reason if choices else ""

        if "Keep" in raw_output or "keep" in raw_output.lower():
            decision = "Keep"
        elif "Discard" in raw_output or "discard" in raw_output.lower():
            decision = "Discard"
        else:
            decision = "Keep"

        append_raw_response_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "title": title,
            "decision": decision,
            "elapsed_sec": round(elapsed_time, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "raw_content": raw_output,
            "raw_reasoning_content": reasoning_output,
            "status": "ok",
        })

        print(f"    [仲裁完成] '{title}' -> 判定: [{decision}] | 耗时: {elapsed_time:.2f} 秒")
        return decision, elapsed_time, prompt_tokens, completion_tokens, total_tokens

    except Exception as e:
        end_time = time.time()  # 异常退出，停止计时
        elapsed_time = end_time - start_time
        append_raw_response_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "title": title,
            "decision": "Keep",
            "elapsed_sec": round(elapsed_time, 4),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "status": "error",
            "error": str(e),
        })
        print(f"    [仲裁失败] '{title}' -> {str(e)}，触发降级保护，默认保留。 | 耗时: {elapsed_time:.2f} 秒")
        time.sleep(2)
        return "Keep", elapsed_time, 0, 0, 0


def cascade_filter_pass1(normalized_title: str, cache_data: dict) -> str:
    """第一阶段静态过滤与缓存检索。"""
    title_no_space = re.sub(r'\s+', '', normalized_title)
    title_lower = title_no_space.lower()

    # 第一级：静态剔除验证
    if any(word in title_no_space for word in DISCARD_WORDS_ZH) or \
            any(word.lower() in title_lower for word in DISCARD_WORDS_EN):
        return "Discard"

    # 第一级：静态保留验证
    if any(word in title_no_space for word in KEEP_WORDS_ZH) or \
            any(word.lower() in title_lower for word in KEEP_WORDS_EN):
        return "Keep"

    # 第二级：本地缓存命中验证
    if normalized_title in cache_data:
        return cache_data[normalized_title]

    return None


def cascade_filter_pass1_with_source(normalized_title: str, cache_data: dict) -> tuple[str | None, str]:
    """返回第一阶段判定结果及其来源标签，便于统计筛选效果。"""
    title_no_space = re.sub(r'\s+', '', normalized_title)
    title_lower = title_no_space.lower()

    if any(word in title_no_space for word in DISCARD_WORDS_ZH) or \
            any(word.lower() in title_lower for word in DISCARD_WORDS_EN):
        return "Discard", "static_discard"

    if any(word in title_no_space for word in KEEP_WORDS_ZH) or \
            any(word.lower() in title_lower for word in KEEP_WORDS_EN):
        return "Keep", "static_keep"

    if normalized_title in cache_data:
        cache_decision = cache_data[normalized_title]
        if cache_decision == "Keep":
            return "Keep", "cache_keep"
        if cache_decision == "Discard":
            return "Discard", "cache_discard"
        return cache_decision, "cache_other"

    return None, "pending_llm"


def adaptive_chunking(content: str, max_length: int = 2000, min_length: int = 50) -> list[str]:
    """语料自适应切片机制。"""
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(p) < min_length:
            if current_chunk:
                current_chunk += " " + p
            continue

        if len(p) > max_length:
            sentences = re.split(r'([。！？.!?])', p)
            temp_chunk = ""
            for i in range(0, len(sentences) - 1, 2):
                sentence_full = sentences[i] + sentences[i + 1]
                if len(temp_chunk) + len(sentence_full) > max_length:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = sentence_full
                else:
                    temp_chunk += sentence_full
            if temp_chunk:
                chunks.append(temp_chunk.strip())
            continue

        if len(current_chunk) + len(p) <= max_length:
            current_chunk += "\n" + p if current_chunk else p
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def parse_markdown_robust(file_path: str, paper_id: str) -> list[dict]:
    """鲁棒性 Markdown 文档解析器。强制截断于 H3，拦截列表项伪标题与目录引文，聚合上下文。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sections = []
    current_section = "文献元数据与引言"
    current_content = []

    for line in lines:
        match = re.match(r'^#{1,3}\s+(.*)', line)
        if match:
            raw_title = match.group(1).strip()

            if re.match(r'^[\(（][0-9]+[\)）]\s*|^[0-9]+[\)）]\s*', raw_title) or \
                    re.search(r'\.{3,}\s*\d+$|…+\s*\d+$', raw_title):
                current_content.append(raw_title)
                continue

            if current_content:
                sections.append({"title": current_section, "content": "\n".join(current_content)})
            current_section = raw_title
            current_content = []

        else:
            if re.match(r'^#{4,6}\s+', line):
                clean_line = re.sub(r'^#{4,6}\s+', '', line).strip()
                if clean_line:
                    current_content.append(clean_line)
            elif line.strip():
                current_content.append(line.strip())

    if current_content:
        sections.append({"title": current_section, "content": "\n".join(current_content)})

    return sections


def summarize_lengths(lengths: list[int]) -> dict:
    """统计 chunk 长度分布，便于后续实验分析直接引用。"""
    if not lengths:
        return {
            "count": 0,
            "avg_char_len": 0.0,
            "median_char_len": 0.0,
            "min_char_len": 0,
            "max_char_len": 0,
            "too_short_chunk_total": 0,
            "too_long_chunk_total": 0,
        }

    return {
        "count": len(lengths),
        "avg_char_len": round(sum(lengths) / len(lengths), 3),
        "median_char_len": float(statistics.median(lengths)),
        "min_char_len": min(lengths),
        "max_char_len": max(lengths),
        "too_short_chunk_total": sum(1 for x in lengths if x < SHORT_CHUNK_CHAR_THRESHOLD),
        "too_long_chunk_total": sum(1 for x in lengths if x > LONG_CHUNK_CHAR_THRESHOLD),
    }


def build_openai_client() -> OpenAI:
    """构建 OpenAI 兼容客户端。"""
    return OpenAI(
        api_key=API_KEY,
        base_url=OPENAI_COMPAT_BASE_URL,
        timeout=API_TIMEOUT_SECONDS,
    )


# ==========================================
# 自动化执行流 (Pipeline Execution)
# ==========================================
def main():
    pipeline_start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_MD_DIR):
        print(f"[系统异常] 数据源目录 {INPUT_MD_DIR} 不存在，请先执行上游的文档解析流程。")
        return

    print("--- 阶段一：多源文献清洗与结构化预处理 (Claude API 并发版 / 钱多多兼容接口) ---")
    print(f"[推理设置] THINKING_MODE={THINKING_MODE}, THINKING_EFFORT={THINKING_EFFORT}, THINKING_BUDGET_TOKENS={THINKING_BUDGET_TOKENS}")
    print(f"[并发设置] MAX_CONCURRENT_REQUESTS={MAX_CONCURRENT_REQUESTS}")
    print(f"[批次设置] START_INDEX={START_INDEX}, MAX_FILES={MAX_FILES}")

    cache_data = load_cache(CACHE_FILE)
    final_corpus = []
    paper_stats = []
    all_chunk_lengths = []
    filter_counts = {
        "section_total": 0,
        "static_keep_total": 0,
        "static_discard_total": 0,
        "cache_keep_total": 0,
        "cache_discard_total": 0,
        "llm_keep_total": 0,
        "llm_discard_total": 0,
        "final_kept_section_total": 0,
        "final_discarded_section_total": 0,
    }
    filter_stats_by_paper = []
    chunk_stats_by_paper = []

    # 性能统计变量
    total_api_calls = 0
    total_api_time = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    md_files = sorted([f for f in os.listdir(INPUT_MD_DIR) if f.lower().endswith('.md')])

    if not md_files:
        print(f"[状态空闲] 目录 {INPUT_MD_DIR} 中未检测到待处理的 Markdown 文件。")
        return

    end_index = None if MAX_FILES == 0 else START_INDEX + MAX_FILES
    selected_md_files = md_files[START_INDEX:end_index]
    if not selected_md_files:
        print(f"[状态空闲] 当前批次未选中任何 Markdown 文件。START_INDEX={START_INDEX}, MAX_FILES={MAX_FILES}")
        return

    print(f"[输入统计] input_md_total={len(md_files)} | selected_md_total={len(selected_md_files)}")

    for filename in selected_md_files:
        paper_started = time.time()
        paper_id = os.path.splitext(filename)[0]
        file_path = os.path.join(INPUT_MD_DIR, filename)

        print(f"\n[装载文献] 正在处理: {paper_id}")
        sections = parse_markdown_robust(file_path, paper_id)

        section_records = []
        pending_titles = []
        pending_set = set()
        paper_filter_counts = {
            "section_total": 0,
            "static_keep_total": 0,
            "static_discard_total": 0,
            "cache_keep_total": 0,
            "cache_discard_total": 0,
            "llm_keep_total": 0,
            "llm_discard_total": 0,
            "final_kept_section_total": 0,
            "final_discarded_section_total": 0,
        }

        for sec in sections:
            orig_title = sec["title"]
            norm_title = normalize_title(orig_title)

            # 静态规则与缓存拦截
            decision, decision_source = cascade_filter_pass1_with_source(norm_title, cache_data)
            section_records.append({
                "section": sec,
                "orig_title": orig_title,
                "norm_title": norm_title,
                "decision": decision,
                "decision_source": decision_source,
            })
            paper_filter_counts["section_total"] += 1
            filter_counts["section_total"] += 1
            if decision_source == "static_keep":
                paper_filter_counts["static_keep_total"] += 1
                filter_counts["static_keep_total"] += 1
            elif decision_source == "static_discard":
                paper_filter_counts["static_discard_total"] += 1
                filter_counts["static_discard_total"] += 1
            elif decision_source == "cache_keep":
                paper_filter_counts["cache_keep_total"] += 1
                filter_counts["cache_keep_total"] += 1
            elif decision_source == "cache_discard":
                paper_filter_counts["cache_discard_total"] += 1
                filter_counts["cache_discard_total"] += 1

            # 待仲裁标题先收集，后续并发处理
            if decision is None and norm_title not in pending_set:
                pending_set.add(norm_title)
                pending_titles.append(norm_title)

        pending_results = {}
        paper_api_calls = 0
        paper_api_time = 0.0
        paper_prompt_tokens = 0
        paper_completion_tokens = 0
        paper_total_tokens = 0
        if pending_titles:
            print(f"    [并发仲裁] 待判定标题 {len(pending_titles)} 个，使用 {MAX_CONCURRENT_REQUESTS} 并发线程调用 API...")

            def _worker(title: str) -> tuple[str, str, float, int, int, int]:
                worker_client = build_openai_client()
                decision, elapsed, prompt_tokens, completion_tokens, total_tokens_item = single_llm_arbitration(title, worker_client)
                return title, decision, elapsed, prompt_tokens, completion_tokens, total_tokens_item

            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                future_to_title = {
                    executor.submit(_worker, title): title for title in pending_titles
                }
                for future in as_completed(future_to_title):
                    fallback_title = future_to_title[future]
                    try:
                        title, decision, elapsed_time, prompt_tokens, completion_tokens, total_tokens_item = future.result()
                    except Exception as e:
                        title = fallback_title
                        decision = "Keep"
                        elapsed_time = 0.0
                        prompt_tokens = 0
                        completion_tokens = 0
                        total_tokens_item = 0
                        print(f"\n    [并发异常] 标题 '{title}' 任务失败: {str(e)}，默认保留。")

                    pending_results[title] = decision
                    cache_data[title] = decision

                    # 累加统计数据
                    total_api_calls += 1
                    total_api_time += elapsed_time
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_tokens += total_tokens_item
                    paper_api_calls += 1
                    paper_api_time += elapsed_time
                    paper_prompt_tokens += prompt_tokens
                    paper_completion_tokens += completion_tokens
                    paper_total_tokens += total_tokens_item
            if pending_results:
                save_cache(cache_data, CACHE_FILE)

        # 根据最终判定结果执行切片落盘
        paper_chunk_count = 0
        kept_section_count = 0
        paper_chunk_lengths = []
        for item in section_records:
            decision = item["decision"]
            if decision is None:
                decision = pending_results.get(item["norm_title"], cache_data.get(item["norm_title"], "Keep"))
                if decision == "Keep":
                    paper_filter_counts["llm_keep_total"] += 1
                    filter_counts["llm_keep_total"] += 1
                else:
                    paper_filter_counts["llm_discard_total"] += 1
                    filter_counts["llm_discard_total"] += 1

            if decision == "Keep":
                kept_section_count += 1
                paper_filter_counts["final_kept_section_total"] += 1
                filter_counts["final_kept_section_total"] += 1
                chunks = adaptive_chunking(item["section"]["content"])
                for chunk in chunks:
                    paper_chunk_count += 1
                    chunk_length = len(chunk)
                    paper_chunk_lengths.append(chunk_length)
                    all_chunk_lengths.append(chunk_length)
                    final_corpus.append({
                        "chunk_id": f"{paper_id}__C{paper_chunk_count:04d}",
                        "paper_id": paper_id,
                        "section": item["orig_title"],  # 保留带前缀的原始标题输出
                        "content": chunk
                    })
            else:
                paper_filter_counts["final_discarded_section_total"] += 1
                filter_counts["final_discarded_section_total"] += 1

        paper_elapsed = round(time.time() - paper_started, 3)
        kept_section_ratio = round(
            paper_filter_counts["final_kept_section_total"] / paper_filter_counts["section_total"], 4
        ) if paper_filter_counts["section_total"] > 0 else 0.0
        avg_chunk_length = round(sum(paper_chunk_lengths) / len(paper_chunk_lengths), 3) if paper_chunk_lengths else 0.0
        paper_stats.append({
            "paper_id": paper_id,
            "source_md_file": filename,
            "section_total": len(section_records),
            "pending_title_total": len(pending_titles),
            "kept_section_total": kept_section_count,
            "kept_section_ratio": kept_section_ratio,
            "output_chunk_total": paper_chunk_count,
            "avg_chunk_length": avg_chunk_length,
            "api_call_total": paper_api_calls,
            "api_elapsed_sec": round(paper_api_time, 3),
            "prompt_tokens": paper_prompt_tokens,
            "completion_tokens": paper_completion_tokens,
            "total_tokens": paper_total_tokens,
            "wall_elapsed_sec": paper_elapsed,
        })
        filter_stats_by_paper.append({
            "paper_id": paper_id,
            **paper_filter_counts,
            "kept_section_ratio": kept_section_ratio,
        })
        chunk_stats_by_paper.append({
            "paper_id": paper_id,
            "chunk_total": len(paper_chunk_lengths),
            **summarize_lengths(paper_chunk_lengths),
        })
        print(
            f"    [文献完成] {paper_id} | section_total={len(section_records)} | "
            f"kept_section_total={kept_section_count} | output_chunk_total={paper_chunk_count} | "
            f"api_calls={paper_api_calls} | tokens={paper_total_tokens} | wall_elapsed={paper_elapsed:.2f}s"
        )

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_corpus, f, ensure_ascii=False, indent=4)

    kept_section_ratio = round(
        filter_counts["final_kept_section_total"] / filter_counts["section_total"], 4
    ) if filter_counts["section_total"] > 0 else 0.0
    filter_stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "section_total": filter_counts["section_total"],
        "static_keep_total": filter_counts["static_keep_total"],
        "static_discard_total": filter_counts["static_discard_total"],
        "cache_keep_total": filter_counts["cache_keep_total"],
        "cache_discard_total": filter_counts["cache_discard_total"],
        "llm_keep_total": filter_counts["llm_keep_total"],
        "llm_discard_total": filter_counts["llm_discard_total"],
        "final_kept_section_total": filter_counts["final_kept_section_total"],
        "final_discarded_section_total": filter_counts["final_discarded_section_total"],
        "final_kept_section_ratio": kept_section_ratio,
        "by_paper": filter_stats_by_paper,
    }
    chunk_stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "short_chunk_char_threshold": SHORT_CHUNK_CHAR_THRESHOLD,
        "long_chunk_char_threshold": LONG_CHUNK_CHAR_THRESHOLD,
        "chunk_total": len(all_chunk_lengths),
        **summarize_lengths(all_chunk_lengths),
        "by_paper": chunk_stats_by_paper,
    }
    write_json(OUTPUT_FILTER_STATS, filter_stats)
    write_json(OUTPUT_CHUNK_STATS, chunk_stats)

    total_wall_time = time.time() - pipeline_start_time
    avg_api_elapsed = (total_api_time / total_api_calls) if total_api_calls > 0 else 0.0
    avg_total_tokens_per_api_call = (total_tokens / total_api_calls) if total_api_calls > 0 else 0.0
    avg_paper_elapsed = (
        sum(float(x.get("wall_elapsed_sec", 0.0)) for x in paper_stats) / len(paper_stats)
        if paper_stats else 0.0
    )
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "input_md_dir": INPUT_MD_DIR,
        "output_dir": OUTPUT_DIR,
        "output_json": OUTPUT_JSON,
        "start_index": START_INDEX,
        "max_files": MAX_FILES,
        "filter_stats_file": OUTPUT_FILTER_STATS,
        "chunk_stats_file": OUTPUT_CHUNK_STATS,
        "cache_file": CACHE_FILE,
        "raw_response_log": RAW_RESPONSE_LOG,
        "chunk_id_enabled": True,
        "model": {
            "name": CLAUDE_MODEL,
            "base_url": OPENAI_COMPAT_BASE_URL,
            "api_timeout_seconds": API_TIMEOUT_SECONDS,
            "thinking_mode": THINKING_MODE,
            "thinking_effort": THINKING_EFFORT,
            "thinking_budget_tokens": THINKING_BUDGET_TOKENS,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        },
        "counts": {
            "input_md_total": len(md_files),
            "selected_md_total": len(selected_md_files),
            "output_chunk_total": len(final_corpus),
            "api_call_total": total_api_calls,
            "final_kept_section_total": filter_counts["final_kept_section_total"],
        },
        "timing": {
            "api_elapsed_total_sec": round(total_api_time, 3),
            "avg_api_elapsed_sec": round(avg_api_elapsed, 3),
            "wall_elapsed_total_sec": round(total_wall_time, 3),
            "avg_paper_elapsed_sec": round(avg_paper_elapsed, 3),
        },
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_total_tokens_per_api_call": round(avg_total_tokens_per_api_call, 3),
        },
        "paper_stats": paper_stats,
    }
    write_json(OUTPUT_MANIFEST, manifest)

    print(f"\n" + "=" * 50)
    print(f"[处理竣工] 语料抽取完毕。有效语料块总量: {len(final_corpus)}")
    print(f"[文件输出] 已生成清洗理论结构化数据集: {OUTPUT_JSON}")
    print(f"[文件输出] 标题筛选统计：{OUTPUT_FILTER_STATS}")
    print(f"[文件输出] chunk 分布统计：{OUTPUT_CHUNK_STATS}")
    print(f"[缓存更新] 标题判定记忆库已同步至: {CACHE_FILE}")
    print(f"[原始返回] 大模型原始响应日志: {RAW_RESPONSE_LOG}")
    print(f"[性能统计] 共发起 {total_api_calls} 次大模型 API 调用，累计 API 网络耗时: {total_api_time:.2f} 秒")
    print(
        f"[性能统计] token_usage -> prompt={total_prompt_tokens} | completion={total_completion_tokens} | "
        f"total={total_tokens} | avg_call={avg_total_tokens_per_api_call:.2f}"
    )
    print(f"[性能统计] 任务实际总用时（并发墙钟时间）: {total_wall_time:.2f} 秒")
    print(f"[性能统计] 平均每次仲裁耗时: {avg_api_elapsed:.2f} 秒 | 平均每篇文献耗时: {avg_paper_elapsed:.2f} 秒")
    print(f"[运行摘要] 已输出: {OUTPUT_MANIFEST}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    log_path = setup_console_log("phase1_02_markdown_clean_and_chunk")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    main()
