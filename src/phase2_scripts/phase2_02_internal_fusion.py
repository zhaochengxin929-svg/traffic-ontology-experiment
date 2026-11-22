import atexit
import json
import math
import re
import sys
import time
import statistics
from collections import Counter
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


# ==========================================
# 配置区（常改）
# ==========================================
# 当前阶段目录：大规模实验/phase2
BASE_DIR = Path(__file__).resolve().parent.parent
# 项目根目录：大规模实验
PROJECT_ROOT = BASE_DIR.parent
# 输入文件：Agent1 输出的 TBox 放行结果
INPUT_JSON_PATH = BASE_DIR / "outputs_phase2_01" / "agent1_passed_tbox.json"

# 输出目录：Agent2-内部融合全部产物
OUTPUT_DIR = BASE_DIR / "outputs_phase2_02"
# 内部聚类分组结果
OUTPUT_CLUSTERS = OUTPUT_DIR / "agent2_internal_groups.json"
# 归并后的 canonical 概念结果
OUTPUT_CANONICAL = OUTPUT_DIR / "agent2_internal_canonical.json"
# 错误明细
OUTPUT_ERRORS = OUTPUT_DIR / "agent2_internal_errors.json"
# 运行摘要与统计信息
OUTPUT_MANIFEST = OUTPUT_DIR / "agent2_internal_manifest.json"
# 内部融合统计摘要：供论文分析使用
OUTPUT_CLUSTER_STATS = OUTPUT_DIR / "agent2_internal_cluster_stats.json"
# 是否将控制台输出同时写入日志文件
ENABLE_FILE_LOG = True
# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"

# 本地嵌入模型名称
EMBED_MODEL_NAME = "BAAI/bge-m3"  # 本地嵌入模型名称。
# 计算设备：当前默认 CPU；后续可改为 "cuda"
EMBED_DEVICE = "cpu"  # 当前固定 CPU 跑小样本；后续可改 "cuda"。
# 向量化批大小
EMBED_BATCH_SIZE = 32  # 向量化批大小。
# 是否归一化向量；True 时可直接用点积近似余弦相似度
EMBED_NORMALIZE = True  # 是否归一化向量；True 时可直接用点积当余弦相似度。

# 内部语义合并阈值
INTERNAL_MERGE_THRESHOLD = 0.95  # Step1 内部合并阈值。

# 本轮处理条数，0 表示处理全部
MAX_ITEMS = 0  # 处理条数；0 表示处理全部。
# 起始索引，用于断点续跑
START_INDEX = 0  # 起始偏移，用于断点续跑。


# ==========================================
# Pydantic 输入模型
# ==========================================
ElementType = Literal["Class", "ObjectProperty", "DatatypeProperty"]
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


class SourceInfo(BaseModel):
    """证据来源定位信息。"""
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    section: str
    chunk_id: str


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


class Agent1Element(BaseModel):
    """承接 Agent1 输出并做名称字段兼容的输入模型。"""
    model_config = ConfigDict(extra="allow")

    name_raw: str = ""
    name_zh: str = ""
    name_en: str = ""
    element_type: ElementType
    definition: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    source: SourceInfo
    mapping_suggestion: Dict[str, Any] = Field(default_factory=dict)
    is_instance: Optional[bool] = None

    @model_validator(mode="after")
    def fill_name_triplet(self) -> "Agent1Element":
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

        ms = dict(self.mapping_suggestion or {})
        if self.element_type == "ObjectProperty":
            ms["candidate_super_property"] = normalize_property_reference(ms.get("candidate_super_property"))
        else:
            ms.pop("candidate_super_property", None)
        self.mapping_suggestion = ms
        return self


class ValidatedRecord(BaseModel):
    """带原始索引的已校验记录，用于后续聚类回溯。"""
    # 保留原始输入索引，便于聚类结果回溯到源文件位置。
    model_config = ConfigDict(extra="forbid")

    original_index: int
    item: Agent1Element


# ==========================================
# 工具函数
# ==========================================
def write_json(path: Path, data: Any) -> None:
    """将对象写入 UTF-8 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_records(path: Path) -> List[Dict[str, Any]]:
    """读取 Agent1 输出结果，并兼容 UTF-8 BOM 文件。"""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("输入文件必须是 JSON 数组。")
    return [x for x in payload if isinstance(x, dict)]


def normalize_for_match(value: Optional[str]) -> str:
    """统一字符串比较口径：去空白、大小写折叠。"""
    if value is None:
        return ""
    # 统一去除空白和大小写，提升同名命中率。
    return re.sub(r"\s+", "", str(value)).casefold().strip()


def clean_optional_text(value: Optional[str]) -> Optional[str]:
    """清理可空文本字段：空串 -> None。"""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_class_reference(value: Optional[str]) -> str:
    """归一化类引用字段：若非空且非 UNSURE，则统一为 :ClassName。"""
    text = clean_optional_text(value)
    if text is None:
        return "UNSURE"
    norm = normalize_for_match(text)
    if norm in {"", "unsure", ":unsure"}:
        return "UNSURE"
    return text if text.startswith(":") else f":{text}"


def normalize_property_reference(value: Optional[str]) -> Optional[str]:
    """归一化属性引用字段：空值返回 None，非空统一为 :propertyName。"""
    text = clean_optional_text(value)
    if text is None:
        return None
    norm = normalize_for_match(text)
    if norm in {"", "unsure", ":unsure", "none", ":none", "n/a", ":n/a", "na", ":na", "null", ":null"}:
        return None
    while text.startswith(":"):
        text = text[1:].strip()
    if not text:
        return None
    return f":{text[0].lower() + text[1:]}"


def preferred_name_zh(item: Agent1Element) -> str:
    """优先返回中文展示名，不足时回退到原名或英文名。"""
    return (item.name_zh or item.name_raw or item.name_en).strip()


def preferred_name_en(item: Agent1Element) -> str:
    """优先返回英文 IRI 名。"""
    return (item.name_en or "").strip()


def display_name(item: Agent1Element) -> str:
    """返回用于日志和聚类展示的优先名称。"""
    return preferred_name_zh(item) or preferred_name_en(item)


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


def build_cluster_size_distribution(cluster_records: List[Dict[str, Any]]) -> Dict[str, int]:
    """按簇成员数统计 size_1/2/3/4_plus 分布。"""
    dist = {
        "size_1_total": 0,
        "size_2_total": 0,
        "size_3_total": 0,
        "size_4_plus_total": 0,
    }
    for rec in cluster_records:
        size = int(rec.get("member_count", 0) or 0)
        if size <= 1:
            dist["size_1_total"] += 1
        elif size == 2:
            dist["size_2_total"] += 1
        elif size == 3:
            dist["size_3_total"] += 1
        else:
            dist["size_4_plus_total"] += 1
    return dist


def build_internal_cluster_stats(
    valid_items: List[ValidatedRecord],
    clusters: List[Dict[str, Any]],
    canonical: List[Dict[str, Any]],
    per_type_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """生成内部融合统计摘要，不影响 canonical 主记录结构。"""
    element_types = ["Class", "ObjectProperty", "DatatypeProperty"]
    input_counter = Counter(rec.item.element_type for rec in valid_items)
    cluster_records_by_type: Dict[str, List[Dict[str, Any]]] = {tp: [] for tp in element_types}
    canonical_records_by_type: Dict[str, List[Dict[str, Any]]] = {tp: [] for tp in element_types}
    for rec in clusters:
        cluster_records_by_type.setdefault(str(rec.get("element_type", "")), []).append(rec)
    for rec in canonical:
        canonical_records_by_type.setdefault(str(rec.get("element_type", "")), []).append(rec)

    per_type_counts: Dict[str, Dict[str, Any]] = {}
    per_type_cluster_size_distribution: Dict[str, Dict[str, int]] = {}
    per_type_internal_similarity_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_max_sims: List[float] = []
    all_avg_sims: List[float] = []
    for tp in element_types:
        tp_clusters = cluster_records_by_type.get(tp, [])
        tp_canonical = canonical_records_by_type.get(tp, [])
        base_stats = per_type_stats.get(tp, {})
        concept_count = int(input_counter.get(tp, 0))
        cluster_count = int(base_stats.get("cluster_count", len(tp_clusters)))
        canonical_total = len(tp_canonical)
        merged_cluster_total = int(base_stats.get("merged_cluster_count", 0))
        singleton_cluster_total = int(base_stats.get("singleton_cluster_count", 0))
        compression_ratio = round(canonical_total / concept_count, 6) if concept_count > 0 else 0.0
        per_type_counts[tp] = {
            "input_total": concept_count,
            "cluster_total": cluster_count,
            "canonical_total": canonical_total,
            "merged_cluster_total": merged_cluster_total,
            "singleton_cluster_total": singleton_cluster_total,
            "compression_ratio": compression_ratio,
        }
        per_type_cluster_size_distribution[tp] = build_cluster_size_distribution(tp_clusters)
        tp_max_sims = [float(rec["internal_similarity"]["max_pair_similarity"]) for rec in tp_canonical]
        tp_avg_sims = [float(rec["internal_similarity"]["avg_pair_similarity"]) for rec in tp_canonical]
        all_max_sims.extend(tp_max_sims)
        all_avg_sims.extend(tp_avg_sims)
        per_type_internal_similarity_stats[tp] = {
            "max_pair_similarity": summarize_similarity(tp_max_sims),
            "avg_pair_similarity": summarize_similarity(tp_avg_sims),
        }

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cluster_size_distribution": build_cluster_size_distribution(clusters),
        "per_type_cluster_size_distribution": per_type_cluster_size_distribution,
        "per_type_counts": per_type_counts,
        "internal_similarity_stats": {
            "overall": {
                "max_pair_similarity": summarize_similarity(all_max_sims),
                "avg_pair_similarity": summarize_similarity(all_avg_sims),
            },
            "per_type": per_type_internal_similarity_stats,
        },
    }


def name_keys_for_match(item: Agent1Element) -> set[str]:
    """收集名称匹配键，供硬规则同名合并使用。"""
    keys: set[str] = set()
    for candidate in [item.name_zh, item.name_raw, item.name_en]:
        key = normalize_for_match(candidate)
        if key:
            keys.add(key)
    return keys


def normalize_mapping_value(value: Optional[str]) -> str:
    """归一化类引用映射值；空值和 unsure 变体统一为 'UNSURE'。"""
    return normalize_class_reference(value)


def normalize_xsd_range(value: Optional[str]) -> str:
    """归一化 xsd range；空值和 unsure 变体统一为 'UNSURE'。"""
    text = clean_optional_text(value)
    if text is None:
        return "UNSURE"
    norm = normalize_for_match(text)
    if norm in {"", "unsure", ":unsure"}:
        return "UNSURE"
    return text.lower()


def mapping_signature(item: Agent1Element) -> str:
    """把 mapping_suggestion 映射为可计数、可比较的签名字符串。"""
    ms = item.mapping_suggestion if isinstance(item.mapping_suggestion, dict) else {}
    if item.element_type == "Class":
        parent = normalize_mapping_value(ms.get("suggested_parent_class"))
        return f"Class|parent={parent}"
    if item.element_type == "ObjectProperty":
        domain = normalize_mapping_value(ms.get("suggested_domain"))
        rng = normalize_mapping_value(ms.get("suggested_range"))
        return f"ObjectProperty|domain={domain}|range={rng}"
    if item.element_type == "DatatypeProperty":
        domain = normalize_mapping_value(ms.get("suggested_domain"))
        xsd_range = normalize_xsd_range(ms.get("suggested_xsd_range"))
        return f"DatatypeProperty|domain={domain}|xsd={xsd_range}"
    return "UNKNOWN"


def is_allowed_class_parent_near_match(p1: str, p2: str) -> bool:
    """判断两个 Class 父类是否属于允许豁免的近邻父类组合。"""
    pair = {p1, p2}
    return pair == {":SegmentIndicator", ":IntersectionIndicator"}


def mapping_conflict(a: Agent1Element, b: Agent1Element) -> bool:
    """判断两条记录是否存在“硬冲突映射”，冲突则禁止合并。"""
    if a.element_type != b.element_type:
        return True

    msa = a.mapping_suggestion if isinstance(a.mapping_suggestion, dict) else {}
    msb = b.mapping_suggestion if isinstance(b.mapping_suggestion, dict) else {}

    def hard_conflict(v1: str, v2: str) -> bool:
        return v1 != "UNSURE" and v2 != "UNSURE" and normalize_for_match(v1) != normalize_for_match(v2)

    if a.element_type == "Class":
        p1 = normalize_mapping_value(msa.get("suggested_parent_class"))
        p2 = normalize_mapping_value(msb.get("suggested_parent_class"))
        if p1 == "UNSURE" or p2 == "UNSURE":
            return False
        if p1 == p2:
            return False
        if is_allowed_class_parent_near_match(p1, p2):
            return False
        return True

    if a.element_type == "ObjectProperty":
        d1 = normalize_mapping_value(msa.get("suggested_domain"))
        d2 = normalize_mapping_value(msb.get("suggested_domain"))
        r1 = normalize_mapping_value(msa.get("suggested_range"))
        r2 = normalize_mapping_value(msb.get("suggested_range"))
        return hard_conflict(d1, d2) or hard_conflict(r1, r2)

    if a.element_type == "DatatypeProperty":
        d1 = normalize_mapping_value(msa.get("suggested_domain"))
        d2 = normalize_mapping_value(msb.get("suggested_domain"))
        x1 = normalize_xsd_range(msa.get("suggested_xsd_range"))
        x2 = normalize_xsd_range(msb.get("suggested_xsd_range"))
        return hard_conflict(d1, d2) or hard_conflict(x1, x2)

    return False


def is_pure_unsure_mapping(item: Agent1Element) -> bool:
    """判断一条记录的 mapping_suggestion 是否属于纯 UNSURE 状态。"""
    ms = item.mapping_suggestion if isinstance(item.mapping_suggestion, dict) else {}

    if item.element_type == "Class":
        parent = normalize_mapping_value(ms.get("suggested_parent_class"))
        return parent == "UNSURE"

    if item.element_type == "ObjectProperty":
        domain = normalize_mapping_value(ms.get("suggested_domain"))
        rng = normalize_mapping_value(ms.get("suggested_range"))
        return domain == "UNSURE" and rng == "UNSURE"

    if item.element_type == "DatatypeProperty":
        domain = normalize_mapping_value(ms.get("suggested_domain"))
        xsd_range = normalize_xsd_range(ms.get("suggested_xsd_range"))
        return domain == "UNSURE" and xsd_range == "UNSURE"

    return True


def select_majority_candidate_super_property(
    items: List[Agent1Element],
    canonical_item: Agent1Element,
) -> Optional[str]:
    """对 ObjectProperty 的 candidate_super_property 做字段级单独汇总。"""
    non_empty_values: List[str] = []
    for item in items:
        ms = item.mapping_suggestion if isinstance(item.mapping_suggestion, dict) else {}
        value = normalize_property_reference(ms.get("candidate_super_property"))
        if value:
            non_empty_values.append(value)

    if not non_empty_values:
        return None

    counts = Counter(non_empty_values)
    winner_count = max(counts.values())
    winner_values = {value for value, cnt in counts.items() if cnt == winner_count}

    canonical_ms = canonical_item.mapping_suggestion if isinstance(canonical_item.mapping_suggestion, dict) else {}
    canonical_value = normalize_property_reference(canonical_ms.get("candidate_super_property"))
    if canonical_value in winner_values:
        return canonical_value
    return sorted(winner_values)[0]


def _select_majority_mapping_legacy(items: List[Agent1Element], canonical_item: Agent1Element) -> Dict[str, Any]:
    """从簇内选择票数最多的 mapping_suggestion。"""
    if not items:
        return canonical_item.mapping_suggestion

    signature_to_items: Dict[str, List[Agent1Element]] = {}
    for item in items:
        sig = mapping_signature(item)
        signature_to_items.setdefault(sig, []).append(item)

    freq = Counter({sig: len(bucket) for sig, bucket in signature_to_items.items()})
    concrete_signatures = {
        sig for sig, bucket in signature_to_items.items() if not is_pure_unsure_mapping(bucket[0])
    }
    candidate_freq = {sig: cnt for sig, cnt in freq.items() if sig in concrete_signatures} or dict(freq)

    winner_count = max(candidate_freq.values())
    winner_signatures = {sig for sig, cnt in candidate_freq.items() if cnt == winner_count}
    canonical_sig = mapping_signature(canonical_item)

    # 平票时优先保留 canonical 的映射建议，保证结果稳定。
    if canonical_sig in winner_signatures:
        winner_sig = canonical_sig
    else:
        winner_sig = sorted(winner_signatures)[0]

    if winner_sig == canonical_sig:
        winner_item = canonical_item
    else:
        winner_item = signature_to_items.get(winner_sig, [canonical_item])[0]
    winner_ms = winner_item.mapping_suggestion if isinstance(winner_item.mapping_suggestion, dict) else {}

    # 归一为稳定字段结构，便于 Step2/Step3 直接读取。
    return {
        "suggested_parent_class": clean_optional_text(winner_ms.get("suggested_parent_class")),
        "suggested_domain": clean_optional_text(winner_ms.get("suggested_domain")),
        "suggested_range": clean_optional_text(winner_ms.get("suggested_range")),
        "suggested_xsd_range": clean_optional_text(winner_ms.get("suggested_xsd_range")),
        "candidate_super_property": normalize_property_reference(winner_ms.get("candidate_super_property")),
        "reason": clean_optional_text(winner_ms.get("reason")) or "",
    }


def select_majority_mapping(items: List[Agent1Element], canonical_item: Agent1Element) -> Dict[str, Any]:
    """从簇内选择票数最多的 mapping_suggestion，并对字段做类型化汇总。"""
    if not items:
        return canonical_item.mapping_suggestion

    signature_to_items: Dict[str, List[Agent1Element]] = {}
    for item in items:
        sig = mapping_signature(item)
        signature_to_items.setdefault(sig, []).append(item)

    freq = Counter({sig: len(bucket) for sig, bucket in signature_to_items.items()})
    concrete_signatures = {
        sig for sig, bucket in signature_to_items.items() if not is_pure_unsure_mapping(bucket[0])
    }
    candidate_freq = {sig: cnt for sig, cnt in freq.items() if sig in concrete_signatures} or dict(freq)

    winner_count = max(candidate_freq.values())
    winner_signatures = {sig for sig, cnt in candidate_freq.items() if cnt == winner_count}
    canonical_sig = mapping_signature(canonical_item)

    if canonical_sig in winner_signatures:
        winner_sig = canonical_sig
    else:
        winner_sig = sorted(winner_signatures)[0]

    if winner_sig == canonical_sig:
        winner_item = canonical_item
    else:
        winner_item = signature_to_items.get(winner_sig, [canonical_item])[0]
    winner_ms = winner_item.mapping_suggestion if isinstance(winner_item.mapping_suggestion, dict) else {}

    if canonical_item.element_type == "Class":
        return {
            "suggested_parent_class": clean_optional_text(winner_ms.get("suggested_parent_class")),
            "reason": clean_optional_text(winner_ms.get("reason")) or "",
        }

    if canonical_item.element_type == "ObjectProperty":
        return {
            "suggested_domain": clean_optional_text(winner_ms.get("suggested_domain")),
            "suggested_range": clean_optional_text(winner_ms.get("suggested_range")),
            "candidate_super_property": select_majority_candidate_super_property(items, canonical_item),
            "reason": clean_optional_text(winner_ms.get("reason")) or "",
        }

    if canonical_item.element_type == "DatatypeProperty":
        return {
            "suggested_domain": clean_optional_text(winner_ms.get("suggested_domain")),
            "suggested_xsd_range": clean_optional_text(winner_ms.get("suggested_xsd_range")),
            "reason": clean_optional_text(winner_ms.get("reason")) or "",
        }

    return {
        "reason": clean_optional_text(winner_ms.get("reason")) or "",
    }


def build_embedding_text(item: Agent1Element) -> str:
    """组装概念向量化文本，供内部语义聚类使用。"""
    return (
        f"name_zh: {preferred_name_zh(item)}\n"
        f"name_en: {preferred_name_en(item)}\n"
        f"name_raw: {item.name_raw}\n"
        f"definition: {item.definition}\n"
        f"evidence_text: {item.evidence_text}"
    )


def cosine_similarity_matrix(embeddings: np.ndarray, normalized: bool) -> np.ndarray:
    """计算同批向量的两两余弦相似度矩阵。"""
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=float)

    if normalized:
        sim = embeddings @ embeddings.T
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        denom = np.clip(norms @ norms.T, a_min=1e-12, a_max=None)
        sim = (embeddings @ embeddings.T) / denom

    sim = np.clip(sim, -1.0, 1.0)
    return sim


class UnionFind:
    """并查集工具：用于相似度连通分量聚类。"""
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def cluster_by_threshold(sim: np.ndarray, threshold: float) -> List[List[int]]:
    """基于相似度阈值做连通分量聚类。"""
    n = sim.shape[0]
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                uf.union(i, j)

    components: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        components[uf.find(i)].append(i)
    clusters = [sorted(v) for _, v in components.items()]
    clusters.sort(key=lambda xs: xs[0] if xs else math.inf)
    return clusters


def score_for_canonical(item: Agent1Element) -> float:
    """为簇内候选打分，选出信息最完整的 canonical 概念。"""
    # 优先选择语义信息更完整的概念作为簇代表。
    return float(
        len(display_name(item)) * 2
        + len(item.definition)
        + len(item.evidence_text) * 0.5
        + (15 if preferred_name_en(item) else 0)
    )


def safe_avg(values: List[float]) -> float:
    """计算列表平均值；空列表时返回 0。"""
    return float(sum(values) / len(values)) if values else 0.0


def dedupe_preserve_order(values: List[str]) -> List[str]:
    """按规范化键去重，但保留原始出现顺序。"""
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        key = normalize_for_match(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def ensure_sentence_transformers():
    """延迟导入 sentence-transformers，避免启动阶段硬失败。"""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as e:  # noqa: F841
        raise RuntimeError(
            "未安装 sentence-transformers，请先执行: pip install -U sentence-transformers"
        )
    return SentenceTransformer


def run_internal_fusion(records: List[ValidatedRecord]) -> Dict[str, Any]:
    """Agent2-Step1：内部自融合主流程（不调用 LLM）。"""
    SentenceTransformer = ensure_sentence_transformers()
    model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        item = rec.item
        by_type[item.element_type].append(
            {
                "global_index": rec.original_index,
                "item": item,
                "embedding_text": build_embedding_text(item),
            }
        )

    all_clusters: List[Dict[str, Any]] = []
    all_canonical: List[Dict[str, Any]] = []
    per_type_stats: Dict[str, Dict[str, Any]] = {}

    print(f"[向量模型] model={EMBED_MODEL_NAME} | device={EMBED_DEVICE} | batch_size={EMBED_BATCH_SIZE}")
    print(f"[融合参数] threshold={INTERNAL_MERGE_THRESHOLD}")

    cluster_counter = 0
    for element_type, bucket in by_type.items():
        count = len(bucket)
        if count == 0:
            continue
        print(f"\n[类型处理] {element_type} | 概念数={count}")

        texts = [x["embedding_text"] for x in bucket]
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=EMBED_NORMALIZE,
            show_progress_bar=False,
        )
        sim = cosine_similarity_matrix(embeddings, EMBED_NORMALIZE)

        # 两段式连边：
        # 1) 硬规则同名命中；2) 向量相似度 >= 阈值；
        # 两者都必须先通过 mapping 冲突检查。
        uf = UnionFind(count)
        for i in range(count):
            for j in range(i + 1, count):
                item_i = bucket[i]["item"]
                item_j = bucket[j]["item"]
                if mapping_conflict(item_i, item_j):
                    continue
                hard_name_match = bool(name_keys_for_match(item_i) & name_keys_for_match(item_j))
                vector_match = float(sim[i, j]) >= INTERNAL_MERGE_THRESHOLD
                if hard_name_match or vector_match:
                    uf.union(i, j)

        components: Dict[int, List[int]] = defaultdict(list)
        for i in range(count):
            components[uf.find(i)].append(i)
        clusters = [sorted(v) for _, v in components.items()]
        clusters.sort(key=lambda xs: xs[0] if xs else math.inf)

        merged_cluster_count = 0
        singleton_count = 0

        for local_indices in clusters:
            cluster_counter += 1
            cluster_id = f"A2C_{cluster_counter:05d}"
            members = [bucket[i] for i in local_indices]
            items = [m["item"] for m in members]
            global_indices = [m["global_index"] for m in members]

            if len(local_indices) == 1:
                max_sim = 1.0
                avg_sim = 1.0
                singleton_count += 1
            else:
                pair_sims: List[float] = []
                for i in range(len(local_indices)):
                    for j in range(i + 1, len(local_indices)):
                        pair_sims.append(float(sim[local_indices[i], local_indices[j]]))
                max_sim = max(pair_sims) if pair_sims else 1.0
                avg_sim = safe_avg(pair_sims)
                merged_cluster_count += 1

            canonical_local_idx = max(
                range(len(items)),
                key=lambda i: score_for_canonical(items[i]),
            )
            canonical_item = items[canonical_local_idx]
            majority_mapping = select_majority_mapping(items, canonical_item)

            aliases: List[str] = []
            canonical_name_norm = normalize_for_match(display_name(canonical_item))
            for i, it in enumerate(items):
                if i == canonical_local_idx:
                    continue
                # aliases 保留“非 canonical 本名”的中英文别称。
                for alias in [preferred_name_zh(it), preferred_name_en(it)]:
                    if not alias:
                        continue
                    if normalize_for_match(alias) == canonical_name_norm:
                        continue
                    if alias not in aliases:
                        aliases.append(alias)
            aliases = dedupe_preserve_order(aliases)

            merged_definitions = []
            definition_seen: set[str] = set()
            for it in items:
                k = normalize_for_match(it.definition)
                if k in definition_seen:
                    continue
                definition_seen.add(k)
                if it.definition not in merged_definitions:
                    merged_definitions.append(it.definition)

            merged_evidence = []
            evidence_seen: set[str] = set()
            for it in items:
                source_dump = it.source.model_dump()
                # 同名 + 同证据文本 + 同来源位置视为同一条证据。
                evidence_key = "|".join(
                    [
                        normalize_for_match(display_name(it)),
                        normalize_for_match(it.evidence_text),
                        normalize_for_match(source_dump.get("paper_id")),
                        normalize_for_match(source_dump.get("section")),
                        normalize_for_match(source_dump.get("chunk_id")),
                    ]
                )
                if evidence_key in evidence_seen:
                    continue
                evidence_seen.add(evidence_key)
                merged_evidence.append(
                    {
                        "name_zh": preferred_name_zh(it),
                        "name_en": preferred_name_en(it),
                        "evidence_text": it.evidence_text,
                        "source": source_dump,
                    }
                )

            cluster_record = {
                "cluster_id": cluster_id,
                "element_type": element_type,
                "member_count": len(items),
                "member_global_indices": global_indices,
                "member_name_zh": [preferred_name_zh(it) for it in items],
                "canonical_name_zh": preferred_name_zh(canonical_item),
                "canonical_name_en": preferred_name_en(canonical_item),
                "max_pair_similarity": round(max_sim, 6),
                "avg_pair_similarity": round(avg_sim, 6),
                "source_paper_count": len({it.source.paper_id for it in items}),
                "source_papers": sorted({it.source.paper_id for it in items}),
            }
            all_clusters.append(cluster_record)

            canonical_dump = canonical_item.model_dump()
            canonical_dump["mapping_suggestion"] = majority_mapping
            canonical_record = {
                "cluster_id": cluster_id,
                "element_type": element_type,
                "member_count": len(items),
                "internal_status": "merged" if len(items) > 1 else "singleton",
                "internal_similarity": {
                    "max_pair_similarity": round(max_sim, 6),
                    "avg_pair_similarity": round(avg_sim, 6),
                    "threshold": INTERNAL_MERGE_THRESHOLD,
                },
                "canonical_element": canonical_dump,
                "aliases": aliases,
                "merged_definitions": merged_definitions,
                "merged_evidence": merged_evidence,
            }
            all_canonical.append(canonical_record)

        per_type_stats[element_type] = {
            "concept_count": count,
            "cluster_count": len(clusters),
            "merged_cluster_count": merged_cluster_count,
            "singleton_cluster_count": singleton_count,
        }

        print(
            f"[类型完成] {element_type} | 聚类数={len(clusters)} | "
            f"合并簇={merged_cluster_count} | 单例簇={singleton_count}"
        )

    return {
        "clusters": all_clusters,
        "canonical": all_canonical,
        "per_type_stats": per_type_stats,
    }


def main() -> None:
    """主流程：读取 Agent1 结果，完成内部融合并写出 canonical 结果。"""
    started = time.time()
    log_path = setup_console_log("phase2_02_internal_fusion")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")
    if not INPUT_JSON_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_JSON_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_records = load_records(INPUT_JSON_PATH)
    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    selected_raw = raw_records[START_INDEX:end_index]

    print("=" * 72)
    print("[运行启动] Agent2 Step1 内部融合启动")
    print(f"[输入信息] input={INPUT_JSON_PATH}")
    print(f"[输入信息] input_total={len(raw_records)} | selected={len(selected_raw)} | START_INDEX={START_INDEX} | MAX_ITEMS={MAX_ITEMS}")
    print(f"[参数配置] merge_threshold={INTERNAL_MERGE_THRESHOLD}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print("=" * 72)

    valid_items: List[ValidatedRecord] = []
    errors: List[Dict[str, Any]] = []
    for idx, raw in enumerate(selected_raw):
        original_index = START_INDEX + idx
        try:
            item = Agent1Element.model_validate(raw)
            if item.is_instance is True:
                errors.append(
                    {
                        "index": original_index,
                        "error_type": "instance_blocked",
                        "error": "is_instance=true，不允许进入 Agent2 Step1 的 TBox 内部自融合。",
                        "raw": raw,
                    }
                )
                continue
            if item.is_instance is None:
                errors.append(
                    {
                        "index": original_index,
                        "error_type": "missing_is_instance",
                        "error": "is_instance 缺失或为 null，无法确认 TBox 安全性。",
                        "raw": raw,
                    }
                )
                continue
            valid_items.append(ValidatedRecord(original_index=original_index, item=item))
        except ValidationError as e:
            errors.append(
                {
                    "index": original_index,
                    "error_type": "validation_error",
                    "error": str(e),
                    "raw": raw,
                }
            )

    print(f"[校验结果] 有效记录={len(valid_items)} | 无效记录={len(errors)}")
    write_json(OUTPUT_ERRORS, errors)

    if not valid_items:
        write_json(OUTPUT_CLUSTERS, [])
        write_json(OUTPUT_CANONICAL, [])
        write_json(OUTPUT_CLUSTER_STATS, {})
        manifest = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "no_valid_items",
            "input_file": str(INPUT_JSON_PATH),
            "output_dir": str(OUTPUT_DIR),
            "cluster_stats_file": str(OUTPUT_CLUSTER_STATS),
            "input_total": len(raw_records),
            "selected_total": len(selected_raw),
            "valid_total": 0,
            "invalid_total": len(errors),
            "elapsed_sec": round(time.time() - started, 3),
        }
        write_json(OUTPUT_MANIFEST, manifest)
        print("[运行结束] 没有可处理记录，已输出空结果。")
        return

    result = run_internal_fusion(valid_items)
    clusters = result["clusters"]
    canonical = result["canonical"]
    per_type_stats = result["per_type_stats"]
    cluster_stats = build_internal_cluster_stats(valid_items, clusters, canonical, per_type_stats)

    write_json(OUTPUT_CLUSTERS, clusters)
    write_json(OUTPUT_CANONICAL, canonical)
    write_json(OUTPUT_CLUSTER_STATS, cluster_stats)

    elapsed = round(time.time() - started, 3)
    member_total_check = sum(c["member_count"] for c in clusters)
    canonical_total_check = len(canonical) == len(clusters)
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(INPUT_JSON_PATH),
        "output_dir": str(OUTPUT_DIR),
        "embedding": {
            "model_name": EMBED_MODEL_NAME,
            "device": EMBED_DEVICE,
            "batch_size": EMBED_BATCH_SIZE,
            "normalize_embeddings": EMBED_NORMALIZE,
        },
        "thresholds": {
            "internal_merge_threshold": INTERNAL_MERGE_THRESHOLD,
        },
        "run_range": {
            "start_index": START_INDEX,
            "max_items": MAX_ITEMS,
            "selected_total": len(selected_raw),
        },
        "counts": {
            "input_total": len(raw_records),
            "valid_total": len(valid_items),
            "invalid_total": len(errors),
            "cluster_total": len(clusters),
            "canonical_total": len(canonical),
            "merged_cluster_total": sum(1 for c in canonical if c["internal_status"] == "merged"),
            "singleton_cluster_total": sum(1 for c in canonical if c["internal_status"] == "singleton"),
            "compression_ratio": round(len(canonical) / len(valid_items), 6) if valid_items else 0.0,
        },
        "consistency_checks": {
            "member_total_equals_valid_total": member_total_check == len(valid_items),
            "member_total": member_total_check,
            "canonical_total_equals_cluster_total": canonical_total_check,
        },
        "per_type_stats": per_type_stats,
        "cluster_stats_file": str(OUTPUT_CLUSTER_STATS),
        "elapsed_sec": elapsed,
    }
    write_json(OUTPUT_MANIFEST, manifest)

    print("\n" + "=" * 72)
    print("[运行结束] Agent2 Step1 内部融合完成")
    print(f"[运行摘要] 有效记录={len(valid_items)} | 无效记录={len(errors)}")
    print(f"[运行摘要] 聚类总数={len(clusters)} | canonical总数={len(canonical)}")
    print(f"[性能统计] 总耗时={elapsed}s")
    print(
        f"[一致性检查] member_total_equals_valid_total={member_total_check == len(valid_items)} | "
        f"canonical_total_equals_cluster_total={canonical_total_check}"
    )
    print(f"[关键输出] {OUTPUT_CANONICAL} -> Agent2 外部对齐输入主文件")
    print(f"[关键输出] {OUTPUT_CLUSTERS} -> 内部融合分组结果")
    print(f"[关键输出] {OUTPUT_CLUSTER_STATS} -> 内部融合统计摘要")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要")
    print(f"[关键输出] {OUTPUT_ERRORS} -> 无效记录与错误清单")
    print("=" * 72)


if __name__ == "__main__":
    main()
