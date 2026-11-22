import atexit
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==========================================
# 配置区
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent  # phase2 根目录
PROJECT_ROOT = BASE_DIR.parent  # 大规模实验根目录
RESOURCE_VERSION = "V3"

ENABLE_FILE_LOG = True  # 是否把控制台输出同步写入 logs
LOG_DIR = PROJECT_ROOT / "logs"  # 统一日志目录

START_INDEX = 0  # 本次处理起始下标，用于小样本测试或断点续跑
MAX_ITEMS = 0  # 本次最多处理多少条属性；设为 0 表示从 START_INDEX 起全量处理

# 属性候选输入，来自 phase2_03 keep_new 的属性子集
INPUT_KEEP_NEW_PROPERTIES_PATH = BASE_DIR / "outputs_phase2_03" / "agent2_external_keep_new_properties.json"
# 04a 类写入判定结果，用于筛出可作为 pending_new 锚点的新类
INPUT_CLASS_POSITIONING_DECISIONS_PATH = BASE_DIR / "outputs_phase2_04a" / "agent3_class_positioning_decisions.json"
# 轻量 class 索引，用于按名称/别名查找本批新类
INPUT_CLASS_INDEX_PATH = BASE_DIR / "outputs_phase2_03" / "agent2_external_keep_new_class_index.json"
# Agent3 使用的本体 TTL，用于抽取旧本体已有类锚点
ONTOLOGY_TTL_PATH = PROJECT_ROOT / "shared_resources" / f"TrafficKG_{RESOURCE_VERSION}.ttl"

# 04b 正式输出目录
OUTPUT_DIR = BASE_DIR / "outputs_phase2_04b"
# 给 04c 使用的属性候选锚点集主文件
OUTPUT_PROPERTY_ANCHOR_CANDIDATES = OUTPUT_DIR / "property_anchor_candidates.json"
# 输入校验失败或候选构造异常记录
OUTPUT_ERRORS = OUTPUT_DIR / "property_anchor_candidate_errors.json"
# 本次运行摘要、输入输出路径与统计信息
OUTPUT_MANIFEST = OUTPUT_DIR / "property_anchor_candidate_manifest.json"
OUTPUT_STATS = OUTPUT_DIR / "property_anchor_candidate_stats.json"

_LOG_FILE_HANDLE = None
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None


class TeeStream:
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Input file must be a JSON array: {path}")
    return [x for x in payload if isinstance(x, dict)]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_for_match(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", "", str(value)).casefold().strip()
    return text


def ensure_anchor_name(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.upper() == "UNSURE":
        return "UNSURE"
    return text if text.startswith(":") else f":{text}"


def is_unsure(value: Optional[str]) -> bool:
    return str(value or "").strip().upper() == "UNSURE"


def normalize_anchor_name(raw: Optional[str]) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.strip("`'\"")
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.lstrip(":").strip()


def extract_ttl_class_anchors(ttl_text: str) -> List[str]:
    """从本体 TTL 中提取旧本体已有类名，作为 existing 锚点集合。"""
    pattern = re.compile(r"^\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+rdf:type\s+owl:Class\b", re.MULTILINE)
    return sorted({m.group("name") for m in pattern.finditer(ttl_text)})


def extract_ttl_class_labels(ttl_text: str) -> Dict[str, str]:
    """从本体 TTL 中提取类的中文标签，供 existing 锚点补充 name_zh。"""
    block_pattern = re.compile(
        r"^\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+rdf:type\s+owl:Class\b(?P<body>.*?)(?=^\s*:[A-Za-z_][A-Za-z0-9_]*\s+rdf:type\s+owl:Class\b|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    zh_label_pattern = re.compile(r'rdfs:label\s+"(?P<label>[^"]+)"@zh\b')

    labels: Dict[str, str] = {}
    for match in block_pattern.finditer(ttl_text):
        name = match.group("name")
        body = match.group("body")
        label_match = zh_label_pattern.search(body)
        if label_match:
            labels[name] = label_match.group("label").strip()
    return labels


def build_existing_anchor_lookup(
    class_anchors: List[str],
    class_labels: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """把旧本体类锚点转成可查表的 existing 锚点字典。"""
    lookup: Dict[str, Dict[str, Any]] = {}
    for name in class_anchors:
        class_name = ensure_anchor_name(name)
        payload = {
            "class_name": class_name,
            "name_zh": class_labels.get(name, ""),
            "anchor_status": "existing",
            "parent_class": None,
            "source_cluster_id": None,
        }
        for raw_key in {class_name, class_name[1:]}:
            key = normalize_for_match(raw_key)
            if key:
                lookup[key] = payload
    return lookup


def build_positioned_class_lookup(
    class_decisions: List[Dict[str, Any]],
    class_index_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """把 04a 中 decision=POSITIONED 的新类转成 pending_new 锚点字典。"""
    index_by_cluster = {
        str(row.get("cluster_id", "")).strip(): row
        for row in class_index_rows
        if str(row.get("cluster_id", "")).strip()
    }

    lookup: Dict[str, Dict[str, Any]] = {}
    for decision in class_decisions:
        if str(decision.get("element_type", "")).strip() != "Class":
            continue
        if str(decision.get("decision", "")).strip() != "POSITIONED":
            continue

        cluster_id = str(decision.get("cluster_id", "")).strip()
        if not cluster_id:
            continue

        idx = index_by_cluster.get(cluster_id, {})
        name_zh = str(idx.get("name_zh") or decision.get("name_zh") or "").strip()
        name_en = str(idx.get("name_en") or decision.get("name_en") or "").strip()
        aliases = idx.get("aliases") or []
        aliases = [str(x).strip() for x in aliases if str(x).strip()]
        canonical_anchor_name = ensure_anchor_name(
            idx.get("canonical_anchor_name") or name_en or name_zh
        )
        parent_class = ensure_anchor_name(decision.get("parent_class"))

        if not canonical_anchor_name or canonical_anchor_name == "UNSURE":
            continue

        payload = {
            "class_name": canonical_anchor_name,
            "name_zh": name_zh,
            "anchor_status": "pending_new",
            "parent_class": parent_class if parent_class and parent_class != "UNSURE" else None,
            "source_cluster_id": cluster_id,
        }

        raw_keys = {canonical_anchor_name, canonical_anchor_name[1:], name_zh, name_en, *aliases}
        for raw_key in raw_keys:
            key = normalize_for_match(raw_key)
            if key:
                lookup[key] = payload

    return lookup


def deduplicate_anchor_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按锚点身份去重，避免同一候选重复写入属性记录。"""
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("class_name", "")),
            str(item.get("anchor_status", "")),
            str(item.get("anchor_source", "")),
            str(item.get("source_cluster_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def collect_anchor_candidates(
    raw_value: Optional[str],
    existing_lookup: Dict[str, Dict[str, Any]],
    pending_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """为单个 domain/range 引用构造候选锚点，包含 exact_match 和 parent_fallback。"""
    text = str(raw_value or "").strip()
    if not text or is_unsure(text):
        return []

    candidates: List[Dict[str, Any]] = []
    lookup_keys = {
        normalize_for_match(text),
        normalize_for_match(ensure_anchor_name(text)),
        normalize_for_match(normalize_anchor_name(text)),
    }

    existing_match = None
    pending_match = None
    for key in lookup_keys:
        if not key:
            continue
        if existing_match is None and key in existing_lookup:
            existing_match = existing_lookup[key]
        if pending_match is None and key in pending_lookup:
            pending_match = pending_lookup[key]

    if existing_match is not None:
        candidates.append(
            {
                "class_name": existing_match["class_name"],
                "name_zh": existing_match.get("name_zh", ""),
                "anchor_status": "existing",
                "anchor_source": "exact_match",
                "parent_class": None,
                "source_cluster_id": None,
            }
        )

    if pending_match is not None:
        candidates.append(
            {
                "class_name": pending_match["class_name"],
                "name_zh": pending_match.get("name_zh", ""),
                "anchor_status": "pending_new",
                "anchor_source": "exact_match",
                "parent_class": pending_match.get("parent_class"),
                "source_cluster_id": pending_match.get("source_cluster_id"),
            }
        )
        parent_class = pending_match.get("parent_class")
        if parent_class:
            parent_existing_match = None
            for parent_key in {
                normalize_for_match(parent_class),
                normalize_for_match(ensure_anchor_name(parent_class)),
                normalize_for_match(normalize_anchor_name(parent_class)),
            }:
                if parent_key and parent_key in existing_lookup:
                    parent_existing_match = existing_lookup[parent_key]
                    break
            candidates.append(
                {
                    "class_name": parent_class,
                    "name_zh": (parent_existing_match or {}).get("name_zh", ""),
                    "anchor_status": "existing",
                    "anchor_source": "parent_fallback",
                    "parent_class": None,
                    "source_cluster_id": pending_match.get("source_cluster_id"),
                }
            )

    return deduplicate_anchor_candidates(candidates)


def build_candidate_record(
    prop_record: Dict[str, Any],
    existing_lookup: Dict[str, Dict[str, Any]],
    pending_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """在原属性记录上补充 domain/range 的候选锚点集。"""
    output = json.loads(json.dumps(prop_record, ensure_ascii=False))
    canonical_record = output.get("canonical_record") or {}
    canonical_element = canonical_record.get("canonical_element") or {}
    mapping_suggestion = canonical_element.get("mapping_suggestion") or {}
    element_type = str(output.get("element_type", "")).strip()

    domain_candidates = collect_anchor_candidates(
        mapping_suggestion.get("suggested_domain"),
        existing_lookup,
        pending_lookup,
    )
    range_candidates: List[Dict[str, Any]] = []
    if element_type == "ObjectProperty":
        range_candidates = collect_anchor_candidates(
            mapping_suggestion.get("suggested_range"),
            existing_lookup,
            pending_lookup,
        )

    output["domain_candidate_anchors"] = domain_candidates
    output["range_candidate_anchors"] = range_candidates
    return output


def build_property_anchor_candidate_stats(candidate_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    property_type_counts = {"ObjectProperty": 0, "DatatypeProperty": 0}
    anchor_status_counts = {"existing_total": 0, "pending_new_total": 0}
    anchor_source_counts = {"exact_match_total": 0, "parent_fallback_total": 0}
    coverage_stats = {
        "domain_anchor_present_total": 0,
        "range_anchor_present_total": 0,
        "xsd_range_present_total": 0,
        "no_candidate_anchor_total": 0,
    }

    for record in candidate_records:
        element_type = str(record.get("element_type") or "").strip()
        if element_type in property_type_counts:
            property_type_counts[element_type] += 1

        domain_candidates = record.get("domain_candidate_anchors") or []
        range_candidates = record.get("range_candidate_anchors") or []
        if domain_candidates:
            coverage_stats["domain_anchor_present_total"] += 1
        if element_type == "ObjectProperty":
            if range_candidates:
                coverage_stats["range_anchor_present_total"] += 1
        elif element_type == "DatatypeProperty":
            mapping = (((record.get("canonical_record") or {}).get("canonical_element") or {}).get("mapping_suggestion") or {})
            if str(mapping.get("suggested_xsd_range") or "").strip().upper() not in {"", "UNSURE"}:
                coverage_stats["xsd_range_present_total"] += 1
        if not domain_candidates and not range_candidates:
            coverage_stats["no_candidate_anchor_total"] += 1

        for candidate in [*domain_candidates, *range_candidates]:
            status = str(candidate.get("anchor_status") or "").strip()
            source = str(candidate.get("anchor_source") or "").strip()
            if status == "existing":
                anchor_status_counts["existing_total"] += 1
            elif status == "pending_new":
                anchor_status_counts["pending_new_total"] += 1
            if source == "exact_match":
                anchor_source_counts["exact_match_total"] += 1
            elif source == "parent_fallback":
                anchor_source_counts["parent_fallback_total"] += 1

    return {
        "property_type_counts": property_type_counts,
        "anchor_status_counts": anchor_status_counts,
        "anchor_source_counts": anchor_source_counts,
        "coverage_stats": coverage_stats,
    }


def ensure_paths() -> None:
    required = [
        INPUT_KEEP_NEW_PROPERTIES_PATH,
        INPUT_CLASS_POSITIONING_DECISIONS_PATH,
        INPUT_CLASS_INDEX_PATH,
        ONTOLOGY_TTL_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def main() -> None:
    started = time.time()
    log_path = setup_console_log("phase2_04b_build_property_anchor_candidates")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")

    ensure_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_properties = load_json_list(INPUT_KEEP_NEW_PROPERTIES_PATH)
    raw_class_decisions = load_json_list(INPUT_CLASS_POSITIONING_DECISIONS_PATH)
    raw_class_index = load_json_list(INPUT_CLASS_INDEX_PATH)
    end_index = None if MAX_ITEMS == 0 else START_INDEX + MAX_ITEMS
    selected_properties = raw_properties[START_INDEX:end_index]

    ttl_text = read_text(ONTOLOGY_TTL_PATH)
    class_anchors = extract_ttl_class_anchors(ttl_text)
    class_labels = extract_ttl_class_labels(ttl_text)
    existing_lookup = build_existing_anchor_lookup(class_anchors, class_labels)
    pending_lookup = build_positioned_class_lookup(raw_class_decisions, raw_class_index)

    print("=" * 84)
    print("[运行启动] Agent3 04b 属性候选锚点构造启动")
    print(f"[输入信息] keep_new_properties输入={INPUT_KEEP_NEW_PROPERTIES_PATH}")
    print(f"[输入信息] class_decisions输入={INPUT_CLASS_POSITIONING_DECISIONS_PATH}")
    print(f"[输入信息] class_index输入={INPUT_CLASS_INDEX_PATH}")
    print(f"[输入信息] 本次处理={len(selected_properties)} (START_INDEX={START_INDEX}, MAX_ITEMS={MAX_ITEMS})")
    print(f"[参数配置] existing_class_anchor_count={len(class_anchors)} | pending_new_class_count={len({v['source_cluster_id'] for v in pending_lookup.values() if v.get('source_cluster_id')})}")
    print(f"[输出目录] {OUTPUT_DIR}")
    print("=" * 84)

    errors: List[Dict[str, Any]] = []
    candidate_records: List[Dict[str, Any]] = []
    domain_hit_total = 0
    range_hit_total = 0

    for idx, record in enumerate(selected_properties, start=1):
        element_type = str(record.get("element_type", "")).strip()
        if element_type not in {"ObjectProperty", "DatatypeProperty"}:
            errors.append(
                {
                    "phase": "validate_input",
                    "index": START_INDEX + idx - 1,
                    "error_type": "unexpected_element_type",
                    "error": f"phase2_04b only accepts properties, got {element_type}",
                    "raw": record,
                }
            )
            continue

        output = build_candidate_record(record, existing_lookup, pending_lookup)
        if output["domain_candidate_anchors"]:
            domain_hit_total += 1
        if output["range_candidate_anchors"]:
            range_hit_total += 1
        candidate_records.append(output)

        print(
            f"[{idx}/{len(selected_properties)}] {record.get('cluster_id', '')} | {element_type} | "
            f"domain_candidates={len(output['domain_candidate_anchors'])} | "
            f"range_candidates={len(output['range_candidate_anchors'])}"
        )

    write_json(OUTPUT_PROPERTY_ANCHOR_CANDIDATES, candidate_records)
    write_json(OUTPUT_ERRORS, errors)
    stats = build_property_anchor_candidate_stats(candidate_records)
    write_json(OUTPUT_STATS, stats)

    elapsed = round(time.time() - started, 3)
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_keep_new_properties_file": str(INPUT_KEEP_NEW_PROPERTIES_PATH),
        "input_class_positioning_decisions_file": str(INPUT_CLASS_POSITIONING_DECISIONS_PATH),
        "input_class_index_file": str(INPUT_CLASS_INDEX_PATH),
        "ontology_ttl_file": str(ONTOLOGY_TTL_PATH),
        "output_dir": str(OUTPUT_DIR),
        "run_range": {
            "start_index": START_INDEX,
            "max_items": MAX_ITEMS,
            "selected_total": len(selected_properties),
        },
        "counts": {
            "input_property_total": len(raw_properties),
            "selected_property_total": len(selected_properties),
            "output_property_total": len(candidate_records),
            "error_total": len(errors),
            "existing_class_anchor_total": len(class_anchors),
            "positioned_class_total": len({v['source_cluster_id'] for v in pending_lookup.values() if v.get('source_cluster_id')}),
            "property_with_domain_candidates_total": domain_hit_total,
            "property_with_range_candidates_total": range_hit_total,
        },
        "output_property_anchor_candidates_file": str(OUTPUT_PROPERTY_ANCHOR_CANDIDATES),
        "anchor_candidate_stats_file": str(OUTPUT_STATS),
        "elapsed_sec": elapsed,
    }
    write_json(OUTPUT_MANIFEST, manifest)

    print("\n" + "=" * 84)
    print("[运行结束] Agent3 04b 属性候选锚点构造完成")
    print(
        f"[运行摘要] 输出属性={len(candidate_records)} | 错误={len(errors)} | "
        f"domain命中={domain_hit_total} | range命中={range_hit_total}"
    )
    print(f"[性能统计] 总耗时={elapsed}s")
    print(f"[关键输出] {OUTPUT_PROPERTY_ANCHOR_CANDIDATES} -> 属性候选锚点集")
    print(f"[关键输出] {OUTPUT_MANIFEST} -> 本次运行摘要")
    print("=" * 84)


if __name__ == "__main__":
    main()
