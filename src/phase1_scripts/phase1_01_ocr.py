import atexit
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# =========================
# 配置区（常改）
# =========================
# 当前阶段目录：大规模实验/phase1
BASE_DIR = Path(__file__).resolve().parent.parent
# 主实验工程根目录：main_experiment
PROJECT_ROOT = BASE_DIR.parent
# 工作区根目录：Traffic_KG_Project
WORKSPACE_ROOT = PROJECT_ROOT.parent
# 智谱 API Key，可直接填写
API_KEY = ""
# 输入目录：正式主实验数据集根目录，脚本会递归扫描其下全部 PDF
INPUT_DIR = WORKSPACE_ROOT / "dataset_main"
# 输出目录：OCR 识别后生成的 Markdown 文件
OUTPUT_DIR = BASE_DIR / "outputs_ocr" / "papers_md"
# OCR 单篇文件统计：用于估算每篇 PDF 的时间成本
OUTPUT_FILE_STATS = BASE_DIR / "outputs_ocr" / "ocr_file_stats.json"
# OCR 本次运行摘要：用于估算总时间与平均速度
OUTPUT_MANIFEST = BASE_DIR / "outputs_ocr" / "ocr_run_manifest.json"
# 单次 OCR 请求处理的页数；文档过大时可适当调小
PAGE_CHUNK_SIZE = 30
# OCR 并发数：按 PDF 文件级并发，建议不超过账号并发限制
MAX_CONCURRENT_REQUESTS = 2
# 起始文件索引：按排序后的 PDF 列表位置计数，0 表示从第一篇开始
START_INDEX = 78
# 本轮最多处理多少篇 PDF；0 表示处理全部
MAX_FILES = 5
# 是否将控制台输出同时写入日志文件
ENABLE_FILE_LOG = True
# 日志目录：自动保存本次运行的控制台输出
LOG_DIR = PROJECT_ROOT / "logs"
# 固定使用的 OCR 模型名称
OCR_MODEL = "glm-ocr"
# 是否覆盖已存在的 Markdown；False=跳过已完成文件
FORCE_OVERWRITE = False

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


def get_client(api_key: str):
    """延迟导入智谱 SDK，只有真正调用 API 时才要求安装依赖。"""
    from zai import ZhipuAiClient
    return ZhipuAiClient(api_key=api_key)


def write_json(path: Path, data) -> None:
    """将对象写入 UTF-8 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_pdf_page_count(pdf_path: Path) -> int:
    """读取 PDF 总页数，用于后续按页分段请求 OCR。"""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def encode_pdf_data_uri(pdf_path: Path) -> str:
    """将 PDF 文件编码为 data URI，符合智谱 OCR 接口的 file 入参格式。"""
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def _get_field(obj, key: str, default=None):
    """兼容 SDK 对象和字典两种返回格式。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_usage_tokens(resp) -> tuple[int, int, int]:
    """从智谱 OCR 响应中提取 prompt/completion/total token。"""
    usage = _get_field(resp, "usage")
    if usage is None:
        return 0, 0, 0

    prompt_tokens = int(_get_field(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_get_field(usage, "completion_tokens", 0) or 0)
    total_tokens = int(_get_field(usage, "total_tokens", 0) or 0)
    return prompt_tokens, completion_tokens, total_tokens


def ocr_single_pdf(client, pdf_path: Path, md_path: Path, page_chunk_size: int) -> dict:
    """
    对单个 PDF 执行 OCR，并输出为 Markdown 文件。
    设计要点：
    1. 按页分块请求，避免大文档单次请求过重。
    2. 每块返回 Markdown 后拼接，最后一次性写盘。
    """
    started = time.time()
    total_pages = get_pdf_page_count(pdf_path)
    file_uri = encode_pdf_data_uri(pdf_path)
    md_parts = []
    chunk_count = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0

    for start in range(1, total_pages + 1, page_chunk_size):
        end = min(start + page_chunk_size - 1, total_pages)
        chunk_count += 1
        print(f"[OCR] 正在识别：{pdf_path.name}，页码 {start}-{end}")

        # 调用智谱 OCR 的版面解析接口，返回该页段对应的 Markdown 文本
        resp = client.layout_parsing.create(
            model=OCR_MODEL,
            file=file_uri,
            start_page_id=start,
            end_page_id=end,
        )
        prompt_tokens, completion_tokens, total_tokens = extract_usage_tokens(resp)
        prompt_tokens_total += prompt_tokens
        completion_tokens_total += completion_tokens
        total_tokens_total += total_tokens
        md_result = getattr(resp, "md_results", "")
        if md_result:
            md_parts.append(md_result)

    # 输出目录不存在则自动创建
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n\n".join(md_parts), encoding="utf-8")
    elapsed = round(time.time() - started, 3)
    return {
        "file_name": pdf_path.name,
        "page_count": total_pages,
        "chunk_count": chunk_count,
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total,
        "total_tokens": total_tokens_total,
        "status": "success",
    }


def process_single_pdf(index: int, total: int, pdf_file: Path, output_md_dir: Path, api_key: str) -> dict:
    """处理单篇 PDF，供线程池调用；每个任务独立创建 client，避免线程共享风险。"""
    md_file = output_md_dir / f"{pdf_file.stem}.md"
    print(f"\n[{index}/{total}] 准备处理：{pdf_file.name}")

    if md_file.exists() and not FORCE_OVERWRITE:
        print(f"[跳过] 已存在 Markdown：{md_file.name}")
        page_count = None
        try:
            page_count = get_pdf_page_count(pdf_file)
        except Exception:
            page_count = None
        return {
            "index": index,
            "stats": {
                "file_name": pdf_file.name,
                "page_count": page_count,
                "chunk_count": None if page_count is None else (page_count + PAGE_CHUNK_SIZE - 1) // PAGE_CHUNK_SIZE,
                "elapsed_sec": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "status": "skipped",
            },
        }

    try:
        client = get_client(api_key)
        stats = ocr_single_pdf(client, pdf_file, md_file, PAGE_CHUNK_SIZE)
        print(
            f"[完成] 已输出：{md_file.name} | 页数={stats['page_count']} | "
            f"分块={stats['chunk_count']} | 耗时={stats['elapsed_sec']}s | "
            f"tokens={stats['total_tokens']}"
        )
        return {"index": index, "stats": stats}
    except Exception as exc:
        print(f"[OCR-FAIL] {pdf_file.name}: {exc}")
        return {
            "index": index,
            "stats": {
                "file_name": pdf_file.name,
                "page_count": None,
                "chunk_count": None,
                "elapsed_sec": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "status": "failed",
                "error": str(exc),
            },
        }


def main() -> int:
    """
    主流程：
    1. 读取配置区并检查输入目录
    2. 检查 API Key
    3. 递归遍历 PDF，按需执行 OCR
    4. 输出统计信息
    """
    input_pdf_dir: Path = INPUT_DIR
    output_md_dir: Path = OUTPUT_DIR
    output_md_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_console_log("phase1_01_ocr")

    print("[开始] 阶段一 OCR：PDF -> Markdown")
    print(f"[配置] 输入目录：{input_pdf_dir}")
    print(f"[配置] 输出目录：{output_md_dir}")
    print("[配置] PDF 扫描方式：递归扫描输入根目录下全部子目录")
    print(f"[配置] 分页大小：{PAGE_CHUNK_SIZE} 页/次")
    print(f"[配置] PDF 并发数：{MAX_CONCURRENT_REQUESTS}")
    print(f"[配置] 起始索引：{START_INDEX}")
    print(f"[配置] 最大处理篇数：{MAX_FILES if MAX_FILES > 0 else '全部'}")
    print(f"[配置] 覆盖已存在结果：{FORCE_OVERWRITE}")
    if log_path:
        print(f"[日志] 控制台输出已同步保存：{log_path}")

    all_pdf_files = sorted(input_pdf_dir.rglob("*.pdf")) if input_pdf_dir.exists() else []
    if not all_pdf_files:
        print(f"[ERROR] 未找到 PDF 文件: {input_pdf_dir}")
        return 1

    if START_INDEX < 0:
        print(f"[ERROR] START_INDEX 不能为负数：{START_INDEX}")
        return 1

    if START_INDEX >= len(all_pdf_files):
        print(f"[ERROR] START_INDEX 超出 PDF 总数范围：{START_INDEX} >= {len(all_pdf_files)}")
        return 1

    selected_pdf_files = all_pdf_files[START_INDEX:]
    if MAX_FILES > 0:
        selected_pdf_files = selected_pdf_files[:MAX_FILES]

    if not selected_pdf_files:
        print("[ERROR] 根据当前 START_INDEX / MAX_FILES 组合，未选中任何 PDF 文件。")
        return 1

    print(f"[配置] 扫描到 PDF 总数：{len(all_pdf_files)}")
    print(f"[配置] 本次选中 PDF 数：{len(selected_pdf_files)}")
    print(f"[配置] 本次起始文件：{selected_pdf_files[0].name}")
    print(f"[配置] 本次结束文件：{selected_pdf_files[-1].name}")

    api_key = (API_KEY or "").strip()
    if not api_key:
        print("[ERROR] 未填写 API_KEY，请在脚本配置区填写。")
        return 1

    success = 0
    skipped = 0
    failed = 0
    started = time.time()
    file_stats = []
    max_workers = max(1, min(MAX_CONCURRENT_REQUESTS, len(selected_pdf_files)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_single_pdf, index, len(selected_pdf_files), pdf_file, output_md_dir, api_key): index
            for index, pdf_file in enumerate(selected_pdf_files, start=1)
        }
        for future in as_completed(future_map):
            result = future.result()
            file_stats.append(result["stats"])

    file_stats.sort(key=lambda item: item["file_name"])
    for item in file_stats:
        status = item.get("status")
        if status == "success":
            success += 1
        elif status == "skipped":
            skipped += 1
        elif status == "failed":
            failed += 1

    total_elapsed = round(time.time() - started, 3)
    success_rows = [x for x in file_stats if x.get("status") == "success"]
    total_pages = sum(int(x.get("page_count") or 0) for x in success_rows)
    total_prompt_tokens = sum(int(x.get("prompt_tokens") or 0) for x in success_rows)
    total_completion_tokens = sum(int(x.get("completion_tokens") or 0) for x in success_rows)
    total_tokens = sum(int(x.get("total_tokens") or 0) for x in success_rows)
    avg_file_elapsed = round(sum(float(x.get("elapsed_sec") or 0.0) for x in success_rows) / len(success_rows), 3) if success_rows else 0.0
    avg_page_elapsed = round(sum(float(x.get("elapsed_sec") or 0.0) for x in success_rows) / total_pages, 3) if total_pages > 0 else 0.0
    avg_tokens_per_file = round(total_tokens / len(success_rows), 3) if success_rows else 0.0
    avg_tokens_per_page = round(total_tokens / total_pages, 3) if total_pages > 0 else 0.0

    write_json(OUTPUT_FILE_STATS, file_stats)
    write_json(
        OUTPUT_MANIFEST,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_dir": str(input_pdf_dir),
            "input_scan_mode": "recursive_rglob_pdf",
            "output_dir": str(output_md_dir),
            "ocr_model": OCR_MODEL,
            "page_chunk_size": PAGE_CHUNK_SIZE,
            "start_index": START_INDEX,
            "max_files": MAX_FILES,
            "force_overwrite": FORCE_OVERWRITE,
            "counts": {
                "input_pdf_total": len(all_pdf_files),
                "selected_pdf_total": len(selected_pdf_files),
                "success_total": success,
                "skipped_total": skipped,
                "failed_total": failed,
            },
            "timing": {
                "total_elapsed_sec": total_elapsed,
                "avg_file_elapsed_sec": avg_file_elapsed,
                "avg_page_elapsed_sec": avg_page_elapsed,
                "success_page_total": total_pages,
            },
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "avg_total_tokens_per_file": avg_tokens_per_file,
                "avg_total_tokens_per_page": avg_tokens_per_page,
            },
            "output_file_stats": str(OUTPUT_FILE_STATS),
        },
    )

    print(f"[统计] 成功={success}，跳过={skipped}，失败={failed}")
    print(f"[统计] 总耗时={total_elapsed}s | 平均每篇={avg_file_elapsed}s | 平均每页={avg_page_elapsed}s")
    print(
        f"[统计] OCR token -> prompt={total_prompt_tokens} | completion={total_completion_tokens} | "
        f"total={total_tokens} | 平均每篇={avg_tokens_per_file}"
    )
    print(f"[输出] Markdown 目录：{output_md_dir}")
    print(f"[输出] 单篇统计：{OUTPUT_FILE_STATS}")
    print(f"[输出] 运行摘要：{OUTPUT_MANIFEST}")
    print("[结束] OCR 流程完成")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
