from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "system_prediction"
DEFAULT_PAPER_DIR = ANALYSIS_DIR.parent / "original_papers_reconstructed"

CLUSTER_ID_RE = re.compile(r"A2C_\d+")

TYPE_CONFIGS = {
    "class": {
        "mapping_file": ANALYSIS_DIR / "system_mapping" / "class_target_cluster_mapping.csv",
        "members_file": ANALYSIS_DIR / "outputs" / "class" / "class_global_cluster_members.csv",
        "result_file": "class_prediction_by_paper.csv",
        "evidence_file": "class_prediction_evidence_detail.csv",
        "label_fields": [("standard_name", ["标准名称", "standard_name"])],
    },
    "object_property": {
        "mapping_file": ANALYSIS_DIR / "system_mapping" / "object_property_target_cluster_mapping.csv",
        "members_file": ANALYSIS_DIR / "outputs" / "object_property" / "object_property_global_cluster_members.csv",
        "result_file": "object_property_prediction_by_paper.csv",
        "evidence_file": "object_property_prediction_evidence_detail.csv",
        "label_fields": [
            ("standard_object_A", ["标准对象A", "standard_object_a", "object_a"]),
            ("relation", ["关系", "relation"]),
            ("standard_object_B", ["标准对象B", "standard_object_b", "object_b"]),
        ],
    },
    "datatype_property": {
        "mapping_file": ANALYSIS_DIR / "system_mapping" / "datatype_property_target_cluster_mapping.csv",
        "members_file": ANALYSIS_DIR / "outputs" / "datatype_property" / "datatype_property_global_cluster_members.csv",
        "result_file": "datatype_property_prediction_by_paper.csv",
        "evidence_file": "datatype_property_prediction_evidence_detail.csv",
        "label_fields": [
            ("standard_object", ["标准对象", "standard_object", "object"]),
            ("property", ["属性", "property"]),
        ],
    },
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat_row: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    flat_row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    flat_row[key] = value
            writer.writerow(flat_row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def first_value(row: dict[str, str], candidates: list[str]) -> str:
    for key in candidates:
        if key in row:
            return str(row.get(key, "") or "").strip()
    return ""


def parse_cluster_ids(cell_text: str) -> list[str]:
    text = str(cell_text or "").strip()
    if not text:
        return []
    lowered = text.casefold()
    if lowered in {"无", "none", "n/a", "na", "null"}:
        return []
    ids = CLUSTER_ID_RE.findall(text)
    seen: set[str] = set()
    out: list[str] = []
    for cluster_id in ids:
        if cluster_id in seen:
            continue
        seen.add(cluster_id)
        out.append(cluster_id)
    return out


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_paper_ids(paper_dir: Path, member_rows: list[dict[str, str]]) -> list[str]:
    if paper_dir.exists():
        paper_ids = sorted(path.stem for path in paper_dir.glob("*.md"))
        if paper_ids:
            return paper_ids
    return sorted({str(row.get("paper_id", "")).strip() for row in member_rows if str(row.get("paper_id", "")).strip()})


def load_mapping_rows(entity_type: str, mapping_file: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(mapping_file)
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        target_id = first_value(row, ["target_id", "Target ID", "target"])
        include_cluster_ids = parse_cluster_ids(first_value(row, ["主纳入 cluster", "include", "include_clusters"]))
        cautious_cluster_ids = parse_cluster_ids(first_value(row, ["谨慎纳入 cluster", "cautious", "cautious_clusters"]))
        parsed = {
            "target_id": target_id,
            "include_cluster_ids": include_cluster_ids,
            "cautious_cluster_ids": cautious_cluster_ids,
            "loose_cluster_ids": dedupe_preserve_order(include_cluster_ids + cautious_cluster_ids),
            "strict_cluster_ids": include_cluster_ids,
        }
        for output_name, candidates in TYPE_CONFIGS[entity_type]["label_fields"]:
            parsed[output_name] = first_value(row, candidates)
        parsed_rows.append(parsed)
    return parsed_rows


def display_member_name(row: dict[str, str]) -> str:
    for key in ("name_zh", "name_en", "name_raw"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def build_prediction_outputs(
    entity_type: str,
    mapping_rows: list[dict[str, Any]],
    member_rows: list[dict[str, str]],
    paper_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    members_by_cluster_and_paper: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in member_rows:
        cluster_id = str(row.get("global_cluster_id", "")).strip()
        paper_id = str(row.get("paper_id", "")).strip()
        if not cluster_id or not paper_id:
            continue
        members_by_cluster_and_paper[cluster_id][paper_id].append(row)

    result_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    strict_positive_total = 0
    loose_positive_total = 0

    for mapping in mapping_rows:
        target_id = mapping["target_id"]
        include_cluster_ids = mapping["include_cluster_ids"]
        cautious_cluster_ids = mapping["cautious_cluster_ids"]
        loose_cluster_ids = mapping["loose_cluster_ids"]
        strict_cluster_ids = mapping["strict_cluster_ids"]

        for paper_id in paper_ids:
            strict_evidence: list[dict[str, str]] = []
            loose_evidence: list[dict[str, str]] = []
            strict_cluster_hits: list[str] = []
            loose_cluster_hits: list[str] = []

            for cluster_id in strict_cluster_ids:
                rows = members_by_cluster_and_paper.get(cluster_id, {}).get(paper_id, [])
                if rows:
                    strict_cluster_hits.append(cluster_id)
                    strict_evidence.extend(rows)

            for cluster_id in loose_cluster_ids:
                rows = members_by_cluster_and_paper.get(cluster_id, {}).get(paper_id, [])
                if rows:
                    loose_cluster_hits.append(cluster_id)
                    loose_evidence.extend(rows)

            pred_label_strict = 1 if strict_evidence else 0
            pred_label_loose = 1 if loose_evidence else 0
            strict_positive_total += pred_label_strict
            loose_positive_total += pred_label_loose

            matched_chunk_ids_strict = dedupe_preserve_order([row.get("chunk_id", "") for row in strict_evidence])
            matched_chunk_ids_loose = dedupe_preserve_order([row.get("chunk_id", "") for row in loose_evidence])
            matched_member_names_strict = dedupe_preserve_order([display_member_name(row) for row in strict_evidence])
            matched_member_names_loose = dedupe_preserve_order([display_member_name(row) for row in loose_evidence])

            result_row: dict[str, Any] = {
                "paper_id": paper_id,
                "target_id": target_id,
                "pred_label_loose": pred_label_loose,
                "pred_label_strict": pred_label_strict,
                "matched_cluster_ids_loose": dedupe_preserve_order(loose_cluster_hits),
                "matched_cluster_ids_strict": dedupe_preserve_order(strict_cluster_hits),
                "matched_chunk_ids_loose": matched_chunk_ids_loose,
                "matched_chunk_ids_strict": matched_chunk_ids_strict,
                "matched_member_names_loose": matched_member_names_loose,
                "matched_member_names_strict": matched_member_names_strict,
                "evidence_count_loose": len(loose_evidence),
                "evidence_count_strict": len(strict_evidence),
            }
            for output_name, _ in TYPE_CONFIGS[entity_type]["label_fields"]:
                result_row[output_name] = mapping.get(output_name, "")
            result_rows.append(result_row)

            for cluster_id in include_cluster_ids:
                for row in members_by_cluster_and_paper.get(cluster_id, {}).get(paper_id, []):
                    evidence_row: dict[str, Any] = {
                        "paper_id": paper_id,
                        "target_id": target_id,
                        "cluster_id": cluster_id,
                        "mapping_type": "include",
                        "chunk_id": row.get("chunk_id", ""),
                        "member_name_zh": row.get("name_zh", ""),
                        "member_name_en": row.get("name_en", ""),
                        "member_name_raw": row.get("name_raw", ""),
                        "evidence_text": row.get("evidence_text", ""),
                        "round_id": row.get("round_id", ""),
                        "section": row.get("section", ""),
                    }
                    for output_name, _ in TYPE_CONFIGS[entity_type]["label_fields"]:
                        evidence_row[output_name] = mapping.get(output_name, "")
                    evidence_rows.append(evidence_row)

            for cluster_id in cautious_cluster_ids:
                for row in members_by_cluster_and_paper.get(cluster_id, {}).get(paper_id, []):
                    evidence_row = {
                        "paper_id": paper_id,
                        "target_id": target_id,
                        "cluster_id": cluster_id,
                        "mapping_type": "cautious",
                        "chunk_id": row.get("chunk_id", ""),
                        "member_name_zh": row.get("name_zh", ""),
                        "member_name_en": row.get("name_en", ""),
                        "member_name_raw": row.get("name_raw", ""),
                        "evidence_text": row.get("evidence_text", ""),
                        "round_id": row.get("round_id", ""),
                        "section": row.get("section", ""),
                    }
                    for output_name, _ in TYPE_CONFIGS[entity_type]["label_fields"]:
                        evidence_row[output_name] = mapping.get(output_name, "")
                    evidence_rows.append(evidence_row)

    stats = {
        "entity_type": entity_type,
        "target_total": len(mapping_rows),
        "paper_total": len(paper_ids),
        "paper_target_total": len(result_rows),
        "strict_positive_total": strict_positive_total,
        "loose_positive_total": loose_positive_total,
        "evidence_row_total": len(evidence_rows),
    }
    return result_rows, evidence_rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper x target system prediction matrices.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--class-mapping-file", type=Path, default=TYPE_CONFIGS["class"]["mapping_file"])
    parser.add_argument("--class-members-file", type=Path, default=TYPE_CONFIGS["class"]["members_file"])
    parser.add_argument("--object-property-mapping-file", type=Path, default=TYPE_CONFIGS["object_property"]["mapping_file"])
    parser.add_argument("--object-property-members-file", type=Path, default=TYPE_CONFIGS["object_property"]["members_file"])
    parser.add_argument("--datatype-property-mapping-file", type=Path, default=TYPE_CONFIGS["datatype_property"]["mapping_file"])
    parser.add_argument("--datatype-property-members-file", type=Path, default=TYPE_CONFIGS["datatype_property"]["members_file"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime_configs = {
        "class": {
            "mapping_file": args.class_mapping_file,
            "members_file": args.class_members_file,
        },
        "object_property": {
            "mapping_file": args.object_property_mapping_file,
            "members_file": args.object_property_members_file,
        },
        "datatype_property": {
            "mapping_file": args.datatype_property_mapping_file,
            "members_file": args.datatype_property_members_file,
        },
    }

    manifest: dict[str, Any] = {
        "paper_dir": str(args.paper_dir),
        "output_dir": str(args.output_dir),
        "runs": [],
    }

    for entity_type in ("class", "object_property", "datatype_property"):
        mapping_file = runtime_configs[entity_type]["mapping_file"]
        members_file = runtime_configs[entity_type]["members_file"]

        mapping_rows = load_mapping_rows(entity_type, mapping_file)
        member_rows = read_csv_rows(members_file)
        paper_ids = load_paper_ids(args.paper_dir, member_rows)

        result_rows, evidence_rows, stats = build_prediction_outputs(
            entity_type=entity_type,
            mapping_rows=mapping_rows,
            member_rows=member_rows,
            paper_ids=paper_ids,
        )

        write_csv(args.output_dir / TYPE_CONFIGS[entity_type]["result_file"], result_rows)
        write_csv(args.output_dir / TYPE_CONFIGS[entity_type]["evidence_file"], evidence_rows)

        manifest["runs"].append(
            {
                "entity_type": entity_type,
                "mapping_file": str(mapping_file),
                "members_file": str(members_file),
                "result_file": str(args.output_dir / TYPE_CONFIGS[entity_type]["result_file"]),
                "evidence_file": str(args.output_dir / TYPE_CONFIGS[entity_type]["evidence_file"]),
                **stats,
            }
        )

        print(
            f"[{entity_type}] target_total={stats['target_total']} | "
            f"paper_total={stats['paper_total']} | paper_target_total={stats['paper_target_total']} | "
            f"loose_positive_total={stats['loose_positive_total']} | "
            f"strict_positive_total={stats['strict_positive_total']} | "
            f"evidence_row_total={stats['evidence_row_total']}"
        )

    write_json(args.output_dir / "system_prediction_manifest.json", manifest)


if __name__ == "__main__":
    main()
