from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from openpyxl import load_workbook


BASE_DIR = Path(__file__).resolve().parents[3]
VALIDATION_DIR = BASE_DIR / "validation" / "global_cluster_analysis"
TRUTH_DIR = VALIDATION_DIR / "Truth Annotation Results"
PREDICTION_DIR = VALIDATION_DIR / "system_prediction"
OUTPUT_DIR = VALIDATION_DIR / "evaluation_results"
REGISTRY_PATH = BASE_DIR / "dataset_main" / "paper_registry_formal.xlsx"


ENTITY_CONFIGS = [
    {
        "entity_type": "class",
        "gold_file": TRUTH_DIR / "class_annotation_v2.csv",
        "pred_file": PREDICTION_DIR / "class_prediction_by_paper_with_round.csv",
        "detail_file": OUTPUT_DIR / "class_eval_detail.csv",
        "overall_file": OUTPUT_DIR / "class_eval_summary.csv",
        "by_round_file": OUTPUT_DIR / "class_eval_by_round.csv",
        "by_target_file": OUTPUT_DIR / "class_eval_by_target.csv",
        "label_fields": ["standard_name"],
    },
    {
        "entity_type": "object_property",
        "gold_file": TRUTH_DIR / "object_property_annotation_filled.csv",
        "pred_file": PREDICTION_DIR / "object_property_prediction_by_paper_with_round.csv",
        "detail_file": OUTPUT_DIR / "object_property_eval_detail.csv",
        "overall_file": OUTPUT_DIR / "object_property_eval_summary.csv",
        "by_round_file": OUTPUT_DIR / "object_property_eval_by_round.csv",
        "by_target_file": OUTPUT_DIR / "object_property_eval_by_target.csv",
        "label_fields": ["object_A", "relation", "object_B"],
    },
    {
        "entity_type": "datatype_property",
        "gold_file": TRUTH_DIR / "datatype_property_annotation_filled.csv",
        "pred_file": PREDICTION_DIR / "datatype_property_prediction_by_paper_with_round.csv",
        "detail_file": OUTPUT_DIR / "datatype_property_eval_detail.csv",
        "overall_file": OUTPUT_DIR / "datatype_property_eval_summary.csv",
        "by_round_file": OUTPUT_DIR / "datatype_property_eval_by_round.csv",
        "by_target_file": OUTPUT_DIR / "datatype_property_eval_by_target.csv",
        "label_fields": ["object", "property"],
    },
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_round_mapping(path: Path) -> Dict[str, str]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    index = {name: idx for idx, name in enumerate(header)}
    mapping: Dict[str, str] = {}
    for row in rows:
        paper_id = str(row[index["paper_id"]]).strip()
        round_id = str(row[index["round_id"]]).strip()
        if paper_id in mapping and mapping[paper_id] != round_id:
            raise ValueError(f"Conflicting round_id for {paper_id}: {mapping[paper_id]} vs {round_id}")
        mapping[paper_id] = round_id
    return mapping


def to_int01(value: str) -> int:
    text = "" if value is None else str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError(f"Expected 0/1 label, got: {value!r}")
    return int(text)


def eval_type(gold: int, pred: int) -> str:
    if gold == 1 and pred == 1:
        return "TP"
    if gold == 0 and pred == 1:
        return "FP"
    if gold == 1 and pred == 0:
        return "FN"
    return "TN"


def safe_div(num: int, den: int) -> str:
    if den == 0:
        return "NA"
    return f"{num / den:.6f}"


def build_metric_row(rows: List[Dict[str, object]]) -> Dict[str, object]:
    sample_total = len(rows)
    gold_positive_total = sum(int(row["gold_label"]) for row in rows)
    gold_negative_total = sample_total - gold_positive_total

    result: Dict[str, object] = {
        "sample_total": sample_total,
        "gold_positive_total": gold_positive_total,
        "gold_negative_total": gold_negative_total,
    }

    for mode in ["loose", "strict"]:
        tp = sum(1 for row in rows if row[f"eval_type_{mode}"] == "TP")
        fp = sum(1 for row in rows if row[f"eval_type_{mode}"] == "FP")
        fn = sum(1 for row in rows if row[f"eval_type_{mode}"] == "FN")
        tn = sum(1 for row in rows if row[f"eval_type_{mode}"] == "TN")
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision == "NA" or recall == "NA":
            f1 = "NA"
        else:
            p = float(precision)
            r = float(recall)
            f1 = "NA" if (p + r) == 0 else f"{(2 * p * r) / (p + r):.6f}"
        result.update(
            {
                f"pred_positive_total_{mode}": tp + fp,
                f"tp_{mode}": tp,
                f"fp_{mode}": fp,
                f"fn_{mode}": fn,
                f"tn_{mode}": tn,
                f"precision_{mode}": precision,
                f"recall_{mode}": recall,
                f"f1_{mode}": f1,
            }
        )
    return result


def build_micro_metric_row(metric_rows: List[Dict[str, object]]) -> Dict[str, object]:
    sample_total = sum(int(row["sample_total"]) for row in metric_rows)
    gold_positive_total = sum(int(row["gold_positive_total"]) for row in metric_rows)
    gold_negative_total = sum(int(row["gold_negative_total"]) for row in metric_rows)

    result: Dict[str, object] = {
        "sample_total": sample_total,
        "gold_positive_total": gold_positive_total,
        "gold_negative_total": gold_negative_total,
    }

    for mode in ["loose", "strict"]:
        tp = sum(int(row[f"tp_{mode}"]) for row in metric_rows)
        fp = sum(int(row[f"fp_{mode}"]) for row in metric_rows)
        fn = sum(int(row[f"fn_{mode}"]) for row in metric_rows)
        tn = sum(int(row[f"tn_{mode}"]) for row in metric_rows)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision == "NA" or recall == "NA":
            f1 = "NA"
        else:
            p = float(precision)
            r = float(recall)
            f1 = "NA" if (p + r) == 0 else f"{(2 * p * r) / (p + r):.6f}"
        result.update(
            {
                f"pred_positive_total_{mode}": tp + fp,
                f"tp_{mode}": tp,
                f"fp_{mode}": fp,
                f"fn_{mode}": fn,
                f"tn_{mode}": tn,
                f"precision_{mode}": precision,
                f"recall_{mode}": recall,
                f"f1_{mode}": f1,
            }
        )
    return result


def sort_key(value: str) -> Tuple[str, int]:
    prefix = "".join(ch for ch in value if not ch.isdigit())
    digits = "".join(ch for ch in value if ch.isdigit())
    return prefix, int(digits or 0)


def main() -> None:
    paper_round_mapping = load_round_mapping(REGISTRY_PATH)
    combined_overall: List[Dict[str, object]] = []
    combined_by_round: List[Dict[str, object]] = []

    for config in ENTITY_CONFIGS:
        gold_rows = read_csv(config["gold_file"])
        pred_rows = read_csv(config["pred_file"])

        gold_by_key = {(row["paper_id"], row["target_id"]): row for row in gold_rows}
        pred_by_key = {(row["paper_id"], row["target_id"]): row for row in pred_rows}
        if set(gold_by_key) != set(pred_by_key):
            missing_in_pred = sorted(set(gold_by_key) - set(pred_by_key))
            missing_in_gold = sorted(set(pred_by_key) - set(gold_by_key))
            raise ValueError(
                f"Key mismatch for {config['entity_type']}: missing_in_pred={missing_in_pred[:5]}, "
                f"missing_in_gold={missing_in_gold[:5]}"
            )

        detail_rows: List[Dict[str, object]] = []
        for paper_id, target_id in sorted(gold_by_key, key=lambda item: (sort_key(item[0]), sort_key(item[1]))):
            gold_row = gold_by_key[(paper_id, target_id)]
            pred_row = pred_by_key[(paper_id, target_id)]
            gold_label = to_int01(gold_row["gold_label"])
            pred_loose = to_int01(pred_row["pred_label_loose"])
            pred_strict = to_int01(pred_row["pred_label_strict"])
            round_id = (pred_row.get("round_id") or "").strip() or paper_round_mapping.get(paper_id, "")
            if not round_id:
                raise ValueError(f"Missing round_id for {paper_id}")

            detail_row: Dict[str, object] = {
                "paper_id": paper_id,
                "target_id": target_id,
                "round_id": round_id,
                "gold_label": gold_label,
                "pred_label_loose": pred_loose,
                "pred_label_strict": pred_strict,
                "eval_type_loose": eval_type(gold_label, pred_loose),
                "eval_type_strict": eval_type(gold_label, pred_strict),
                "gold_evidence_text": gold_row.get("evidence_text", ""),
                "gold_note": gold_row.get("note", ""),
            }
            for field in config["label_fields"]:
                detail_row[field] = gold_row.get(field, "")
            for field in [
                "matched_cluster_ids_loose",
                "matched_cluster_ids_strict",
                "matched_chunk_ids_loose",
                "matched_chunk_ids_strict",
                "matched_member_names_loose",
                "matched_member_names_strict",
                "evidence_count_loose",
                "evidence_count_strict",
            ]:
                detail_row[field] = pred_row.get(field, "")
            detail_rows.append(detail_row)

        detail_fieldnames = [
            "paper_id",
            "target_id",
            "round_id",
            *config["label_fields"],
            "gold_label",
            "pred_label_loose",
            "pred_label_strict",
            "eval_type_loose",
            "eval_type_strict",
            "gold_evidence_text",
            "gold_note",
            "matched_cluster_ids_loose",
            "matched_cluster_ids_strict",
            "matched_chunk_ids_loose",
            "matched_chunk_ids_strict",
            "matched_member_names_loose",
            "matched_member_names_strict",
            "evidence_count_loose",
            "evidence_count_strict",
        ]
        write_csv(config["detail_file"], detail_rows, detail_fieldnames)

        overall_row = {"entity_type": config["entity_type"], **build_metric_row(detail_rows)}
        overall_fields = list(overall_row.keys())
        write_csv(config["overall_file"], [overall_row], overall_fields)
        combined_overall.append(overall_row)

        by_round_rows: List[Dict[str, object]] = []
        for round_id in sorted({row["round_id"] for row in detail_rows}, key=sort_key):
            round_rows = [row for row in detail_rows if row["round_id"] == round_id]
            row = {"entity_type": config["entity_type"], "round_id": round_id, **build_metric_row(round_rows)}
            by_round_rows.append(row)
            combined_by_round.append(row)
        by_round_fields = list(by_round_rows[0].keys())
        write_csv(config["by_round_file"], by_round_rows, by_round_fields)

        by_target_rows: List[Dict[str, object]] = []
        for target_id in sorted({row["target_id"] for row in detail_rows}, key=sort_key):
            target_rows = [row for row in detail_rows if row["target_id"] == target_id]
            row = {
                "entity_type": config["entity_type"],
                "target_id": target_id,
                **{field: target_rows[0].get(field, "") for field in config["label_fields"]},
                **build_metric_row(target_rows),
            }
            by_target_rows.append(row)
        by_target_fields = list(by_target_rows[0].keys())
        write_csv(config["by_target_file"], by_target_rows, by_target_fields)

    overall_all_types = {"entity_type": "all_types_micro", **build_micro_metric_row(combined_overall)}
    overall_rows = combined_overall + [overall_all_types]
    write_csv(OUTPUT_DIR / "overall_eval_summary_all.csv", overall_rows, list(overall_rows[0].keys()))

    by_round_all: List[Dict[str, object]] = list(combined_by_round)
    for round_id in sorted({row["round_id"] for row in combined_by_round}, key=sort_key):
        round_rows = [row for row in combined_by_round if row["round_id"] == round_id]
        by_round_all.append(
            {
                "entity_type": "all_types_micro",
                "round_id": round_id,
                **build_micro_metric_row(round_rows),
            }
        )
    write_csv(OUTPUT_DIR / "eval_by_round_all.csv", by_round_all, list(by_round_all[0].keys()))


if __name__ == "__main__":
    main()
