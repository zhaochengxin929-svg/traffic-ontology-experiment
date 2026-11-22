import argparse
import csv
from collections import Counter
from itertools import product
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGET_COUNTS = {"class": 20, "object_property": 5, "datatype_property": 5}
POSITIVE_COUNTS = {"class": 288, "object_property": 14, "datatype_property": 50}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate():
    papers = load_csv(DATA / "papers.csv")
    papers_by_id = {row["paper_id"]: row for row in papers}
    require(len(papers) == len(papers_by_id) == 120, "Expected 120 unique paper IDs")
    require(sorted(Counter(row["round_id"] for row in papers).values()) == [30] * 4,
            "Expected four rounds with 30 papers each")

    url_counts = Counter()
    for row in papers:
        status = row["url_status"]
        url = urlparse(row["article_url"])
        if status in {"verified", "verified_metadata_differs"}:
            require(url.scheme == "https" and url.netloc and row["verification_source"],
                    f"Missing HTTPS link or verification source for {row['paper_id']}")
        else:
            require(status == "pending" and not row["article_url"],
                    f"Invalid link status for {row['paper_id']}")
        url_counts[status] += 1

    targets = set()
    for kind, count in TARGET_COUNTS.items():
        rows = load_csv(DATA / "targets" / f"{kind}_target_spec.csv")
        kind_targets = {(kind, row["target_id"]) for row in rows}
        require(len(rows) == len(kind_targets) == count, f"Unexpected target count for {kind}")
        targets.update(kind_targets)
    expected = {(paper_id, kind, target_id)
                for paper_id, (kind, target_id) in product(papers_by_id, targets)}

    matrices = (
        ("gold_labels.csv", ("gold_label",)),
        ("archived_predictions.csv", ("pred_label_loose", "pred_label_strict")),
    )
    for name, fields in matrices:
        rows = load_csv(DATA / name)
        keys = {(row["paper_id"], row["target_type"], row["target_id"]) for row in rows}
        require(len(rows) == len(keys) and keys == expected, f"Incomplete or duplicated keys in {name}")
        require(all(row[field] in {"0", "1"} for row in rows for field in fields),
                f"Non-binary value in {name}")

    gold = load_csv(DATA / "gold_labels.csv")
    positive = Counter(row["target_type"] for row in gold if row["gold_label"] == "1")
    require(positive == POSITIVE_COUNTS, "Gold label totals differ from the frozen archive")
    return papers_by_id, gold, url_counts


def main():
    parser = argparse.ArgumentParser(description="Check relationships in the distributed experiment data.")
    parser.add_argument("--paper", help="Print source metadata and labels for a paper ID")
    parser.add_argument("--write-index", action="store_true",
                        help="Refresh the one-row-per-paper viewing CSV from the source tables")
    args = parser.parse_args()
    papers, gold, url_counts = validate()
    print(f"PASS: {len(papers)} papers, {len(gold)} gold labels, matching prediction keys.")
    print(f"Article link statuses: {dict(sorted(url_counts.items()))}")
    if args.write_index:
        labels = {paper_id: {} for paper_id in papers}
        for row in gold:
            column = f"{row['target_type']}_{row['target_id']}"
            labels[row["paper_id"]][column] = row["gold_label"]
        fields = ["paper_id", "round_id", "title", "article_url", "url_status"]
        path = DATA / "paper_label_index.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + list(next(iter(labels.values()))))
            writer.writeheader()
            for paper_id, paper in papers.items():
                writer.writerow({
                    "paper_id": paper_id, "round_id": paper["round_id"],
                    "title": paper["article_title"] or paper["registry_title"],
                    "article_url": paper["article_url"], "url_status": paper["url_status"],
                    **labels[paper_id],
                })
        print(f"Wrote {path.name}")
    if args.paper:
        paper_id = args.paper.upper()
        if paper_id not in papers:
            parser.error(f"unknown paper ID {paper_id}")
        paper = papers[paper_id]
        title = paper["article_title"] or paper["registry_title"]
        print(f"\n{paper_id} | {title}")
        print(f"{paper['url_status']} | {paper['article_url']}")
        print("target_type,target_id,gold_label")
        for row in gold:
            if row["paper_id"] == paper_id:
                print(f"{row['target_type']},{row['target_id']},{row['gold_label']}")


if __name__ == "__main__":
    main()
