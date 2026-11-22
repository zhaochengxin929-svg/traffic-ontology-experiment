# Dataset and annotation guide

## Scope

The experiment register contains 120 paper records across four rounds of 30 records each. Each record has 30 label decisions linked to the study’s predefined traffic ontology targets. The labels apply to the cleaned text retained for each paper in the experiment.

| Target type | Targets per paper | Labels | Positive | Negative |
|---|---:|---:|---:|---:|
| Class | 20 | 2,400 | 288 | 2,112 |
| Object property | 5 | 600 | 14 | 586 |
| Datatype property | 5 | 600 | 50 | 550 |
| Total | 30 | 3,600 | 352 | 3,248 |

## Trace a label to its source

[paper_label_index.csv](data/paper_label_index.csv) is a viewing table with one row per paper and 30 label columns named by target type and ID. It is generated from the paper register and the gold labels. After updating verified publication details, refresh it with `python tools/check_release.py --write-index`.

1. Find a record in [papers.csv](data/papers.csv) by `paper_id` and open its `article_url` if available.
2. Filter [gold_labels.csv](data/gold_labels.csv) by the same `paper_id`. Each row gives the target type, target ID and gold label.
3. Use `target_type` and `target_id` to look up the definition and exclusions in the relevant [target specification](data/targets).
4. To inspect the historical system prediction for that decision, join [archived_predictions.csv](data/archived_predictions.csv) on `paper_id`, `target_type` and `target_id`.

The [inspection command](README.md#inspect-a-paper-and-its-labels) can print all 30 decisions for one paper.

## Paper register

| Field | Meaning |
|---|---|
| `paper_id` | Stable ID from the original experiment register |
| `round_id` | Original experimental round |
| `registry_title`, `registry_year`, `registry_venue` | Metadata recorded in the historical register |
| `language`, `corpus_group` | Language and sampling group from the register |
| `article_title`, `article_years`, `article_venue` | Metadata confirmed from the cited publication source. Multiple publication years are separated by a vertical bar |
| `doi`, `article_url` | DOI where available and a verified article link where matched |
| `url_status`, `verification_source` | Verification status and the metadata source or matching basis |

`verified` indicates a confirmed match. `verified_metadata_differs` indicates a confirmed article with a difference in the register’s bibliographic fields. Abbreviations, subtitle variants and differences between online and issue dates can contribute to those differences. `pending` indicates that a publication link still requires corroboration. The experimental ID, original register fields and labels are preserved in all cases.

## Annotation decisions

- **1** means the cleaned corpus clearly supports the specified target semantics.
- **0** means the cleaned corpus lacks clear support under the original rubric. This includes ambiguous or insufficient evidence.
- The unit is a paper and a predefined target. Each paper contributes 20 class decisions, five object property decisions and five datatype property decisions. The target files supply definitions, common expressions, and exclusions.

The archived annotation materials include task instructions and an LLM-assisted annotation workflow. The files here preserve the resulting reference labels. Publisher PDFs, cleaned text, evidence excerpts and free-text notes are outside the scope of this distribution.

## Archived predictions and scores

`pred_label_loose` and `pred_label_strict` preserve the two original evaluation settings. The published [results](results) are archived aggregates for those settings. The repository inspection script checks key coverage and label consistency without recalculating the original performance results.
