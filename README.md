# Urban Traffic Ontology Experiment

This repository contains research artifacts for a four-round urban traffic ontology expansion experiment. It includes the source-paper register, gold labels and archived predictions for the extraction evaluation, and the historical code, prompts and ontology versions.

## Repository contents

| Location | Contents |
|---|---|
| [DATASET.md](DATASET.md) | Dataset scope, annotation rules and instructions for tracing a label to its source paper |
| [data](data) | Paper register, 30 target definitions, 3,600 gold labels and the corresponding archived predictions |
| [src](src) and [prompts](prompts) | Historical processing scripts and prompts used in the original experiment |
| [ontology](ontology) | Ontology versions V0–V4 and their experiment input resources |
| [results](results) | Frozen evaluation summaries by target type and by round |

## Inspect a paper and its labels

Open [paper_label_index.csv](data/paper_label_index.csv) to see each paper link next to its 30 target labels. For a terminal view, use Python 3.9 or later from the repository directory. The inspection script uses the standard library and runs offline.

```text
python tools/check_release.py
python tools/check_release.py --paper A013
```

The second command prints the article link, followed by its 30 target labels. The target files provide the definitions for those IDs.

## Historical scripts

The phase directories correspond to document preparation, extraction and alignment, and ontology updates. [CODE.md](CODE.md) maps the script files to the study workflow and describes the original runtime dependencies. The published artifacts allow inspection of the historical inputs available here and comparison of archived predictions with archived labels. Re-executing the model pipeline requires the original corpus inputs, adapted paths, and access to the relevant model services.

The paper register distinguishes links verified against bibliographic sources from records under review. See [DATASET.md](DATASET.md) for the status codes.
