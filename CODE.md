# Experimental code

| Study step | Archived implementation |
|---|---|
| OCR and source text preparation | [Phase 1](src/phase1_scripts) |
| Ontology element extraction | [phase2_01_concept_extraction.py](src/phase2_scripts/phase2_01_concept_extraction.py) |
| Candidate fusion and alignment | [phase2_02_internal_fusion.py](src/phase2_scripts/phase2_02_internal_fusion.py) and [phase2_03_external_alignment.py](src/phase2_scripts/phase2_03_external_alignment.py) |
| Class and property positioning | [Phase 2 modules 04a–04c](src/phase2_scripts) |
| Review queue, validation, change sets and ontology merge | [Phase 3](src/phase3_scripts) |
| Prediction matrix construction and evaluation summaries | [Archived evaluation scripts](src/evaluation) |

The [prompts](prompts) document the task instructions for screening, extraction, alignment and class and property positioning. The [ontology](ontology) directory provides the V0–V4 snapshots and the reduced schema and vocabulary resources used as inputs in rounds 1–4.

The source scripts retain their historical path and service configuration. Some stages use model or OCR services, and the original workflow used Python libraries including `requests`, `openai`, `numpy`, `sentence-transformers`, `pydantic` and `openpyxl`. Running the research pipeline requires the original input layout or equivalent path edits, input text at the same preprocessing stage, and the appropriate services and environments for each stage. The archived evaluation scripts describe the scoring procedure and expect the original validation directory layout and intermediate mapping files.

For an offline check of the distributed artifacts, run `python tools/check_release.py`. This is the supported entry point for key, label and file checks in this repository.
