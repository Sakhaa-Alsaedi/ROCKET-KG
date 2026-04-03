![Image](https://github.com/user-attachments/assets/17fe1a1d-ea02-4fed-bcf7-5d2d099b0553)
# ROCKET — Detailed Technical Reference

This document captures the full architecture, data pipeline, and knowledge graph details for the ROCKET / GraphCare system.

---

## 1. ROCKET Repository Overview

**ROCKET** = **R**eproducible, **O**rganised **C**linical **K**nowledge-graph **E**valuation and **T**esting

A framework for clinical outcome prediction using knowledge graphs and graph neural networks, evaluated on MIMIC-III and MIMIC-IV electronic health record data.

### Project Purpose

Clinical outcome prediction tasks:
- **Mortality** — binary in-hospital mortality prediction
- **Readmission** — binary 15-day readmission prediction
- **Drug Recommendation** — multi-label drug recommendation
- **Length of Stay** — 10-class length-of-stay prediction

All tasks use MIMIC-III / MIMIC-IV EHR data enriched with LLM-generated knowledge graphs.

---

## 2. Directory Structure

```
ROCKET/
├── src/
│   ├── data/                  # MIMIC data pipeline (NEW)
│   │   ├── config.py          #   Centralized path configuration
│   │   ├── task_fn.py         #   PyHealth task functions (MIMIC-III + IV)
│   │   ├── data_prepare.py    #   Full pipeline: load → cluster → graph → annotate
│   │   └── graph_dataset.py   #   GraphDataset + get_subgraph + get_dataloader
│   ├── models/                # CADI, CAT (causal GNNs) + baselines
│   │   ├── cadi.py            #   Causal Attention Dual Inference
│   │   ├── cat.py             #   Causal Attention Transformer
│   │   └── baselines/
│   │       ├── bat.py         #   Bi-Attention GNN
│   │       ├── gnns.py        #   GAT, GIN
│   │       └── ehr_baselines.py  # RNN, Transformer, RETAIN, MLP
│   ├── kg_construction/       # GPT triple generation + embeddings + clustering
│   │   ├── build_kg.py
│   │   ├── build_embeddings.py
│   │   ├── run_clustering.py
│   │   └── attention_weights.py
│   ├── rocket_score/          # 5-dimensional KG quality metric (S1–S5)
│   │   └── score.py
│   ├── causal_discovery/      # Ensemble: NOTEARS, GOLEM, PC, LiNGAM
│   │   └── ensemble.py
│   ├── evaluation/            # Task-specific metrics
│   │   ├── metrics.py
│   │   └── evaluate.py
│   └── agent/                 # ReAct clinical QA agent
│       └── rocket_agent.py
├── scripts/                   # Numbered end-to-end pipeline scripts
│   ├── 01_generate_kg_triples.py   # Step 1: GPT → per-code triple files
│   ├── 02_build_embeddings.py      # Step 2: ada-002 embeddings per type
│   ├── 03_merge_embeddings.py      # Step 3: merge into task-group sets
│   └── 04_run_data_prepare.py      # Step 4: PyHealth load → graph → annotate
├── data/
│   ├── resources/             # Code-mapping CSVs: CCSCM.csv, CCSPROC.csv, ATC.csv
│   ├── mimic3_csv/            # Decompressed MIMIC-III tables (gitignored)
│   ├── mimic4_csv/            # Decompressed MIMIC-IV tables (gitignored)
│   ├── graphs/                # GPT-generated KG triple files (per code)
│   │   ├── condition/CCSCM/{code}.txt
│   │   ├── procedure/CCSPROC/{code}.txt
│   │   ├── drug/ATC3/{code}.txt
│   │   ├── cond_proc/CCSCM_CCSPROC/      # merged embeddings (drugrec/lenofstay)
│   │   └── cond_proc_drug/…              # merged embeddings (mortality/readmission)
│   ├── exp_data/              # Processed datasets + cluster graphs
│   │   ├── ccscm_ccsproc/     # drugrec, lenofstay
│   │   └── ccscm_ccsproc_atc3/ # mortality, readmission
│   ├── mimic_iii/             # MIMIC-III download instructions
│   └── mimic_iv/              # MIMIC-IV download instructions
├── tests/                     # pytest test suite (~133 tests)
│   ├── test_models.py
│   ├── test_rocket_score.py
│   ├── test_causal_discovery.py
│   ├── test_evaluation.py
│   └── test_data_pipeline.py  # 46 unit + 10 integration tests
├── checkpoints/               # Pre-trained weights + benchmark results
│   ├── gpt35_kg/
│   └── gpt4_kg/
├── configs/default.yaml       # Training config template
├── pytest.ini
├── requirements.txt
├── README.md
└── more_details.md            # This file
```

---

## 3. Key Components

| Component | File | Description |
|-----------|------|-------------|
| **CADI** | `src/models/cadi.py` | Causal Attention Dual Inference — dual-path factual + counterfactual GNN |
| **CAT** | `src/models/cat.py` | Causal Attention Transformer — gradient-feedback causal weighting |
| **BAT** | `src/models/baselines/bat.py` | Bi-Attention GNN — node-level α + visit-decay β attention |
| **GAT / GIN** | `src/models/baselines/gnns.py` | Graph Attention Network, Graph Isomorphism Network |
| **EHR Baselines** | `src/models/baselines/ehr_baselines.py` | RNN (GRU), Transformer, RETAIN, MLP |
| **ROCKET Score** | `src/rocket_score/score.py` | S1 (structural) + S2 (semantic) + S3 (task relevance) + S4 (causal confidence) + S5 (clinical coverage) |
| **KG Builder** | `src/kg_construction/build_kg.py` | OpenAI GPT → triples → ada-002 embeddings → agglomerative clustering |
| **Causal Discovery** | `src/causal_discovery/ensemble.py` | Ensemble of NOTEARS, GOLEM, PC, LiNGAM with majority voting |
| **ReAct Agent** | `src/agent/rocket_agent.py` | KG search + patient history + drug interactions + similarity tools |
| **Data Pipeline** | `src/data/` | Full MIMIC → per-patient graph pipeline |

---

## 4. Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 1.12.0 | Deep learning framework |
| PyTorch Geometric | 2.3.0 | Graph neural network operations |
| PyHealth | 1.1.2 | EHR dataset loading and code mapping |
| OpenAI API | 0.27.4 | KG triple generation + ada-002 embeddings |
| scikit-learn | — | Agglomerative clustering |
| NetworkX | — | Global cluster graph construction |
| causal-learn | — | PC algorithm causal discovery |
| lingam | — | DirectLiNGAM causal discovery |
| pytest | — | Testing framework |

---

## 5. Benchmark Results (MIMIC-III, GPT-3.5 KG)

| Task | Best AUROC | Best Model |
|------|-----------|------------|
| Mortality | 0.6370 | CADI |
| Readmission | 0.6893 | GAT |
| Drug Recommendation | 0.9485 | GIN |
| Length of Stay | 0.7836 | GIN |

Pre-trained weights are stored in `checkpoints/gpt35_kg/` and `checkpoints/gpt4_kg/`.

---

## 6. MIMIC Data Paths (on IBEX cluster)

| Purpose | Path |
|---------|------|
| MIMIC-III source (compressed) | `/ibex/user/alsaedsb/GhraphCARE/Data/physionet.org/files/mimiciii/1.4` |
| MIMIC-IV source (compressed) | `/ibex/user/alsaedsb/DeepCARES_DT/MIMIC-IV-Data-Pipeline/mimiciv/3.1/hosp` |
| MIMIC-III working CSVs | `/ibex/user/alsaedsb/GhraphCARE/GraphCare/mimic3_csv/` |
| MIMIC-IV working CSVs | `/ibex/user/alsaedsb/GhraphCARE/GraphCare/mimic4_csv/` |

### Which MIMIC CSV Files Are Actually Used

Out of 6 files per dataset, the pipeline directly reads only **3**:

**MIMIC-III (`mimic3_csv/`) — 798 MB total**

| File | Size | Used? | Why |
|------|------|-------|-----|
| `DIAGNOSES_ICD.csv` | 19 MB | **YES** | ICD9CM diagnosis codes → mapped to CCSCM |
| `PROCEDURES_ICD.csv` | 6.5 MB | **YES** | ICD9PROC codes → mapped to CCSPROC |
| `PRESCRIPTIONS.csv` | 735 MB | **YES** | NDC drug codes → mapped to ATC-3 |
| `ADMISSIONS.csv` | 12 MB | NO | PyHealth uses internally for timestamps |
| `PATIENTS.csv` | 2.6 MB | NO | PyHealth uses internally for demographics |
| `TRANSFERS.csv` | 24 MB | NO | Not used at all |

**MIMIC-IV (`mimic4_csv/`) — 3.8 GB total**

| File | Size | Used? | Why |
|------|------|-------|-----|
| `diagnoses_icd.csv` | 174 MB | **YES** | ICD9/10CM → CCSCM |
| `procedures_icd.csv` | 33 MB | **YES** | ICD9/10PROC → CCSPROC |
| `prescriptions.csv` | 3.3 GB | **YES** | NDC → ATC-3 |
| `admissions.csv` | 90 MB | NO | PyHealth uses internally |
| `patients.csv` | 12 MB | NO | PyHealth uses internally |
| `transfers.csv` | 196 MB | NO | Not used at all |

The exact loading code (`src/data/data_prepare.py`):

```python
# MIMIC-III
ds = MIMIC3Dataset(
    root=MIMIC3_CSV_DIR,
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={
        "NDC":      ("ATC", {"target_kwargs": {"level": 3}}),
        "ICD9CM":   "CCSCM",
        "ICD9PROC": "CCSPROC",
    }
)

# MIMIC-IV
ds = MIMIC4Dataset(
    root=MIMIC4_CSV_DIR,
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    code_mapping={
        "NDC":       ("ATC", {"target_kwargs": {"level": 3}}),
        "ICD9CM":    "CCSCM",   "ICD10CM":   "CCSCM",
        "ICD9PROC":  "CCSPROC", "ICD10PROC": "CCSPROC",
    }
)
```

---

## 7. The ROCKET KG (with Gene Information)

### What is the ROCKET KG?

The ROCKET KG is an **evidence-based, causally-scored knowledge graph** that enriches LLM-generated concept KGs with biological relationships (gene–disease, protein–protein, biomarker–disease, etc.).

**Main file:**
```
/ibex/user/alsaedsb/ROCKET/Data/Intgrated_KGs/output/scoring/GraphCRAE_ROCKET.csv
```
- Size: **811 MB**
- Rows: **5,962,902**
- Format: `relation, x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source, causal_score`

### Gene-related relation types in the ROCKET KG

| Relation Type | Records | Source |
|---------------|---------|--------|
| `protein_protein` | 642,150 | NCBI / UniProt |
| `disease_protein` | 883,370 | NCBI |
| `gene_disease` | 10,452 | ClinVar |
| `biomarker_disease` | 10,943 | custom biomarker DB |
| `risk_disease` | 7,164 | GWAS catalog |
| `pathway_protein` | 85,292 | Reactome |
| `bioprocess_protein` | 289,610 | Gene Ontology |
| `cellcomp_protein` | 166,804 | Gene Ontology |
| `molfunc_protein` | 139,060 | Gene Ontology |

### Raw source files that built the ROCKET KG

Location: `/ibex/user/alsaedsb/ROCKET/Data/Intgrated_KGs/data/raw/`

| File | Size | Records | Content |
|------|------|---------|---------|
| `Risk_gene_disease_associations_NCBI.tsv` | 87 MB | 722,549 | NCBI gene-disease risk associations with scores |
| `padded_kg.csv` | 960 MB | 8,100,499 | Protein-protein interactions (multi-source) |
| `mr_causal_database_high_confidence.csv` | 3.9 MB | 8,855 | Mendelian Randomization causal links |
| `clngene.csv` | 1.5 MB | 12,497 | ClinVar clinical gene-disease associations |
| `biomarkers_updated.csv` | 1.2 MB | 8,182 | Biomarker-disease associations |
| `Dis_Cause_Dis.csv` | 425 KB | 3,093 | Disease-disease causal relationships |

Additional root-level files at `/ibex/user/alsaedsb/ROCKET/`:

| File | Size | Content |
|------|------|---------|
| `rocket_kg_dump.csv` | 762 MB | Full ROCKET KG dump |
| `merged_GWAS_SNP_df.csv` | 668 MB | 4.6M SNP → gene → disease mappings |
| `Final_snps_gene_df.csv` | 3.2 MB | 93K processed SNP-to-gene mappings |

### ROCKET KG key statistics

- **Total integrated records:** 8,851,729
- **Unique entity types:** Disease, Drug, Biomarker, Risk Gene, Gene/Protein
- **Records with causal scores:** 731,403 (8.3%)
- **High confidence relations (score ≥ 0.7):** 8,759 (in `archive/high_confidence_relations.tsv`)
- **Data sources:** 15 (NCBI, DrugBank, UniProt, BEFREE, HPO, CTD, ClinVar, MR, GWASCAT, Reactome, GO, …)

### How the ROCKET KG is integrated into GraphCare

Integration script: `scripts/rocket_kg_integration.py` (in the parent GraphCare repo)

Activated when `KG_VERSION` contains `"rocket"` (e.g. `rocket_gpt35`, `rocket_gpt4`).

**4-phase integration:**

1. **Map** — CCS condition codes → ROCKET disease IDs (fuzzy name matching, score ≥ 65); ATC3 drug codes → ROCKET drug IDs
2. **Extract** — For each mapped concept, extract 1-hop neighborhood from `GraphCRAE_ROCKET.csv`, top-30 highest-scoring edges, filtered to: Disease, Drug, Biomarker, Risk Gene, Gene/Protein
3. **Merge** — Append ROCKET triples to existing GPT `.txt` files → written to `graphs_rocket_{version}/`
4. **Re-embed** — Re-run OpenAI ada-002 embeddings over the enriched triple set

---

## 8. Full Pipeline: Raw MIMIC → Per-Patient Graph

### Summary Diagram

```
MIMIC CSVs (ICD/NDC codes)
        ↓
  PyHealth: code mapping (ICD→CCSCM, NDC→ATC3)
        ↓
  GPT-3.5/4 per code → triples (.txt files)
  [+ ROCKET KG enrichment if rocket version]
        ↓
  OpenAI ada-002 embeddings → [N, 1536]
        ↓
  Agglomerative clustering (τ=0.15, cosine) → cluster IDs
        ↓
  Global cluster graph (NetworkX → PyG)
        ↓
  Per patient: node_set from their codes
        ↓
  At training: 2-hop subgraph extraction
        ↓
  PyG Data object → fed to CADI / CAT / BAT / GAT / GIN model
```

### Step 1 — Load Raw MIMIC

**File:** `src/data/data_prepare.py`, function `load_dataset()`

- Uses PyHealth `MIMIC3Dataset` / `MIMIC4Dataset` on working CSV directories
- Loads 3 tables: `DIAGNOSES_ICD`, `PROCEDURES_ICD`, `PRESCRIPTIONS`
- Applies code mappings:
  - `ICD9CM` / `ICD10CM` → **CCSCM** (CCS condition codes, ~283 categories)
  - `ICD9PROC` / `ICD10PROC` → **CCSPROC** (CCS procedure codes, ~231 categories)
  - `NDC` → **ATC-3** (4-character drug codes, e.g. `A10A`, `B01A`)
- Applies task function to produce structured samples
- Output cached to: `data/exp_data[_variant]/ccscm_ccsproc[_atc3]/sample_dataset_{dataset}_{task}_th015.pkl`

**Task functions** (`src/data/task_fn.py`):

| Task | Function (MIMIC-III) | Function (MIMIC-IV) | Label |
|------|---------------------|---------------------|-------|
| drugrec | `drug_recommendation_fn` | `drug_recommendation_mimic4_fn` | Multi-hot drug vector |
| mortality | `mortality_prediction_mimic3_fn` | `mortality_prediction_mimic4_fn` | 0/1 binary |
| readmission | `readmission_prediction_mimic3_fn` | `readmission_prediction_mimic4_fn` | 0/1 binary (15-day window) |
| lenofstay | `length_of_stay_prediction_mimic3_fn` | `length_of_stay_prediction_mimic4_fn` | 0–9 class |

Length-of-stay categories:

| Class | Duration |
|-------|----------|
| 0 | < 1 day |
| 1–7 | Days 1–7 (one class per day) |
| 8 | 7–14 days |
| 9 | > 14 days |

---

### Step 2 — Generate Concept KGs per Medical Code

**Script:** `scripts/01_generate_kg_triples.py`

For every unique CCS/ATC code across all patients:
- Calls **GPT-3.5-turbo** (or GPT-4 if `KG_VERSION=gpt4`) with a few-shot prompt
- Generates ~100 RDF-like triples per code: `[ENTITY1, RELATIONSHIP, ENTITY2]`
- Saves as tab-separated lines:

```
head\trelation\ttail
```

Output directories (per `KG_VERSION`):
- `data/graphs/condition/CCSCM/{code}.txt`
- `data/graphs/procedure/CCSPROC/{code}.txt`
- `data/graphs/drug/ATC3/{code}.txt`

Resumes automatically — codes with existing files ≥ 100 lines are skipped.

---

### Step 3 — Optionally Enrich with ROCKET KG

**Script:** `scripts/rocket_kg_integration.py` (GraphCare parent repo)

Only activated when `KG_VERSION` contains `"rocket"` (e.g. `rocket_gpt35`).

1. **Map** — CCS codes → ROCKET disease IDs (fuzzy name matching, score ≥ 65); ATC3 → ROCKET drug IDs
2. **Extract** — 1-hop neighborhood from `GraphCRAE_ROCKET.csv`, top-30 highest-scoring edges per concept
3. **Merge** — Append ROCKET triples to GPT `.txt` files → written to `graphs_rocket_{version}/`
4. Relation types: `gene_disease`, `drug_disease`, `disease_disease`, etc. → natural language text with causal confidence score

---

### Step 4 — Build OpenAI Embeddings

**Script:** `scripts/02_build_embeddings.py`

For all entities and relations across all concept KG files:
- Embeds using **OpenAI `text-embedding-ada-002`** → **1536-dim** vectors
- 20 concurrent threads for throughput
- Creates per-type mappings:
  - `ent2id.json`, `id2ent.json`
  - `rel2id.json`, `id2rel.json`
  - `entity_embedding.pkl` → `[N_entities, 1536]`
  - `relation_embedding.pkl` → `[N_relations, 1536]`
- Written into each type directory (`condition/CCSCM/`, `procedure/CCSPROC/`, `drug/ATC3/`)

---

### Step 5 — Merge Embeddings by Task Group

**Script:** `scripts/03_merge_embeddings.py`

Two merged sets are created with offset-indexed global IDs:

| Merged set | Task group | Directory |
|-----------|------------|-----------|
| condition + procedure | `drugrec`, `lenofstay` | `data/graphs/cond_proc/CCSCM_CCSPROC/` |
| condition + procedure + drug | `mortality`, `readmission` | `data/graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/` |

---

### Step 6 — Cluster Entities

**Function:** `clustering()` in `src/data/data_prepare.py`

- **Algorithm:** Agglomerative Clustering with cosine affinity, distance threshold **τ = 0.15**
- Groups semantically similar entities into clusters
- **Cluster centroid** = mean embedding of all member entities
- Outputs (in `data/exp_data/ccscm_ccsproc[_atc3]/`):
  - `clusters_th015.json` — entity_id → cluster metadata + centroid embedding
  - `clusters_inv_th015.json` — entity_id → cluster_id (inverse lookup)
  - `clusters_rel_th015.json` — relation_id → cluster metadata
  - `clusters_inv_rel_th015.json` — relation_id → cluster_id

---

### Step 7 — Build Global Cluster Graph

**Function:** `process_graph()` in `src/data/data_prepare.py`

- Iterates over all patients, reads their concept KG `.txt` files
- Maps each triple: `(head_text, relation_text, tail_text)` → `(head_ent_id, rel_id, tail_ent_id)` → `(head_cluster_id, rel_cluster_id, tail_cluster_id)`
- Builds a **NetworkX undirected graph G** where:
  - **Nodes** = cluster IDs, attribute `x` = centroid embedding (1536-dim), attribute `y` = cluster ID
  - **Edges** = between clusters, attribute `relation` = relation cluster ID
- Converts to **PyTorch Geometric** via `from_networkx(G)`
- Saved to: `data/exp_data/.../graph_{dataset}_{task}_th015.pkl`

---

### Step 8 — Annotate Patients with Node Sets

**Function:** `process_sample_dataset()` in `src/data/data_prepare.py`

For each patient visit, resolves their medical codes to cluster node IDs. Annotates each patient record with:
- `node_set` — list of all unique cluster nodes touched across all visits
- `visit_padded_node` — tensor `[max_visits × num_cluster_nodes]`, multi-hot node presence per visit (most recent visit first, zero-padded)

---

### Step 9 — Extract Per-Patient Subgraph at Training Time

**Function:** `get_subgraph()` in `src/data/graph_dataset.py`, called by `GraphDataset.__getitem__()`

- Takes patient's `node_set` as **seed nodes**
- Extracts **2-hop neighborhood** from global graph `G` via `k_hop_subgraph()`
- Returns a **PyTorch Geometric `Data` object**:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `x` | `[num_nodes, 1536]` | Cluster centroid embeddings |
| `edge_index` | `[2, num_edges]` | COO edge connectivity |
| `relation` | `[num_edges]` | Relation cluster IDs per edge |
| `y` | `[num_nodes]` | Cluster node IDs |
| `label` | varies | Task label: drug multi-hot / binary scalar / 10-class one-hot |
| `visit_padded_node` | `[max_visits, num_nodes]` | Multi-hot node presence per visit |
| `ehr_nodes` | `[num_nodes]` | Binary mask of patient's direct EHR nodes |
| `patient_id` | scalar | MIMIC patient identifier |

**Typical graph statistics per patient:**
- Nodes: 50–200 (cluster nodes in 2-hop neighborhood)
- Edges: 100–500
- Node embedding dim: 1536
- Max visits: varies by dataset (up to 30+)

---

## 9. Dataset and DataLoader Setup

```python
from src.data import GraphDataset, get_dataloader

# GraphDataset wraps global graph + patient list
train_set = GraphDataset(G_tg, train_patients, task="mortality")

# get_dataloader builds train/val/test loaders
train_loader, val_loader, test_loader = get_dataloader(
    G_tg, train_patients, val_patients, test_patients,
    task="mortality",
    batch_size=16,
)
```

Patient splitting is done by patient ID (not visit) to prevent data leakage, with default ratio 70/10/20 (train/val/test).

---

## 10. KG_VERSION Environment Variable

Controls which graph directory is used throughout the pipeline:

| `KG_VERSION` | Graph directory | Embeddings directory | Description |
|-------------|----------------|---------------------|-------------|
| `gpt35` (default) | `data/graphs/` | `data/exp_data/` | GPT-3.5-turbo triples only |
| `gpt4` | `data/graphs_gpt4/` | `data/exp_data_gpt4/` | GPT-4 triples only |
| `rocket_gpt35` | `data/graphs_rocket_gpt35/` | `data/exp_data_rocket_gpt35/` | GPT-3.5 + ROCKET KG enrichment |
| `rocket_gpt4` | `data/graphs_rocket_gpt4/` | `data/exp_data_rocket_gpt4/` | GPT-4 + ROCKET KG enrichment |

Set before running any script:
```bash
export KG_VERSION=rocket_gpt4
```

---

## 11. Running the Tests

```bash
# All tests (133 pass, 10 skipped = integration tests waiting for MIMIC-IV demo)
pytest tests/ -v

# Unit tests only (no data needed — always run)
pytest tests/ -v -m "not integration"

# Integration tests (requires MIMIC-IV demo CSVs)
# Demo download: https://physionet.org/content/mimic-iv-demo/2.2/
MIMIC4_CSV_DIR=/path/to/demo/csv pytest tests/test_data_pipeline.py -v -m integration
```

**Test breakdown:**

| File | Tests | Data Required |
|------|-------|---------------|
| `test_models.py` | ~20 | None (synthetic) |
| `test_rocket_score.py` | ~19 | None (numpy only) |
| `test_causal_discovery.py` | ~9 | None (synthetic) |
| `test_evaluation.py` | ~16 | None (sklearn only) |
| `test_data_pipeline.py` | 46 unit + 10 integration | Unit: none; Integration: MIMIC-IV demo CSVs |

---

## 12. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests (no data needed)
pytest tests/ -v -m "not integration"

# 3. Set paths
export MIMIC4_CSV_DIR=/path/to/mimic4_csv
export OPENAI_API_KEY=sk-...
export KG_VERSION=gpt35

# 4. Generate KG triples (Step 1)
python scripts/01_generate_kg_triples.py

# 5. Build embeddings (Step 2)
python scripts/02_build_embeddings.py

# 6. Merge embeddings (Step 3)
python scripts/03_merge_embeddings.py

# 7. Prepare MIMIC data and build graphs (Step 4)
python scripts/04_run_data_prepare.py --dataset mimic4 --task mortality

# 8. Train a model
python train.py --config configs/default.yaml --model CADI --dataset mimic4 --task mortality
```

# ROCKET-KG 
ROCKET provides:
- Clean, tested implementations of **CADI** and **CAT** causal GNN models
- Baseline models: **BAT, GAT, GIN, RNN, Transformer, RETAIN, MLP**
- A multi-source **KG construction pipeline** (LLM + UMLS + Rocket ontology)
- **ROCKET Score** (S1–S5): a five-dimensional KG quality metric
- **Ensemble causal discovery** (NOTEARS, GOLEM, PC, LiNGAM)
- A **ReAct agent** for clinical QA
- Full test suite with `pytest`

---

## Quick Start

### 1. Install dependencies

```bash
cd ROCKET
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch==1.12.0+cu116 \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    torch-geometric==2.3.0 \
    -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

### 2. Run the test suite

```bash
# From ROCKET directory
cd /path/to/ROCKET
pytest tests/ -v
```

Expected results:
- `test_models.py`          — ~20 tests, all pass (no data required)
- `test_rocket_score.py`    — ~19 tests, all pass (numpy only)
- `test_causal_discovery.py`— ~9 tests, all pass
- `test_evaluation.py`      — ~16 tests, all pass (sklearn only)

### 3. Build the Knowledge Graph

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=<ADD_YOUR_OPENAI_KEY_HERE>  # ← Add your key here

python -c "
from src.kg_construction import KGBuilder
b = KGBuilder()
b.build_from_csv('resources/CCSCM.csv', 'code', 'name', 'condition', 'CCSCM')
"
```

### 4. Download MIMIC data

See `data/mimic_iii/README.md` and `data/mimic_iv/README.md` for
step-by-step PhysioNet download instructions.

### 5. Train a model

```bash
# Example: train CADI on MIMIC-III mortality
python train.py \
    --config configs/default.yaml \
    --model CADI \
    --dataset mimic3 \
    --task mortality
```

---
## ROCKET framework for Medical Digital Twin


![Image](https://github.com/user-attachments/assets/c943f1af-b9a1-4963-b32c-28a0526a35ce)
ROCKET framework. (A) Multi-source causal KG construction: genetic risk integration (CAUSALdb, ClinVar, Open Targets) and biomarker-driven causal modeling from independent cohorts (eICU) feed into the ROCKET Score (5-component confidence across 13.3M edges from 15 data sources). (B) Patient subgraph construction: MIMIC-IV records are mapped to ontologies and expanded into typed subgraphs unified via Risk GraphRAG. (C) Task-specific routing: the ROCKET Agent and causal attention architectures (CADI/CAT) perform clinical prediction through causally scored pathways.

## Causal Attention Model Architectures


Figure 2 CADI and CAT causal attention framework for clinical prediction. (Left) Patient graph composition: dynamic data (lab tests, medications, vitals, diagnoses) and static data (demographics, comorbidities, lifestyle) are organized into sequential condition, procedure, and medication event streams across temporal visits, forming a multi-relational patient graph. (Center) Two causal attention architectures operate over the patient graph. CADI (top): a dual-path architecture where GNN(l) processes the original graph and GNN(l) processes a counterfactual graph generated by the CF Module; per-layer contrast fact cf ∆(l) → w(l) modulates factual representations, and a Fusion Gate blends both paths. CAT (bottom): a single-path architecture where a Causal Gate causal modulates attention using gradient-based node importance (∆causal), periodically updated via exponential moving average (EMA(l)). Both architectures share a downstream readout: patient node and graph embeddings are concatenated, passed through convolution, activation, dropout, and an MLP. (Right) Four healthcare prediction tasks: mortality prediction, readmission prediction, drug recommendation, and length-of-stay estimation.


### CADI — Causal Attention Dual Inference

```
                     ┌─── Factual path ───────────────────────────────┐
patient graph  →  perturb  →  Counterfactual path                     │
                     │                                                 │
                     └── Δ (contrast) → CausalContrastModule → gate ──┤
                                                                       ▼
                                                               DualFusionGate → MLP
```

### CAT — Causal Attention Transformer

```
                                  ┌── gradient importance (every K epochs) ──┐
                                  │                                            │
patient graph  →  GNN forward  →  CausalImportanceModule (EMA gate)  →  MLP  ←┘
```

---

## ROCKET Score
<img width="5367" height="4143" alt="Image" src="https://github.com/user-attachments/assets/d75b95bd-c0c4-4e2b-9a94-4410508e8359" />

The ROCKET Score evaluates knowledge graph quality across five dimensions:

| Sub-score | What it measures | Range |
|-----------|-----------------|-------|
| **S1** Structural Quality | Graph density × LCC fraction × clustering coefficient | [0, 1] |
| **S2** Semantic Coherence | Inter-cluster embedding separation (cosine distance) | [0, 1] |
| **S3** Task Relevance | Cosine alignment with task-specific query terms | [0, 1] |
| **S4** Causal Confidence | Ensemble agreement across causal discovery methods | [0, 1] |
| **S5** Clinical Coverage | Fraction of medical codes represented in KG | [0, 1] |

```python
from src.rocket_score import RocketScore
import numpy as np

rs = RocketScore()
result = rs.compute_all(
    graph=G,                         # networkx graph
    cluster_embeddings=emb,          # [K, D] centroids
    task_query_embeddings=task_emb,  # [Q, D] task terms
    causal_adj_matrices=adj_list,    # List of [V, V] from causal discovery
    total_codes=285,
    covered_codes=280,
)
print(result)  # RocketScore(S1=0.71, S2=0.63, S3=0.58, S4=0.74, S5=0.98, ROCKET=0.73)
```

---

## Benchmark Results (MIMIC-III, GPT-3.5 KG)

| Task | Best AUROC | Best Model |
|------|-----------|------------|
| Mortality | 0.6370 | CADI |
| Readmission | 0.6893 | GAT |
| Drug Rec | 0.9485 | GIN |
| Length of Stay | 0.7836 | GIN |

See `results/` for full results tables.

---

## Data Access

Both MIMIC-III and MIMIC-IV require PhysioNet credentialed access.

- MIMIC-III: https://physionet.org/content/mimiciii/1.4/
- MIMIC-IV: https://physionet.org/content/mimiciv/3.1/

See `data/mimic_iii/README.md` and `data/mimic_iv/README.md` for
detailed download and preprocessing instructions.

## Knowledge Graph (ROCKET-KG)

The ROCKET knowledge graph and its curated subsets are publicly available via Zenodo:

- Zenodo: https://zenodo.org/records/19401685  
- DOI: 10.5281/zenodo.19401685  

### Download ROCKET-KG
```bash
wget https://zenodo.org/records/19401685/files/ROCKET-KG.zip -O data/rocket_kg.zip
unzip data/rocket_kg.zip -d data/rocket_kg
```

## Citation

If you use ROCKET-KG, please cite:

```bibtex
@dataset{rocketkg_2026,
  author       = {Alsaedi, Sakhaa et al.},
  title        = {ROCKET-KG: Risk-Oriented Causal Biomedical Knowledge Graph},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19401685}
}
```

---


## License

MIT License — see `LICENSE` for details.

