![Image](https://github.com/user-attachments/assets/17fe1a1d-ea02-4fed-bcf7-5d2d099b0553)
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

## Directory Structure

```
ROCKET/
├── data/
│   ├── rocket_kg/       # KG triples, embeddings, clusters, attention weights
│   ├── mimic_iii/       # MIMIC-III download instructions & processing notes
│   └── mimic_iv/        # MIMIC-IV download instructions & processing notes
├── src/
│   ├── kg_construction/ # Multi-source KG building pipeline
│   │   ├── build_kg.py           # GPT-3.5/4 triple generation
│   │   ├── build_embeddings.py   # OpenAI ada-002 embeddings + merge
│   │   ├── run_clustering.py     # Agglomerative clustering (τ=0.15)
│   │   └── attention_weights.py  # Task-specific attention initialisation
│   ├── rocket_score/    # ROCKET Score computation (S1–S5)
│   │   └── score.py
│   ├── causal_discovery/# Ensemble causal discovery
│   │   └── ensemble.py          # NOTEARS + GOLEM + PC + LiNGAM
│   ├── models/
│   │   ├── cadi.py               # CADI architecture
│   │   ├── cat.py                # CAT architecture
│   │   └── baselines/
│   │       ├── bat.py            # BAT
│   │       ├── gnns.py           # GAT, GIN
│   │       └── ehr_baselines.py  # RNN, Transformer, RETAIN, MLP
│   ├── agent/           # ROCKET Agent
│   │   └── rocket_agent.py
│   └── evaluation/      # Evaluation metrics + Evaluator
│       ├── metrics.py
│       └── evaluate.py
├── configs/
│   └── default.yaml     # Training & experiment configuration
├── notebooks/           # Reproducibility notebooks
├── figures/             # Paper figures placeholder
├── tests/               # pytest test suite
│   ├── test_models.py
│   ├── test_rocket_score.py
│   ├── test_causal_discovery.py
│   └── test_evaluation.py
├── requirements.txt
└── README.md
```

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

## Model Architectures

### BAT — Bi-Attention GNN

```
visit_node  →  α attention (node-level)  ─┐
             →  β attention (visit-decay)  ─┴→  α⊙β  →  BiAttnConv (L layers)  →  MLP
node_emb    →  lin(x)  ──────────────────────────────────────────────────────────────┘
```

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

---


## License

MIT License — see `LICENSE` for details.

