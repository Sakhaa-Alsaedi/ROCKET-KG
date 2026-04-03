# MIMIC-IV Data

## Overview

MIMIC-IV (Medical Information Mart for Intensive Care IV) is the successor
to MIMIC-III, covering ~400,000 hospital admissions at BIDMC from 2008–2022.
It supports both ICD-9 and ICD-10 coded visits.

**Version used:** v3.1
**License:** PhysioNet Credentialed Health Data License 1.5.0
**Citation:** Johnson A et al. (2023). *Scientific Data*, 10:1.

---

## How to Download

### Step 1 — Obtain PhysioNet credentials

Same process as MIMIC-III — complete CITI training and request access:
https://physionet.org/content/mimiciv/3.1/

### Step 2 — Download

```bash
wget -r -N -c -np --user YOUR_USERNAME \
     --ask-password \
     https://physionet.org/files/mimiciv/3.1/

# Or via PhysioNet CLI:
pip install wfdb
physionet download mimiciv/3.1
```

### Step 3 — Decompress required tables

```bash
# MIMIC-IV uses lowercase table names
for table in admissions diagnoses_icd procedures_icd prescriptions patients transfers; do
    gunzip -k mimic4_raw/hosp/${table}.csv.gz
done
```

Place decompressed CSVs in `ROCKET/data/mimic_iv/csv/`.

> **Disk space warning:** MIMIC-IV is significantly larger than MIMIC-III.
> The 6 required CSV files total ~3.8 GB uncompressed.
> The `lenofstay` sample dataset can reach ~272 GB — prepare MIMIC-III first.

---

## Data Processing Pipeline

```bash
python scripts/prepare_data.py \
    --dataset mimic4 \
    --task mortality \
    --graphs_dir data/rocket_kg/graphs \
    --clustering_dir data/rocket_kg/clustering \
    --out_dir data/rocket_kg/exp_data
```

### What happens internally

| Step | Description |
|------|-------------|
| PyHealth load | `MIMIC4Dataset` reads lowercase CSV tables |
| ICD mapping | ICD-9 + ICD-10 diagnosis → CCSCM; ICD-9 + ICD-10 procedure → CCSPROC; NDC → ATC3 |
| Task function | Applies `mortality_prediction_mimic4_fn` / etc. |
| KG subgraph | Per-patient cluster-level subgraph construction |
| Save | `sample_dataset_mimic4_{task}_th015.pkl` + `graph_mimic4_{task}_th015.pkl` |

### Key differences from MIMIC-III

| Aspect | MIMIC-III | MIMIC-IV |
|--------|-----------|----------|
| Patients | ~23,692 | ~400,000 |
| Code system | ICD-9 only | ICD-9 + ICD-10 |
| Table names | UPPERCASE | lowercase |
| Visit minimum | 1 | 2 (some tasks) |
| Disk (sample dataset) | ~50 GB | ~272 GB (LOS) |

---

## File Structure

```
data/mimic_iv/
├── csv/                    # Decompressed MIMIC-IV tables (gitignored)
│   ├── admissions.csv
│   ├── diagnoses_icd.csv
│   ├── procedures_icd.csv
│   ├── prescriptions.csv
│   ├── patients.csv
│   └── transfers.csv
└── README.md               # This file
```

> **Note:** CSV files are excluded from version control.
> Download them using your PhysioNet credentials.
