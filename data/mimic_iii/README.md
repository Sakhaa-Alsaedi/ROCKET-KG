# MIMIC-III Data

## Overview

MIMIC-III (Medical Information Mart for Intensive Care III) is a large,
freely available database comprising de-identified health-related data
associated with over 40,000 patients who stayed in critical care units
of the Beth Israel Deaconess Medical Center between 2001 and 2012.

**License:** PhysioNet Credentialed Health Data License 1.5.0
**Citation:** Johnson AEW et al. (2016). *Scientific Data*, 3:160035.

---

## How to Download

### Step 1 — Complete CITI training

You must complete the CITI Data or Specimens Only Research training.
Visit: https://physionet.org/about/citi-course/

### Step 2 — Request access

1. Create a PhysioNet account at https://physionet.org/register/
2. Sign the data use agreement for MIMIC-III:
   https://physionet.org/content/mimiciii/1.4/
3. Access is typically granted within a few days.

### Step 3 — Download

```bash
# Using wget (after authentication)
wget -r -N -c -np --user YOUR_PHYSIONET_USERNAME \
     --ask-password \
     https://physionet.org/files/mimiciii/1.4/

# Or using the PhysioNet CLI:
pip install wfdb
physionet download mimiciii/1.4
```

### Step 4 — Decompress required tables

Only 6 tables are needed for ROCKET:

```bash
for table in ADMISSIONS DIAGNOSES_ICD PROCEDURES_ICD PRESCRIPTIONS PATIENTS TRANSFERS; do
    gunzip -k mimic3_raw/${table}.csv.gz
done
```

Place the decompressed CSVs in `ROCKET/data/mimic_iii/csv/`.

### Step 5 — Uppercase column headers (PyHealth requirement)

PyHealth requires uppercase column names:

```python
import pandas as pd, os, glob

for f in glob.glob("data/mimic_iii/csv/*.csv"):
    df = pd.read_csv(f, nrows=0)
    df.columns = df.columns.str.upper()
    # rewrite headers in-place
    full = pd.read_csv(f)
    full.columns = full.columns.str.upper()
    full.to_csv(f, index=False)
    print(f"Fixed: {os.path.basename(f)}")
```

---

## Data Processing Pipeline

After downloading, run data preparation via:

```bash
python scripts/prepare_data.py \
    --dataset mimic3 \
    --task mortality \
    --graphs_dir data/rocket_kg/graphs \
    --clustering_dir data/rocket_kg/clustering \
    --out_dir data/rocket_kg/exp_data
```

### What happens internally

| Step | Description |
|------|-------------|
| PyHealth load | `MIMIC3Dataset` reads ADMISSIONS + DIAGNOSES_ICD + PROCEDURES_ICD + PRESCRIPTIONS |
| ICD-9 mapping | ICD-9 diagnosis → CCSCM codes; ICD-9 procedure → CCSPROC codes; NDC/ATC → ATC3 |
| Task function | Applies `mortality_prediction_mimic3_fn` / `readmission_prediction_mimic3_fn` / etc. |
| KG subgraph | For each patient visit, builds a cluster-level subgraph from pre-built KG triples |
| Save | `sample_dataset_mimic3_{task}_th015.pkl` + `graph_mimic3_{task}_th015.pkl` |

### Expected cohort sizes (after task filtering)

| Task | Approx. Patients | Approx. Samples |
|------|-----------------|----------------|
| Mortality | ~19,000 | ~37,000 |
| Readmission | ~19,000 | ~37,000 |
| Drug Rec | ~23,000 | ~43,000 |
| Length of Stay | ~23,000 | ~48,000 |

---

## File Structure

```
data/mimic_iii/
├── csv/                    # Decompressed MIMIC-III tables (gitignored)
│   ├── ADMISSIONS.csv
│   ├── DIAGNOSES_ICD.csv
│   ├── PROCEDURES_ICD.csv
│   ├── PRESCRIPTIONS.csv
│   ├── PATIENTS.csv
│   └── TRANSFERS.csv
└── README.md               # This file
```

> **Note:** CSV files are excluded from version control via `.gitignore`.
> They must be downloaded separately using your PhysioNet credentials.
