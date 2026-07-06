# Flight-Delay

Predicting a US domestic flight's **arrival delay** (in minutes) from **pre-departure** information only. The project is a five-notebook pipeline with shared helpers in `src/`.

## Repository structure

```
Flight-Delay/
├── notebooks/
│   ├── Introduction.ipynb         # overview & plan (read first)
│   ├── Eda.ipynb                  # cleaning + exploration  → data/processed/flights_clean.parquet
│   ├── Feature_Engineering.ipynb  # features + time split   → data/processed/flights_features.parquet
│   ├── Modeling.ipynb             # baselines → linear → LightGBM (Optuna tuning, MLflow tracking)
│   └── Conclusion.ipynb           # results & findings (read last)
├── src/
│   ├── metrics.py                 # MAE / RMSE / R² helpers
│   └── models.py                  # preprocessing, pipelines, tuning objective, model loader
├── tests/                         # pytest unit tests for src/
├── data/                          # git-ignored
│   ├── raw/                       # put the downloaded CSV here
│   └── processed/                 # parquet artifacts written by the notebooks
├── requirements.txt
└── README.md
```

**Data flow:** `Eda` writes `flights_clean.parquet`, which `Feature_Engineering` reads and turns into `flights_features.parquet`, which `Modeling` reads. `Introduction` and `Conclusion` are prose (they produce no artifacts).

## Getting the data

The dataset is git-ignored (too large for the repo). Download it from Kaggle and place the CSV in `data/raw/`:

- **Source:** [Flight Data 2024 — Kaggle](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)
- **Expected path:** `data/raw/flight_data_2024.csv`

## Running

From the project root:

**1. Install** — Python 3.10+:

```bash
pip install -r requirements.txt
```

**2. Get the data** — download from Kaggle (see *Getting the data* above) to `data/raw/flight_data_2024.csv`.

**3. Run the notebooks in order, top to bottom** — the middle three each write a parquet the next one reads:

| # | notebook | produces / does |
|---|---|---|
| 1 | `Introduction.ipynb` | overview & plan — *read* |
| 2 | `Eda.ipynb` | → `data/processed/flights_clean.parquet` |
| 3 | `Feature_Engineering.ipynb` | → `data/processed/flights_features.parquet` |
| 4 | `Modeling.ipynb` | baselines → linear → LightGBM (Optuna, MLflow), then the held-out test |
| 5 | `Conclusion.ipynb` | results & findings — *read* |

**4. Run the tests** (optional):

```bash
pytest
```

> Run `Modeling.ipynb` **start-to-finish in one session**: its training cells log models to a local `mlruns/` store that the final cells load back (git-ignored, regenerated on each run).
