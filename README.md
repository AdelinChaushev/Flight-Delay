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
│   └── models.py                  # preprocessing, pipeline, and the LightGBM tuning objective
├── data/                          # git-ignored
│   ├── raw/                       # put the downloaded CSV here
│   └── processed/                 # parquet artifacts written by the notebooks
└── README.md
```

**Data flow:** `Eda` writes `flights_clean.parquet`, which `Feature_Engineering` reads and turns into `flights_features.parquet`, which `Modeling` reads. `Introduction` and `Conclusion` are prose (they produce no artifacts).

## Getting the data

The dataset is git-ignored (too large for the repo). Download it from Kaggle and place the CSV in `data/raw/`:

- **Source:** [Flight Data 2024 — Kaggle](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)
- **Expected path:** `data/raw/flight_data_2024.csv`

## Setup

Python 3.10+. Install the dependencies:

```bash
pip install pandas numpy "scikit-learn>=1.4" scipy feature-engine lightgbm optuna mlflow matplotlib pyarrow
```

## Running

Run the notebooks **in order, top to bottom**. Each of the middle three writes a file the next one reads:

1. `Introduction.ipynb` — read (no code to run)
2. `Eda.ipynb` — writes `data/processed/flights_clean.parquet`
3. `Feature_Engineering.ipynb` — writes `data/processed/flights_features.parquet`
4. `Modeling.ipynb` — trains, tunes with Optuna, tracks with MLflow, scores the held-out test set
5. `Conclusion.ipynb` — read (no code to run)

> **Note:** run `Modeling.ipynb` start-to-finish in a single session — its training cells log models to MLflow that the final test cell loads back. The local `mlruns/` store is git-ignored and is regenerated on each run.
