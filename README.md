<p align="center">
  <img src="assets/wattmosaic-banner.svg" alt="WattMosaic — domain-aware campus energy forecasting" width="100%" />
</p>

<p align="center">
  <a href="notebooks/WattMosaic.ipynb"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter Notebook" /></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Public%20RMSE-3.17093-16A085" alt="Public RMSE 3.17093" />
  <img src="https://img.shields.io/badge/License-MIT-2F4858" alt="MIT License" />
</p>

<p align="center">
  <b>Forecasting campus energy demand when real-world sensor data is incomplete.</b>
</p>

---

## ✨ Overview

**WattMosaic** is a regression pipeline for forecasting campus facility energy consumption from historical usage, occupancy, environmental conditions, building context, and calendar signals.

The project was originally developed for **NTU Datathon / Deep Learning Week 2026 — Track 1: Smart Campus Analytics**. The central modelling challenge was not simply regression accuracy: the public test distribution contained **different combinations of missing sensor readings**, so a single generic imputation strategy left performance on the table.

WattMosaic addresses that with a **domain-aware mixture of specialists**. Instead of forcing every incomplete row through the same model, it detects which of the four sensor channels are unavailable and routes the observation to a specialist trained for that missingness pattern.

> **Best verified public leaderboard RMSE: `3.17093`**

## 🧠 Why WattMosaic?

The four sensor channels — **temperature, humidity, occupancy, and previous usage** — create 16 possible availability states. WattMosaic uses the full energy model when all sensors are present and specialized models when one or more sensors are missing.

<p align="center">
  <img src="assets/architecture.svg" alt="WattMosaic architecture" width="100%" />
</p>

### Core ideas

- **Mask-aware routing** — recognizes the exact missing-sensor pattern for each row.
- **Domain-aware specialists** — adapts missing-sensor relationships using unlabeled reference features while never using hidden energy labels.
- **Strong structural baseline** — preserves building, calendar, weather, occupancy, and historical-usage relationships.
- **Prediction-oriented missing-data handling** — optimizes downstream energy forecasts rather than treating imputation accuracy as the end goal.
- **Portable inference** — the final competition model was exported to plain Python tree data so inference did not require CatBoost in the fixed runtime.
- **Defensive output validation** — predictions were aligned by ID and checked for schema, row order, finite values, and unseen-ID behavior.

## 📊 Results

| Evaluation | RMSE | MAE | R² |
|---|---:|---:|---:|
| Development audit — natural missingness | **3.0229** | **2.3810** | **0.9732** |
| Development audit — test-like joint missingness | **3.2143** | **2.5070** | **0.9696** |
| **Hackathon public leaderboard** | **3.17093** | — | — |

The leaderboard score is reported exactly as observed during the event. Development and leaderboard metrics are kept separate because they evaluate different observations.

## 🧩 Modelling pipeline

```text
Raw campus telemetry
        │
        ├── building + building type
        ├── hour / weekday / month
        ├── temperature + humidity
        ├── occupancy
        └── previous energy usage
        │
        ▼
Validation + deterministic feature construction
        │
        ▼
Missing-sensor mask (4 sensors → 16 states)
        │
        ├── complete row ─────────────► energy core
        │
        └── incomplete row ───────────► mask-specific domain-aware specialist
                                         │
                                         ▼
                               combined energy forecast
```

## 📁 Repository structure

```text
WattMosaic/
├── assets/
│   ├── architecture.svg
│   └── wattmosaic-banner.svg
├── notebooks/
│   └── WattMosaic.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Only public-facing project files are included. Competition datasets, serialized model weights, generated predictions, leaderboard-probing artifacts, intermediate experiment rounds, and submission-only files are intentionally excluded.

## 🚀 Explore the notebook

The cleaned notebook contains the final feature interface, portable inference implementation, missingness routing logic, concise EDA, and verified evaluation summary:

**[Open `notebooks/WattMosaic.ipynb`](notebooks/WattMosaic.ipynb)**

The original competition data is **not redistributed** because the provided materials did not explicitly grant public redistribution rights. If you have a legitimate copy of the event dataset, place it under `data/` using the original filenames to activate the optional EDA cells.

## 🛠️ Environment

The final event runtime was fixed to **Python 3.12** and the following package versions:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The pinned environment includes NumPy, pandas, scikit-learn, LightGBM, XGBoost, SciPy, Matplotlib, Seaborn, and Joblib. The published notebook does not include the competition's trained `model.pkl`.

## 🔬 What I learned

This project reinforced several practical ML lessons:

1. **Missingness is part of the data-generating process.** Treating every missing value identically can be worse than modelling different failure modes explicitly.
2. **Better imputation is not automatically better prediction.** The useful objective is downstream forecast error.
3. **Validation must resemble deployment.** Random CV alone can miss train/test differences in sensor availability.
4. **Complexity only earns its place when it transfers.** Several larger boosted and ensemble alternatives were rejected when they failed to improve robust validation or leaderboard performance.
5. **Reproducibility matters under deadline pressure.** Runtime pinning, portable model export, and output-schema checks became part of the solution—not afterthoughts.

## 🏁 Hackathon context

WattMosaic was built for **NTU Datathon / Deep Learning Week 2026**, Track 1 — **Sustainability / Smart Campus Analytics**. The task was to predict a campus facility's continuous energy consumption from historical usage, environmental conditions, building information, occupancy, and time-based patterns.

The repository is a cleaned portfolio release of the final project rather than a dump of the competition workspace.

## 📄 License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

---

<p align="center">
  Built with Python, careful validation, and far too many experiments on missing sensors ⚡
</p>
