# Christine Wanjiku Kungu | B01806008 | MSc IT with Data Analytics | UWS | Feb 2026

# An Explainable AI-Driven Decision Support System with Interactive Dashboard for Evaluating Carbon Pricing Policies in the Energy Sector

**Student:** Christine Wanjiku Kungu | B01806008 | MSc IT with Data Analytics | University of the West of Scotland | February 2026

## File Table
| File | Purpose |
|---|---|
| `notebook.ipynb` | Colab preprocessing pipeline: merge, clean, feature engineering, SHAP pre-analysis |
| `app.py` | Flask REST API for predictions, explainability, scenario simulation, dashboard data |
| `frontend/` | React + Vite interactive dashboard (6 tabs) |
| `requirements.txt` | Pinned Python dependencies |
| `Procfile` | Railway process entrypoint |
| `railway.toml` | Railway deployment settings |
| `.env.example` | Backend environment variables |
| `Dockerfile` | Optional containerized deployment |
| `frontend/.env.example` | Frontend environment variable template |

## Part A - Google Colab Instructions
1. Open `notebook.ipynb` in Google Colab.
2. Upload `thesis_data_only_csv_xlsx.zip` to `/content/`.
3. In Cell 2, verify `ZIP_PATH` is `/content/thesis_data_only_csv_xlsx.zip`.
4. Run **Runtime -> Run all**.
5. Wait for data harmonization, target engineering, and SHAP pre-analysis to complete.
6. Download outputs from Cell 25:
   - `merged_dataset.csv`
   - `feature_cols.json`
   - `target_cols.json`
   - `feature_importance_xgb.csv`

## Part B - Local Development Setup
1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure environment:
```bash
cp .env.example .env
```
4. Run Flask backend:
```bash
python app.py
```
5. Run frontend:
```bash
cd frontend
npm install
npm run dev
```
6. Open `http://localhost:5173`.

## Part C - Railway Deployment
1. Authenticate Railway:
```bash
railway login
```
2. Deploy:
```bash
railway up
```
3. Set environment variables in Railway dashboard:
   - `MODEL_DIR=models`
   - `DATA_PATH=data/master_dataset_engineered.csv`
   - `PORT=8000`
   - `FRONTEND_URL=https://aidecisionsupport-production.up.railway.app`
   - `CORS_ALLOWED_ORIGINS=https://aidecisionsupport-production.up.railway.app`
   - `CORS_ALLOW_ALL=0`
   - `RUN_PIPELINE_ON_STARTUP=1`
   - `FORCE_PIPELINE_RETRAIN=0`
   - `PIPELINE_TIMEOUT_SECONDS=0`
4. Verify health check:
   - `GET /api/health`

Pipeline bootstrap note: `startup.py` runs before API start and will execute `pipeline.py` automatically when model artifacts are missing (or when `FORCE_PIPELINE_RETRAIN=1`).

## Dataset Sources
| Source | Dataset | URL |
|---|---|---|
| IEA | GHG Highlights + World Energy Balances | https://www.iea.org/data-and-statistics |
| EU ETS | Data Viewer + cube extracts | https://climate.ec.europa.eu/eu-action/eu-emissions-trading-system-eu-ets_en |
| World Bank | Climate indicators + country metadata | https://data.worldbank.org |
| UK DESNZ | DUKES and GHG tables | https://www.gov.uk/government/organisations/department-for-energy-security-and-net-zero |

## Target Variables
| Name | Derivation | Unit | Expected Range |
|---|---|---|---|
| `eu_ets_carbon_price_eur` | EU ETS price records filtered by unit | EUR/tCO2 | >= 0 |
| `ghg_reduction_pct` | Year-on-year % change in `GHG_FC` multiplied by -1 | % | ~ -50 to +50 |
| `policy_effectiveness_score` | Normalized composite of reduction vs carbon price | 0-100 | 0-100 |

## Models
| Model | Algorithm | Strengths | Citation |
|---|---|---|---|
| RandomForest | Bagged trees | Robust nonlinearity, low tuning sensitivity | Breiman (2001) |
| XGBoost | Gradient boosting | Strong tabular performance | Chen & Guestrin (2016) |
| LightGBM | Histogram GBDT | Fast training on large data | Ke et al. (2017) |
| CatBoost | Ordered boosting | Handles categorical effects | Prokhorenkova et al. (2018) |
| FFNN | Dense neural network | Learns high-order interactions | Goodfellow et al. (2016) |
| LSTM | Recurrent sequence model | Captures temporal dependencies | Hochreiter & Schmidhuber (1997) |
| Ensemble | Weighted blend | Variance reduction, stability | Zhou (2012) |

## XAI Methods
| Method | Scope | Output | Citation |
|---|---|---|---|
| SHAP attribution | Global + local explainability | Beeswarm, bar, waterfall, dependence, `shap_values.csv` | Lundberg & Lee (2017) |
| LIME local approximation | Local sample explanations | `lime_explanations.json`, local feature weights | Ribeiro et al. (2016) |

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service + model/data readiness |
| POST | `/api/predict` | 3-target prediction with confidence interval |
| POST | `/api/explain/shap` | SHAP attribution response |
| POST | `/api/explain/lime` | LIME local approximation response |
| POST | `/api/simulate` | Scenario simulation |
| GET | `/api/scenarios` | List saved scenarios |
| POST | `/api/scenarios/compare` | Compare selected scenarios |
| GET | `/api/feature-importance` | Cached global SHAP ranking |
| GET | `/api/model-metrics` | Cached model metric summary |
| GET | `/api/data/timeseries` | Indicator time series |
| GET | `/api/data/countries` | Country code list |
| GET | `/api/data/kpis` | Dashboard KPI aggregates |

## Troubleshooting
- SHAP import errors:
  - Reinstall compatible versions: `pip install shap==0.45.0 numpy==1.26.4`.
  - Exponential backoff is implemented; reduce request concurrency.
- Railway memory limits:
  - Use fewer workers (`--workers 1`) and reduce model loading footprint.
- IEA header row detection:
  - Notebook scans first rows for `World` or known country names before parsing.
- CatBoost Windows DLL issues:
  - Use latest Visual C++ Redistributable and reinstall `catboost==1.2.5`.

## Ethical Note
All data are publicly available aggregate statistics and contain no personal data. 
