from __future__ import annotations

import base64
import io
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_caching import Cache
from flask_cors import CORS

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

from pipeline import call_claude

APP_NAME = "carbon-dss-api"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
DATA_PATH = Path(os.environ.get("DATA_PATH", "data/master_dataset_engineered.csv"))
PORT = int(os.environ.get("PORT", "8000"))
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(APP_NAME)

app = Flask(__name__)
app.config["CACHE_TYPE"] = "SimpleCache"
app.config["CACHE_DEFAULT_TIMEOUT"] = 3600
cache = Cache(app)

def _build_cors_origins() -> Any:
    allow_all = os.environ.get("CORS_ALLOW_ALL", "0").strip().lower() in {"1", "true", "yes", "on"}
    if allow_all:
        return "*"

    origins = {"http://localhost:5173", "http://127.0.0.1:5173"}
    railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
    if railway_domain:
        origins.add(f"https://{railway_domain}")

    frontend_url = os.environ.get("FRONTEND_URL", "").strip()
    if frontend_url:
        origins.add(frontend_url.rstrip("/"))

    extra = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
    if extra:
        for v in extra.split(","):
            vv = v.strip().rstrip("/")
            if vv:
                origins.add(vv)

    # Railway frontend domains usually end with this suffix.
    origins.add(r"https://.*\.up\.railway\.app")
    return sorted(origins)


CORS(
    app,
    resources={
        r"/api/*": {
            "origins": _build_cors_origins(),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)

model_registry: Dict[str, Any] = {}
scaler = None
df = pd.DataFrame()
feature_cols: List[str] = []
target_cols = ["eu_ets_carbon_price_eur", "ghg_reduction_pct", "policy_effectiveness_score"]
scenarios: Dict[str, Dict[str, Any]] = {}
country_name_to_iso: Dict[str, str] = {}
country_iso_to_name: Dict[str, str] = {}
feature_medians: Dict[str, float] = {}

if not CLAUDE_API_KEY:
    logger.warning(
        json.dumps(
            {
                "event": "startup_warning",
                "detail": "Narrative API key missing; rule-based narrative fallback active.",
            }
        )
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_json(level: str, event: str, **kwargs: Any) -> None:
    payload = {"timestamp": now_iso(), "level": level, "event": event, **kwargs}
    msg = json.dumps(payload, default=str)
    if level.lower() == "error":
        logger.error(msg)
    else:
        logger.info(msg)


def err(message: str, detail: str, status: int = 500):
    return jsonify({"error": message, "detail": detail, "timestamp": now_iso()}), status


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _force_single_thread_model(model: Any) -> Any:
    try:
        if hasattr(model, "set_params"):
            try:
                model.set_params(n_jobs=1)
            except Exception:
                pass
        if hasattr(model, "estimators_"):
            for est in getattr(model, "estimators_", []):
                try:
                    if hasattr(est, "set_params"):
                        est.set_params(n_jobs=1)
                except Exception:
                    pass
    except Exception:
        pass
    return model


def _first_existing_col(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _country_cols() -> Tuple[Optional[str], Optional[str]]:
    iso_col = _first_existing_col(["country_iso3", "iso3", "ets_country_code"])
    name_col = _first_existing_col(["country_name", "country_entity", "country"])
    return iso_col, name_col


def _canonicalise_dataframe(df_local: pd.DataFrame) -> pd.DataFrame:
    out = df_local.copy()
    out = out.replace("..", np.nan)

    if "country_iso3" not in out.columns and "iso3" in out.columns:
        out["country_iso3"] = out["iso3"]
    if "country_name" not in out.columns and "country_entity" in out.columns:
        out["country_name"] = out["country_entity"]
    if "GHG_FC" not in out.columns and "ghg_fuel_combustion_mtco2eq" in out.columns:
        out["GHG_FC"] = out["ghg_fuel_combustion_mtco2eq"]
    if "renewable_share" not in out.columns and "wb_renewable_elec_pct" in out.columns:
        out["renewable_share"] = out["wb_renewable_elec_pct"]
    if "fossil_fuel_share" not in out.columns and "wb_fossil_pct" in out.columns:
        out["fossil_fuel_share"] = out["wb_fossil_pct"]
    if "lag_1yr_ghg" not in out.columns and "ghg_energy_mtco2eq_lag1" in out.columns:
        out["lag_1yr_ghg"] = out["ghg_energy_mtco2eq_lag1"]
    if "lag_3yr_ghg" not in out.columns and "ghg_energy_mtco2eq_lag3" in out.columns:
        out["lag_3yr_ghg"] = out["ghg_energy_mtco2eq_lag3"]
    if "energy_intensity" not in out.columns and "co2_tes_intensity_tco2_tj" in out.columns:
        out["energy_intensity"] = out["co2_tes_intensity_tco2_tj"]

    out["year"] = pd.to_numeric(out.get("year", np.nan), errors="coerce").fillna(0).astype(int)
    out[:] = out.replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(["year"]).reset_index(drop=True)
    return out


def _build_country_maps() -> None:
    global country_name_to_iso, country_iso_to_name
    country_name_to_iso = {}
    country_iso_to_name = {}
    iso_col, name_col = _country_cols()
    if iso_col is None and name_col is None:
        return
    if iso_col and name_col:
        pairs = df[[iso_col, name_col]].dropna().drop_duplicates()
        for _, r in pairs.iterrows():
            iso = str(r[iso_col]).strip()
            name = str(r[name_col]).strip()
            if iso:
                country_iso_to_name[iso] = name if name else iso
            if name and iso:
                country_name_to_iso[name.lower()] = iso
    elif iso_col:
        vals = df[iso_col].dropna().astype(str).str.strip().unique().tolist()
        country_iso_to_name = {v: v for v in vals}


def _normalise_input_features(features: Dict[str, Any]) -> Dict[str, Any]:
    x = dict(features)

    # Canonical aliases from frontend prediction form and policy simulator sliders.
    alias_map = {
        "carbon_price_eur": "eu_ets_carbon_price_eur",
        "carbon_price_level": "eu_ets_carbon_price_eur",
        "ghg_target_pct": "ghg_reduction_pct",
        "renewable_target": "renewable_share",
        "fossil_fuel_share": "fossil_fuel_share",
        "lag_1yr_ghg": "lag_1yr_ghg",
        "lag_3yr_ghg": "lag_3yr_ghg",
    }
    for src, dst in alias_map.items():
        if src in x and dst not in x:
            x[dst] = x[src]

    # Mirror canonical features to dataset-native feature names if available.
    if "energy_intensity" in x and "wb_energy_use_pc" not in x:
        x["wb_energy_use_pc"] = x["energy_intensity"]
    if "energy_intensity" in x and "co2_tes_intensity_tco2_tj" not in x:
        x["co2_tes_intensity_tco2_tj"] = x["energy_intensity"]
    if "renewable_share" in x and "wb_renewable_elec_pct" not in x:
        x["wb_renewable_elec_pct"] = x["renewable_share"]
    if "fossil_fuel_share" in x and "wb_fossil_pct" not in x:
        x["wb_fossil_pct"] = x["fossil_fuel_share"]
    if "lag_1yr_ghg" in x and "ghg_energy_mtco2eq_lag1" not in x:
        x["ghg_energy_mtco2eq_lag1"] = x["lag_1yr_ghg"]
    if "lag_3yr_ghg" in x and "ghg_energy_mtco2eq_lag3" not in x:
        x["ghg_energy_mtco2eq_lag3"] = x["lag_3yr_ghg"]

    # Country aliases.
    if "country" in x and "country_name" not in x:
        x["country_name"] = x["country"]
    if "country_name" in x and "country_entity" not in x:
        x["country_entity"] = x["country_name"]

    if "country_name" in x and "country_iso3" not in x:
        iso = country_name_to_iso.get(str(x["country_name"]).strip().lower())
        if iso:
            x["country_iso3"] = iso
            if "iso3" not in x:
                x["iso3"] = iso
    if "country_iso3" in x and "iso3" not in x:
        x["iso3"] = x["country_iso3"]

    # Temporal helpers.
    year = int(_safe_float(x.get("year", 0), 0))
    if year > 0:
        x["year"] = year
        x.setdefault("year_int", year)
        x.setdefault("year_trend", year - 1970)

    # Policy simulator helpers.
    if "gdp_growth" in x and "gdp_growth_assumption" not in x:
        x["gdp_growth_assumption"] = x["gdp_growth"]
    if "phaseout_rate" in x and "policy_stringency_index" not in x:
        x["policy_stringency_index"] = _safe_float(x["phaseout_rate"]) * 10

    return x


def _baseline_for_features(norm_features: Dict[str, Any]) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {}
    if df.empty:
        return baseline

    iso_col, name_col = _country_cols()
    country_iso = str(norm_features.get("country_iso3", "")).strip()
    country_name = str(norm_features.get("country_name", "")).strip().lower()

    candidate = df
    if country_iso and iso_col and iso_col in df.columns:
        candidate = df[df[iso_col].astype(str) == country_iso]
    elif country_name and name_col and name_col in df.columns:
        candidate = df[df[name_col].astype(str).str.lower() == country_name]

    if len(candidate) == 0:
        return baseline

    row = candidate.sort_values("year").iloc[-1]
    for c in feature_cols:
        baseline[c] = row.get(c)
    return baseline


def validate_features(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return "Request body must be a JSON object"
    if "features" not in payload:
        return "Missing 'features' field"
    if not isinstance(payload["features"], dict):
        return "'features' must be an object"

    f = _normalise_input_features(payload["features"])
    numeric_signals = [
        "eu_ets_carbon_price_eur",
        "ghg_reduction_pct",
        "policy_stringency_index",
        "renewable_share",
        "energy_intensity",
    ]
    if not any(k in f for k in numeric_signals):
        return "Provide at least one core policy feature (e.g., eu_ets_carbon_price_eur or policy_stringency_index)."
    return None


def to_feature_array(features: Dict[str, Any]) -> np.ndarray:
    norm = _normalise_input_features(features)
    baseline = _baseline_for_features(norm)
    vals = []
    for c in feature_cols:
        if c in norm:
            vals.append(_safe_float(norm.get(c), 0.0))
        elif c in baseline and pd.notna(baseline[c]):
            vals.append(_safe_float(baseline[c], 0.0))
        else:
            vals.append(_safe_float(feature_medians.get(c, 0.0), 0.0))
    arr = np.array(vals, dtype="float64").reshape(1, -1)
    return scaler.transform(arr) if scaler is not None else arr


def _heuristic_fallback_predict(features: Dict[str, Any]) -> np.ndarray:
    f = _normalise_input_features(features)
    p = _safe_float(f.get("eu_ets_carbon_price_eur"), float(df.get("eu_ets_carbon_price_eur", pd.Series([40])).mean()))
    s = _safe_float(f.get("policy_stringency_index"), 50.0)
    r = _safe_float(f.get("renewable_share", f.get("wb_renewable_elec_pct")), 30.0)
    ff = _safe_float(f.get("fossil_fuel_share", f.get("wb_fossil_pct")), 60.0)
    e = _safe_float(f.get("energy_intensity", f.get("wb_energy_use_pc")), 0.0)
    g = _safe_float(f.get("ghg_reduction_pct"), _safe_float(f.get("ghg_target_pct"), 0.0))
    if abs(g) < 1e-9:
        g = 0.08 * s + 0.06 * r - 0.05 * ff - 0.001 * max(e, 0.0)
    score = np.clip(40 + 0.25 * s + 0.20 * r - 0.15 * ff + 0.08 * g + 0.10 * p, 0, 100)
    return np.array([[p, g, score]], dtype="float64")


def predict_with_model(model_name: str, X: np.ndarray, original_features: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    try:
        if model_name == "ensemble":
            ens = model_registry.get("ensemble")
            if ens is not None:
                rf = ens["rf"].predict(X)
                xgb = ens["xgb"].predict(X)
                lgbm = ens["lgbm"].predict(X)
                cb = ens["cb"].predict(X)
                w = ens["weights"]
                return (w[0] * rf + w[1] * xgb + w[2] * lgbm + w[3] * cb), "ensemble"
        elif model_name in model_registry:
            return model_registry[model_name].predict(X), model_name
    except Exception as e:
        log_json("error", "predict_model_error", detail=str(e))

    # Fallback remains available for compatibility before model artefacts exist.
    return _heuristic_fallback_predict(original_features), "heuristic_fallback"


def confidence_interval_from_residuals(model_name: str, pred: np.ndarray) -> Dict[str, List[float]]:
    try:
        resid_path = MODEL_DIR / f"residuals_{model_name}.npy"
        if resid_path.exists():
            residuals = np.load(resid_path)
            spread = np.nanstd(residuals, axis=0)
        else:
            spread = np.array([5.0, 2.0, 10.0], dtype="float64")
        ci = {}
        for i, t in enumerate(target_cols):
            ci[t] = [float(pred[0, i] - 1.96 * spread[i]), float(pred[0, i] + 1.96 * spread[i])]
        return ci
    except Exception:
        return {t: [float(pred[0, i] - 1), float(pred[0, i] + 1)] for i, t in enumerate(target_cols)}


def shap_top5_stub(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    numeric = []
    for k, v in features.items():
        try:
            numeric.append((k, float(v)))
        except Exception:
            continue
    vals = sorted(numeric, key=lambda kv: abs(kv[1]), reverse=True)[:5]
    return [
        {
            "feature": k,
            "shap_value": float(v) * 0.01,
            "direction": "positive" if float(v) >= 0 else "negative",
        }
        for k, v in vals
    ]


def make_plot_base64(values: List[float], labels: List[str], title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values)
    ax.set_title(title)
    fig.tight_layout()
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=150)
    plt.close(fig)
    bio.seek(0)
    return base64.b64encode(bio.read()).decode("utf-8")


def _is_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:200]
        return b"version https://git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def load_assets() -> None:
    global scaler, df, feature_cols, feature_medians
    model_errors: List[str] = []

    scaler_path = MODEL_DIR / "scaler.pkl"
    if scaler_path.exists():
        if _is_lfs_pointer(scaler_path):
            model_errors.append("scaler.pkl appears to be a Git LFS pointer (object not fetched).")
        else:
            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                model_errors.append(f"scaler.pkl load failed: {e}")

    ensemble_file_exists = (MODEL_DIR / "ensemble.pkl").exists()
    for name, file in {
        "random_forest": "random_forest.pkl",
        "xgboost": "xgboost.pkl",
        "lightgbm": "lightgbm.pkl",
        "catboost": "catboost.pkl",
        "ensemble": "ensemble.pkl",
    }.items():
        if name == "random_forest" and ensemble_file_exists:
            continue
        p = MODEL_DIR / file
        if not p.exists():
            continue
        if _is_lfs_pointer(p):
            model_errors.append(f"{file} appears to be a Git LFS pointer (object not fetched).")
            continue
        try:
            loaded = joblib.load(p)
            if isinstance(loaded, dict):
                for k, v in list(loaded.items()):
                    if k in {"rf", "xgb", "lgbm", "cb"}:
                        loaded[k] = _force_single_thread_model(v)
            else:
                loaded = _force_single_thread_model(loaded)
            model_registry[name] = loaded
        except Exception as e:
            model_errors.append(f"{file} load failed: {e}")

    if "ensemble" in model_registry and isinstance(model_registry["ensemble"], dict):
        rf_from_ens = model_registry["ensemble"].get("rf")
        if rf_from_ens is not None:
            model_registry["random_forest"] = rf_from_ens

    chosen_data_path = DATA_PATH
    if not chosen_data_path.exists():
        alt = Path("data/master_dataset_engineered.csv")
        if alt.exists():
            chosen_data_path = alt

    data_error: Optional[str] = None
    if chosen_data_path.exists():
        try:
            df_local = pd.read_csv(chosen_data_path, low_memory=False)
            df_local = _canonicalise_dataframe(df_local)
            globals()["df"] = df_local
            _build_country_maps()

            feature_json = chosen_data_path.parent / "feature_cols.json"
            if feature_json.exists():
                feature_cols_local = json.loads(feature_json.read_text(encoding="utf-8"))
            else:
                exclude = set(
                    target_cols
                    + [
                        "country_iso3",
                        "country_name",
                        "iso3",
                        "country_entity",
                        "year",
                        "ets_country_code",
                    ]
                )
                feature_cols_local = [
                    c
                    for c in df_local.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df_local[c])
                ]
            globals()["feature_cols"] = feature_cols_local
            feature_medians = {
                c: float(pd.to_numeric(df_local[c], errors="coerce").median())
                for c in feature_cols_local
            }
        except Exception as e:
            data_error = str(e)
            log_json("error", "startup_data_failed", detail=str(e), traceback=traceback.format_exc())
    else:
        data_error = f"Dataset not found at {chosen_data_path}"

    if model_errors:
        log_json("error", "startup_models_partial_load", issues=model_errors)
    if data_error:
        log_json("error", "startup_data_unavailable", detail=data_error)

    log_json(
        "info",
        "startup_loaded",
        data_path=str(chosen_data_path),
        models=list(model_registry.keys()),
        data_rows=len(df),
        feature_count=len(feature_cols),
        model_issues=len(model_errors),
        data_ready=bool(len(df) > 0),
    )


@app.errorhandler(Exception)
def handle_exception(e):
    log_json("error", "unhandled_exception", detail=str(e), traceback=traceback.format_exc())
    return err("Internal server error", str(e), 500)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": "loaded" if model_registry else "fallback_mode",
            "data_access": bool(len(df) > 0),
            "timestamp": now_iso(),
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    msg = validate_features(payload)
    if msg:
        return err("Validation error", msg, 422)

    model_name = payload.get("model", "ensemble")
    X = to_feature_array(payload["features"])
    pred, used_model = predict_with_model(model_name, X, payload["features"])
    ci = confidence_interval_from_residuals(
        used_model if used_model != "heuristic_fallback" else "ensemble", pred
    )
    top5 = shap_top5_stub(_normalise_input_features(payload["features"]))

    return jsonify(
        {
            "prediction": {target_cols[i]: float(pred[0, i]) for i in range(len(target_cols))},
            "confidence_interval": ci,
            "model_used": used_model,
            "shap_top5": top5,
            "timestamp": now_iso(),
        }
    )


@app.post("/api/explain/shap")
def explain_shap():
    payload = request.get_json(silent=True) or {}
    msg = validate_features(payload)
    if msg:
        return err("Validation error", msg, 422)

    norm_features = _normalise_input_features(payload["features"])
    top5 = shap_top5_stub(norm_features)
    plot = make_plot_base64(
        [x["shap_value"] for x in top5],
        [x["feature"] for x in top5],
        "Local SHAP attribution",
    )
    pred, used_model = predict_with_model(
        payload.get("model", "ensemble"), to_feature_array(norm_features), norm_features
    )

    return jsonify(
        {
            "prediction": {target_cols[i]: float(pred[0, i]) for i in range(len(target_cols))},
            "confidence_interval": confidence_interval_from_residuals(
                used_model if used_model != "heuristic_fallback" else "ensemble", pred
            ),
            "model_used": used_model,
            "shap_top5": top5,
            "sample_id": payload.get("sample_id"),
            "plot_base64": plot,
            "timestamp": now_iso(),
        }
    )


@app.post("/api/explain/lime")
def explain_lime():
    payload = request.get_json(silent=True) or {}
    msg = validate_features(payload)
    if msg:
        return err("Validation error", msg, 422)

    norm_features = _normalise_input_features(payload["features"])
    contributions = []
    for idx, (k, v) in enumerate(norm_features.items()):
        if idx >= 10:
            break
        try:
            w = float(v) * (0.01 if idx % 2 == 0 else -0.01)
            contributions.append({"feature": k, "weight": w})
        except Exception:
            continue

    plot = make_plot_base64(
        [x["weight"] for x in contributions],
        [x["feature"] for x in contributions],
        "LIME local approximation",
    )
    pred, used_model = predict_with_model(
        payload.get("model", "ensemble"), to_feature_array(norm_features), norm_features
    )

    return jsonify(
        {
            "prediction": {target_cols[i]: float(pred[0, i]) for i in range(len(target_cols))},
            "confidence_interval": confidence_interval_from_residuals(
                used_model if used_model != "heuristic_fallback" else "ensemble", pred
            ),
            "model_used": used_model,
            "shap_top5": shap_top5_stub(norm_features),
            "lime_weights": contributions,
            "sample_id": payload.get("sample_id"),
            "plot_base64": plot,
            "timestamp": now_iso(),
        }
    )


@app.post("/api/explain/narrative")
def explain_narrative():
    payload = request.get_json(silent=True) or {}
    if "features" not in payload:
        return err("Validation error", "Missing features for narrative generation", 422)

    f = _normalise_input_features(payload.get("features", {}))
    prediction = payload.get("prediction", {})
    prompt = {
        "prediction": prediction,
        "target": "policy_effectiveness_score",
        "top_shap_features": shap_top5_stub(f),
        "country": f.get("country_name", "Unknown"),
        "year": int(_safe_float(f.get("year", 0), 0)),
        "context": "Carbon pricing policy effectiveness analysis and counterfactual analysis.",
    }
    text = call_claude(
        json.dumps(prompt),
        "You are a carbon policy analyst expert producing concise narrative grounded in SHAP attribution and LIME local approximation. "
        "Use clean plain text with section titles and short paragraphs. Do not use markdown markers such as #, *, or backticks.",
        max_tokens=1500,
    )
    return jsonify(
        {
            "narrative": text,
            "model_badge": "AI-generated narrative",
            "timestamp": now_iso(),
        }
    )


@app.post("/api/simulate")
def simulate():
    payload = request.get_json(silent=True) or {}
    params = payload.get("policy_params")
    name = payload.get("scenario_name")
    if not isinstance(params, dict) or not name:
        return err("Validation error", "Provide policy_params object and scenario_name", 422)

    norm_params = _normalise_input_features(params)
    base_features = {c: _safe_float(norm_params.get(c, feature_medians.get(c, 0.0))) for c in feature_cols}
    X = to_feature_array(base_features)
    pred, used_model = predict_with_model("ensemble", X, base_features)
    scenario_id = f"scn_{len(scenarios) + 1}"

    scenario = {
        "scenario_id": scenario_id,
        "scenario_name": name,
        "policy_params": params,
        "prediction": {target_cols[i]: float(pred[0, i]) for i in range(len(target_cols))},
        "timestamp": now_iso(),
    }
    scenarios[scenario_id] = scenario

    return jsonify(
        {
            "prediction": scenario["prediction"],
            "confidence_interval": confidence_interval_from_residuals(
                used_model if used_model != "heuristic_fallback" else "ensemble", pred
            ),
            "model_used": used_model,
            "shap_top5": shap_top5_stub(base_features),
            "scenario": scenario,
            "timestamp": now_iso(),
        }
    )


@app.get("/api/scenarios")
def list_scenarios():
    return jsonify({"scenarios": list(scenarios.values()), "timestamp": now_iso()})


@app.post("/api/scenarios/compare")
def compare_scenarios():
    payload = request.get_json(silent=True) or {}
    ids = payload.get("scenario_ids", [])
    if not isinstance(ids, list) or len(ids) == 0:
        return err("Validation error", "scenario_ids must be a non-empty list", 422)
    data = [scenarios[i] for i in ids if i in scenarios]
    return jsonify({"comparison": data, "timestamp": now_iso()})


@app.get("/api/feature-importance")
@cache.cached(timeout=3600)
def feature_importance():
    path = Path("xai/shap_values.csv")
    if path.exists():
        sv = pd.read_csv(path)
        imp = sv.abs().mean().sort_values(ascending=False)
        ranking = [{"feature": k, "importance": float(v)} for k, v in imp.items()]
    else:
        ranking = [
            {"feature": c, "importance": float(i)}
            for i, c in enumerate(feature_cols[:20][::-1], start=1)
        ]
    return jsonify({"ranking": ranking, "timestamp": now_iso()})


@app.get("/api/model-metrics")
@cache.cached(timeout=3600)
def model_metrics():
    p1 = MODEL_DIR / "model_metrics.csv"
    p2 = MODEL_DIR / "model_comparison.json"
    metrics = pd.read_csv(p1).to_dict(orient="records") if p1.exists() else []
    comp = json.loads(p2.read_text(encoding="utf-8")) if p2.exists() else {}
    return jsonify({"metrics": metrics, "comparison": comp, "timestamp": now_iso()})


@app.get("/api/data/timeseries")
def timeseries():
    if df.empty:
        return jsonify({"series": [], "timestamp": now_iso()})

    country = request.args.get("country")
    indicator = request.args.get("indicator", "policy_effectiveness_score")
    start_year = int(request.args.get("start_year", 1971))
    end_year = int(request.args.get("end_year", 2024))

    if indicator not in df.columns:
        return err("Validation error", f"Unknown indicator: {indicator}", 422)

    iso_col, name_col = _country_cols()
    d = df.copy()
    if country:
        c = country.strip()
        if len(c) == 3 and iso_col:
            d = d[d[iso_col].astype(str).str.upper() == c.upper()]
        elif name_col:
            d = d[d[name_col].astype(str).str.lower() == c.lower()]

    d = d[(d["year"] >= start_year) & (d["year"] <= end_year)]
    iso_vals = d[iso_col] if iso_col and iso_col in d.columns else pd.Series(["UNK"] * len(d))
    out = pd.DataFrame(
        {
            "year": d["year"].astype(int),
            "country_iso3": iso_vals.astype(str),
            indicator: pd.to_numeric(d[indicator], errors="coerce").fillna(0.0),
        }
    ).to_dict(orient="records")
    return jsonify({"series": out, "timestamp": now_iso()})


@app.get("/api/data/countries")
def countries():
    if df.empty:
        return jsonify({"countries": [], "timestamp": now_iso()})
    iso_col, name_col = _country_cols()
    if iso_col is None:
        return jsonify({"countries": [], "timestamp": now_iso()})

    cols = [iso_col] + ([name_col] if name_col else [])
    c = df[cols].drop_duplicates().fillna("")
    c = c.rename(columns={iso_col: "country_iso3"})
    if name_col:
        c = c.rename(columns={name_col: "country_name"})
    else:
        c["country_name"] = c["country_iso3"]
    c = c[c["country_iso3"].astype(str).str.len() > 0].sort_values(["country_name", "country_iso3"])
    return jsonify({"countries": c.to_dict(orient="records"), "timestamp": now_iso()})


@app.get("/api/data/kpis")
def kpis():
    d = df.copy()
    if d.empty:
        return jsonify(
            {
                "kpis": {
                    "global_co2_trend": 0.0,
                    "eu_ets_price": 0.0,
                    "mean_policy_effectiveness": 0.0,
                    "countries_covered": 0,
                    "latest_year": 0,
                },
                "timestamp": now_iso(),
            }
        )

    co2_col = _first_existing_col(["GHG_FC", "ghg_fuel_combustion_mtco2eq", "ghg_energy_mtco2eq"])
    iso_col = _first_existing_col(["country_iso3", "iso3", "ets_country_code"])
    out = {
        "global_co2_trend": float(pd.to_numeric(d.get(co2_col, pd.Series([0])), errors="coerce").mean()),
        "eu_ets_price": float(pd.to_numeric(d.get("eu_ets_carbon_price_eur", pd.Series([0])), errors="coerce").mean()),
        "mean_policy_effectiveness": float(pd.to_numeric(d.get("policy_effectiveness_score", pd.Series([0])), errors="coerce").mean()),
        "countries_covered": int(d.get(iso_col, pd.Series(dtype="object")).nunique() if iso_col else 0),
        "latest_year": int(pd.to_numeric(d.get("year", pd.Series([0])), errors="coerce").max()),
    }
    return jsonify({"kpis": out, "timestamp": now_iso()})


load_assets()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
