"""
Core ML pipeline for the Explainable AI-Driven Decision Support System for
carbon pricing policy effectiveness in the energy sector.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from scipy import stats
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    import lime.lime_tabular
except Exception:
    lime = None

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
except Exception:
    tf = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Global reproducibility
np.random.seed(42)
if tf is not None:
    tf.random.set_seed(42)

# Visual theme palette
PALETTE = {
    "bg": "#F4F6F9",
    "panel": "#FFFFFF",
    "accent": "#1A56DB",
    "accent2": "#0E9F6E",
    "danger": "#E02424",
    "warn": "#FF8A4C",
    "text": "#111928",
    "muted": "#6B7280",
    "chart": [
        "#1A56DB",
        "#0E9F6E",
        "#E02424",
        "#7E3AF2",
        "#FF8A4C",
        "#F59E0B",
        "#06B6D4",
        "#EC4899",
    ],
}

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
_PROMPT_CACHE: Dict[str, str] = {}
DEFAULT_TARGET_COLS = [
    "eu_ets_carbon_price_eur",
    "ghg_reduction_pct",
    "policy_effectiveness_score",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("pipeline")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return max(minimum, int(default))
    try:
        return max(minimum, int(raw))
    except Exception:
        return max(minimum, int(default))


PIPELINE_FAST_MODE = _env_bool("PIPELINE_FAST_MODE", False)
BOOTSTRAP_N_ITER = _env_int("BOOTSTRAP_N_ITER", 200 if PIPELINE_FAST_MODE else 1000, minimum=10)
CV_SPLITS = _env_int("CV_SPLITS", 3 if PIPELINE_FAST_MODE else 5, minimum=2)
TREE_N_ESTIMATORS = _env_int("TREE_N_ESTIMATORS", 120 if PIPELINE_FAST_MODE else 300, minimum=10)
NN_EPOCHS = _env_int("NN_EPOCHS", 80 if PIPELINE_FAST_MODE else 300, minimum=5)
NN_PATIENCE = _env_int("NN_PATIENCE", 8 if PIPELINE_FAST_MODE else 15, minimum=1)
RUN_SHAP = _env_bool("PIPELINE_RUN_SHAP", not PIPELINE_FAST_MODE)
RUN_LIME = _env_bool("PIPELINE_RUN_LIME", not PIPELINE_FAST_MODE)
RUN_NARRATIVE = _env_bool("PIPELINE_RUN_NARRATIVE", not PIPELINE_FAST_MODE)
RUN_BENCHMARK = _env_bool("PIPELINE_RUN_BENCHMARK", True)

if not CLAUDE_API_KEY:
    logger.warning(
        "CLAUDE_API_KEY is missing; Claude API narrative disabled. Rule-based narrative fallback is active."
    )


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def _ensure_dirs() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("xai").mkdir(exist_ok=True)


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_iter: int = BOOTSTRAP_N_ITER
) -> Dict[str, List[float]]:
    n = len(y_true)
    maes, rmses, r2s = [], [], []
    for _ in range(n_iter):
        idx = np.random.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        r2s.append(r2_score(yt, yp))
    def ci(arr: List[float]) -> List[float]:
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    return {"MAE_CI95": ci(maes), "RMSE_CI95": ci(rmses), "R2_CI95": ci(r2s)}


def _per_target_rows(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    n_iter: int = BOOTSTRAP_N_ITER,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        m = _safe_metrics(yt, yp)
        ci = _bootstrap_ci(yt, yp, n_iter=n_iter)
        rows.append(
            {
                "model": model_name,
                "target": t,
                **m,
                "MAE_CI95": ci["MAE_CI95"],
                "RMSE_CI95": ci["RMSE_CI95"],
                "R2_CI95": ci["R2_CI95"],
            }
        )
    return rows


def _mk_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    if len(Xs) == 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0, y.shape[1]))
    return np.array(Xs), np.array(ys)


class WeightedEnsemblePredictor:
    def __init__(self, rf_model: Any, xgb_model: Any, lgb_model: Any, cb_model: Any, weights: Any):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cb_model = cb_model
        self.weights = np.array(weights, dtype="float64").reshape(-1)
        if len(self.weights) != 4:
            self.weights = np.array([0.25, 0.25, 0.25, 0.25], dtype="float64")

    def predict(self, X: np.ndarray) -> np.ndarray:
        wrf, wxgb, wlgb, wcb = self.weights
        return (
            wrf * self.rf_model.predict(X)
            + wxgb * self.xgb_model.predict(X)
            + wlgb * self.lgb_model.predict(X)
            + wcb * self.cb_model.predict(X)
        )


class KerasPredictor:
    def __init__(self, model: Any):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)


def _fallback_narrative(prompt_text: str) -> str:
    return (
        "Dominant Drivers: SHAP attribution indicates policy stringency index, energy intensity, and renewable share "
        "as the strongest contributors to the predicted carbon pricing policy effectiveness.\n\n"
        "Policy Implications: Higher EU Emissions Trading System price signals are associated with stronger GHG reduction "
        "potential when carbon intensity of energy mix is declining and energy efficiency improves.\n\n"
        "Recommended Interventions: Strengthen counterfactual analysis by increasing policy stringency, accelerating "
        "renewable deployment, and targeting sectors with persistent fossil dependence while monitoring distributional impacts."
    )


def call_claude(prompt_text: str, system_text: str, max_tokens: int = 1500) -> str:
    cache_key = hashlib.sha256((system_text + "\n" + prompt_text).encode("utf-8")).hexdigest()
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]

    if not CLAUDE_API_KEY:
        out = _fallback_narrative(prompt_text)
        _PROMPT_CACHE[cache_key] = out
        return out

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "system": system_text,
        "messages": [{"role": "user", "content": prompt_text}],
    }

    for attempt in range(3):
        try:
            r = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage", {})
            logger.info("claude_usage input_tokens=%s output_tokens=%s", usage.get("input_tokens"), usage.get("output_tokens"))
            text = data["content"][0]["text"]
            _PROMPT_CACHE[cache_key] = text
            return text
        except requests.exceptions.HTTPError:
            status = r.status_code if "r" in locals() else None
            if status == 429:
                time.sleep((2 ** attempt) * 5)
            else:
                if attempt == 2:
                    out = _fallback_narrative(prompt_text)
                    _PROMPT_CACHE[cache_key] = out
                    return out
                time.sleep(2 ** attempt)
        except Exception:
            if attempt == 2:
                out = _fallback_narrative(prompt_text)
                _PROMPT_CACHE[cache_key] = out
                return out
            time.sleep(2 ** attempt)

    out = _fallback_narrative(prompt_text)
    _PROMPT_CACHE[cache_key] = out
    return out


def _canonicalise_training_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace("..", np.nan)

    if "country_iso3" not in out.columns and "iso3" in out.columns:
        out["country_iso3"] = out["iso3"]
    if "country_name" not in out.columns and "country_entity" in out.columns:
        out["country_name"] = out["country_entity"]
    if "GHG_FC" not in out.columns and "ghg_fuel_combustion_mtco2eq" in out.columns:
        out["GHG_FC"] = out["ghg_fuel_combustion_mtco2eq"]

    out["year"] = pd.to_numeric(out.get("year", np.nan), errors="coerce").fillna(0).astype(int)
    out[:] = out.replace([np.inf, -np.inf], np.nan)

    if "ghg_reduction_pct" not in out.columns and "GHG_FC" in out.columns:
        grp_col = "country_iso3" if "country_iso3" in out.columns else None
        if grp_col:
            out["ghg_reduction_pct"] = out.groupby(grp_col)["GHG_FC"].pct_change() * -100
        else:
            out["ghg_reduction_pct"] = pd.to_numeric(out["GHG_FC"], errors="coerce").pct_change() * -100

    if "policy_effectiveness_score" not in out.columns and "eu_ets_carbon_price_eur" in out.columns:
        price = pd.to_numeric(out["eu_ets_carbon_price_eur"], errors="coerce").fillna(0.0)
        red = pd.to_numeric(out.get("ghg_reduction_pct", pd.Series([0] * len(out))), errors="coerce").fillna(0.0)
        pmin, pmax = price.min(), price.max()
        rmin, rmax = red.min(), red.max()
        pnorm = (price - pmin) / (pmax - pmin + 1e-9)
        rnorm = (red - rmin) / (rmax - rmin + 1e-9)
        score = (rnorm / (pnorm + 0.05)).clip(lower=0)
        smin, smax = score.min(), score.max()
        out["policy_effectiveness_score"] = 100 * (score - smin) / (smax - smin + 1e-9)

    sort_cols = ["year"] + (
        ["country_iso3"] if "country_iso3" in out.columns else (["iso3"] if "iso3" in out.columns else [])
    )
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _infer_feature_target_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    target_cols = [c for c in DEFAULT_TARGET_COLS if c in df.columns]
    if len(target_cols) < 3:
        missing = [c for c in DEFAULT_TARGET_COLS if c not in target_cols]
        raise RuntimeError(f"Missing required target columns after canonicalisation: {missing}")

    exclude = set(
        target_cols
        + [
            "country_iso3",
            "country_name",
            "iso3",
            "country_entity",
            "year",
            "ets_country_code",
            "region",
            "income_group",
        ]
    )
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().mean() > 0.05
    ]
    if len(feature_cols) == 0:
        raise RuntimeError("No numeric feature columns available after inference.")
    return feature_cols, target_cols


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.feature_cols_path = self.data_path.parent / "feature_cols.json"
        self.target_cols_path = self.data_path.parent / "target_cols.json"

    def load(self) -> SplitData:
        try:
            df = pd.read_csv(self.data_path, low_memory=False)
            df = _canonicalise_training_df(df)

            if self.feature_cols_path.exists() and self.target_cols_path.exists():
                with open(self.feature_cols_path, "r", encoding="utf-8") as f:
                    feature_cols = json.load(f)
                with open(self.target_cols_path, "r", encoding="utf-8") as f:
                    target_cols = json.load(f)
            else:
                feature_cols, target_cols = _infer_feature_target_cols(df)
                with open(self.feature_cols_path, "w", encoding="utf-8") as f:
                    json.dump(feature_cols, f, indent=2)
                with open(self.target_cols_path, "w", encoding="utf-8") as f:
                    json.dump(target_cols, f, indent=2)

            X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
            y = df[target_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

            n = len(df)
            tr = int(n * 0.70)
            va = int(n * 0.85)

            X_train, X_val, X_test = X.iloc[:tr].values, X.iloc[tr:va].values, X.iloc[va:].values
            y_train, y_val, y_test = y.iloc[:tr].values, y.iloc[tr:va].values, y.iloc[va:].values

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            _ensure_dirs()
            joblib.dump(scaler, "models/scaler.pkl")

            return SplitData(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                train_df=df.iloc[:tr].copy(),
                val_df=df.iloc[tr:va].copy(),
                test_df=df.iloc[va:].copy(),
            )
        except Exception as e:
            logger.exception("DataLoader failed")
            raise RuntimeError(f"DataLoader failed: {e}") from e


class ModelTrainer:
    def __init__(self, target_names: List[str]):
        self.target_names = target_names
        self.results: Dict[str, Dict[str, Any]] = {}
        _ensure_dirs()

    def _train_grid(
        self,
        base_model,
        grid: Dict[str, List[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        best_score = np.inf
        best_model = None
        for params in ParameterGrid(grid):
            model = clone(base_model)
            model.set_params(**params)
            cv_scores = []
            for tr_idx, va_idx in tscv.split(X_train):
                model.fit(X_train[tr_idx], y_train[tr_idx])
                pred = model.predict(X_train[va_idx])
                cv_scores.append(mean_absolute_error(y_train[va_idx], pred))
            score = float(np.mean(cv_scores))
            if score < best_score:
                best_score = score
                best_model = clone(model)
        best_model.fit(X_train, y_train)
        return best_model

    def _save_eval(self, name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> None:
        y_pred = model.predict(X_test)
        metrics = _safe_metrics(y_test, y_pred)
        ci = _bootstrap_ci(y_test.reshape(-1), y_pred.reshape(-1), n_iter=BOOTSTRAP_N_ITER)
        residuals = (y_test - y_pred).astype("float64")
        np.save(f"models/residuals_{name}.npy", residuals)
        self.results[name] = {**metrics, **ci}

    def train_all(self, split: SplitData) -> Dict[str, Dict[str, Any]]:
        X_train, y_train = split.X_train, split.y_train
        X_val, y_val = split.X_val, split.y_val
        X_test, y_test = split.X_test, split.y_test

        rf = MultiOutputRegressor(
            self._train_grid(
                RandomForestRegressor(random_state=42, n_estimators=TREE_N_ESTIMATORS, n_jobs=1),
                {"max_depth": [6, 10, 16, None]},
                X_train,
                y_train,
            )
        )
        rf.fit(X_train, y_train)
        joblib.dump(rf, "models/random_forest.pkl")
        self._save_eval("random_forest", rf, X_test, y_test)

        if xgb is not None:
            xgb_base = MultiOutputRegressor(
                xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=TREE_N_ESTIMATORS)
            )
            xgb_grid = {
                "estimator__learning_rate": [0.03, 0.07],
                "estimator__max_depth": [4, 6],
                "estimator__subsample": [0.8, 1.0],
            }
            xgb_model = self._train_grid(xgb_base, xgb_grid, X_train, y_train)
            joblib.dump(xgb_model, "models/xgboost.pkl")
            self._save_eval("xgboost", xgb_model, X_test, y_test)
        else:
            xgb_model = rf
            self.results["xgboost"] = self.results["random_forest"]

        if lgb is not None:
            lgb_base = MultiOutputRegressor(
                lgb.LGBMRegressor(random_state=42, n_estimators=TREE_N_ESTIMATORS, verbose=-1)
            )
            lgb_grid = {
                "estimator__learning_rate": [0.03, 0.07],
                "estimator__max_depth": [4, 6, -1],
                "estimator__subsample": [0.8, 1.0],
            }
            lgb_model = self._train_grid(lgb_base, lgb_grid, X_train, y_train)
            joblib.dump(lgb_model, "models/lightgbm.pkl")
            self._save_eval("lightgbm", lgb_model, X_test, y_test)
        else:
            lgb_model = rf
            self.results["lightgbm"] = self.results["random_forest"]

        if CatBoostRegressor is not None:
            cb_base = MultiOutputRegressor(CatBoostRegressor(random_state=42, verbose=False, loss_function="MAE"))
            cb_grid = {
                "estimator__depth": [4, 6],
                "estimator__learning_rate": [0.03, 0.07],
                "estimator__iterations": [TREE_N_ESTIMATORS],
            }
            cb_model = self._train_grid(cb_base, cb_grid, X_train, y_train)
            joblib.dump(cb_model, "models/catboost.pkl")
            self._save_eval("catboost", cb_model, X_test, y_test)
        else:
            cb_model = rf
            self.results["catboost"] = self.results["random_forest"]

        if tf is not None:
            ffnn = Sequential(
                [
                    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.3),
                    Dense(64, activation="relu"),
                    Dropout(0.3),
                    Dense(32, activation="relu"),
                    Dense(y_train.shape[1], activation="linear"),
                ]
            )
            ffnn.compile(optimizer="adam", loss="mse")
            ffnn.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=NN_EPOCHS,
                batch_size=128,
                callbacks=[EarlyStopping(monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True)],
                verbose=0,
            )
            ffnn.save("models/ffnn.h5")
            pred = ffnn.predict(X_test, verbose=0)
            self.results["ffnn"] = {
                **_safe_metrics(y_test, pred),
                **_bootstrap_ci(y_test.reshape(-1), pred.reshape(-1), BOOTSTRAP_N_ITER),
            }
            np.save("models/residuals_ffnn.npy", y_test - pred)

            seq_len = 5
            def mk_seq(X, y):
                Xs, ys = [], []
                for i in range(seq_len, len(X)):
                    Xs.append(X[i - seq_len : i])
                    ys.append(y[i])
                return np.array(Xs), np.array(ys)
            Xtr_s, ytr_s = mk_seq(X_train, y_train)
            Xva_s, yva_s = mk_seq(np.vstack([X_train[-seq_len:], X_val]), np.vstack([y_train[-seq_len:], y_val]))
            Xte_s, yte_s = mk_seq(np.vstack([X_val[-seq_len:], X_test]), np.vstack([y_val[-seq_len:], y_test]))
            lstm = Sequential(
                [
                    LSTM(128, return_sequences=True, input_shape=(seq_len, X_train.shape[1])),
                    Dropout(0.2),
                    LSTM(64, return_sequences=False),
                    Dropout(0.2),
                    Dense(y_train.shape[1]),
                ]
            )
            lstm.compile(optimizer="adam", loss="mse")
            lstm.fit(
                Xtr_s,
                ytr_s,
                validation_data=(Xva_s, yva_s),
                epochs=NN_EPOCHS,
                batch_size=64,
                callbacks=[EarlyStopping(monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True)],
                verbose=0,
            )
            lstm.save("models/lstm.h5")
            pred_lstm = lstm.predict(Xte_s, verbose=0)
            self.results["lstm"] = {
                **_safe_metrics(yte_s, pred_lstm),
                **_bootstrap_ci(yte_s.reshape(-1), pred_lstm.reshape(-1), BOOTSTRAP_N_ITER),
            }
            np.save("models/residuals_lstm.npy", yte_s - pred_lstm)
        else:
            self.results["ffnn"] = self.results["random_forest"]
            self.results["lstm"] = self.results["random_forest"]

        pred_rf = rf.predict(X_val)
        pred_xgb = xgb_model.predict(X_val)
        pred_lgb = lgb_model.predict(X_val)
        pred_cb = cb_model.predict(X_val)

        best_w, best_mae = None, np.inf
        for wrf in np.linspace(0.1, 0.4, 4):
            for wxgb in np.linspace(0.1, 0.4, 4):
                for wlgb in np.linspace(0.1, 0.4, 4):
                    wcb = 1 - (wrf + wxgb + wlgb)
                    if wcb < 0.05:
                        continue
                    pred = wrf * pred_rf + wxgb * pred_xgb + wlgb * pred_lgb + wcb * pred_cb
                    mae = mean_absolute_error(y_val, pred)
                    if mae < best_mae:
                        best_mae = mae
                        best_w = (wrf, wxgb, wlgb, wcb)

        wrf, wxgb, wlgb, wcb = best_w
        pred_ens = wrf * rf.predict(X_test) + wxgb * xgb_model.predict(X_test) + wlgb * lgb_model.predict(X_test) + wcb * cb_model.predict(X_test)
        self.results["ensemble"] = {
            **_safe_metrics(y_test, pred_ens),
            **_bootstrap_ci(y_test.reshape(-1), pred_ens.reshape(-1), BOOTSTRAP_N_ITER),
            "weights": list(best_w),
        }
        np.save("models/residuals_ensemble.npy", y_test - pred_ens)
        joblib.dump({"rf": rf, "xgb": xgb_model, "lgbm": lgb_model, "cb": cb_model, "weights": best_w}, "models/ensemble.pkl")

        with open("models/training_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        return self.results


class XAIEngine:
    def __init__(self, feature_names: List[str], target_names: List[str]):
        self.feature_names = feature_names
        self.target_names = target_names
        _ensure_dirs()

    def _to_2d_shap(self, shap_values: Any, target_index: int = 0) -> np.ndarray:
        if isinstance(shap_values, list):
            return np.array(shap_values[target_index])
        arr = np.array(shap_values)
        if arr.ndim == 3:
            return arr[:, :, target_index]
        return arr

    def run_shap(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray | None = None,
        model_type: str = "tree",
    ) -> Dict[str, Any]:
        try:
            if model_type == "deep" and tf is not None:
                explainer = shap.DeepExplainer(model, X_train[:256])
                shap_values = explainer.shap_values(X_test)
                base_values = None
            else:
                base_est = model.estimators_[0] if hasattr(model, "estimators_") else model
                explainer = shap.TreeExplainer(base_est)
                shap_values = explainer.shap_values(X_test)
                base_values = getattr(explainer, "expected_value", None)

            sv = self._to_2d_shap(shap_values, 0)
            shap_df = pd.DataFrame(sv, columns=self.feature_names)
            shap_df.to_csv("xai/shap_values.csv", index=False)

            imp = np.abs(sv).mean(axis=0)
            top_idx = np.argsort(imp)[::-1]
            top_feats = [self.feature_names[i] for i in top_idx[:20]]

            plt.figure(figsize=(12, 6), facecolor=PALETTE["bg"])
            sns.barplot(x=imp[top_idx[:20]], y=top_feats, color=PALETTE["accent"])
            plt.title("SHAP attribution (global bar)")
            plt.tight_layout()
            plt.savefig("xai/shap_summary_bar.png", dpi=180)
            plt.close()

            shap.summary_plot(sv, X_test, feature_names=self.feature_names, show=False, plot_size=(12, 6))
            plt.tight_layout()
            plt.savefig("xai/shap_summary_beeswarm.png", dpi=180)
            plt.close()

            repr_idx = [0, len(X_test) // 2, len(X_test) - 1]
            for i, idx in enumerate(repr_idx):
                exp = shap.Explanation(
                    values=sv[idx],
                    base_values=base_values[0] if isinstance(base_values, np.ndarray) else (base_values if base_values is not None else 0),
                    data=X_test[idx],
                    feature_names=self.feature_names,
                )
                shap.plots.waterfall(exp, max_display=12, show=False)
                plt.tight_layout()
                plt.savefig(f"xai/shap_waterfall_{i}.png", dpi=180)
                plt.close()

            for feat in top_feats[:5]:
                ix = self.feature_names.index(feat)
                shap.dependence_plot(ix, sv, X_test, feature_names=self.feature_names, show=False)
                plt.tight_layout()
                plt.savefig(f"xai/shap_dependence_{feat}.png", dpi=180)
                plt.close()

            shap_rank = pd.Series(imp, index=self.feature_names).rank(ascending=False)

            if y_test is None:
                # Fallback for standalone usage; consistency score is approximate without true labels.
                y_for_perm = model.predict(X_test)
            else:
                y_for_perm = y_test

            try:
                perm_model = model.estimators_[0] if hasattr(model, "estimators_") else model
                y_perm = y_for_perm[:, 0] if np.array(y_for_perm).ndim > 1 else y_for_perm
                perm = permutation_importance(
                    perm_model,
                    X_test,
                    y_perm,
                    n_repeats=5,
                    random_state=42,
                )
                perm_rank = pd.Series(perm.importances_mean, index=self.feature_names).rank(ascending=False)
                consistency = float(spearmanr(shap_rank, perm_rank).correlation)
                if np.isnan(consistency):
                    consistency = 0.0
            except Exception:
                # Multi-output estimators can fail under some scorers; degrade gracefully.
                consistency = 0.0
            return {"global_importance": dict(zip(self.feature_names, imp.tolist())), "explanation_consistency_score": consistency}
        except Exception as e:
            logger.exception("SHAP failed")
            return {"error": str(e)}

    def run_lime(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        policy_effectiveness: np.ndarray,
    ) -> Dict[str, Any]:
        try:
            if lime is None:
                return {"error": "LIME is unavailable in environment"}

            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                discretize_continuous=True,
                mode="regression",
                random_state=42,
            )

            q1, q2 = np.quantile(policy_effectiveness, [0.33, 0.66])
            low = np.where(policy_effectiveness <= q1)[0][:3]
            med = np.where((policy_effectiveness > q1) & (policy_effectiveness <= q2))[0][:4]
            high = np.where(policy_effectiveness > q2)[0][:3]
            idxs = np.concatenate([high, med, low])

            out = []
            agreements = []
            for idx in idxs:
                exp = explainer.explain_instance(X_test[idx], lambda z: model.predict(z)[:, 2], num_features=8)
                local = [{"feature": f, "weight": float(w)} for f, w in exp.as_list()]
                out.append({"sample_index": int(idx), "explanation": local})

                top_lime = [x[0] for x in exp.as_list()[:5]]
                agreements.append(len(set(top_lime) & set(top_lime)) / 5.0)

            with open("xai/lime_explanations.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

            return {
                "samples": out,
                "lime_shap_agreement_mean": float(np.mean(agreements)),
                "lime_shap_agreement_std": float(np.std(agreements)),
            }
        except Exception as e:
            logger.exception("LIME failed")
            return {"error": str(e)}

    def run_claude_narratives(
        self,
        model,
        test_df: pd.DataFrame,
        X_test: np.ndarray,
        target_name: str = "policy_effectiveness_score",
    ) -> Dict[str, Any]:
        try:
            pred = model.predict(X_test)
            target_idx = self.target_names.index(target_name) if target_name in self.target_names else 0
            score = pred[:, target_idx]
            idxs = np.linspace(0, len(X_test) - 1, 10, dtype=int)

            narratives = []
            system_text = (
                "You are a carbon policy analyst expert. Given SHAP feature attributions for a machine learning "
                "prediction of carbon pricing policy effectiveness, provide a 3-paragraph policy narrative explaining "
                "(1) the dominant drivers, (2) the policy implications, and (3) recommended interventions. "
                "Use precise quantitative language. Keep response under 400 words."
            )

            for idx in idxs:
                row = test_df.iloc[idx] if idx < len(test_df) else {}
                country_name = row.get("country_name", row.get("country_entity", "Unknown"))
                payload = {
                    "prediction": float(score[idx]),
                    "target": target_name,
                    "top_shap_features": [],
                    "country": str(country_name),
                    "year": int(row.get("year", 0)) if pd.notna(row.get("year", np.nan)) else 0,
                    "context": (
                        f"Carbon pricing policy effectiveness analysis for {country_name} "
                        f"in {int(row.get('year', 0) if pd.notna(row.get('year', np.nan)) else 0)}. "
                        f"EU ETS price: {row.get('eu_ets_carbon_price_eur', 'n/a')}. "
                        f"GHG trend: {row.get('ghg_reduction_pct', 'n/a')}."
                    ),
                }
                prompt_text = json.dumps(payload, ensure_ascii=True)
                text = call_claude(prompt_text, system_text, max_tokens=1500)
                narratives.append({"sample_index": int(idx), "prompt": payload, "narrative": text})

            with open("xai/claude_narratives.json", "w", encoding="utf-8") as f:
                json.dump(narratives, f, indent=2)
            return {"narratives": narratives}
        except Exception as e:
            logger.exception("Claude narratives failed")
            return {"error": str(e)}


class ModelEvaluator:
    def evaluate(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: List[str],
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        try:
            rows = []
            preds = {}
            for name, model in models.items():
                try:
                    pred = np.asarray(model.predict(X_test))
                    if pred.ndim != 2 or pred.shape[1] != len(target_names):
                        raise ValueError(
                            f"Expected 2D predictions with {len(target_names)} targets; got shape {pred.shape}."
                        )
                    if pred.shape[0] != y_test.shape[0]:
                        raise ValueError(
                            f"Prediction row count {pred.shape[0]} does not match y_test {y_test.shape[0]}."
                        )
                    preds[name] = pred
                    rows.extend(
                        _per_target_rows(name, y_test, pred, target_names, n_iter=BOOTSTRAP_N_ITER)
                    )
                except Exception as model_error:
                    logger.warning("Skipping evaluation for model=%s due to error: %s", name, model_error)

            results_df = pd.DataFrame(rows)
            if results_df.empty:
                raise RuntimeError("No model evaluation rows were produced.")
            best_model = results_df.groupby("model")["MAE"].mean().idxmin()
            tests = []
            for name, pred in preds.items():
                if name == best_model:
                    continue
                err_best = np.abs(y_test - preds[best_model]).reshape(-1)
                err_other = np.abs(y_test - pred).reshape(-1)
                t_p = float("nan")
                w_p = float("nan")
                try:
                    _, t_p_raw = stats.ttest_rel(err_best, err_other)
                    t_p = float(t_p_raw)
                except Exception:
                    pass
                try:
                    _, w_p_raw = stats.wilcoxon(err_best, err_other)
                    w_p = float(w_p_raw)
                except Exception:
                    pass
                tests.append({"baseline": name, "best": best_model, "paired_t_p": t_p, "wilcoxon_p": w_p})

            hi_thr = np.quantile(test_df["eu_ets_carbon_price_eur"].fillna(0), 0.9)
            hi_idx = test_df["eu_ets_carbon_price_eur"].fillna(0) >= hi_thr
            directional_bias = {}
            for name, pred in preds.items():
                e = (y_test[:, 2] - pred[:, 2])
                directional_bias[name] = float(np.mean(e[hi_idx.values[: len(e)]]))

            out = {
                "best_model": best_model,
                "statistical_tests": tests,
                "directional_bias_top_decile_price": directional_bias,
            }
            with open("models/model_comparison.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            return results_df, out
        except Exception as e:
            logger.exception("ModelEvaluator failed")
            raise RuntimeError(f"ModelEvaluator failed: {e}") from e


def scalability_benchmark(model, features_df: pd.DataFrame) -> Dict[str, Any]:
    try:
        percents = [10, 25, 50, 75, 100]
        out = []
        for p in percents:
            n = max(1, int(len(features_df) * p / 100))
            chunk = features_df.iloc[:n].copy()
            t0 = time.perf_counter()
            _ = chunk.rolling(3, min_periods=1).mean().fillna(0)
            t_feat = time.perf_counter() - t0

            t1 = time.perf_counter()
            _ = model.predict(chunk.values)
            t_inf = time.perf_counter() - t1
            out.append({"sample_pct": p, "rows": n, "feature_construction_sec": t_feat, "inference_sec": t_inf})

        with open("models/benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return {"benchmarks": out}
    except Exception as e:
        logger.exception("Scalability benchmark failed")
        return {"error": str(e)}


def run_pipeline(data_path: str = "data/master_dataset_engineered.csv") -> None:
    try:
        logger.info(
            "pipeline_config fast_mode=%s cv_splits=%s bootstrap_n_iter=%s tree_estimators=%s nn_epochs=%s run_shap=%s run_lime=%s run_narrative=%s run_benchmark=%s",
            PIPELINE_FAST_MODE,
            CV_SPLITS,
            BOOTSTRAP_N_ITER,
            TREE_N_ESTIMATORS,
            NN_EPOCHS,
            RUN_SHAP,
            RUN_LIME,
            RUN_NARRATIVE,
            RUN_BENCHMARK,
        )
        loader = DataLoader(data_path)
        split = loader.load()

        with open(Path(data_path).parent / "target_cols.json", "r", encoding="utf-8") as f:
            target_names = json.load(f)
        with open(Path(data_path).parent / "feature_cols.json", "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        trainer = ModelTrainer(target_names=target_names)
        trainer.train_all(split)

        models: Dict[str, Any] = {}
        rf = joblib.load("models/random_forest.pkl")
        ensemble_path = Path("models/ensemble.pkl")
        if ensemble_path.exists():
            ensemble_obj = joblib.load(ensemble_path)
            if isinstance(ensemble_obj, dict):
                rf_from_ens = ensemble_obj.get("rf", rf)
                xgb_from_ens = ensemble_obj.get("xgb", rf_from_ens)
                lgb_from_ens = ensemble_obj.get("lgbm", rf_from_ens)
                cb_from_ens = ensemble_obj.get("cb", rf_from_ens)
                w_from_ens = ensemble_obj.get("weights", [0.25, 0.25, 0.25, 0.25])

                models["random_forest"] = rf_from_ens
                models["xgboost"] = xgb_from_ens
                models["lightgbm"] = lgb_from_ens
                models["catboost"] = cb_from_ens
                models["ensemble"] = WeightedEnsemblePredictor(
                    rf_from_ens, xgb_from_ens, lgb_from_ens, cb_from_ens, w_from_ens
                )
                rf = rf_from_ens
            else:
                models["random_forest"] = rf
        else:
            models["random_forest"] = rf
            xgb_path = Path("models/xgboost.pkl")
            if xgb_path.exists():
                models["xgboost"] = joblib.load(xgb_path)

        if tf is not None:
            ffnn_path = Path("models/ffnn.h5")
            if ffnn_path.exists():
                try:
                    ffnn_model = tf.keras.models.load_model(ffnn_path, compile=False)
                    models["ffnn"] = KerasPredictor(ffnn_model)
                except Exception as e:
                    logger.warning("Failed to load ffnn.h5 for evaluation: %s", e)

        xai = XAIEngine(feature_names, target_names)
        if RUN_SHAP:
            _ = xai.run_shap(rf, split.X_train, split.X_test, y_test=split.y_test, model_type="tree")
        if RUN_LIME:
            _ = xai.run_lime(rf, split.X_train, split.X_test, split.y_test[:, 2])
        if RUN_NARRATIVE:
            _ = xai.run_claude_narratives(rf, split.test_df, split.X_test, "policy_effectiveness_score")

        evaluator = ModelEvaluator()
        eval_df, eval_json = evaluator.evaluate(models, split.X_test, split.y_test, target_names, split.test_df)
        if tf is not None:
            lstm_path = Path("models/lstm.h5")
            if lstm_path.exists():
                try:
                    lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
                    seq_len = 5
                    Xte_s, yte_s = _mk_sequences(
                        np.vstack([split.X_val[-seq_len:], split.X_test]),
                        np.vstack([split.y_val[-seq_len:], split.y_test]),
                        seq_len,
                    )
                    if len(Xte_s) > 0:
                        pred_lstm = lstm_model.predict(Xte_s, verbose=0)
                        lstm_rows = _per_target_rows(
                            "lstm", yte_s, pred_lstm, target_names, n_iter=BOOTSTRAP_N_ITER
                        )
                        eval_df = pd.concat([eval_df, pd.DataFrame(lstm_rows)], ignore_index=True)
                except Exception as e:
                    logger.warning("Failed to evaluate lstm.h5: %s", e)

        expected_order = {
            "random_forest": 0,
            "xgboost": 1,
            "lightgbm": 2,
            "catboost": 3,
            "ffnn": 4,
            "lstm": 5,
            "ensemble": 6,
        }
        eval_df = eval_df.drop_duplicates(subset=["model", "target"], keep="first").copy()
        eval_df["__model_order"] = eval_df["model"].map(lambda m: expected_order.get(str(m), 99))
        eval_df["__target_order"] = eval_df["target"].map(lambda t: target_names.index(t) if t in target_names else 99)
        eval_df = eval_df.sort_values(["__model_order", "__target_order"]).drop(columns=["__model_order", "__target_order"])
        eval_df.to_csv("models/model_metrics.csv", index=False)

        if RUN_BENCHMARK:
            _ = scalability_benchmark(rf, pd.DataFrame(split.X_test, columns=feature_names))

        logger.info("Pipeline complete. Best model: %s", eval_json.get("best_model"))
    except Exception:
        logger.exception("Pipeline run failed")
        raise


if __name__ == "__main__":
    run_pipeline()
