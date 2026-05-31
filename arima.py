"""
arima.py — Real ARIMA / SARIMA forecasts with auto-order selection via AIC
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from cachetools import TTLCache
from cachetools.keys import hashkey
import threading
import warnings
warnings.filterwarnings("ignore")

_lock  = threading.Lock()
_cache = TTLCache(maxsize=100, ttl=3600)  # 1h cache


def _is_stationary(series: pd.Series, significance: float = 0.05) -> bool:
    try:
        p = adfuller(series.dropna(), autolag="AIC")[1]
        return p < significance
    except:
        return False


def _auto_arima_order(series: pd.Series):
    """
    Grid search ARIMA(p,d,q) orders by minimising AIC.
    p, q ∈ {0,1,2}, d ∈ {0,1}
    Returns (p, d, q) and model object.
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_model = None

    d = 0 if _is_stationary(series) else 1
    s = series.diff(d).dropna() if d else series

    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0:
                continue
            try:
                m = SARIMAX(series,
                            order=(p, d, q),
                            seasonal_order=(0, 0, 0, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
                if m.aic < best_aic:
                    best_aic   = m.aic
                    best_order = (p, d, q)
                    best_model = m
            except:
                pass

    return best_order, best_model


def _auto_sarima_order(series: pd.Series, m: int = 5):
    """
    Try SARIMA with weekly seasonality (m=5 trading days).
    Returns best model among ARIMA and SARIMA(p,d,q)(1,0,1,5).
    """
    base_order, base_model = _auto_arima_order(series)
    best_aic   = base_model.aic if base_model else np.inf
    best_model = base_model
    best_info  = {"order": base_order, "seasonal": False}

    d = base_order[1]
    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0:
                continue
            try:
                sm = SARIMAX(series,
                             order=(p, d, q),
                             seasonal_order=(1, 0, 1, m),
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False)
                if sm.aic < best_aic:
                    best_aic   = sm.aic
                    best_model = sm
                    best_info  = {"order": (p, d, q), "seasonal": True, "m": m}
            except:
                pass

    return best_info, best_model


def _confidence_intervals(forecast_result, steps: int, alpha_80=0.20, alpha_95=0.05):
    """Extract 80% and 95% confidence intervals from SARIMAX forecast."""
    try:
        ci_80 = forecast_result.conf_int(alpha=alpha_80)
        ci_95 = forecast_result.conf_int(alpha=alpha_95)
        result = []
        for i in range(min(steps, len(ci_80))):
            result.append({
                "ci80_low":  round(float(ci_80.iloc[i, 0]), 2),
                "ci80_high": round(float(ci_80.iloc[i, 1]), 2),
                "ci95_low":  round(float(ci_95.iloc[i, 0]), 2),
                "ci95_high": round(float(ci_95.iloc[i, 1]), 2),
            })
        return result
    except:
        return []


def _backtest_rmse(series: pd.Series, order: tuple, seasonal: bool, m: int = 5) -> float:
    """
    Walk-forward backtest over last 24 months.
    Returns RMSE% (relative to mean price).
    """
    try:
        n = min(len(series), 120)  # use up to 6 months
        train = series.iloc[:-20]
        test  = series.iloc[-20:]

        seasonal_order = (1, 0, 1, m) if seasonal else (0, 0, 0, 0)
        model = SARIMAX(train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
        fc = model.forecast(steps=len(test))
        rmse = np.sqrt(np.mean((fc.values - test.values) ** 2))
        return round((rmse / series.mean()) * 100, 2)
    except:
        return 0.0


def forecast(ticker: str, history: pd.Series, horizons: list = [5, 10, 20]) -> dict:
    """
    Full ARIMA forecast pipeline.
    Returns dict with model info, forecasts for each horizon, CIs, diagnostics.
    """
    key = hashkey(ticker)
    with _lock:
        if key in _cache:
            return _cache[key]

    if history is None or len(history) < 30:
        return {"error": "Insufficient history for ARIMA modelling"}

    try:
        series = history.copy()

        # Auto-select best model
        info, model = _auto_sarima_order(series)

        if model is None:
            return {"error": "Model fitting failed"}

        order    = info["order"]
        seasonal = info["seasonal"]
        m        = info.get("m", 5)

        # Forecast max horizon
        max_h    = max(horizons)
        fc_res   = model.get_forecast(steps=max_h)
        fc_mean  = fc_res.predicted_mean
        ci_list  = _confidence_intervals(fc_res, max_h)

        last_price = float(series.iloc[-1])

        forecasts = {}
        for h in horizons:
            if h <= len(fc_mean):
                pred_price = float(fc_mean.iloc[h - 1])
                change_pct = ((pred_price - last_price) / last_price) * 100
                ci = ci_list[h - 1] if h - 1 < len(ci_list) else {}
                forecasts[f"d{h}"] = {
                    "price":      round(pred_price, 2),
                    "change_pct": round(change_pct, 2),
                    "up":         change_pct >= 0,
                    "ci80_low":   ci.get("ci80_low"),
                    "ci80_high":  ci.get("ci80_high"),
                    "ci95_low":   ci.get("ci95_low"),
                    "ci95_high":  ci.get("ci95_high"),
                }

        # Sparkline pts (historical normalised 0-40 + forecast extension)
        spark_hist = series.iloc[-8:].values.tolist()
        spark_pred = [float(fc_mean.iloc[i]) for i in range(3)]
        all_pts    = spark_hist + spark_pred
        mn, mx     = min(all_pts), max(all_pts)
        def norm(v): return round(((v - mn) / (mx - mn + 1e-9)) * 38 + 1, 1)
        pts = [norm(v) for v in all_pts]

        # Diagnostics
        rmse = _backtest_rmse(series, order, seasonal, m)
        p_str = f"ARIMA({order[0]},{order[1]},{order[2]})"
        if seasonal:
            p_str = f"SARIMA({order[0]},{order[1]},{order[2]})({m})"

        result = {
            "ticker":        ticker,
            "model":         p_str,
            "aic":           round(model.aic, 1),
            "bic":           round(model.bic, 1),
            "backtest_rmse": rmse,
            "seasonal":      seasonal,
            "forecasts":     forecasts,
            "pts":           pts,
            "last_price":    last_price,
            "generated_at":  pd.Timestamp.now().isoformat(),
        }

        with _lock:
            _cache[key] = result
        return result

    except Exception as e:
        return {"error": str(e)}
