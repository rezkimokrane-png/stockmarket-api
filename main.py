"""
main.py — StockMarket Pro API
FastAPI backend with live yfinance data + real ARIMA forecasts
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import concurrent.futures
from typing import Optional
import numpy as np
import pandas as pd
import time

from data import (
    INDEX_MAP, STOCK_UNIVERSE,
    get_price_data, get_fundamentals, get_income_history,
    get_sparkline, get_history,
    compute_snowflake_score, compute_rsi,
    verdict_from_score_and_signal,
    _fmt_price, _fmt_cap, _safe,
)
from arima import forecast

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="StockMarket Pro API",
    description="Live market data + real ARIMA forecasts",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _get_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


# ── Signal scoring ────────────────────────────────────────
def _ts_signal(rsi: float, change_pct: float, score: int) -> str:
    if score >= 72 and rsi < 65 and change_pct >= 0:
        return "buy"
    if score <= 45 or rsi > 72 or (rsi < 35 and change_pct < -1):
        return "sell"
    return "hold"


def _sig_label(sig: str) -> str:
    return {"buy": "BUY", "sell": "SELL", "hold": "NEUTRAL"}.get(sig, "NEUTRAL")


# ── Health ────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "StockMarket Pro API v2.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}


# ── INDICES ───────────────────────────────────────────────
def _build_index(name: str) -> dict:
    meta   = INDEX_MAP[name]
    ticker = meta["ticker"]

    price  = get_price_data(ticker)
    hist   = get_history(ticker, "1mo")

    if hist.empty:
        raise ValueError(f"No data for {ticker}")

    price_val  = hist.iloc[-1]
    prev_val   = hist.iloc[-2] if len(hist) > 1 else price_val
    chg_pct    = ((price_val - prev_val) / prev_val) * 100
    up         = chg_pct >= 0

    rsi = compute_rsi(hist)

    # Sparkline (11 pts, normalised)
    spark = get_sparkline(ticker, 11)

    # Quick ARIMA forecast (D+5 and D+10 only, lightweight)
    hist_3m = get_history(ticker, "3mo")
    fc = forecast(ticker, hist_3m, horizons=[5, 10])

    j5  = fc.get("forecasts", {}).get("d5",  {}).get("price", price_val * 1.005)
    j10 = fc.get("forecasts", {}).get("d10", {}).get("price", price_val * 1.01)
    model_str = fc.get("model", "ARIMA(1,1,1)")

    # Pseudo-score for signal
    score = 60  # indices don't have fundamentals, use RSI-based scoring
    if rsi < 40:  score -= 15
    if rsi > 70:  score += 10
    if up:        score += 8

    sig = _ts_signal(rsi, chg_pct, score)

    # Format values
    if "=" in ticker:
        v_fmt = f"{price_val:.4f}" if "USD" in ticker else f"${price_val:,.2f}"
    elif ticker.startswith("^") and ("FCHI" in ticker or "GDAXI" in ticker or "SSMI" in ticker or "IBEX" in ticker):
        v_fmt = f"{price_val:,.2f}"
    else:
        v_fmt = f"{price_val:,.2f}"

    chg_fmt = f"{'+' if up else ''}{chg_pct:.2f}%"
    j5_fmt  = f"{j5:,.0f}" if j5 > 100 else f"{j5:.4f}"
    j10_fmt = f"{j10:,.0f}" if j10 > 100 else f"{j10:.4f}"

    # Compute forecast prediction string
    fc10_chg = ((j10 - price_val) / price_val) * 100
    pred_str = f"{'+' if fc10_chg >= 0 else ''}{fc10_chg:.1f}%"

    return {
        "n":     name,
        "r":     meta["r"],
        "reg":   meta["reg"],
        "v":     v_fmt,
        "c":     chg_fmt,
        "up":    up,
        "sig":   sig,
        "model": model_str,
        "pred":  pred_str,
        "j5":    j5_fmt,
        "j10":   j10_fmt,
        "rsi":   str(rsi),
        "pts":   spark,
        "tv":    f"https://fr.tradingview.com/symbols/{ticker.replace('^','').replace('=F','').replace('=X','')}",
        "inv":   f"https://fr.investing.com/search/?q={name}",
    }


@app.get("/indices")
async def get_indices(region: Optional[str] = None):
    """Get all live indices with ARIMA signals."""
    async def fetch_one(name):
        try:
            loop = _get_loop()
            return await loop.run_in_executor(executor, _build_index, name)
        except Exception as e:
            return None

    names = list(INDEX_MAP.keys())
    if region:
        names = [n for n in names if INDEX_MAP[n]["r"] == region]

    tasks   = [fetch_one(n) for n in names]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


@app.get("/index/{name}")
async def get_index(name: str):
    """Get single index with full ARIMA forecast."""
    if name not in INDEX_MAP:
        raise HTTPException(404, f"Index '{name}' not found")
    loop = _get_loop()
    result = await loop.run_in_executor(executor, _build_index, name)
    # Add full forecast
    ticker = INDEX_MAP[name]["ticker"]
    hist   = get_history(ticker, "3mo")
    fc     = forecast(ticker, hist, horizons=[5, 10, 20])
    result["forecast"] = fc
    return result


# ── STOCKS ───────────────────────────────────────────────
def _build_stock(name: str) -> dict:
    meta   = STOCK_UNIVERSE[name]
    ticker = meta["ticker"]

    hist   = get_history(ticker, "1mo")
    if hist.empty:
        raise ValueError(f"No data for {ticker}")

    price_val = hist.iloc[-1]
    prev_val  = hist.iloc[-2] if len(hist) > 1 else price_val
    chg_pct   = ((price_val - prev_val) / prev_val) * 100
    up        = chg_pct >= 0

    fund = get_fundamentals(ticker)
    rsi  = compute_rsi(hist)

    pe     = _safe(fund.get("pe_ratio"), 0)
    pb     = _safe(fund.get("pb_ratio"), 0)
    roe    = _safe(fund.get("roe"), 0)
    de     = _safe(fund.get("debt_equity"), 0)
    margin = _safe(fund.get("net_margin"), 0)
    div    = _safe(fund.get("dividend_yield"), 0)
    cap    = fund.get("market_cap")

    score   = compute_snowflake_score(fund)
    sig     = _ts_signal(rsi, chg_pct, score)
    verdict = verdict_from_score_and_signal(score, rsi, pe)

    spark = get_sparkline(ticker, 11)

    v_fmt   = _fmt_price(price_val, ticker)
    chg_fmt = f"{'+' if up else ''}{chg_pct:.2f}%"
    cap_fmt = _fmt_cap(cap)

    return {
        "n":       name,
        "tick":    ticker,
        "cat":     meta["cat"],
        "sec":     meta["sec"],
        "v":       v_fmt,
        "c":       chg_fmt,
        "up":      up,
        "cap":     cap_fmt,
        "per":     pe,
        "pb":      pb,
        "roe":     roe,
        "de":      de,
        "margin":  margin,
        "div":     div,
        "score":   score,
        "sig":     sig,
        "verdict": verdict,
        "pts":     spark,
        "rsi":     rsi,
        "moat":    "N/A",
        "risk":    "N/A",
        "tv":      f"https://fr.tradingview.com/symbols/{ticker.replace('.', '-')}/",
        "inv":     f"https://fr.investing.com/search/?q={name}",
    }


@app.get("/stocks")
async def get_stocks(cat: Optional[str] = None):
    """Get all live stocks with fundamentals."""
    names = list(STOCK_UNIVERSE.keys())
    if cat:
        names = [n for n in names if STOCK_UNIVERSE[n]["cat"] == cat]

    async def fetch_one(name):
        try:
            loop = _get_loop()
            return await loop.run_in_executor(executor, _build_stock, name)
        except Exception as e:
            return None

    tasks   = [fetch_one(n) for n in names]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


@app.get("/stock/{name}")
async def get_stock(name: str):
    """Get single stock with full fundamentals + financial history."""
    # Try exact match first, then fuzzy
    matched = None
    for k in STOCK_UNIVERSE:
        if k.lower() == name.lower() or STOCK_UNIVERSE[k]["ticker"].lower() == name.lower():
            matched = k
            break
    if not matched:
        raise HTTPException(404, f"Stock '{name}' not found")

    loop   = _get_loop()
    result = await loop.run_in_executor(executor, _build_stock, matched)

    # Add income history
    ticker  = STOCK_UNIVERSE[matched]["ticker"]
    income  = get_income_history(ticker)
    result["rev"] = income.get("revenue",    ["—","—","—","—"])
    result["ni"]  = income.get("net_income", ["—","—","—","—"])
    result["eps"] = income.get("eps",        ["—","—","—","—"])
    result["fcf"] = income.get("fcf",        ["—","—","—","—"])

    return result


# ── FORECAST ─────────────────────────────────────────────
@app.get("/forecast/{ticker}")
async def get_forecast(ticker: str, horizons: str = "5,10,20"):
    """
    Real ARIMA/SARIMA forecast for any ticker.
    Query: horizons=5,10,20
    """
    h_list = [int(x) for x in horizons.split(",") if x.strip().isdigit()]
    if not h_list:
        h_list = [5, 10, 20]

    loop = _get_loop()
    hist = await loop.run_in_executor(executor, get_history, ticker, "6mo")

    if hist.empty:
        raise HTTPException(404, f"No history for ticker '{ticker}'")

    result = await loop.run_in_executor(executor, forecast, ticker, hist, h_list)

    if "error" in result:
        raise HTTPException(500, result["error"])

    return result


# ── SEARCH ────────────────────────────────────────────────
@app.get("/search/{query}")
async def search(query: str):
    """Search stocks and indices by name or ticker."""
    q = query.lower().strip()

    matches = []
    for name, meta in STOCK_UNIVERSE.items():
        if q in name.lower() or q in meta["ticker"].lower():
            matches.append({
                "type":   "stock",
                "name":   name,
                "ticker": meta["ticker"],
                "sector": meta["sec"],
            })
    for name in INDEX_MAP:
        if q in name.lower() or q in INDEX_MAP[name]["ticker"].lower():
            matches.append({
                "type":   "index",
                "name":   name,
                "ticker": INDEX_MAP[name]["ticker"],
            })
    return matches[:10]


# ── NEWSLETTER ────────────────────────────────────────────
@app.get("/market-summary")
async def market_summary():
    """
    Weekly market summary: top movers, signals, quick ARIMA view.
    Used for newsletter generation.
    """
    loop = _get_loop()

    # Fetch 3 key indices
    key_indices = ["S&P 500", "CAC 40", "Nasdaq 100"]
    idx_tasks   = [loop.run_in_executor(executor, _build_index, n) for n in key_indices]
    idx_results = await asyncio.gather(*idx_tasks, return_exceptions=True)

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "indices":      [r for r in idx_results if not isinstance(r, Exception)],
    }
