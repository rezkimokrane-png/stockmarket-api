"""
data.py — yfinance fetcher with smart TTL cache
Prices cached 5min, fundamentals 6h, history 1h
"""
import yfinance as yf
import pandas as pd
import numpy as np
from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from datetime import datetime
import threading

# ── Caches ──────────────────────────────────────────────
_lock_price = threading.Lock()
_lock_info  = threading.Lock()
_lock_hist  = threading.Lock()

_cache_price = TTLCache(maxsize=200, ttl=300)    # 5 min
_cache_info  = TTLCache(maxsize=200, ttl=21600)  # 6 h
_cache_hist  = TTLCache(maxsize=200, ttl=3600)   # 1 h

# ── Index tickers ────────────────────────────────────────
INDEX_MAP = {
    "CAC 40":        {"ticker": "^FCHI",  "reg": "France · Euronext",    "r": "europe"},
    "DAX 40":        {"ticker": "^GDAXI", "reg": "Germany · Xetra",      "r": "europe"},
    "FTSE 100":      {"ticker": "^FTSE",  "reg": "UK · LSE",             "r": "europe"},
    "SMI":           {"ticker": "^SSMI",  "reg": "Switzerland · SIX",    "r": "europe"},
    "IBEX 35":       {"ticker": "^IBEX",  "reg": "Spain · BME",          "r": "europe"},
    "S&P 500":       {"ticker": "^GSPC",  "reg": "USA · NYSE",           "r": "us"},
    "Nasdaq 100":    {"ticker": "^NDX",   "reg": "USA · Nasdaq",         "r": "us"},
    "Dow Jones":     {"ticker": "^DJI",   "reg": "USA · NYSE",           "r": "us"},
    "Nikkei 225":    {"ticker": "^N225",  "reg": "Japan · TSE",          "r": "asia"},
    "Hang Seng":     {"ticker": "^HSI",   "reg": "Hong Kong · HKEX",     "r": "asia"},
    "Sensex":        {"ticker": "^BSESN", "reg": "India · BSE",          "r": "asia"},
    "ASX 200":       {"ticker": "^AXJO",  "reg": "Australia · ASX",      "r": "asia"},
    "Gold (XAU/USD)":{"ticker": "GC=F",   "reg": "Commodity",            "r": "commodity"},
    "Brent Crude":   {"ticker": "BZ=F",   "reg": "Commodity",            "r": "commodity"},
    "EUR/USD":       {"ticker": "EURUSD=X","reg": "Forex",               "r": "commodity"},
}

STOCK_UNIVERSE = {
    # CAC 40
    "LVMH":        {"ticker": "MC.PA",      "sec": "Luxury",           "cat": "cac"},
    "TotalEnergies":{"ticker": "TTE.PA",    "sec": "Energy",           "cat": "cac"},
    "Sanofi":      {"ticker": "SAN.PA",     "sec": "Pharma",           "cat": "cac"},
    "BNP Paribas": {"ticker": "BNP.PA",     "sec": "Banking",          "cat": "cac"},
    "Airbus":      {"ticker": "AIR.PA",     "sec": "Aerospace",        "cat": "cac"},
    # S&P 500
    "Apple":       {"ticker": "AAPL",       "sec": "Technology",       "cat": "sp500"},
    "Microsoft":   {"ticker": "MSFT",       "sec": "Technology",       "cat": "sp500"},
    "Tesla":       {"ticker": "TSLA",       "sec": "Auto/Tech",        "cat": "sp500"},
    # Nasdaq
    "Nvidia":      {"ticker": "NVDA",       "sec": "Semiconductors",   "cat": "nasdaq"},
    "Amazon":      {"ticker": "AMZN",       "sec": "E-commerce/Cloud", "cat": "nasdaq"},
    "Meta":        {"ticker": "META",       "sec": "Tech/Social",      "cat": "nasdaq"},
    # World
    "Nestlé":      {"ticker": "NESN.SW",    "sec": "Food & Beverage",  "cat": "world"},
    "Samsung":     {"ticker": "005930.KS",  "sec": "Tech/Semi",        "cat": "world"},
}


def _fmt_price(v, ticker=""):
    """Format price with currency symbol."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if ".PA" in ticker or "AIR" in ticker:
        return f"€{v:,.2f}"
    if "NESN" in ticker:
        return f"CHF {v:,.2f}"
    if "005930" in ticker:
        return f"₩{v:,.0f}"
    if "EURUSD" in ticker:
        return f"{v:.4f}"
    if "=F" in ticker or "=X" in ticker:
        return f"${v:,.2f}"
    if ticker.startswith("^"):
        return f"{v:,.2f}"
    return f"${v:,.2f}"


def _fmt_cap(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


def _safe(v, default=0.0):
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else round(f, 2)
    except:
        return default


def get_history(ticker: str, period: str = "3mo") -> pd.Series:
    key = hashkey(ticker, period)
    with _lock_hist:
        if key in _cache_hist:
            return _cache_hist[key]
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)["Close"].dropna()
        with _lock_hist:
            _cache_hist[key] = hist
        return hist
    except:
        return pd.Series(dtype=float)


def get_price_data(ticker: str) -> dict:
    key = hashkey(ticker)
    with _lock_price:
        if key in _cache_price:
            return _cache_price[key]
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if hist.empty:
            return {}
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else last
        chg = ((last - prev) / prev) * 100
        result = {
            "price": last,
            "change_pct": chg,
            "up": chg >= 0,
        }
        with _lock_price:
            _cache_price[key] = result
        return result
    except:
        return {}


def get_fundamentals(ticker: str) -> dict:
    key = hashkey(ticker)
    with _lock_info:
        if key in _cache_info:
            return _cache_info[key]
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        result = {
            "market_cap":        info.get("marketCap"),
            "pe_ratio":          _safe(info.get("trailingPE"), 0),
            "pb_ratio":          _safe(info.get("priceToBook"), 0),
            "roe":               _safe(info.get("returnOnEquity", 0) * 100, 0),
            "debt_equity":       _safe(info.get("debtToEquity", 0) / 100, 0),
            "net_margin":        _safe(info.get("profitMargins", 0) * 100, 0),
            "dividend_yield":    _safe(info.get("dividendYield", 0) * 100, 0),
            "revenue":           info.get("totalRevenue"),
            "sector":            info.get("sector", ""),
            "industry":          info.get("industry", ""),
            "short_name":        info.get("shortName", ""),
            "currency":          info.get("currency", "USD"),
        }
        with _lock_info:
            _cache_info[key] = result
        return result
    except:
        return {}


def get_income_history(ticker: str) -> dict:
    """Get 4 years of revenue and net income."""
    key = hashkey("income", ticker)
    with _lock_info:
        if key in _cache_info:
            return _cache_info[key]
    try:
        t = yf.Ticker(ticker)
        fin = t.financials  # annual
        if fin is None or fin.empty:
            return {}
        rev_row = None
        ni_row  = None
        for idx in fin.index:
            il = str(idx).lower()
            if "total revenue" in il:
                rev_row = idx
            if "net income" in il:
                ni_row = idx

        def fmt_row(row):
            if row is None or row not in fin.index:
                return ["—", "—", "—", "—"]
            vals = fin.loc[row].head(4)
            return [f"{v/1e9:.1f}" if not (np.isnan(v) if isinstance(v, float) else False) else "—" for v in vals]

        result = {
            "revenue":    fmt_row(rev_row),
            "net_income": fmt_row(ni_row),
        }
        # EPS
        try:
            eps_hist = t.earnings_history
            if eps_hist is not None and not eps_hist.empty:
                eps_vals = eps_hist["epsActual"].head(4).tolist()
                result["eps"] = [f"{v:.2f}" if v else "—" for v in eps_vals]
            else:
                result["eps"] = ["—","—","—","—"]
        except:
            result["eps"] = ["—","—","—","—"]
        # FCF
        try:
            cf = t.cashflow
            fcf_row = None
            for idx in cf.index:
                if "free cash flow" in str(idx).lower():
                    fcf_row = idx
                    break
            if fcf_row:
                vals = cf.loc[fcf_row].head(4)
                result["fcf"] = [f"{v/1e9:.1f}" if not np.isnan(v) else "—" for v in vals]
            else:
                result["fcf"] = ["—","—","—","—"]
        except:
            result["fcf"] = ["—","—","—","—"]

        with _lock_info:
            _cache_info[key] = result
        return result
    except:
        return {}


def compute_snowflake_score(f: dict) -> int:
    """0-100 Snowflake score from fundamentals."""
    score = 50  # base
    pe   = f.get("pe_ratio", 0)
    roe  = f.get("roe", 0)
    de   = f.get("debt_equity", 999)
    mgn  = f.get("net_margin", 0)
    div  = f.get("dividend_yield", 0)

    # Valuation (0-25)
    if   pe > 0 and pe < 15:  score += 12
    elif pe > 0 and pe < 25:  score += 8
    elif pe > 0 and pe < 40:  score += 3
    elif pe >= 40:             score -= 5

    # Profitability (0-25)
    if   roe > 20:  score += 12
    elif roe > 10:  score += 7
    elif roe > 0:   score += 2

    if   mgn > 25:  score += 10
    elif mgn > 12:  score += 6
    elif mgn > 5:   score += 3

    # Financial health (0-25)
    if   de < 0.3:  score += 10
    elif de < 0.7:  score += 6
    elif de < 1.5:  score += 2
    else:           score -= 5

    # Dividend bonus (0-25)
    if   div > 4:   score += 5
    elif div > 2:   score += 3
    elif div > 0:   score += 1

    return max(0, min(100, score))


def get_sparkline(ticker: str, n: int = 11) -> list:
    """Return n normalised price points (0-40) for sparkline."""
    hist = get_history(ticker, "1mo")
    if hist.empty:
        return list(range(n))
    # Resample to n points
    idx  = np.linspace(0, len(hist)-1, n, dtype=int)
    vals = hist.iloc[idx].values.tolist()
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [20] * n
    norm = [round(((v - mn) / (mx - mn)) * 38 + 1, 1) for v in vals]
    return norm


def verdict_from_score_and_signal(score: int, rsi: float, pe: float) -> str:
    if score >= 75 and rsi < 70:
        return "buy"
    if score <= 45 or rsi > 75:
        return "sell"
    return "hold"


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return round(float(val), 1) if not np.isnan(val) else 50.0
