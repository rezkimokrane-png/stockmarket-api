"""
data.py — Multi-source market data fetcher
Primary: pandas_datareader (Stooq) — free, no API key, works on cloud
Fallback: hardcoded demo data
"""
import pandas as pd
import numpy as np
from cachetools import TTLCache
from cachetools.keys import hashkey
import threading
import logging
import requests
from io import StringIO
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_lock_hist  = threading.Lock()
_lock_info  = threading.Lock()
_cache_hist = TTLCache(maxsize=200, ttl=3600)
_cache_info = TTLCache(maxsize=200, ttl=21600)

# ── Ticker maps ──────────────────────────────────────────
# Stooq uses different ticker format
STOOQ_MAP = {
    "^FCHI":    "^fchi",    # CAC 40
    "^GDAXI":   "^dax",     # DAX
    "^FTSE":    "^ftse",    # FTSE 100
    "^SSMI":    "^ssmi",    # SMI
    "^IBEX":    "^ibex",    # IBEX
    "^GSPC":    "^spx",     # S&P 500
    "^NDX":     "^ndx",     # Nasdaq 100
    "^DJI":     "^dji",     # Dow Jones
    "^N225":    "^n225",    # Nikkei
    "^HSI":     "^hsi",     # Hang Seng
    "^BSESN":   "^bsesn",  # Sensex
    "^AXJO":    "^axjo",    # ASX 200
    "GC=F":     "gc.f",     # Gold
    "BZ=F":     "bz.f",     # Brent
    "EURUSD=X": "eurusd",   # EUR/USD
    "MC.PA":    "mc.fr",    # LVMH
    "TTE.PA":   "tte.fr",   # Total
    "SAN.PA":   "san.fr",   # Sanofi
    "BNP.PA":   "bnp.fr",   # BNP
    "AIR.PA":   "air.fr",   # Airbus
    "AAPL":     "aapl.us",  # Apple
    "MSFT":     "msft.us",  # Microsoft
    "TSLA":     "tsla.us",  # Tesla
    "NVDA":     "nvda.us",  # Nvidia
    "AMZN":     "amzn.us",  # Amazon
    "META":     "meta.us",  # Meta
    "NESN.SW":  "nesn.ch",  # Nestlé
    "005930.KS":"005930.kr",# Samsung
}

INDEX_MAP = {
    "CAC 40":         {"ticker": "^FCHI",    "reg": "France · Euronext",   "r": "europe"},
    "DAX 40":         {"ticker": "^GDAXI",   "reg": "Germany · Xetra",     "r": "europe"},
    "FTSE 100":       {"ticker": "^FTSE",    "reg": "UK · LSE",            "r": "europe"},
    "SMI":            {"ticker": "^SSMI",    "reg": "Switzerland · SIX",   "r": "europe"},
    "IBEX 35":        {"ticker": "^IBEX",    "reg": "Spain · BME",         "r": "europe"},
    "S&P 500":        {"ticker": "^GSPC",    "reg": "USA · NYSE",          "r": "us"},
    "Nasdaq 100":     {"ticker": "^NDX",     "reg": "USA · Nasdaq",        "r": "us"},
    "Dow Jones":      {"ticker": "^DJI",     "reg": "USA · NYSE",          "r": "us"},
    "Nikkei 225":     {"ticker": "^N225",    "reg": "Japan · TSE",         "r": "asia"},
    "Hang Seng":      {"ticker": "^HSI",     "reg": "Hong Kong · HKEX",    "r": "asia"},
    "Sensex":         {"ticker": "^BSESN",   "reg": "India · BSE",         "r": "asia"},
    "ASX 200":        {"ticker": "^AXJO",    "reg": "Australia · ASX",     "r": "asia"},
    "Gold (XAU/USD)": {"ticker": "GC=F",     "reg": "Commodity",           "r": "commodity"},
    "Brent Crude":    {"ticker": "BZ=F",     "reg": "Commodity",           "r": "commodity"},
    "EUR/USD":        {"ticker": "EURUSD=X", "reg": "Forex",               "r": "commodity"},
}

STOCK_UNIVERSE = {
    "LVMH":          {"ticker": "MC.PA",     "sec": "Luxury",           "cat": "cac"},
    "TotalEnergies": {"ticker": "TTE.PA",    "sec": "Energy",           "cat": "cac"},
    "Sanofi":        {"ticker": "SAN.PA",    "sec": "Pharma",           "cat": "cac"},
    "BNP Paribas":   {"ticker": "BNP.PA",    "sec": "Banking",          "cat": "cac"},
    "Airbus":        {"ticker": "AIR.PA",    "sec": "Aerospace",        "cat": "cac"},
    "Apple":         {"ticker": "AAPL",      "sec": "Technology",       "cat": "sp500"},
    "Microsoft":     {"ticker": "MSFT",      "sec": "Technology",       "cat": "sp500"},
    "Tesla":         {"ticker": "TSLA",      "sec": "Auto/Tech",        "cat": "sp500"},
    "Nvidia":        {"ticker": "NVDA",      "sec": "Semiconductors",   "cat": "nasdaq"},
    "Amazon":        {"ticker": "AMZN",      "sec": "E-commerce/Cloud", "cat": "nasdaq"},
    "Meta":          {"ticker": "META",      "sec": "Tech/Social",      "cat": "nasdaq"},
    "Nestlé":        {"ticker": "NESN.SW",   "sec": "Food & Beverage",  "cat": "world"},
    "Samsung":       {"ticker": "005930.KS", "sec": "Tech/Semi",        "cat": "world"},
}

# ── Demo fallback data (when all sources fail) ───────────
DEMO_PRICES = {
    "^FCHI": 7842.15, "^GDAXI": 18220.33, "^FTSE": 8312.44,
    "^SSMI": 13220.17, "^IBEX": 11432.20, "^GSPC": 5302.47,
    "^NDX": 18643.90, "^DJI": 39180.55, "^N225": 38460.08,
    "^HSI": 17764.50, "^BSESN": 74302.60, "^AXJO": 7840.10,
    "GC=F": 2418.50, "BZ=F": 83.42, "EURUSD=X": 1.0842,
    "MC.PA": 624.80, "TTE.PA": 58.32, "SAN.PA": 92.14,
    "BNP.PA": 68.42, "AIR.PA": 162.44, "AAPL": 192.53,
    "MSFT": 415.32, "TSLA": 176.88, "NVDA": 875.40,
    "AMZN": 182.66, "META": 492.36, "NESN.SW": 78.42,
    "005930.KS": 78400,
}


def _stooq_fetch(ticker: str, days: int = 90) -> pd.Series:
    """Fetch from Stooq — free, no API key, works on cloud servers."""
    stooq_ticker = STOOQ_MAP.get(ticker, ticker.lower())
    end   = datetime.now()
    start = end - timedelta(days=days)
    url   = (
        f"https://stooq.com/q/d/l/"
        f"?s={stooq_ticker}"
        f"&d1={start.strftime('%Y%m%d')}"
        f"&d2={end.strftime('%Y%m%d')}"
        f"&i=d"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    if "No data" in resp.text or len(resp.text) < 50:
        raise ValueError(f"No data from Stooq for {stooq_ticker}")
    df = pd.read_csv(StringIO(resp.text), parse_dates=["Date"])
    df = df.sort_values("Date")
    return df.set_index("Date")["Close"].dropna()


def _demo_series(ticker: str, n: int = 60) -> pd.Series:
    """Generate realistic demo price series when all sources fail."""
    base  = DEMO_PRICES.get(ticker, 100.0)
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")
    np.random.seed(abs(hash(ticker)) % 9999)
    returns = np.random.normal(0.0003, 0.012, n)
    prices  = base * np.exp(np.cumsum(returns))
    prices[-1] = base  # anchor last point to known price
    return pd.Series(prices, index=dates)


def get_history(ticker: str, period: str = "3mo") -> pd.Series:
    days_map = {"5d": 7, "1mo": 35, "3mo": 95, "6mo": 185}
    days = days_map.get(period, 95)
    key  = hashkey(ticker, period)

    with _lock_hist:
        if key in _cache_hist:
            return _cache_hist[key]

    # Try Stooq first
    try:
        log.info(f"Stooq fetch: {ticker}")
        series = _stooq_fetch(ticker, days)
        if len(series) >= 5:
            with _lock_hist:
                _cache_hist[key] = series
            return series
    except Exception as e:
        log.warning(f"Stooq failed for {ticker}: {e}")

    # Fallback: demo data
    log.info(f"Using demo data for {ticker}")
    series = _demo_series(ticker, min(days, 60))
    with _lock_hist:
        _cache_hist[key] = series
    return series


def get_fundamentals(ticker: str) -> dict:
    """Return fundamentals — try multiple sources, fallback to estimates."""
    key = hashkey("fund", ticker)
    with _lock_info:
        if key in _cache_info:
            return _cache_info[key]

    # Try yfinance (sometimes works)
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        info = t.get_info() or {}
        if info and info.get("trailingPE"):
            roe_raw = info.get("returnOnEquity", 0) or 0
            de_raw  = info.get("debtToEquity",  0) or 0
            mg_raw  = info.get("profitMargins",  0) or 0
            dy_raw  = info.get("dividendYield",  0) or 0
            result = {
                "market_cap":     info.get("marketCap"),
                "pe_ratio":       _safe(info.get("trailingPE"), 0),
                "pb_ratio":       _safe(info.get("priceToBook"), 0),
                "roe":            _safe(roe_raw * 100, 0),
                "debt_equity":    _safe(de_raw / 100 if de_raw > 5 else de_raw, 0),
                "net_margin":     _safe(mg_raw * 100, 0),
                "dividend_yield": _safe(dy_raw * 100, 0),
                "revenue":        info.get("totalRevenue"),
                "sector":         info.get("sector", ""),
                "currency":       info.get("currency", "USD"),
            }
            with _lock_info:
                _cache_info[key] = result
            return result
    except Exception as e:
        log.warning(f"yfinance fundamentals failed for {ticker}: {e}")

    # Fallback estimates based on known data
    FUND_FALLBACK = {
        "AAPL":     {"pe_ratio":31.2,"pb_ratio":48.6,"roe":160.8,"debt_equity":1.77,"net_margin":25.3,"dividend_yield":0.5,"market_cap":2980000000000},
        "MSFT":     {"pe_ratio":36.4,"pb_ratio":14.2,"roe":38.4,"debt_equity":0.36,"net_margin":36.4,"dividend_yield":0.8,"market_cap":3090000000000},
        "NVDA":     {"pe_ratio":68.4,"pb_ratio":38.2,"roe":55.8,"debt_equity":0.41,"net_margin":55.0,"dividend_yield":0.1,"market_cap":2160000000000},
        "TSLA":     {"pe_ratio":58.4,"pb_ratio":8.2, "roe":13.4,"debt_equity":0.08,"net_margin":8.2, "dividend_yield":0.0,"market_cap":564000000000},
        "AMZN":     {"pe_ratio":42.8,"pb_ratio":8.6, "roe":20.4,"debt_equity":0.62,"net_margin":8.6, "dividend_yield":0.0,"market_cap":1920000000000},
        "META":     {"pe_ratio":24.8,"pb_ratio":7.2, "roe":30.8,"debt_equity":0.12,"net_margin":34.2,"dividend_yield":0.4,"market_cap":1260000000000},
        "MC.PA":    {"pe_ratio":22.4,"pb_ratio":5.8, "roe":18.2,"debt_equity":0.42,"net_margin":19.8,"dividend_yield":2.1,"market_cap":314000000000},
        "TTE.PA":   {"pe_ratio":8.6, "pb_ratio":1.4, "roe":16.8,"debt_equity":0.28,"net_margin":12.4,"dividend_yield":4.8,"market_cap":148000000000},
        "SAN.PA":   {"pe_ratio":15.2,"pb_ratio":2.1, "roe":13.6,"debt_equity":0.31,"net_margin":16.2,"dividend_yield":3.5,"market_cap":118000000000},
        "BNP.PA":   {"pe_ratio":7.8, "pb_ratio":0.72,"roe":9.8, "debt_equity":4.20,"net_margin":28.4,"dividend_yield":6.2,"market_cap":88000000000},
        "AIR.PA":   {"pe_ratio":24.8,"pb_ratio":8.4, "roe":34.2,"debt_equity":0.62,"net_margin":8.4, "dividend_yield":1.2,"market_cap":128000000000},
        "NESN.SW":  {"pe_ratio":18.4,"pb_ratio":5.2, "roe":28.4,"debt_equity":0.58,"net_margin":14.2,"dividend_yield":3.4,"market_cap":212000000000},
        "005930.KS":{"pe_ratio":16.8,"pb_ratio":1.42,"roe":8.42,"debt_equity":0.18,"net_margin":12.4,"dividend_yield":2.8,"market_cap":452000000000},
    }
    base = FUND_FALLBACK.get(ticker, {
        "pe_ratio":20.0,"pb_ratio":3.0,"roe":12.0,"debt_equity":0.5,
        "net_margin":10.0,"dividend_yield":2.0,"market_cap":50000000000
    })
    base["revenue"] = None
    base["sector"]  = ""
    base["currency"] = "USD"
    with _lock_info:
        _cache_info[key] = base
    return base


def get_income_history(ticker: str) -> dict:
    return {
        "revenue":    ["—","—","—","—"],
        "net_income": ["—","—","—","—"],
        "eps":        ["—","—","—","—"],
        "fcf":        ["—","—","—","—"],
    }


def _safe(v, default=0.0):
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else round(f, 2)
    except:
        return default


def _fmt_price(v, ticker=""):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if ".PA" in ticker or ".fr" in ticker:
            return f"€{v:,.2f}"
        if "NESN" in ticker or ".ch" in ticker:
            return f"CHF {v:,.2f}"
        if "005930" in ticker or ".kr" in ticker:
            return f"₩{v:,.0f}"
        if "EURUSD" in ticker:
            return f"{v:.4f}"
        if ticker.startswith("^"):
            return f"{v:,.2f}"
        return f"${v:,.2f}"
    except:
        return str(round(v, 2))


def _fmt_cap(v):
    try:
        if not v:
            return "N/A"
        v = float(v)
        if v >= 1e12: return f"${v/1e12:.2f}T"
        if v >= 1e9:  return f"${v/1e9:.1f}B"
        return f"${v/1e6:.0f}M"
    except:
        return "N/A"


def get_sparkline(ticker: str, n: int = 11) -> list:
    try:
        hist = get_history(ticker, "1mo")
        if hist.empty:
            return list(range(1, n+1))
        idx  = np.linspace(0, len(hist)-1, n, dtype=int)
        vals = [float(hist.iloc[i]) for i in idx]
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [20] * n
        return [round(((v - mn) / (mx - mn)) * 38 + 1, 1) for v in vals]
    except:
        return list(range(1, n+1))


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    try:
        if len(series) < period + 1:
            return 50.0
        delta = series.diff().dropna()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        val   = rsi.iloc[-1]
        return round(float(val), 1) if not np.isnan(val) else 50.0
    except:
        return 50.0


def compute_snowflake_score(f: dict) -> int:
    score = 50
    pe  = f.get("pe_ratio", 0)
    roe = f.get("roe", 0)
    de  = f.get("debt_equity", 999)
    mgn = f.get("net_margin", 0)
    div = f.get("dividend_yield", 0)
    if pe > 0 and pe < 15:   score += 12
    elif pe > 0 and pe < 25: score += 8
    elif pe > 0 and pe < 40: score += 3
    elif pe >= 40:            score -= 5
    if roe > 20:   score += 12
    elif roe > 10: score += 7
    elif roe > 0:  score += 2
    if mgn > 25:   score += 10
    elif mgn > 12: score += 6
    elif mgn > 5:  score += 3
    if de < 0.3:   score += 10
    elif de < 0.7: score += 6
    elif de < 1.5: score += 2
    else:          score -= 5
    if div > 4:    score += 5
    elif div > 2:  score += 3
    elif div > 0:  score += 1
    return max(0, min(100, score))


def verdict_from_score_and_signal(score: int, rsi: float, pe: float) -> str:
    if score >= 75 and rsi < 70:
        return "buy"
    if score <= 45 or rsi > 75:
        return "sell"
    return "hold"
