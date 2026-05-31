"""
data.py — yfinance fetcher with robust error handling and TTL cache
"""
import yfinance as yf
import pandas as pd
import numpy as np
from cachetools import TTLCache
from cachetools.keys import hashkey
import threading
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Caches ──────────────────────────────────────────────
_lock_price = threading.Lock()
_lock_info  = threading.Lock()
_lock_hist  = threading.Lock()

_cache_price = TTLCache(maxsize=200, ttl=300)
_cache_info  = TTLCache(maxsize=200, ttl=21600)
_cache_hist  = TTLCache(maxsize=200, ttl=3600)

# ── Index tickers ────────────────────────────────────────
INDEX_MAP = {
    "CAC 40":         {"ticker": "^FCHI",   "reg": "France · Euronext",   "r": "europe"},
    "DAX 40":         {"ticker": "^GDAXI",  "reg": "Germany · Xetra",     "r": "europe"},
    "FTSE 100":       {"ticker": "^FTSE",   "reg": "UK · LSE",            "r": "europe"},
    "SMI":            {"ticker": "^SSMI",   "reg": "Switzerland · SIX",   "r": "europe"},
    "IBEX 35":        {"ticker": "^IBEX",   "reg": "Spain · BME",         "r": "europe"},
    "S&P 500":        {"ticker": "^GSPC",   "reg": "USA · NYSE",          "r": "us"},
    "Nasdaq 100":     {"ticker": "^NDX",    "reg": "USA · Nasdaq",        "r": "us"},
    "Dow Jones":      {"ticker": "^DJI",    "reg": "USA · NYSE",          "r": "us"},
    "Nikkei 225":     {"ticker": "^N225",   "reg": "Japan · TSE",         "r": "asia"},
    "Hang Seng":      {"ticker": "^HSI",    "reg": "Hong Kong · HKEX",    "r": "asia"},
    "Sensex":         {"ticker": "^BSESN",  "reg": "India · BSE",         "r": "asia"},
    "ASX 200":        {"ticker": "^AXJO",   "reg": "Australia · ASX",     "r": "asia"},
    "Gold (XAU/USD)": {"ticker": "GC=F",    "reg": "Commodity",           "r": "commodity"},
    "Brent Crude":    {"ticker": "BZ=F",    "reg": "Commodity",           "r": "commodity"},
    "EUR/USD":        {"ticker": "EURUSD=X","reg": "Forex",               "r": "commodity"},
}

STOCK_UNIVERSE = {
    "LVMH":         {"ticker": "MC.PA",     "sec": "Luxury",           "cat": "cac"},
    "TotalEnergies":{"ticker": "TTE.PA",    "sec": "Energy",           "cat": "cac"},
    "Sanofi":       {"ticker": "SAN.PA",    "sec": "Pharma",           "cat": "cac"},
    "BNP Paribas":  {"ticker": "BNP.PA",    "sec": "Banking",          "cat": "cac"},
    "Airbus":       {"ticker": "AIR.PA",    "sec": "Aerospace",        "cat": "cac"},
    "Apple":        {"ticker": "AAPL",      "sec": "Technology",       "cat": "sp500"},
    "Microsoft":    {"ticker": "MSFT",      "sec": "Technology",       "cat": "sp500"},
    "Tesla":        {"ticker": "TSLA",      "sec": "Auto/Tech",        "cat": "sp500"},
    "Nvidia":       {"ticker": "NVDA",      "sec": "Semiconductors",   "cat": "nasdaq"},
    "Amazon":       {"ticker": "AMZN",      "sec": "E-commerce/Cloud", "cat": "nasdaq"},
    "Meta":         {"ticker": "META",      "sec": "Tech/Social",      "cat": "nasdaq"},
    "Nestlé":       {"ticker": "NESN.SW",   "sec": "Food & Beverage",  "cat": "world"},
    "Samsung":      {"ticker": "005930.KS", "sec": "Tech/Semi",        "cat": "world"},
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
        if ".PA" in ticker:
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
    except:
        return str(v)


def _fmt_cap(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        if v >= 1e9:
            return f"${v/1e9:.1f}B"
        return f"${v/1e6:.0f}M"
    except:
        return "N/A"


def get_history(ticker: str, period: str = "3mo") -> pd.Series:
    key = hashkey(ticker, period)
    with _lock_hist:
        if key in _cache_hist:
            return _cache_hist[key]
    try:
        log.info(f"Fetching history: {ticker} {period}")
        # Use yf.download() — more reliable on cloud servers
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            timeout=20,
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)
        close = df["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.squeeze()
        with _lock_hist:
            _cache_hist[key] = close
        return close
    except Exception as e:
        log.error(f"get_history error {ticker}: {e}")
        return pd.Series(dtype=float)


def get_price_data(ticker: str) -> dict:
    key = hashkey("price", ticker)
    with _lock_price:
        if key in _cache_price:
            return _cache_price[key]
    try:
        hist = get_history(ticker, "5d")
        if hist.empty:
            return {}
        last = float(hist.iloc[-1])
        prev = float(hist.iloc[-2]) if len(hist) > 1 else last
        chg  = ((last - prev) / prev) * 100
        result = {"price": last, "change_pct": chg, "up": chg >= 0}
        with _lock_price:
            _cache_price[key] = result
        return result
    except Exception as e:
        log.error(f"get_price_data error {ticker}: {e}")
        return {}


def get_fundamentals(ticker: str) -> dict:
    key = hashkey("fund", ticker)
    with _lock_info:
        if key in _cache_info:
            return _cache_info[key]
    try:
        log.info(f"Fetching fundamentals: {ticker}")
        t = yf.Ticker(ticker)
        # fast_info is lighter and less likely to be blocked
        fi = t.fast_info
        info = {}
        try:
            info = t.info or {}
        except:
            pass

        def _get(key, default=None):
            v = info.get(key)
            if v is None:
                try: v = getattr(fi, key, default)
                except: pass
            return v if v is not None else default

        roe_raw = _get("returnOnEquity", 0) or 0
        de_raw  = _get("debtToEquity",  0) or 0
        mg_raw  = _get("profitMargins", 0) or 0
        dy_raw  = _get("dividendYield", 0) or 0

        # fast_info market cap fallback
        mkt_cap = _get("marketCap")
        if mkt_cap is None:
            try: mkt_cap = fi.market_cap
            except: mkt_cap = None

        result = {
            "market_cap":     mkt_cap,
            "pe_ratio":       _safe(_get("trailingPE"), 0),
            "pb_ratio":       _safe(_get("priceToBook"), 0),
            "roe":            _safe(roe_raw * 100, 0),
            "debt_equity":    _safe(de_raw / 100 if de_raw > 5 else de_raw, 0),
            "net_margin":     _safe(mg_raw * 100, 0),
            "dividend_yield": _safe(dy_raw * 100, 0),
            "revenue":        _get("totalRevenue"),
            "sector":         _get("sector", ""),
            "currency":       _get("currency", "USD"),
        }
        with _lock_info:
            _cache_info[key] = result
        return result
    except Exception as e:
        log.error(f"get_fundamentals error {ticker}: {e}")
        return {}


def get_income_history(ticker: str) -> dict:
    key = hashkey("income", ticker)
    with _lock_info:
        if key in _cache_info:
            return _cache_info[key]
    try:
        t   = yf.Ticker(ticker)
        fin = t.financials
        if fin is None or fin.empty:
            return {"revenue": ["—","—","—","—"], "net_income": ["—","—","—","—"],
                    "eps": ["—","—","—","—"], "fcf": ["—","—","—","—"]}

        def find_row(keywords):
            for idx in fin.index:
                if any(k in str(idx).lower() for k in keywords):
                    return idx
            return None

        def fmt_row(row):
            if row is None or row not in fin.index:
                return ["—","—","—","—"]
            vals = fin.loc[row].head(4)
            return [f"{float(v)/1e9:.1f}" if pd.notna(v) else "—" for v in vals]

        rev_row = find_row(["total revenue"])
        ni_row  = find_row(["net income"])

        result = {
            "revenue":    fmt_row(rev_row),
            "net_income": fmt_row(ni_row),
            "eps":        ["—","—","—","—"],
            "fcf":        ["—","—","—","—"],
        }
        try:
            cf = t.cashflow
            fcf_row = find_row(["free cash flow"])
            if fcf_row and fcf_row in cf.index:
                vals = cf.loc[fcf_row].head(4)
                result["fcf"] = [f"{float(v)/1e9:.1f}" if pd.notna(v) else "—" for v in vals]
        except:
            pass

        with _lock_info:
            _cache_info[key] = result
        return result
    except Exception as e:
        log.error(f"get_income_history error {ticker}: {e}")
        return {"revenue": ["—","—","—","—"], "net_income": ["—","—","—","—"],
                "eps": ["—","—","—","—"], "fcf": ["—","—","—","—"]}


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


def verdict_from_score_and_signal(score: int, rsi: float, pe: float) -> str:
    if score >= 75 and rsi < 70:
        return "buy"
    if score <= 45 or rsi > 75:
        return "sell"
    return "hold"
