import os
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# EMR-6 Final
# ETF Momentum Rotation -> Stock Leader Rotation
# ------------------------------------------------------------
# 핵심 구조
# 1) 넓은 ETF 유니버스
# 2) ETF 모멘텀 점수 계산
# 3) ETF 중복 제거(corr cluster)
# 4) 최종 ETF Top15 선정
# 5) ETF holdings Top10 추출
# 6) 종목 Gate 필터
# 7) M/R/E/V 점수 계산
# 8) Top6 포트폴리오
# 9) 기존 보유 Top12 유지 규칙
# 10) 주간 자동 리밸런싱
#
# 출력
# - data/emr6_current_portfolio.json
# - data/emr6_rebalance_log.csv
# - data/emr6_history.csv
# - 텔레그램 리포트
# ============================================================


# ============================================================
# 경로 / 환경변수
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "emr6_holdings_cache")

CURRENT_PORTFOLIO_FILE = os.path.join(DATA_DIR, "emr6_current_portfolio.json")
REBALANCE_LOG_FILE = os.path.join(DATA_DIR, "emr6_rebalance_log.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "emr6_history.csv")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# ============================================================
# 전략 파라미터
# ============================================================
PRICE_HISTORY_PERIOD = "1y"
PRICE_INTERVAL = "1d"

ETF_TOP_N_RAW = 20
ETF_TOP_N_FINAL = 15
ETF_HOLDINGS_TOP_N = 10

PORTFOLIO_SIZE = 6
HOLD_BUFFER_RANK = 12

ETF_CORR_LOOKBACK_DAYS = 126
ETF_CORR_THRESHOLD = 0.90

MIN_STOCK_PRICE = 10.0
MIN_STOCK_DOLLAR_VOLUME = 20_000_000
MAX_20D_DRAWDOWN_FILTER = -0.20
ETF_OVERHEAT_20D = 0.25

HOLDINGS_CACHE_DAYS = 14

TELEGRAM_MAX_CHARS = 3500


# ============================================================
# ETF 유니버스
# 바이오 / 원자재 / 레버리지 / 인버스 제외
# ============================================================
ETF_UNIVERSE: Dict[str, str] = {
    # core sector
    "XLK": "Technology",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",

    # broad/style
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "MDY": "Mid Cap",
    "RSP": "Equal Weight S&P 500",
    "MTUM": "Momentum",
    "QUAL": "Quality",
    "USMV": "Low Volatility",

    # semis / software / tech sub
    "SMH": "Semiconductors",
    "SOXX": "Semiconductors 2",
    "XSD": "Semiconductors Equal Weight",
    "IGV": "Software",
    "CLOU": "Cloud",
    "SKYY": "Cloud Infra",
    "BOTZ": "Robotics & AI",
    "HACK": "Cybersecurity",
    "DRIV": "Autonomous Vehicles",

    # industrial / infra
    "PAVE": "Infrastructure",
    "ITA": "Aerospace Defense",
    "PKB": "Construction",
    "AIRR": "Industrial Innovation",
    "IYT": "Transportation",
    "XTN": "Transportation Equal Weight",
    "SEA": "Shipping",

    # financial sub
    "KBE": "Banks",
    "KIE": "Insurance",
    "IAI": "Broker Dealers",
    "IPAY": "Digital Payments",
    "FINX": "Fintech",

    # energy / utilities
    "XOP": "Oil & Gas Exploration",
    "OIH": "Oil Services",
    "VDE": "Energy Broad",
    "ICLN": "Clean Energy",
    "GRID": "Smart Grid",
    "XLU": "Utilities",

    # consumer / internet
    "XRT": "Retail",
    "FDIS": "Consumer Disc Alt",
    "ONLN": "Online Retail",
    "IBUY": "E-commerce",
    "SOCL": "Social Media",

    # materials / mining / metals
    "XME": "Metals & Mining",

    # international thematic but not commodity/biotech
    "KWEB": "China Internet",
    "EWJ": "Japan",
    "EWG": "Germany",
    "INDA": "India",
    "EWT": "Taiwan",
    "EZU": "Eurozone",
    "VGK": "Europe",
    "VWO": "Emerging Markets",

    # additional style / momentum-ish
    "SPMO": "S&P Momentum",
    "VUG": "Large Growth",
    "VTV": "Large Value",
    "SCHG": "US Large Growth",
    "SCHD": "Dividend Quality",
    "IWF": "Russell 1000 Growth",
    "IWD": "Russell 1000 Value",
    "VO": "Midcap",
    "VB": "Smallcap",
}


# ============================================================
# 데이터 구조
# ============================================================
@dataclass
class ETFScore:
    ticker: str
    name: str
    close: float
    ret_1m: float
    ret_3m: float
    ma50: float
    ma200: float
    ret_20d: float
    score: float


@dataclass
class StockScore:
    ticker: str
    name: str
    source_etfs: str
    close: float
    dollar_volume_20: float
    ret_1m: float
    ret_3m: float
    ret_6m: float
    rs_3m_vs_spy: float
    accel_proxy: float
    volume_ratio: float
    m_score: float
    r_score: float
    e_score: float
    v_score: float
    total_score: float
    rank: int = 0


# ============================================================
# 유틸
# ============================================================
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_message(text: str) -> None:
    if not telegram_enabled():
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, data=payload, timeout=20)
    except Exception:
        pass


def send_telegram_chunked(title: str, lines: List[str], max_chars: int = TELEGRAM_MAX_CHARS) -> None:
    if not lines:
        send_telegram_message(title)
        return

    chunks: List[str] = []
    current = title

    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_chars:
            chunks.append(current)
            current = f"{title}\n{line}"
        else:
            current = candidate

    if current:
        chunks.append(current)

    total = len(chunks)
    if total == 1:
        send_telegram_message(chunks[0])
        return

    for i, chunk in enumerate(chunks, start=1):
        body_lines = chunk.split("\n")[1:] if "\n" in chunk else []
        send_telegram_message(f"{title} ({i}/{total})\n" + "\n".join(body_lines))


def safe_float(x) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.2f}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def rolling_return(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def minmax_normalize(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() <= 1:
        return pd.Series([0.5] * len(v), index=v.index)
    vmin = v.min()
    vmax = v.max()
    if pd.isna(vmin) or pd.isna(vmax) or math.isclose(vmin, vmax):
        return pd.Series([0.5] * len(v), index=v.index)
    return (v - vmin) / (vmax - vmin)


# ============================================================
# 가격 데이터
# ============================================================
def normalize_downloaded(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "date":
            rename_map[c] = "Date"
        elif cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "volume":
            rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().sort_values("Date").reset_index(drop=True)


def download_history(ticker: str, period: str = PRICE_HISTORY_PERIOD) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=PRICE_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_downloaded(raw)


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma20"] = out["Close"].rolling(20).mean()
    out["ma50"] = out["Close"].rolling(50).mean()
    out["ma90"] = out["Close"].rolling(90).mean()
    out["ma200"] = out["Close"].rolling(200).mean()
    out["ret_20d"] = rolling_return(out["Close"], 20)
    out["ret_1m"] = rolling_return(out["Close"], 21)
    out["ret_3m"] = rolling_return(out["Close"], 63)
    out["ret_6m"] = rolling_return(out["Close"], 126)
    out["volume_ma20"] = out["Volume"].rolling(20).mean()
    out["volume_ma90"] = out["Volume"].rolling(90).mean()
    out["dollar_volume_20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    return out


# ============================================================
# ETF 모멘텀
# ============================================================
def build_etf_scores() -> Tuple[List[ETFScore], Dict[str, pd.DataFrame]]:
    rows: List[ETFScore] = []
    price_map: Dict[str, pd.DataFrame] = {}

    for ticker, name in ETF_UNIVERSE.items():
        try:
            df = download_history(ticker)
            if df.empty or len(df) < 220:
                continue

            df = add_basic_indicators(df)
            price_map[ticker] = df

            row = df.iloc[-1]
            close = safe_float(row["Close"])
            ma50 = safe_float(row["ma50"])
            ma200 = safe_float(row["ma200"])
            ret_1m = safe_float(row["ret_1m"])
            ret_3m = safe_float(row["ret_3m"])
            ret_20d = safe_float(row["ret_20d"])

            if close is None or ma50 is None or ma200 is None or ret_1m is None or ret_3m is None or ret_20d is None:
                continue

            if close <= ma50:
                continue
            if close <= ma200:
                continue
            if ret_20d > ETF_OVERHEAT_20D:
                continue

            score = 0.60 * ret_3m + 0.40 * ret_1m

            rows.append(
                ETFScore(
                    ticker=ticker,
                    name=name,
                    close=close,
                    ret_1m=ret_1m,
                    ret_3m=ret_3m,
                    ma50=ma50,
                    ma200=ma200,
                    ret_20d=ret_20d,
                    score=score,
                )
            )
        except Exception:
            continue

    rows = sorted(rows, key=lambda x: x.score, reverse=True)
    return rows, price_map


def filter_etf_correlation(top_etfs: List[ETFScore], price_map: Dict[str, pd.DataFrame]) -> List[ETFScore]:
    selected: List[ETFScore] = []

    for candidate in top_etfs:
        keep = True
        c_df = price_map.get(candidate.ticker)
        if c_df is None or c_df.empty:
            continue

        c_ret = c_df["Close"].pct_change().dropna().tail(ETF_CORR_LOOKBACK_DAYS)

        for chosen in selected:
            s_df = price_map.get(chosen.ticker)
            if s_df is None or s_df.empty:
                continue
            s_ret = s_df["Close"].pct_change().dropna().tail(ETF_CORR_LOOKBACK_DAYS)

            merged = pd.concat([c_ret.rename("c"), s_ret.rename("s")], axis=1).dropna()
            if len(merged) < 60:
                continue

            corr = merged["c"].corr(merged["s"])
            if pd.notna(corr) and corr >= ETF_CORR_THRESHOLD:
                keep = False
                break

        if keep:
            selected.append(candidate)

        if len(selected) >= ETF_TOP_N_FINAL:
            break

    return selected


# ============================================================
# ETF holdings
# yfinance funds_data / fund_top_holdings 우선 시도
# 실패 시 info/holdings 관련 여러 경로 탐색
# ============================================================
def _read_cache(ticker: str) -> Optional[List[str]]:
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        saved_at = datetime.fromisoformat(payload.get("saved_at"))
        if datetime.now() - saved_at > timedelta(days=HOLDINGS_CACHE_DAYS):
            return None
        holdings = payload.get("holdings", [])
        if isinstance(holdings, list) and holdings:
            return [str(x).upper().strip() for x in holdings if str(x).strip()]
    except Exception:
        return None

    return None


def _write_cache(ticker: str, holdings: List[str]) -> None:
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    payload = {
        "saved_at": datetime.now().isoformat(),
        "holdings": holdings,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _clean_holding_ticker(x: str) -> str:
    return str(x or "").upper().strip().replace(".", "-")


def fetch_etf_holdings(ticker: str, top_n: int = ETF_HOLDINGS_TOP_N) -> List[str]:
    cached = _read_cache(ticker)
    if cached:
        return cached[:top_n]

    holdings: List[str] = []

    try:
        t = yf.Ticker(ticker)

        # 경로 1: funds_data.top_holdings
        try:
            funds_data = getattr(t, "funds_data", None)
            top_holdings = getattr(funds_data, "top_holdings", None) if funds_data is not None else None
            if top_holdings is not None and isinstance(top_holdings, pd.DataFrame) and not top_holdings.empty:
                possible_cols = ["Symbol", "symbol", "Holding", "holding"]
                for col in possible_cols:
                    if col in top_holdings.columns:
                        holdings = [_clean_holding_ticker(x) for x in top_holdings[col].tolist()]
                        holdings = [x for x in holdings if x]
                        break
        except Exception:
            pass

        # 경로 2: quote_type / funds_data 내부 다른 컬럼 탐색
        if not holdings:
            try:
                if funds_data is not None:
                    for attr in dir(funds_data):
                        if attr.startswith("_"):
                            continue
                        value = getattr(funds_data, attr, None)
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            for col in ["Symbol", "symbol", "Holding", "holding"]:
                                if col in value.columns:
                                    holdings = [_clean_holding_ticker(x) for x in value[col].tolist()]
                                    holdings = [x for x in holdings if x]
                                    if holdings:
                                        break
                        if holdings:
                            break
            except Exception:
                pass

        # 경로 3: info 내 holdings 류
        if not holdings:
            try:
                info = t.info
                for key in ["holdings", "topHoldings", "top_holdings"]:
                    value = info.get(key)
                    if isinstance(value, list):
                        temp: List[str] = []
                        for item in value:
                            if isinstance(item, dict):
                                sym = item.get("symbol") or item.get("holding") or item.get("ticker")
                                if sym:
                                    temp.append(_clean_holding_ticker(sym))
                            elif isinstance(item, str):
                                temp.append(_clean_holding_ticker(item))
                        temp = [x for x in temp if x]
                        if temp:
                            holdings = temp
                            break
            except Exception:
                pass

    except Exception:
        holdings = []

    # fallback: ETF 자기 자신을 후보에 넣지 않음
    holdings = [x for x in holdings if x and x != ticker]
    deduped = []
    seen = set()
    for x in holdings:
        if x not in seen:
            seen.add(x)
            deduped.append(x)

    if deduped:
        _write_cache(ticker, deduped[:top_n])

    return deduped[:top_n]


def build_candidate_sources(selected_etfs: List[ETFScore]) -> Dict[str, List[str]]:
    candidate_sources: Dict[str, List[str]] = {}

    for etf in selected_etfs:
        holdings = fetch_etf_holdings(etf.ticker, ETF_HOLDINGS_TOP_N)
        for stock in holdings:
            candidate_sources.setdefault(stock, []).append(etf.ticker)

    return candidate_sources


# ============================================================
# 종목 스코어
# ============================================================
def build_stock_scores(candidate_sources: Dict[str, List[str]], spy_df: pd.DataFrame) -> List[StockScore]:
    spy_df = add_basic_indicators(spy_df)
    spy_row = spy_df.iloc[-1]
    spy_ret_3m = safe_float(spy_row["ret_3m"])
    if spy_ret_3m is None:
        spy_ret_3m = 0.0

    raw_rows: List[dict] = []

    for ticker, source_etfs in candidate_sources.items():
        try:
            df = download_history(ticker)
            if df.empty or len(df) < 220:
                continue

            df = add_basic_indicators(df)
            row = df.iloc[-1]

            close = safe_float(row["Close"])
            ma50 = safe_float(row["ma50"])
            ma200 = safe_float(row["ma200"])
            dollar_vol_20 = safe_float(row["dollar_volume_20"])
            ret_1m = safe_float(row["ret_1m"])
            ret_3m = safe_float(row["ret_3m"])
            ret_6m = safe_float(row["ret_6m"])
            ret_20d = safe_float(row["ret_20d"])
            volume_ma20 = safe_float(row["volume_ma20"])
            volume_ma90 = safe_float(row["volume_ma90"])

            if any(v is None for v in [close, ma50, ma200, dollar_vol_20, ret_1m, ret_3m, ret_6m, ret_20d, volume_ma20, volume_ma90]):
                continue

            # Gate
            if close < MIN_STOCK_PRICE:
                continue
            if dollar_vol_20 < MIN_STOCK_DOLLAR_VOLUME:
                continue
            if close <= ma50:
                continue
            if close <= ma200:
                continue
            if ret_20d <= MAX_20D_DRAWDOWN_FILTER:
                continue

            rs_3m = ret_3m - spy_ret_3m
            accel_proxy = ret_1m - ret_3m
            volume_ratio = (volume_ma20 / volume_ma90) if volume_ma90 and volume_ma90 > 0 else 1.0

            raw_rows.append(
                {
                    "ticker": ticker,
                    "name": ticker,
                    "source_etfs": ",".join(sorted(set(source_etfs))),
                    "close": close,
                    "dollar_volume_20": dollar_vol_20,
                    "ret_1m": ret_1m,
                    "ret_3m": ret_3m,
                    "ret_6m": ret_6m,
                    "rs_3m_vs_spy": rs_3m,
                    "accel_proxy": accel_proxy,
                    "volume_ratio": volume_ratio,
                    "m_raw": 0.5 * ret_3m + 0.3 * ret_1m + 0.2 * ret_6m,
                    "r_raw": rs_3m,
                    "e_raw": accel_proxy,
                    "v_raw": volume_ratio,
                }
            )

        except Exception:
            continue

    if not raw_rows:
        return []

    temp = pd.DataFrame(raw_rows)

    temp["m_score"] = minmax_normalize(temp["m_raw"])
    temp["r_score"] = minmax_normalize(temp["r_raw"])
    temp["e_score"] = minmax_normalize(temp["e_raw"])
    temp["v_score"] = minmax_normalize(temp["v_raw"])

    temp["total_score"] = (
        temp["m_score"] * 0.40
        + temp["r_score"] * 0.30
        + temp["e_score"] * 0.15
        + temp["v_score"] * 0.15
    )

    temp = temp.sort_values(["total_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
    temp["rank"] = range(1, len(temp) + 1)

    results: List[StockScore] = []
    for _, row in temp.iterrows():
        results.append(
            StockScore(
                ticker=str(row["ticker"]),
                name=str(row["name"]),
                source_etfs=str(row["source_etfs"]),
                close=float(row["close"]),
                dollar_volume_20=float(row["dollar_volume_20"]),
                ret_1m=float(row["ret_1m"]),
                ret_3m=float(row["ret_3m"]),
                ret_6m=float(row["ret_6m"]),
                rs_3m_vs_spy=float(row["rs_3m_vs_spy"]),
                accel_proxy=float(row["accel_proxy"]),
                volume_ratio=float(row["volume_ratio"]),
                m_score=float(row["m_score"]),
                r_score=float(row["r_score"]),
                e_score=float(row["e_score"]),
                v_score=float(row["v_score"]),
                total_score=float(row["total_score"]),
                rank=int(row["rank"]),
            )
        )

    return results


# ============================================================
# 포트 상태
# ============================================================
def load_current_portfolio() -> Dict:
    if not os.path.exists(CURRENT_PORTFOLIO_FILE):
        return {"as_of_date": None, "positions": []}
    try:
        with open(CURRENT_PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"as_of_date": None, "positions": []}


def save_current_portfolio(as_of_date: str, positions: List[StockScore]) -> None:
    payload = {
        "as_of_date": as_of_date,
        "positions": [
            {
                "ticker": p.ticker,
                "name": p.name,
                "rank": p.rank,
                "score": round(p.total_score, 4),
                "source_etfs": p.source_etfs,
                "close": round(p.close, 2),
            }
            for p in positions
        ],
    }
    with open(CURRENT_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_rebalance_log(as_of_date: str, out_items: List[Dict], in_items: List[Dict], kept_items: List[Dict]) -> None:
    rows = []

    for item in out_items:
        rows.append(
            {
                "date": as_of_date,
                "action": "OUT",
                "ticker": item.get("ticker"),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "source_etfs": item.get("source_etfs"),
            }
        )

    for item in in_items:
        rows.append(
            {
                "date": as_of_date,
                "action": "IN",
                "ticker": item.get("ticker"),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "source_etfs": item.get("source_etfs"),
            }
        )

    for item in kept_items:
        rows.append(
            {
                "date": as_of_date,
                "action": "KEEP",
                "ticker": item.get("ticker"),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "source_etfs": item.get("source_etfs"),
            }
        )

    if not rows:
        return

    df_new = pd.DataFrame(rows)
    if os.path.exists(REBALANCE_LOG_FILE):
        df_old = pd.read_csv(REBALANCE_LOG_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(REBALANCE_LOG_FILE, index=False, encoding="utf-8-sig")


def append_history(as_of_date: str, stock_scores: List[StockScore], held_tickers: List[str]) -> None:
    rows = []
    held_set = set(held_tickers)

    for s in stock_scores:
        rows.append(
            {
                "date": as_of_date,
                "ticker": s.ticker,
                "name": s.name,
                "rank": s.rank,
                "score": round(s.total_score, 6),
                "ret_1m": round(s.ret_1m, 6),
                "ret_3m": round(s.ret_3m, 6),
                "ret_6m": round(s.ret_6m, 6),
                "rs_3m_vs_spy": round(s.rs_3m_vs_spy, 6),
                "volume_ratio": round(s.volume_ratio, 6),
                "source_etfs": s.source_etfs,
                "is_held": 1 if s.ticker in held_set else 0,
            }
        )

    df_new = pd.DataFrame(rows)
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 리밸런싱
# 기존 보유가 Top12 안이면 유지
# 최종 6종목
# ============================================================
def build_final_portfolio(stock_scores: List[StockScore]) -> Tuple[List[StockScore], List[Dict], List[Dict], List[Dict]]:
    current = load_current_portfolio()
    current_positions = current.get("positions", [])
    current_tickers = [str(x.get("ticker")).upper().strip() for x in current_positions]

    rank_map = {s.ticker: s.rank for s in stock_scores}
    score_map = {s.ticker: s.total_score for s in stock_scores}
    obj_map = {s.ticker: s for s in stock_scores}

    kept: List[StockScore] = []
    for ticker in current_tickers:
        rank = rank_map.get(ticker)
        if rank is not None and rank <= HOLD_BUFFER_RANK and ticker in obj_map:
            kept.append(obj_map[ticker])

    kept_tickers = {x.ticker for x in kept}

    top6 = stock_scores[:PORTFOLIO_SIZE]
    final_positions: List[StockScore] = []

    # 1. 유지 종목 먼저
    for item in kept:
        if item.ticker not in {x.ticker for x in final_positions}:
            final_positions.append(item)

    # 2. 상위 종목으로 채움
    for item in top6:
        if len(final_positions) >= PORTFOLIO_SIZE:
            break
        if item.ticker not in {x.ticker for x in final_positions}:
            final_positions.append(item)

    # 3. 아직 비었으면 전체 랭킹으로 채움
    if len(final_positions) < PORTFOLIO_SIZE:
        for item in stock_scores:
            if len(final_positions) >= PORTFOLIO_SIZE:
                break
            if item.ticker not in {x.ticker for x in final_positions}:
                final_positions.append(item)

    final_tickers = [x.ticker for x in final_positions]
    current_ticker_set = set(current_tickers)
    final_ticker_set = set(final_tickers)

    out_items = []
    for p in current_positions:
        t = str(p.get("ticker")).upper().strip()
        if t not in final_ticker_set:
            out_items.append(
                {
                    "ticker": t,
                    "name": p.get("name", t),
                    "rank": rank_map.get(t),
                    "score": score_map.get(t),
                    "source_etfs": p.get("source_etfs", ""),
                }
            )

    in_items = []
    for p in final_positions:
        if p.ticker not in current_ticker_set:
            in_items.append(
                {
                    "ticker": p.ticker,
                    "name": p.name,
                    "rank": p.rank,
                    "score": round(p.total_score, 4),
                    "source_etfs": p.source_etfs,
                }
            )

    kept_items = []
    for p in final_positions:
        if p.ticker in current_ticker_set:
            kept_items.append(
                {
                    "ticker": p.ticker,
                    "name": p.name,
                    "rank": p.rank,
                    "score": round(p.total_score, 4),
                    "source_etfs": p.source_etfs,
                }
            )

    final_positions = sorted(final_positions, key=lambda x: x.rank)
    return final_positions, out_items, in_items, kept_items


# ============================================================
# 리포트
# ============================================================
def build_report(
    as_of_date: str,
    top_etfs_raw: List[ETFScore],
    top_etfs_final: List[ETFScore],
    stock_scores: List[StockScore],
    final_positions: List[StockScore],
    out_items: List[Dict],
    in_items: List[Dict],
    kept_items: List[Dict],
) -> str:
    lines: List[str] = []
    lines.append(f"[EMR-6 주간 리밸런싱]")
    lines.append(f"기준일: {as_of_date}")
    lines.append("")

    lines.append(f"ETF 통과 수: {len(top_etfs_raw)}")
    lines.append(f"최종 ETF 수: {len(top_etfs_final)}")
    lines.append(f"후보 종목 수: {len(stock_scores)}")
    lines.append("")

    lines.append("[상위 ETF]")
    for i, e in enumerate(top_etfs_final[:10], start=1):
        lines.append(
            f"{i}. {e.ticker} {e.name} | 점수 {e.score:.3f} | 1M {fmt_pct(e.ret_1m)} | 3M {fmt_pct(e.ret_3m)}"
        )

    lines.append("")
    lines.append("[최종 Top6]")
    for i, s in enumerate(final_positions, start=1):
        lines.append(
            f"{i}. {s.ticker} {s.name} | rank {s.rank} | score {s.total_score:.3f} | "
            f"1M {fmt_pct(s.ret_1m)} | 3M {fmt_pct(s.ret_3m)} | RS {fmt_pct(s.rs_3m_vs_spy)} | ETF {s.source_etfs}"
        )

    lines.append("")
    lines.append("[유지]")
    if kept_items:
        for x in kept_items:
            lines.append(f"- {x['ticker']} {x['name']} | rank {x['rank']} | score {x['score']}")
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("[교체 OUT]")
    if out_items:
        for x in out_items:
            lines.append(f"- {x['ticker']} {x['name']} | rank {x['rank']} | score {x['score']}")
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("[교체 IN]")
    if in_items:
        for x in in_items:
            lines.append(f"- {x['ticker']} {x['name']} | rank {x['rank']} | score {x['score']}")
    else:
        lines.append("- 없음")

    return "\n".join(lines)


def build_candidate_lines(stock_scores: List[StockScore]) -> List[str]:
    lines: List[str] = []
    for s in stock_scores:
        lines.append(
            f"{s.rank}. {s.ticker} {s.name} | score {s.total_score:.3f} | "
            f"M {s.m_score:.3f} | R {s.r_score:.3f} | E {s.e_score:.3f} | V {s.v_score:.3f} | ETF {s.source_etfs}"
        )
    return lines


# ============================================================
# 메인
# ============================================================
def main() -> None:
    ensure_dirs()

    # ETF scoring
    etf_scores_all, price_map = build_etf_scores()
    top_etfs_raw = etf_scores_all[:ETF_TOP_N_RAW]
    top_etfs_final = filter_etf_correlation(top_etfs_raw, price_map)

    if not top_etfs_final:
        msg = "[EMR-6 주간 리밸런싱]\nETF 후보가 없습니다."
        send_telegram_message(msg)
        print(msg)
        return

    # ETF holdings -> stock candidates
    candidate_sources = build_candidate_sources(top_etfs_final)

    # SPY benchmark
    spy_df = download_history("SPY")
    if spy_df.empty:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    # stock scoring
    stock_scores = build_stock_scores(candidate_sources, spy_df)
    if not stock_scores:
        msg = "[EMR-6 주간 리밸런싱]\n종목 후보가 없습니다."
        send_telegram_message(msg)
        print(msg)
        return

    final_positions, out_items, in_items, kept_items = build_final_portfolio(stock_scores)

    as_of_date = str(pd.to_datetime(spy_df.iloc[-1]["Date"]).date())

    save_current_portfolio(as_of_date, final_positions)
    append_rebalance_log(as_of_date, out_items, in_items, kept_items)
    append_history(as_of_date, stock_scores, [x.ticker for x in final_positions])

    report = build_report(
        as_of_date=as_of_date,
        top_etfs_raw=top_etfs_raw,
        top_etfs_final=top_etfs_final,
        stock_scores=stock_scores,
        final_positions=final_positions,
        out_items=out_items,
        in_items=in_items,
        kept_items=kept_items,
    )

    send_telegram_message(report)
    send_telegram_chunked("[EMR-6 후보 랭킹]", build_candidate_lines(stock_scores))
    print(report)


if __name__ == "__main__":
    main()
