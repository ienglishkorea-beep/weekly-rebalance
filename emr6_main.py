import os
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# EMR-6 Industry-Leading Rotation
# ------------------------------------------------------------
# 구조
# 1) 산업 대표 10종목 CSV 로드
# 2) 종목 점수 계산
# 3) 산업 점수 계산
# 4) 산업 상위 3개 선택
# 5) 1위 산업 3종목 / 2위 산업 2종목 / 3위 산업 1종목
# 6) 산업 내부 Top3 편입 / Top5 유지 / 6위 이하 교체
#
# 필요 파일
# - data/industry_universe_rbics_style_v1.csv
#
# 출력
# - data/emr6_current_portfolio.json
# - data/emr6_rebalance_log.csv
# - data/emr6_history.csv
# ============================================================


# ============================================================
# 경로 / 환경변수
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CURRENT_PORTFOLIO_FILE = os.path.join(DATA_DIR, "emr6_current_portfolio.json")
REBALANCE_LOG_FILE = os.path.join(DATA_DIR, "emr6_rebalance_log.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "emr6_history.csv")
INDUSTRY_UNIVERSE_FILE = os.path.join(DATA_DIR, "industry_universe_rbics_style_v1.csv")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# ============================================================
# 전략 파라미터
# ============================================================
PRICE_HISTORY_PERIOD = "1y"
PRICE_INTERVAL = "1d"

PORTFOLIO_SIZE = 6

INDUSTRY_SLOTS = {
    1: 3,
    2: 2,
    3: 1,
}

INDUSTRY_UNIVERSE_SIZE = 10
INDUSTRY_ENTRY_RANK = 3
INDUSTRY_HOLD_RANK = 5

MIN_STOCK_PRICE = 10.0
MIN_STOCK_DOLLAR_VOLUME = 20_000_000
MAX_20D_DRAWDOWN_FILTER = -0.20

VOL_LOOKBACK_DAYS = 20
VOL_FLOOR = 0.015
VOL_CAP = 0.08

TELEGRAM_MAX_CHARS = 3500


# ============================================================
# 데이터 구조
# ============================================================
@dataclass
class StockScore:
    industry: str
    ticker: str
    name: str
    close: float
    dollar_volume_20: float
    ret_1m: float
    ret_3m: float
    ret_6m: float
    rs_3m_vs_spy: float
    acceleration: float
    volume_ratio: float
    vol_20d: float
    ma50_gap: float
    m_score: float
    r_score: float
    a_score: float
    v_score: float
    total_score: float
    industry_rank: int = 0
    target_weight: float = 0.0


@dataclass
class IndustryScore:
    industry: str
    score: float
    leader_avg_score: float
    rs_positive_ratio: float
    accel_positive_ratio: float
    volume_expansion_ratio: float
    trend_ratio: float
    member_count: int
    rank: int = 0
    slots: int = 0


# ============================================================
# 유틸
# ============================================================
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


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


def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


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


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ============================================================
# 기본 파일 초기화
# ============================================================
def ensure_base_files(as_of_date: Optional[str] = None) -> None:
    if not os.path.exists(CURRENT_PORTFOLIO_FILE):
        with open(CURRENT_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"as_of_date": as_of_date, "positions": []},
                f,
                ensure_ascii=False,
                indent=2,
            )

    if not os.path.exists(REBALANCE_LOG_FILE):
        pd.DataFrame(
            columns=[
                "date",
                "action",
                "industry",
                "ticker",
                "name",
                "industry_rank",
                "industry_score",
                "stock_score",
                "target_weight",
            ]
        ).to_csv(REBALANCE_LOG_FILE, index=False, encoding="utf-8-sig")

    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(
            columns=[
                "date",
                "industry",
                "ticker",
                "name",
                "industry_rank",
                "score",
                "ret_1m",
                "ret_3m",
                "ret_6m",
                "rs_3m_vs_spy",
                "acceleration",
                "volume_ratio",
                "vol_20d",
                "target_weight",
                "is_held",
            ]
        ).to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 입력 파일
# ============================================================
def load_industry_universe_csv() -> pd.DataFrame:
    if not os.path.exists(INDUSTRY_UNIVERSE_FILE):
        raise FileNotFoundError(
            f"필수 파일 없음: {INDUSTRY_UNIVERSE_FILE}\n"
            f"data 폴더 안에 industry_universe_rbics_style_v1.csv 파일을 넣어야 합니다."
        )

    df = pd.read_csv(INDUSTRY_UNIVERSE_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"industry", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"산업 CSV 필수 컬럼 누락: {sorted(missing)}")

    if "name" not in df.columns:
        df["name"] = df["ticker"]

    df["industry"] = df["industry"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    df["name"] = df["name"].astype(str).str.strip()

    df = df[(df["industry"] != "") & (df["ticker"] != "")]
    df = df.drop_duplicates(subset=["industry", "ticker"]).reset_index(drop=True)

    counts = df.groupby("industry")["ticker"].count().to_dict()
    bad_groups = {k: v for k, v in counts.items() if v != INDUSTRY_UNIVERSE_SIZE}
    if bad_groups:
        raise ValueError(
            f"산업별 종목 수가 10개가 아닌 산업이 있음: {bad_groups}"
        )

    duplicated_tickers = df["ticker"][df["ticker"].duplicated()].unique().tolist()
    if duplicated_tickers:
        raise ValueError(
            f"하나의 종목이 여러 산업에 중복 배치됨: {duplicated_tickers}"
        )

    return df


def build_industry_membership(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    industry_map: Dict[str, List[str]] = {}
    name_map: Dict[str, str] = {}
    ticker_to_industry: Dict[str, str] = {}

    for row in df.itertuples(index=False):
        industry_map.setdefault(row.industry, []).append(row.ticker)
        name_map[row.ticker] = row.name
        ticker_to_industry[row.ticker] = row.industry

    return industry_map, name_map, ticker_to_industry


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
    out["daily_ret"] = out["Close"].pct_change()
    out["vol_20d"] = out["daily_ret"].rolling(VOL_LOOKBACK_DAYS).std()
    return out


# ============================================================
# 종목 점수
# ============================================================
def build_stock_scores(
    industry_map: Dict[str, List[str]],
    name_map: Dict[str, str],
    spy_df: pd.DataFrame,
) -> List[StockScore]:
    spy_df = add_basic_indicators(spy_df)
    spy_row = spy_df.iloc[-1]
    spy_ret_3m = safe_float(spy_row["ret_3m"])
    if spy_ret_3m is None:
        spy_ret_3m = 0.0

    raw_rows: List[dict] = []

    for industry, tickers in industry_map.items():
        for ticker in tickers:
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
                vol_20d = safe_float(row["vol_20d"])

                if any(v is None for v in [
                    close, ma50, ma200, dollar_vol_20, ret_1m, ret_3m,
                    ret_6m, ret_20d, volume_ma20, volume_ma90, vol_20d
                ]):
                    continue

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
                acceleration = ret_1m - (ret_3m / 3.0)
                volume_ratio = (volume_ma20 / volume_ma90) if volume_ma90 and volume_ma90 > 0 else 1.0
                ma50_gap = (close / ma50 - 1.0) if ma50 > 0 else 0.0

                raw_rows.append(
                    {
                        "industry": industry,
                        "ticker": ticker,
                        "name": name_map.get(ticker, ticker),
                        "close": close,
                        "dollar_volume_20": dollar_vol_20,
                        "ret_1m": ret_1m,
                        "ret_3m": ret_3m,
                        "ret_6m": ret_6m,
                        "rs_3m_vs_spy": rs_3m,
                        "acceleration": acceleration,
                        "volume_ratio": volume_ratio,
                        "vol_20d": vol_20d,
                        "ma50_gap": ma50_gap,
                        "m_raw": 0.65 * ret_3m + 0.35 * ret_1m,
                        "r_raw": rs_3m,
                        "a_raw": acceleration,
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
    temp["a_score"] = minmax_normalize(temp["a_raw"])
    temp["v_score"] = minmax_normalize(temp["v_raw"])

    temp["total_score"] = (
        temp["m_score"] * 0.35
        + temp["r_score"] * 0.25
        + temp["a_score"] * 0.25
        + temp["v_score"] * 0.15
    )

    results: List[StockScore] = []
    for industry, grp in temp.groupby("industry", sort=False):
        grp = grp.sort_values(["total_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
        grp["industry_rank"] = range(1, len(grp) + 1)

        for _, row in grp.iterrows():
            results.append(
                StockScore(
                    industry=str(row["industry"]),
                    ticker=str(row["ticker"]),
                    name=str(row["name"]),
                    close=float(row["close"]),
                    dollar_volume_20=float(row["dollar_volume_20"]),
                    ret_1m=float(row["ret_1m"]),
                    ret_3m=float(row["ret_3m"]),
                    ret_6m=float(row["ret_6m"]),
                    rs_3m_vs_spy=float(row["rs_3m_vs_spy"]),
                    acceleration=float(row["acceleration"]),
                    volume_ratio=float(row["volume_ratio"]),
                    vol_20d=float(row["vol_20d"]),
                    ma50_gap=float(row["ma50_gap"]),
                    m_score=float(row["m_score"]),
                    r_score=float(row["r_score"]),
                    a_score=float(row["a_score"]),
                    v_score=float(row["v_score"]),
                    total_score=float(row["total_score"]),
                    industry_rank=int(row["industry_rank"]),
                    target_weight=0.0,
                )
            )

    return results


# ============================================================
# 산업 점수
# ============================================================
def build_industry_scores(stock_scores: List[StockScore]) -> List[IndustryScore]:
    rows: List[IndustryScore] = []

    if not stock_scores:
        return rows

    df = pd.DataFrame(
        [
            {
                "industry": s.industry,
                "ticker": s.ticker,
                "score": s.total_score,
                "industry_rank": s.industry_rank,
                "rs_3m_vs_spy": s.rs_3m_vs_spy,
                "acceleration": s.acceleration,
                "volume_ratio": s.volume_ratio,
                "close_above_ma50": 1,
            }
            for s in stock_scores
        ]
    )

    for industry, grp in df.groupby("industry", sort=False):
        grp = grp.sort_values(["industry_rank"], ascending=[True])

        leaders = grp[grp["industry_rank"] <= 3]
        leader_avg_score = float(leaders["score"].mean()) if not leaders.empty else 0.0
        rs_positive_ratio = float((grp["rs_3m_vs_spy"] > 0).mean())
        accel_positive_ratio = float((grp["acceleration"] > 0).mean())
        volume_expansion_ratio = float((grp["volume_ratio"] > 1.0).mean())
        trend_ratio = float(grp["close_above_ma50"].mean())

        score = (
            leader_avg_score * 0.45
            + rs_positive_ratio * 0.20
            + accel_positive_ratio * 0.15
            + volume_expansion_ratio * 0.10
            + trend_ratio * 0.10
        )

        rows.append(
            IndustryScore(
                industry=industry,
                score=score,
                leader_avg_score=leader_avg_score,
                rs_positive_ratio=rs_positive_ratio,
                accel_positive_ratio=accel_positive_ratio,
                volume_expansion_ratio=volume_expansion_ratio,
                trend_ratio=trend_ratio,
                member_count=int(len(grp)),
                rank=0,
                slots=0,
            )
        )

    rows = sorted(rows, key=lambda x: (x.score, x.leader_avg_score), reverse=True)

    for i, item in enumerate(rows, start=1):
        item.rank = i
        item.slots = INDUSTRY_SLOTS.get(i, 0)

    return rows


# ============================================================
# 비중
# ============================================================
def apply_volatility_weights(positions: List[StockScore]) -> List[StockScore]:
    if not positions:
        return positions

    inverse_vols: List[float] = []
    for p in positions:
        clipped_vol = min(max(p.vol_20d, VOL_FLOOR), VOL_CAP)
        inverse_vols.append(1.0 / clipped_vol)

    total = sum(inverse_vols)
    if total <= 0:
        equal_weight = 1.0 / len(positions)
        for p in positions:
            p.target_weight = equal_weight
        return positions

    for p, inv in zip(positions, inverse_vols):
        p.target_weight = inv / total

    return positions


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


def save_current_portfolio(as_of_date: str, positions: List[StockScore], industry_rank_map: Dict[str, int], industry_score_map: Dict[str, float]) -> None:
    payload = {
        "as_of_date": as_of_date,
        "positions": [
            {
                "industry": p.industry,
                "industry_rank": industry_rank_map.get(p.industry),
                "industry_score": round(industry_score_map.get(p.industry, 0.0), 4),
                "ticker": p.ticker,
                "name": p.name,
                "industry_member_rank": p.industry_rank,
                "score": round(p.total_score, 4),
                "target_weight": round(p.target_weight, 6),
                "close": round(p.close, 2),
            }
            for p in positions
        ],
    }
    with open(CURRENT_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_rebalance_log(
    as_of_date: str,
    out_items: List[Dict],
    in_items: List[Dict],
    kept_items: List[Dict],
) -> None:
    rows = []

    for action, items in [("OUT", out_items), ("IN", in_items), ("KEEP", kept_items)]:
        for item in items:
            rows.append(
                {
                    "date": as_of_date,
                    "action": action,
                    "industry": item.get("industry"),
                    "ticker": item.get("ticker"),
                    "name": item.get("name"),
                    "industry_rank": item.get("industry_rank"),
                    "industry_score": item.get("industry_score"),
                    "stock_score": item.get("stock_score"),
                    "target_weight": item.get("target_weight"),
                }
            )

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
                "industry": s.industry,
                "ticker": s.ticker,
                "name": s.name,
                "industry_rank": s.industry_rank,
                "score": round(s.total_score, 6),
                "ret_1m": round(s.ret_1m, 6),
                "ret_3m": round(s.ret_3m, 6),
                "ret_6m": round(s.ret_6m, 6),
                "rs_3m_vs_spy": round(s.rs_3m_vs_spy, 6),
                "acceleration": round(s.acceleration, 6),
                "volume_ratio": round(s.volume_ratio, 6),
                "vol_20d": round(s.vol_20d, 6),
                "target_weight": round(s.target_weight, 6),
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
# 포트 구성
# ============================================================
def build_final_portfolio(
    stock_scores: List[StockScore],
    industry_scores: List[IndustryScore],
) -> Tuple[List[StockScore], List[Dict], List[Dict], List[Dict]]:
    current = load_current_portfolio()
    current_positions = current.get("positions", [])

    industry_rank_map = {x.industry: x.rank for x in industry_scores}
    industry_score_map = {x.industry: x.score for x in industry_scores}

    top_industries = [x.industry for x in industry_scores if x.rank in INDUSTRY_SLOTS]
    slots_map = {x.industry: x.slots for x in industry_scores if x.rank in INDUSTRY_SLOTS}

    stock_map = {(s.industry, s.ticker): s for s in stock_scores}
    stock_by_industry: Dict[str, List[StockScore]] = {}
    for s in stock_scores:
        stock_by_industry.setdefault(s.industry, []).append(s)

    for industry in stock_by_industry:
        stock_by_industry[industry] = sorted(stock_by_industry[industry], key=lambda x: x.industry_rank)

    final_positions: List[StockScore] = []
    current_key_set = {(str(x.get("industry")), str(x.get("ticker")).upper().strip()) for x in current_positions}

    for industry in top_industries:
        slots = slots_map.get(industry, 0)
        members = stock_by_industry.get(industry, [])
        if not members or slots <= 0:
            continue

        kept_candidates: List[StockScore] = []
        new_candidates: List[StockScore] = []

        for m in members:
            key = (m.industry, m.ticker)
            if key in current_key_set and m.industry_rank <= INDUSTRY_HOLD_RANK:
                kept_candidates.append(m)
            elif m.industry_rank <= INDUSTRY_ENTRY_RANK:
                new_candidates.append(m)

        kept_candidates = sorted(kept_candidates, key=lambda x: x.industry_rank)
        new_candidates = sorted(new_candidates, key=lambda x: x.industry_rank)

        chosen: List[StockScore] = []
        for item in kept_candidates:
            if len(chosen) >= slots:
                break
            if item.ticker not in {x.ticker for x in chosen}:
                chosen.append(item)

        for item in members:
            if len(chosen) >= slots:
                break
            if item.industry_rank <= INDUSTRY_ENTRY_RANK and item.ticker not in {x.ticker for x in chosen}:
                chosen.append(item)

        for item in members:
            if len(chosen) >= slots:
                break
            if item.industry_rank <= INDUSTRY_HOLD_RANK and item.ticker not in {x.ticker for x in chosen}:
                chosen.append(item)

        final_positions.extend(chosen)

    final_positions = sorted(
        final_positions,
        key=lambda x: (industry_rank_map.get(x.industry, 999), x.industry_rank, x.ticker)
    )[:PORTFOLIO_SIZE]

    final_positions = apply_volatility_weights(final_positions)

    final_keys = {(x.industry, x.ticker) for x in final_positions}
    current_keys = {(str(x.get("industry")), str(x.get("ticker")).upper().strip()) for x in current_positions}

    out_items: List[Dict] = []
    for p in current_positions:
        key = (str(p.get("industry")), str(p.get("ticker")).upper().strip())
        if key not in final_keys:
            out_items.append(
                {
                    "industry": p.get("industry"),
                    "ticker": str(p.get("ticker")).upper().strip(),
                    "name": p.get("name"),
                    "industry_rank": industry_rank_map.get(p.get("industry")),
                    "industry_score": industry_score_map.get(p.get("industry")),
                    "stock_score": p.get("score"),
                    "target_weight": p.get("target_weight"),
                }
            )

    in_items: List[Dict] = []
    kept_items: List[Dict] = []
    for p in final_positions:
        item = {
            "industry": p.industry,
            "ticker": p.ticker,
            "name": p.name,
            "industry_rank": industry_rank_map.get(p.industry),
            "industry_score": industry_score_map.get(p.industry),
            "stock_score": round(p.total_score, 4),
            "target_weight": round(p.target_weight, 6),
        }
        if (p.industry, p.ticker) in current_keys:
            kept_items.append(item)
        else:
            in_items.append(item)

    return final_positions, out_items, in_items, kept_items


# ============================================================
# 설명 / 리포트
# ============================================================
def summarize_strengths(s: StockScore) -> str:
    strengths: List[str] = []
    if s.industry_rank <= 2:
        strengths.append("산업 내 최상위")
    if s.ret_3m >= 0.20:
        strengths.append("3개월 모멘텀 강함")
    if s.rs_3m_vs_spy >= 0.10:
        strengths.append("RS 우위")
    if s.acceleration >= 0.05:
        strengths.append("가속도 양호")
    if s.volume_ratio >= 1.05:
        strengths.append("거래량 확장")
    if s.vol_20d <= 0.03:
        strengths.append("변동성 안정")

    if not strengths:
        strengths.append("상대 점수 우위")
    return ", ".join(strengths[:3])


def summarize_risks(s: StockScore) -> str:
    risks: List[str] = []
    if s.industry_rank >= 4:
        risks.append("산업 내 순위 밀림")
    if s.acceleration < 0:
        risks.append("최근 가속 둔화")
    if s.rs_3m_vs_spy < 0.03:
        risks.append("RS 약함")
    if s.volume_ratio < 0.90:
        risks.append("거래량 둔화")
    if s.vol_20d >= 0.06:
        risks.append("변동성 높음")

    if not risks:
        risks.append("특이 경고 없음")
    return ", ".join(risks[:2])


def summarize_industry(ind: IndustryScore) -> str:
    return (
        f"leader {ind.leader_avg_score:.3f} | "
        f"RS+ {fmt_pct(ind.rs_positive_ratio)} | "
        f"ACC+ {fmt_pct(ind.accel_positive_ratio)} | "
        f"VOL+ {fmt_pct(ind.volume_expansion_ratio)}"
    )


def build_report(
    as_of_date: str,
    industry_scores: List[IndustryScore],
    stock_scores: List[StockScore],
    final_positions: List[StockScore],
    out_items: List[Dict],
    in_items: List[Dict],
    kept_items: List[Dict],
) -> str:
    lines: List[str] = []

    top3_industries = [x for x in industry_scores if x.rank in INDUSTRY_SLOTS]
    industry_rank_map = {x.industry: x.rank for x in industry_scores}
    kept_set = {(x["industry"], x["ticker"]) for x in kept_items}

    lines.append("[EMR-6 산업 선행 리밸런싱]")
    lines.append(f"기준일: {as_of_date}")
    lines.append("")
    lines.append("[요약]")
    lines.append(f"- 유지 {len(kept_items)} / 신규 {len(in_items)} / 제외 {len(out_items)}")
    lines.append(f"- 산업 계산 수 {len(industry_scores)} / 종목 계산 수 {len(stock_scores)}")
    lines.append("- 규칙: 산업 1위 3종목 / 2위 2종목 / 3위 1종목")
    lines.append("- 산업 내부 규칙: Top3 편입 / Top5 유지 / 6위 이하 교체")
    lines.append("")

    lines.append("[상위 산업 랭킹]")
    for ind in top3_industries:
        lines.append(
            f"{ind.rank}. {ind.industry} | score {ind.score:.3f} | slots {ind.slots} | {summarize_industry(ind)}"
        )

    lines.append("")
    lines.append("[최종 포트폴리오]")
    if final_positions:
        for i, s in enumerate(final_positions, start=1):
            status = "보유" if (s.industry, s.ticker) in kept_set else "신규 매수"
            lines.append(
                f"{i}. {s.ticker} | {s.name} | {s.industry} | 산업 {industry_rank_map.get(s.industry)}위 | 내부 {s.industry_rank}위 | {status} | 비중 {fmt_pct(s.target_weight)}"
            )
            lines.append(
                f"   - score {s.total_score:.3f} | 1M {fmt_pct(s.ret_1m)} | 3M {fmt_pct(s.ret_3m)} | RS {fmt_pct(s.rs_3m_vs_spy)}"
            )
            lines.append(f"   - 강점: {summarize_strengths(s)}")
            lines.append(f"   - 주의: {summarize_risks(s)}")
    else:
        lines.append("- 최종 포지션 없음")

    lines.append("")
    lines.append("[교체 요약]")
    lines.append("OUT")
    if out_items:
        for x in out_items:
            lines.append(
                f"- {x['ticker']} | {x['name']} | {x['industry']} | 산업 {x['industry_rank']}위 | score {fmt_num(x['stock_score'], 3)}"
            )
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("IN")
    if in_items:
        for x in in_items:
            lines.append(
                f"- {x['ticker']} | {x['name']} | {x['industry']} | 산업 {x['industry_rank']}위 | score {fmt_num(x['stock_score'], 3)} | 비중 {fmt_pct(x['target_weight'])}"
            )
    else:
        lines.append("- 없음")

    return "\n".join(lines)


def build_industry_candidate_lines(industry_scores: List[IndustryScore], stock_scores: List[StockScore]) -> List[str]:
    lines: List[str] = []
    separator = "────────────────"

    by_industry: Dict[str, List[StockScore]] = {}
    for s in stock_scores:
        by_industry.setdefault(s.industry, []).append(s)

    for industry in by_industry:
        by_industry[industry] = sorted(by_industry[industry], key=lambda x: x.industry_rank)

    for ind in industry_scores[:10]:
        lines.append(f"[{ind.rank}] {ind.industry}")
        lines.append(f"- industry score {ind.score:.3f} | slots {ind.slots}")
        lines.append(f"- {summarize_industry(ind)}")
        members = by_industry.get(ind.industry, [])[:5]
        for s in members:
            lines.append(
                f"  · {s.industry_rank}. {s.ticker} | {s.name} | score {s.total_score:.3f} | 1M {fmt_pct(s.ret_1m)} | 3M {fmt_pct(s.ret_3m)} | RS {fmt_pct(s.rs_3m_vs_spy)}"
            )
        lines.append(separator)
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()
    return lines


# ============================================================
# 메인
# ============================================================
def main() -> None:
    ensure_dirs()
    ensure_base_files()

    universe_df = load_industry_universe_csv()
    industry_map, name_map, _ = build_industry_membership(universe_df)

    spy_df = download_history("SPY")
    if spy_df.empty:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    as_of_date = str(pd.to_datetime(spy_df.iloc[-1]["Date"]).date())
    ensure_base_files(as_of_date)

    stock_scores = build_stock_scores(industry_map, name_map, spy_df)

    if not stock_scores:
        save_current_portfolio(as_of_date, [], {}, {})
        append_history(as_of_date, [], [])
        send_telegram_message(
            f"[EMR-6 산업 선행 리밸런싱]\n기준일: {as_of_date}\n종목 후보가 없습니다."
        )
        print("종목 후보가 없습니다.")
        return

    industry_scores = build_industry_scores(stock_scores)

    if not industry_scores:
        save_current_portfolio(as_of_date, [], {}, {})
        append_history(as_of_date, stock_scores, [])
        send_telegram_message(
            f"[EMR-6 산업 선행 리밸런싱]\n기준일: {as_of_date}\n산업 점수 계산 실패"
        )
        print("산업 점수 계산 실패")
        return

    final_positions, out_items, in_items, kept_items = build_final_portfolio(stock_scores, industry_scores)

    industry_rank_map = {x.industry: x.rank for x in industry_scores}
    industry_score_map = {x.industry: x.score for x in industry_scores}

    save_current_portfolio(as_of_date, final_positions, industry_rank_map, industry_score_map)
    append_rebalance_log(as_of_date, out_items, in_items, kept_items)
    append_history(as_of_date, stock_scores, [x.ticker for x in final_positions])

    report = build_report(
        as_of_date=as_of_date,
        industry_scores=industry_scores,
        stock_scores=stock_scores,
        final_positions=final_positions,
        out_items=out_items,
        in_items=in_items,
        kept_items=kept_items,
    )

    send_telegram_message(report)
    send_telegram_chunked("[EMR-6 산업/종목 랭킹]", build_industry_candidate_lines(industry_scores, stock_scores))
    print(report)


if __name__ == "__main__":
    main()
