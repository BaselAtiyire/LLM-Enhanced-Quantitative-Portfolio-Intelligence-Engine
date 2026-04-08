# reasoning_agent.py
from dotenv import load_dotenv
import os
from datetime import datetime
import sys
import csv
import math
import json
import sqlite3
from typing import Optional, List, Dict

import yfinance as yf
import numpy as np
import pandas as pd

load_dotenv()

# -----------------------------
# Optional Agno model setup (used only for CLI chat mode)
# -----------------------------
try:
    from agno.agent import Agent
    if os.getenv("GROQ_API_KEY"):
        from agno.models.groq import Groq
        _MODEL = Groq(id="llama-3.3-70b-versatile")
    else:
        from agno.models.openai import OpenAIResponses
        _MODEL = OpenAIResponses(id="gpt-5.2")

    _AGENT = Agent(
        model=_MODEL,
        reasoning=True,
        instructions=[
            "You are a finance assistant. Use ONLY the provided dataset.",
            "Do not invent metrics or prices.",
            "If the dataset doesn't contain a metric, say it's missing.",
            "Be concise and structured."
        ],
    )
except Exception:
    _AGENT = None


# =============================
# Formatting helpers
# =============================
def _safe_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def fmt_money(x):
    if x is None:
        return "N/A"
    x = float(x)
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if abs(x) >= div:
            return f"${x/div:.2f}{unit}"
    return f"${x:.2f}"


def fmt_price(x):
    return "N/A" if x is None else f"${float(x):.2f}"


def fmt_pct_percent(x):
    return "N/A" if x is None else f"{float(x):.2f}%"


def fmt_pct_decimal(x):
    return "N/A" if x is None else f"{float(x)*100:.2f}%"


def fmt_num(x, nd=2):
    return "N/A" if x is None else f"{float(x):.{nd}f}"


def fmt_vol(x):
    return "N/A" if x is None else f"{float(x)*100:.2f}%"


def fmt_range(low, high):
    if low is None or high is None:
        return "N/A"
    return f"{fmt_price(low)}–{fmt_price(high)}"


# =============================
# Price history + quant metrics
# =============================
def get_close_history(symbol: str, period: str = "1y") -> pd.Series:
    t    = yf.Ticker(symbol)
    hist = t.history(period=period)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float)
    closes      = hist["Close"].dropna()
    closes.name = symbol
    return closes


def returns_from_closes(closes: pd.Series) -> pd.Series:
    if closes is None or closes.empty:
        return pd.Series(dtype=float)
    rets      = closes.pct_change().dropna()
    rets.name = closes.name
    return rets


def ann_vol(daily_rets: pd.Series) -> Optional[float]:
    if daily_rets is None or daily_rets.empty:
        return None
    sd = float(daily_rets.std(ddof=1))
    return sd * math.sqrt(252)


def sharpe_ratio(daily_rets: pd.Series, rf_annual: float = 0.0) -> Optional[float]:
    if daily_rets is None or daily_rets.empty:
        return None
    sd = float(daily_rets.std(ddof=1))
    if sd == 0:
        return None
    rf_daily = rf_annual / 252.0
    mean     = float(daily_rets.mean())
    return ((mean - rf_daily) / sd) * math.sqrt(252)


def max_drawdown_from_returns(daily_rets: pd.Series) -> Optional[float]:
    if daily_rets is None or daily_rets.empty:
        return None
    equity = (1.0 + daily_rets).cumprod()
    peak   = equity.cummax()
    dd     = (equity / peak) - 1.0
    return float(dd.min())


def rolling_vol_20d(daily_rets: pd.Series) -> Optional[float]:
    if daily_rets is None or daily_rets.empty or len(daily_rets) < 25:
        return None
    rv  = daily_rets.rolling(20).std(ddof=1) * math.sqrt(252)
    val = rv.dropna()
    return float(val.iloc[-1]) if not val.empty else None


def compute_quant_pack(symbol: str, period: str = "1y", rf_annual: float = 0.0) -> dict:
    closes = get_close_history(symbol, period=period)
    rets   = returns_from_closes(closes)

    ret_30d      = None
    vol_ann_30d  = None
    if closes is not None and not closes.empty and len(closes) >= 31:
        window = closes.iloc[-31:]
        first  = float(window.iloc[0])
        last   = float(window.iloc[-1])
        if first != 0:
            ret_30d = (last / first) - 1.0
        rets_30     = returns_from_closes(window)
        vol_ann_30d = ann_vol(rets_30)

    return {
        "ret_30d":      _safe_float(ret_30d),
        "vol_ann":      _safe_float(vol_ann_30d),
        "sharpe":       _safe_float(sharpe_ratio(rets, rf_annual=rf_annual)),
        "max_drawdown": _safe_float(max_drawdown_from_returns(rets)),
        "roll_vol_20d": _safe_float(rolling_vol_20d(rets)),
    }


# =============================
# Snapshot (fundamentals + quant pack)
# =============================
def get_stock_snapshot(symbol: str, period: str = "1y", rf_annual: float = 0.0) -> dict:
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    price       = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    market_cap  = info.get("marketCap")
    pe          = info.get("trailingPE")
    week52_low  = info.get("fiftyTwoWeekLow")
    week52_high = info.get("fiftyTwoWeekHigh")

    hist_7d      = t.history(period="7d")
    one_week_pct = None
    if hist_7d is not None and not hist_7d.empty and "Close" in hist_7d.columns:
        closes = hist_7d["Close"].dropna()
        if len(closes) >= 2:
            first = float(closes.iloc[0])
            last  = float(closes.iloc[-1])
            if first != 0:
                one_week_pct = ((last / first) - 1.0) * 100.0

    qp = compute_quant_pack(symbol, period=period, rf_annual=rf_annual)

    return {
        "ticker":       symbol.upper(),
        "price":        _safe_float(price),
        "market_cap":   market_cap,
        "one_week_pct": _safe_float(one_week_pct),
        "trailing_pe":  _safe_float(pe),
        "week52_low":   _safe_float(week52_low),
        "week52_high":  _safe_float(week52_high),
        "ret_30d":      qp["ret_30d"],
        "vol_ann":      qp["vol_ann"],
        "sharpe":       qp["sharpe"],
        "max_drawdown": qp["max_drawdown"],
        "roll_vol_20d": qp["roll_vol_20d"],
    }


# =============================
# Ranking engine
# =============================
def _minmax_scale(values: List[float], higher_is_better: bool) -> Dict[float, float]:
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return {v: 1.0 for v in values}
    out = {}
    for v in values:
        s      = (v - vmin) / (vmax - vmin)
        out[v] = s if higher_is_better else (1.0 - s)
    return out


def add_scores_and_ranks(rows: List[dict], weights: dict) -> List[dict]:
    ws    = {k: float(v) for k, v in weights.items()}
    total = sum(ws.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    ws = {k: v / total for k, v in ws.items()}

    metric_values = {
        m: [float(r[m]) for r in rows if r.get(m) is not None]
        for m in ws
    }

    scalers = {
        "market_cap":   _minmax_scale(metric_values["market_cap"],   True)  if metric_values.get("market_cap")   else {},
        "one_week_pct": _minmax_scale(metric_values["one_week_pct"], True)  if metric_values.get("one_week_pct") else {},
        "ret_30d":      _minmax_scale(metric_values["ret_30d"],      True)  if metric_values.get("ret_30d")      else {},
        "trailing_pe":  _minmax_scale(metric_values["trailing_pe"],  False) if metric_values.get("trailing_pe")  else {},
        "vol_ann":      _minmax_scale(metric_values["vol_ann"],      False) if metric_values.get("vol_ann")      else {},
    }

    scored = []
    for r in rows:
        score  = 0.0
        used_w = 0.0
        for m, w in ws.items():
            v = r.get(m)
            if v is None:
                continue
            sv = scalers[m].get(float(v))
            if sv is None:
                continue
            score  += w * sv
            used_w += w
        score = (score / used_w) if used_w > 0 else None
        rr          = dict(r)
        rr["score"] = score
        scored.append(rr)

    scored.sort(key=lambda x: (x["score"] is None, -(x["score"] or -1)))
    for i, r in enumerate(scored, start=1):
        r["rank"] = i
    return scored


def build_table_markdown(rows: List[dict]) -> str:
    header = (
        "| Rank | Ticker | Score | Price | Mkt Cap | 1W % | 30D Ret | "
        "Ann Vol | Sharpe | Max DD | Roll Vol(20D) | P/E | 52W Range |\n"
    )
    sep = "| --- " * 13 + "|\n"
    lines = [header, sep]
    for r in rows:
        score_str = "N/A" if r.get("score") is None else f"{r['score']:.3f}"
        maxdd     = r.get("max_drawdown")
        maxdd_str = "N/A" if maxdd is None else f"{maxdd*100:.2f}%"
        lines.append(
            f"| {r.get('rank','')} | {r['ticker']} | {score_str} | "
            f"{fmt_price(r.get('price'))} | {fmt_money(r.get('market_cap'))} | "
            f"{fmt_pct_percent(r.get('one_week_pct'))} | {fmt_pct_decimal(r.get('ret_30d'))} | "
            f"{fmt_vol(r.get('vol_ann'))} | {fmt_num(r.get('sharpe'),2)} | "
            f"{maxdd_str} | {fmt_vol(r.get('roll_vol_20d'))} | "
            f"{fmt_num(r.get('trailing_pe'),2)} | "
            f"{fmt_range(r.get('week52_low'), r.get('week52_high'))} |\n"
        )
    return "".join(lines).strip()


# =============================
# CSV export
# =============================
def export_to_csv(rows: List[dict], as_of: str, filename: str | None = None) -> str:
    if filename is None:
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"financial_analysis_{stamp}.csv"

    fieldnames = [
        "as_of","ticker","price","market_cap","one_week_pct",
        "ret_30d","vol_ann","sharpe","max_drawdown","roll_vol_20d",
        "trailing_pe","week52_low","week52_high","score","rank"
    ]

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out         = {k: r.get(k) for k in fieldnames if k != "as_of"}
            out["as_of"] = as_of
            w.writerow(out)

    return filename


# =============================
# SQLite storage (+ migration)
# =============================
DB_PATH = "financial_runs.db"


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def db_init(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        as_of        TEXT NOT NULL,
        source       TEXT NOT NULL,
        tickers_json TEXT NOT NULL,
        weights_json TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL,
        ticker      TEXT NOT NULL,
        price       REAL,
        market_cap  INTEGER,
        one_week_pct REAL,
        trailing_pe  REAL,
        week52_low   REAL,
        week52_high  REAL,
        ret_30d      REAL,
        vol_ann      REAL,
        score        REAL,
        rank         INTEGER,
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );
    """)
    conn.commit()


def db_migrate(conn: sqlite3.Connection):
    """Safe migration — adds new columns if they don't exist."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(snapshots);")
    existing = {row[1] for row in cur.fetchall()}
    wanted   = {
        "sharpe":       "REAL",
        "max_drawdown": "REAL",
        "roll_vol_20d": "REAL",
    }
    for col, col_type in wanted.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {col_type};")
    conn.commit()


def db_insert_run(
    conn: sqlite3.Connection,
    as_of: str,
    source: str,
    tickers: List[str],
    weights: dict
) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (as_of, source, tickers_json, weights_json) VALUES (?, ?, ?, ?)",
        (as_of, source, json.dumps(tickers), json.dumps(weights))
    )
    conn.commit()
    return int(cur.lastrowid)


def db_insert_snapshots(conn: sqlite3.Connection, run_id: int, rows: List[dict]):
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO snapshots (
            run_id, ticker, price, market_cap, one_week_pct, trailing_pe,
            week52_low, week52_high, ret_30d, vol_ann, sharpe,
            max_drawdown, roll_vol_20d, score, rank
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            run_id,
            r.get("ticker"),
            r.get("price"),
            r.get("market_cap"),
            r.get("one_week_pct"),
            r.get("trailing_pe"),
            r.get("week52_low"),
            r.get("week52_high"),
            r.get("ret_30d"),
            r.get("vol_ann"),
            r.get("sharpe"),
            r.get("max_drawdown"),
            r.get("roll_vol_20d"),
            r.get("score"),
            r.get("rank"),
        )
        for r in rows
    ])
    conn.commit()


# =============================
# Backtest — walk-forward monthly rebalance
# =============================
def backtest_walkforward(
    tickers: List[str],
    period: str = "5y",
    rf_annual: float = 0.0,
    lookback_days: int = 252,
    rebalance_freq: str = "ME",   # month-end (pandas ≥ 2.2)
    n_samples_opt: int = 7000,
    seed: int = 42,
) -> dict:
    """
    Walk-forward backtest:
    - Downloads daily close prices for all tickers.
    - At each monthly rebalance date, optimises weights using the prior
      lookback window by searching for the max-Sharpe long-only portfolio
      via Dirichlet random sampling (7,000 samples).
    - Compares against a static equal-weight benchmark rebalanced monthly.
    - Returns equity curves, summary stats, and per-ticker details.

    Parameters
    ----------
    tickers        : list of ticker symbols
    period         : yfinance period string ("1y", "2y", "3y", "5y")
    rf_annual      : annual risk-free rate (decimal)
    lookback_days  : number of trading days used for optimisation window
    rebalance_freq : pandas offset alias for rebalance dates (default "ME")
    n_samples_opt  : number of random portfolios sampled per rebalance
    seed           : random seed for reproducibility
    """
    # ── 1. Download prices ────────────────────────────────────────────────
    closes = []
    valid  = []
    for t in tickers:
        s = get_close_history(t, period=period)
        if not s.empty:
            closes.append(s)
            valid.append(t)

    if len(valid) < 2:
        return {"error": "Need at least 2 tickers with valid price history."}

    df   = pd.concat(closes, axis=1).dropna(how="any")
    rets = df.pct_change().dropna()

    if len(rets) < lookback_days + 30:
        return {
            "error": (
                f"Not enough data ({len(rets)} days) for the chosen "
                f"lookback ({lookback_days} days) and period ({period}). "
                "Try a longer period or shorter lookback."
            )
        }

    # ── 2. Build monthly rebalance dates ──────────────────────────────────
    # Use month-end dates directly from the returns index (pandas-version safe)
    rebal_dates = (
        rets.groupby([rets.index.year, rets.index.month])
            .apply(lambda x: x.index[-1])
            .values
            .tolist()
    )
    rebal_dates = [pd.Timestamp(d) for d in rebal_dates]
    rebal_set   = set(rebal_dates)

    # ── 3. Walk-forward loop ──────────────────────────────────────────────
    np.random.seed(seed)

    eq_opt  = [1.0]
    eq_eqw  = [1.0]
    dates   = [rets.index[0]]

    n = len(valid)
    current_w_opt = np.ones(n) / n
    current_w_eqw = np.ones(n) / n

    def optimize_on_slice(ret_slice: pd.DataFrame) -> np.ndarray:
        mu    = ret_slice.mean().values
        cov   = ret_slice.cov().values
        rf_d  = rf_annual / 252.0
        best_sh = -1e18
        best_w  = None
        for _ in range(n_samples_opt):
            w  = np.random.dirichlet(np.ones(n))
            pr = float(np.dot(w, mu))
            pv = float(np.sqrt(np.dot(w, np.dot(cov, w))))
            if pv <= 0:
                continue
            sh = ((pr - rf_d) / pv) * math.sqrt(252)
            if sh > best_sh:
                best_sh = sh
                best_w  = w
        return best_w if best_w is not None else np.ones(n) / n

    for i in range(len(rets)):
        d = rets.index[i]

        if d in rebal_set and i >= lookback_days:
            look_slice    = rets.iloc[i - lookback_days: i]
            current_w_opt = optimize_on_slice(look_slice)
            current_w_eqw = np.ones(n) / n

        r_vec        = rets.iloc[i].values
        port_r_opt   = float(np.dot(current_w_opt, r_vec))
        port_r_eqw   = float(np.dot(current_w_eqw, r_vec))

        eq_opt.append(eq_opt[-1] * (1.0 + port_r_opt))
        eq_eqw.append(eq_eqw[-1] * (1.0 + port_r_eqw))
        dates.append(d)

    eq_opt_s = pd.Series(eq_opt[1:], index=dates[1:], name="HPIE")
    eq_eqw_s = pd.Series(eq_eqw[1:], index=dates[1:], name="Equal-Weight")

    # ── 4. Compute summary stats ──────────────────────────────────────────
    def summarize(equity: pd.Series) -> dict:
        daily   = equity.pct_change().dropna()
        ar      = float(daily.mean() * 252)
        av      = float(daily.std(ddof=1) * math.sqrt(252))
        sh      = sharpe_ratio(daily, rf_annual=rf_annual)
        mdd     = max_drawdown_from_returns(daily)
        total_r = float(equity.iloc[-1] / equity.iloc[0] - 1)
        return {
            "ann_return":  round(ar,    4),
            "ann_vol":     round(av,    4),
            "sharpe":      round(sh,    3) if sh is not None else None,
            "max_drawdown":round(mdd,   4) if mdd is not None else None,
            "total_return":round(total_r,4),
        }

    return {
        "tickers":       valid,
        "period":        period,
        "lookback_days": lookback_days,
        "rebalance":     rebalance_freq,
        "opt_summary":   summarize(eq_opt_s),
        "eqw_summary":   summarize(eq_eqw_s),
        "equity_opt":    eq_opt_s,
        "equity_eqw":    eq_eqw_s,
    }


# =============================
# CLI (optional)
# =============================
def run_chat_cli(scored_rows: List[dict], as_of: str, source: str):
    if _AGENT is None:
        print("\nAgno not available. Install agno + model deps.")
        return
    dataset = {"as_of": as_of, "source": source, "table": scored_rows}
    print("\nChat mode: type a question (or 'exit')\n")
    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        prompt = f"""
Dataset (use only this):
{dataset}

User question: {q}

Rules:
- Answer using only dataset fields.
- If user asks for a metric not in dataset, say it's missing.
""".strip()
        _AGENT.print_response(prompt)


def parse_cli(argv: List[str]):
    flags = {"--csv": False, "--chat": False, "--store": False}
    period  = "1y"
    rf      = 0.0
    weights = {
        "market_cap":   0.30,
        "one_week_pct": 0.25,
        "ret_30d":      0.15,
        "trailing_pe":  0.15,
        "vol_ann":      0.15,
    }
    tickers = []
    i = 0
    while i < len(argv):
        a = argv[i].strip()
        if a in flags:
            flags[a] = True; i += 1; continue
        if a == "--period":
            period = argv[i + 1]; i += 2; continue
        if a == "--rf":
            rf = float(argv[i + 1]); i += 2; continue
        if a in {"--w_mcap","--w_mom","--w_ret","--w_pe","--w_vol"}:
            val = float(argv[i + 1])
            key = {
                "--w_mcap":"market_cap","--w_mom":"one_week_pct",
                "--w_ret":"ret_30d","--w_pe":"trailing_pe","--w_vol":"vol_ann"
            }[a]
            weights[key] = val; i += 2; continue
        tickers.append(a.upper()); i += 1

    if len(tickers) < 2:
        tickers = ["NVDA", "AMD", "TSLA", "AAPL"]

    return flags, tickers, weights, period, rf


if __name__ == "__main__":
    flags, tickers, weights, period, rf = parse_cli(sys.argv[1:])

    as_of  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source = "yfinance (Yahoo Finance)"

    raw_rows    = [get_stock_snapshot(t, period=period, rf_annual=rf) for t in tickers]
    scored_rows = add_scores_and_ranks(raw_rows, weights=weights)

    print("\n## Ranked Table")
    print(build_table_markdown(scored_rows))
    print(f"\nWeights (normalized internally): {weights}")
    print(f"History period: {period} | rf={rf}")

    if flags["--csv"]:
        fname = export_to_csv(scored_rows, as_of=as_of)
        print(f"\nCSV saved: {fname}")

    if flags["--store"]:
        conn = db_connect()
        db_init(conn)
        db_migrate(conn)
        run_id = db_insert_run(
            conn, as_of=as_of, source=source,
            tickers=tickers, weights=weights
        )
        db_insert_snapshots(conn, run_id=run_id, rows=scored_rows)
        conn.close()
        print(f"\nStored run in SQLite: run_id={run_id} (db={DB_PATH})")

    if flags["--chat"]:
        run_chat_cli(scored_rows, as_of=as_of, source=source)
