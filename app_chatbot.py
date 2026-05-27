import os
import json
from datetime import datetime

import streamlit as st
import yaml
from yaml.loader import SafeLoader

from dotenv import load_dotenv
import pandas as pd
import bcrypt

from agno.agent import Agent

from reasoning_agent import (
    get_stock_snapshot,
    add_scores_and_ranks,
    build_table_markdown,
    export_to_csv,
    db_connect,
    db_init,
    db_migrate,
    db_insert_run,
    db_insert_snapshots,
    backtest_walkforward,
)

# =========================
# Page config FIRST
# =========================
st.set_page_config(page_title="AI Financial Chatbot (Agno)", layout="wide")
st.title("📊 AI Financial Chatbot (Agno)")
st.caption("Sign in to continue.")

load_dotenv()

# =========================
# Load config.yaml
# =========================
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=SafeLoader)

USERS  = config.get("credentials", {}).get("usernames", {})
COOKIE = config.get("cookie", {})

# =========================
# Session auth state
# =========================
if "auth_ok"   not in st.session_state: st.session_state.auth_ok   = False
if "auth_user" not in st.session_state: st.session_state.auth_user = None
if "auth_name" not in st.session_state: st.session_state.auth_name = None
if "auth_role" not in st.session_state: st.session_state.auth_role = None

# =========================
# Auth helpers
# =========================
def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def do_logout():
    st.session_state.auth_ok   = False
    st.session_state.auth_user = None
    st.session_state.auth_name = None
    st.session_state.auth_role = None
    st.rerun()

# =========================
# Login UI
# =========================
if not st.session_state.auth_ok:
    st.subheader("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username", placeholder="e.g., admin or basil")
        p = st.text_input("Password", type="password", placeholder="your password")
        submitted = st.form_submit_button("Login")

    if submitted:
        u    = (u or "").strip()
        user = USERS.get(u)
        if not user:
            st.error("Unknown username. Check config.yaml.")
            st.stop()
        hashed = user.get("password", "")
        if not verify_password(p or "", hashed):
            st.error("Incorrect password.")
            st.stop()
        st.session_state.auth_ok   = True
        st.session_state.auth_user = u
        st.session_state.auth_name = user.get("name", u)
        st.session_state.auth_role = user.get("role", "user")
        st.success("Logged in ✅")
        st.rerun()

    st.info("Enter your username and password above, then click Login.")
    st.stop()

# =========================
# Logged-in sidebar
# =========================
username  = st.session_state.auth_user
name      = st.session_state.auth_name
user_role = st.session_state.auth_role

st.sidebar.success(f"Signed in as: {name} ({user_role})")
if st.sidebar.button("Logout"):
    do_logout()

# =========================
# Agno model selection
# =========================
if os.getenv("GROQ_API_KEY"):
    from agno.models.groq import Groq
    model = Groq(id="llama-3.1-8b-instant")
elif os.getenv("OPENAI_API_KEY"):
    from agno.models.openai import OpenAIResponses
    model = OpenAIResponses(id="gpt-4o")
else:
    st.error(
        "⚠️ No LLM API key found. Please add GROQ_API_KEY to your .env file "
        "and restart the app. Get a free key at https://console.groq.com"
    )
    st.stop()

agent = Agent(
    model=model,
    reasoning=False,
    instructions=[
        "You are a concise finance assistant.",
        "Use ONLY the dataset provided in the user prompt.",
        "Answer in plain English. Never output raw JSON or code blocks.",
        "Never repeat the answer or say 'The final answer is'.",
        "Do not invent prices or metrics.",
        "If a metric is missing, say it is missing.",
        "Format numbers cleanly: percentages as X.XX%, prices as $X.XX.",
    ],
)

# =========================
# Helpers
# =========================
GLOSSARY = {
    "Score":        "Composite score from your weights (higher is better).",
    "1W %":         "Percent price change over the last ~7 trading days.",
    "30D Ret":      "Approx 30-day return (from last ~31 closes).",
    "Ann Vol":      "Annualized volatility from daily returns (risk).",
    "Sharpe":       "Risk-adjusted return; higher is better.",
    "Max DD":       "Maximum drawdown (peak-to-trough decline).",
    "Roll Vol(20D)":"Latest rolling 20-trading-day annualized volatility.",
    "P/E":          "Trailing price-to-earnings ratio.",
    "Mkt Cap":      "Market capitalization (company size).",
}

def dataset_key(ds: dict) -> str:
    if not ds:
        return "no_dataset"
    if ds.get("run_id"):
        return f"run_{ds['run_id']}"
    return f"tmp_{ds.get('as_of','')}_{','.join(ds.get('tickers', []))}"

COLUMN_LABELS = {
    "rank": "Rank", "ticker": "Ticker", "score": "Score",
    "price": "Price", "market_cap": "Mkt Cap",
    "one_week_pct": "1W %", "ret_30d": "30D Ret",
    "vol_ann": "Ann Vol", "sharpe": "Sharpe",
    "max_drawdown": "Max DD", "roll_vol_20d": "Roll Vol(20D)",
    "trailing_pe": "P/E", "week52_low": "52W Low", "week52_high": "52W High",
}

def ds_to_dataframe(ds: dict, display: bool = False) -> pd.DataFrame:
    if not ds or "table" not in ds:
        return pd.DataFrame()
    df = pd.DataFrame(ds["table"]).copy()
    num_cols = ["price","market_cap","one_week_pct","ret_30d","vol_ann",
                "trailing_pe","score","rank","sharpe","max_drawdown","roll_vol_20d"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if display:
        if "price" in df.columns:
            df["price"] = df["price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        if "market_cap" in df.columns:
            df["market_cap"] = df["market_cap"].apply(
                lambda x: f"${x/1e12:.2f}T" if pd.notna(x) and x>=1e12
                else (f"${x/1e9:.2f}B" if pd.notna(x) else ""))
        for pc in ["one_week_pct","ret_30d","vol_ann","max_drawdown","roll_vol_20d"]:
            if pc in df.columns:
                df[pc] = df[pc].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        if "score" in df.columns:
            df["score"] = df["score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        if "sharpe" in df.columns:
            df["sharpe"] = df["sharpe"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        if "trailing_pe" in df.columns:
            df["trailing_pe"] = df["trailing_pe"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        df = df.rename(columns=COLUMN_LABELS)
        priority = ["Rank","Ticker","Score","Price","Mkt Cap","1W %","30D Ret",
                    "Ann Vol","Sharpe","Max DD","Roll Vol(20D)","P/E","52W Low","52W High"]
        cols = [c for c in priority if c in df.columns] + [c for c in df.columns if c not in priority]
        df = df[cols]
    return df

# =========================
# Session defaults
# =========================
if "dataset"         not in st.session_state: st.session_state.dataset         = None
if "chat_by_dataset" not in st.session_state: st.session_state.chat_by_dataset = {}
if "bt_result"       not in st.session_state: st.session_state.bt_result       = None

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["🧠 Chat & Analysis", "🆚 Compare Runs (Admin)", "📈 Backtest (Admin)"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Chat & Analysis
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.sidebar.header("⚙️ Controls")

    tickers_text = st.sidebar.text_input(
        "Tickers (comma-separated)", "NVDA, AMD, TSLA, AAPL"
    )
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    period = st.sidebar.selectbox(
        "History period", ["3mo","6mo","1y","2y","5y"], index=2
    )
    rf = st.sidebar.number_input(
        "Risk-free rate (annual, decimal)", 0.0, 0.20, 0.00, 0.01
    )

    # ── DATE PICKER (reproducibility) ────────────────────────────────────────
    import datetime as _dt
    st.sidebar.subheader("📅 Data as of date")
    selected_date = st.sidebar.date_input(
        label="Fetch market data as of",
        value=_dt.date(2026, 5, 8),
        min_value=_dt.date(2020, 1, 1),
        max_value=_dt.date.today(),
        help="Set to 2026-05-08 to reproduce the exact results reported in the paper.",
    )
    selected_date_str = selected_date.strftime("%Y-%m-%d")
    st.sidebar.caption(
        f"📌 Data as of: **{selected_date_str}**  \n"
        "Set to **2026-05-08** to reproduce paper results."
    )
    # ─────────────────────────────────────────────────────────────────────────

    st.sidebar.subheader("Ranking weights")
    w_mcap = st.sidebar.number_input("Market Cap",               value=0.30, step=0.05)
    w_mom  = st.sidebar.number_input("1W %",                     value=0.25, step=0.05)
    w_ret  = st.sidebar.number_input("30D Return",               value=0.15, step=0.05)
    w_pe   = st.sidebar.number_input("Trailing P/E (lower better)", value=0.15, step=0.05)
    w_vol  = st.sidebar.number_input("Ann Vol (lower better)",   value=0.15, step=0.05)

    weights = {
        "market_cap":   float(w_mcap),
        "one_week_pct": float(w_mom),
        "ret_30d":      float(w_ret),
        "trailing_pe":  float(w_pe),
        "vol_ann":      float(w_vol),
    }

    run_btn = st.sidebar.button("▶ Run")

    if run_btn:
        if len(tickers) < 2:
            st.sidebar.error("Enter at least 2 tickers.")
        else:
            with st.spinner("Fetching data + computing metrics..."):
                raw_rows    = [get_stock_snapshot(t, period=period, rf_annual=rf, end_date=selected_date_str) for t in tickers]
                scored_rows = add_scores_and_ranks(raw_rows, weights=weights)

            as_of  = f"{selected_date_str} {datetime.now().strftime('%H:%M:%S')}"
            source = "yfinance (Yahoo Finance)"

            # ── Save to SQLite ──────────────────────────────────────────────
            conn = db_connect()
            db_init(conn)
            db_migrate(conn)
            run_id = db_insert_run(
                conn, as_of=as_of, source=source,
                tickers=tickers, weights=weights
            )
            db_insert_snapshots(conn, run_id=run_id, rows=scored_rows)
            conn.close()
            # ───────────────────────────────────────────────────────────────

            st.session_state.dataset = {
                "run_id":  run_id,
                "as_of":   as_of,
                "source":  source,
                "weights": weights,
                "tickers": tickers,
                "table":   scored_rows,
                "period":  period,
                "rf":      rf,
            }
            st.session_state.chat_by_dataset = {}
            st.sidebar.success(f"Analysis ready ✅  (run_id={run_id})")

    ds   = st.session_state.dataset
    dkey = dataset_key(ds)
    st.session_state.chat_by_dataset.setdefault(dkey, [])

    # ── Dataset (full width) ──────────────────────────────────────────────────
    st.subheader("📌 Dataset")
    if not ds:
        st.info("Click ▶ Run to generate a dataset.")
    else:
        st.write(f"**As of:** {ds['as_of']}  |  **Run ID:** {ds.get('run_id','—')}")
        st.write(f"**Tickers:** {', '.join(ds['tickers'])}")
        st.write(f"**Weights:** {ds['weights']}")

        # Full-width scrollable dataframe
        df = ds_to_dataframe(ds, display=True)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

        col_csv, col_exp = st.columns([1, 3])
        with col_csv:
            if st.button("⬇️ Export CSV"):
                fname = export_to_csv(ds["table"], as_of=ds["as_of"])
                with open(fname, "rb") as f_:
                    st.download_button(
                        "Download CSV", data=f_, file_name=fname, mime="text/csv"
                    )
        with col_exp:
            with st.expander("📘 Glossary"):
                for k, v in GLOSSARY.items():
                    st.markdown(f"**{k}** — {v}")

        if not df.empty and "Score" in df.columns:
            st.markdown("### 📊 Score chart")
            try:
                import plotly.graph_objects as go
                # Use raw numeric data to avoid string-sort issues
                raw = pd.DataFrame(ds["table"]).copy()
                raw["score"] = pd.to_numeric(raw["score"], errors="coerce")
                raw["rank"]  = pd.to_numeric(raw["rank"],  errors="coerce")
                raw = raw.dropna(subset=["score","rank"]).sort_values("rank")
                fig = go.Figure(go.Bar(
                    x=raw["ticker"],
                    y=raw["score"],
                    marker_color="#1f77b4",
                    text=raw["score"].apply(lambda v: f"{v:.3f}"),
                    textposition="outside",
                ))
                fig.update_layout(
                    xaxis_title="Ticker", yaxis_title="Score",
                    height=400, margin=dict(t=40, b=40),
                    yaxis=dict(range=[0, raw["score"].max() * 1.18])
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.bar_chart(
                    df.sort_values("Rank")[["Ticker","Score"]].set_index("Ticker"),
                    use_container_width=True
                )

    st.divider()

    # ── Chat (full width below dataset) ───────────────────────────────────────
    st.subheader("💬 Chat (uses ONLY dataset facts)")
    for m in st.session_state.chat_by_dataset[dkey]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask about ranks, volatility, Sharpe, drawdown, valuation…")
    if q:
        if not ds:
            st.warning("Run a dataset first (▶ Run in the sidebar).")
        else:
            st.session_state.chat_by_dataset[dkey].append({"role":"user","content":q})
            # Compact dataset — one line per ticker to save tokens
            def _fmt(v, pct=False):
                try:
                    f = float(v)
                    return f"{f*100:.2f}%" if pct else f"{f:.3f}"
                except Exception:
                    return str(v) if v not in (None, "") else "n/a"

            rows = []
            for r in ds["table"]:
                rows.append(
                    f"{r.get('rank','?')}. {r.get('ticker','?')} | "
                    f"Score={_fmt(r.get('score'))} | "
                    f"Price=${float(r.get('price',0)):,.2f} | "
                    f"1W={_fmt(r.get('one_week_pct'),True)} | "
                    f"30D={_fmt(r.get('ret_30d'),True)} | "
                    f"AnnVol={_fmt(r.get('vol_ann'),True)} | "
                    f"Sharpe={_fmt(r.get('sharpe'))} | "
                    f"MaxDD={_fmt(r.get('max_drawdown'),True)} | "
                    f"P/E={_fmt(r.get('trailing_pe'))} | "
                    f"MktCap={r.get('market_cap','n/a')}"
                )
            compact = "\n".join(rows)

            prompt = f"""You are a concise finance assistant. Answer using ONLY the data below.
Rules: plain English only, 2-4 sentences max, no JSON, no code formatting, no backticks, no repetition.
Format numbers cleanly: percentages as X.XX%, prices as $X.XX, market caps as $X.XXB or $X.XXT.

Tickers ranked by score:
{compact}

Question: {q}""".strip()
            with st.chat_message("assistant"):
                resp = agent.run(prompt)
                import re
                if hasattr(resp, "content"):
                    text = resp.content
                else:
                    text = str(resp)
                text = re.sub(r"```[a-z]*\n.*?```", "", text, flags=re.DOTALL).strip()
                lines = text.split("\n")
                seen = set()
                clean = []
                for line in lines:
                    normalized = line.strip().lower()
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        clean.append(line)
                text = "\n".join(clean).strip()
                st.markdown(text)
            st.session_state.chat_by_dataset[dkey].append(
                {"role":"assistant","content":text}
            )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Compare Runs (Admin)
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if user_role != "admin":
        st.info("🔒 Admin only.")
    else:
        st.header("🆚 Compare Runs")
        st.caption("Select any two saved runs to compare rank movement and score changes.")

        conn = db_connect()
        db_init(conn)
        db_migrate(conn)

        runs_df = pd.read_sql(
            "SELECT run_id, as_of, tickers_json, weights_json "
            "FROM runs ORDER BY run_id DESC",
            conn
        )

        if runs_df.empty:
            st.info("No saved runs yet. Use ▶ Run in the Controls panel to create runs.")
            conn.close()
        else:
            st.subheader("📋 All Saved Runs")
            st.dataframe(runs_df, use_container_width=True, hide_index=True)

            run_ids = runs_df["run_id"].tolist()
            col_a, col_b = st.columns(2)
            run_a = col_a.selectbox(
                "Run A (baseline)", run_ids, index=0, key="run_a"
            )
            run_b = col_b.selectbox(
                "Run B (compare)", run_ids,
                index=min(1, len(run_ids) - 1), key="run_b"
            )

            if st.button("🔍 Compare Selected Runs"):
                if run_a == run_b:
                    st.warning("Select two different runs to compare.")
                else:
                    snap_a = pd.read_sql(
                        f"SELECT ticker, rank, score, sharpe, vol_ann, max_drawdown "
                        f"FROM snapshots WHERE run_id={run_a}", conn
                    )
                    snap_b = pd.read_sql(
                        f"SELECT ticker, rank, score, sharpe, vol_ann, max_drawdown "
                        f"FROM snapshots WHERE run_id={run_b}", conn
                    )

                    merged = snap_a.merge(
                        snap_b, on="ticker",
                        suffixes=(f" (Run {run_a})", f" (Run {run_b})")
                    )
                    merged["Rank Δ"]  = (
                        merged[f"rank (Run {run_a})"] - merged[f"rank (Run {run_b})"]
                    ).astype(int)
                    merged["Score Δ"] = (
                        merged[f"score (Run {run_a})"] - merged[f"score (Run {run_b})"]
                    ).round(4)
                    merged = merged.sort_values("Rank Δ")

                    st.subheader("📊 Rank & Score Movement")
                    st.dataframe(merged, use_container_width=True, hide_index=True)

                    st.subheader("📈 Score Comparison Chart")
                    tickers_common = snap_a["ticker"].tolist()
                    scores_a = snap_a.set_index("ticker")["score"]
                    scores_b = snap_b.set_index("ticker")["score"]
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure(data=[
                            go.Bar(name=f"Run {run_a}", x=tickers_common,
                                   y=[scores_a.get(t, 0) for t in tickers_common],
                                   marker_color="#1f77b4"),
                            go.Bar(name=f"Run {run_b}", x=tickers_common,
                                   y=[scores_b.get(t, 0) for t in tickers_common],
                                   marker_color="#aec7e8"),
                        ])
                        fig.update_layout(barmode="group", height=420,
                            xaxis_title="Ticker", yaxis_title="Score",
                            margin=dict(t=20,b=40))
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        chart_df = pd.DataFrame({
                            f"Run {run_a}": scores_a, f"Run {run_b}": scores_b
                        }).dropna()
                        st.bar_chart(chart_df, use_container_width=True)

            conn.close()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Backtest (Admin)
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    if user_role != "admin":
        st.info("🔒 Admin only.")
    else:
        st.header("📈 Walk-Forward Backtest")
        st.caption(
            "Sharpe-optimised portfolio weights · Monthly rebalancing · "
            "252-day rolling lookback · Equal-weight benchmark comparison · "
            "5 bps transaction costs applied"
        )

        # ── Backtest controls ─────────────────────────────────────────────
        with st.expander("⚙️ Backtest Settings", expanded=True):
            bt_tickers_text = st.text_input(
                "Tickers for backtest (comma-separated)",
                value=tickers_text,
                key="bt_tickers_input"
            )
            bt_tickers = [
                t.strip().upper() for t in bt_tickers_text.split(",") if t.strip()
            ]

            col_s1, col_s2, col_s3 = st.columns(3)
            bt_period  = col_s1.selectbox(
                "Backtest period", ["1y","2y","3y","5y"], index=3, key="bt_period"
            )
            bt_rf      = col_s2.number_input(
                "Risk-free rate", 0.0, 0.20, float(rf), 0.01, key="bt_rf"
            )
            bt_lookback = col_s3.number_input(
                "Lookback window (days)", 60, 504, 252, 21, key="bt_lookback"
            )

        run_bt = st.button("▶ Run Backtest", type="primary")

        if run_bt:
            if len(bt_tickers) < 2:
                st.error("Enter at least 2 tickers.")
            else:
                with st.spinner(
                    f"Running walk-forward backtest on {len(bt_tickers)} tickers "
                    f"over {bt_period}… this may take 30–60 seconds."
                ):
                    result = backtest_walkforward(
                        tickers=bt_tickers,
                        period=bt_period,
                        rf_annual=bt_rf,
                        lookback_days=int(bt_lookback),
                    )
                st.session_state.bt_result = result

        # ── Display results ───────────────────────────────────────────────
        result = st.session_state.bt_result

        if result is None:
            st.info("Configure settings above and click ▶ Run Backtest.")

        elif "error" in result:
            st.error(f"Backtest error: {result['error']}")

        else:
            opt = result["opt_summary"]
            eqw = result["eqw_summary"]

            # ── Key metrics ───────────────────────────────────────────────
            st.subheader("📊 Performance Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Sharpe Ratio — HPIE",
                f"{opt['sharpe']:.3f}",
                delta=f"{opt['sharpe'] - eqw['sharpe']:+.3f} vs benchmark"
            )
            c2.metric(
                "Annualised Return — HPIE",
                f"{opt['ann_return']*100:.2f}%",
                delta=f"{(opt['ann_return'] - eqw['ann_return'])*100:+.2f}%"
            )
            c3.metric(
                "Max Drawdown — HPIE",
                f"{opt['max_drawdown']*100:.2f}%",
                delta=f"{(opt['max_drawdown'] - eqw['max_drawdown'])*100:+.2f}%",
                delta_color="inverse"
            )
            c4.metric(
                "Annualised Vol — HPIE",
                f"{opt['ann_vol']*100:.2f}%",
                delta=f"{(opt['ann_vol'] - eqw['ann_vol'])*100:+.2f}%",
                delta_color="inverse"
            )

            # ── Equity curve ──────────────────────────────────────────────
            st.subheader("📈 Equity Curve — HPIE vs Equal-Weight Benchmark")
            equity_df = pd.DataFrame({
                "HPIE (Sharpe-Optimised)": result["equity_opt"].values,
                "Equal-Weight Benchmark":  result["equity_eqw"].values,
            }, index=result["equity_opt"].index)
            st.line_chart(equity_df, use_container_width=True)

            # ── Side-by-side table ────────────────────────────────────────
            st.subheader("📋 Full Comparison Table")
            comp = pd.DataFrame({
                "Metric": [
                    "Sharpe Ratio",
                    "Annualised Return (%)",
                    "Annualised Volatility (%)",
                    "Maximum Drawdown (%)",
                ],
                "HPIE (Sharpe-Optimised)": [
                    f"{opt['sharpe']:.3f}",
                    f"{opt['ann_return']*100:.2f}",
                    f"{opt['ann_vol']*100:.2f}",
                    f"{opt['max_drawdown']*100:.2f}",
                ],
                "Equal-Weight Benchmark": [
                    f"{eqw['sharpe']:.3f}",
                    f"{eqw['ann_return']*100:.2f}",
                    f"{eqw['ann_vol']*100:.2f}",
                    f"{eqw['max_drawdown']*100:.2f}",
                ],
                "Difference (HPIE − Benchmark)": [
                    f"{opt['sharpe'] - eqw['sharpe']:+.3f}",
                    f"{(opt['ann_return'] - eqw['ann_return'])*100:+.2f}",
                    f"{(opt['ann_vol'] - eqw['ann_vol'])*100:+.2f}",
                    f"{(opt['max_drawdown'] - eqw['max_drawdown'])*100:+.2f}",
                ],
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)

            # ── Run details ───────────────────────────────────────────────
            with st.expander("🔍 Backtest Details"):
                st.write(f"**Tickers ({len(result['tickers'])}):** "
                         f"{', '.join(result['tickers'])}")
                st.write(f"**Period:** {result['period']}  |  "
                         f"**Lookback:** {result['lookback_days']} days  |  "
                         f"**Rebalance:** Monthly")
                st.write(
                    "**Methodology:** At each monthly rebalance, the engine scores "
                    "all tickers using the prior lookback window and searches for "
                    "the maximum-Sharpe long-only portfolio via Dirichlet sampling "
                    "(7,000 random portfolios). The benchmark is a static equal-weight "
                    "allocation rebalanced monthly."
                )

            # ── Export backtest results ───────────────────────────────────
            st.subheader("⬇️ Export Backtest Results")
            bt_export = pd.DataFrame({
                "Metric":    comp["Metric"],
                "HPIE":      comp["HPIE (Sharpe-Optimised)"],
                "Benchmark": comp["Equal-Weight Benchmark"],
                "Delta":     comp["Difference (HPIE − Benchmark)"],
            })
            st.download_button(
                label="Download Results CSV",
                data=bt_export.to_csv(index=False),
                file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
