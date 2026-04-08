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
    model = Groq(id="llama-3.3-70b-versatile")
else:
    from agno.models.openai import OpenAIResponses
    model = OpenAIResponses(id="gpt-5.2")

agent = Agent(
    model=model,
    reasoning=True,
    instructions=[
        "You are a finance assistant.",
        "Use ONLY the provided dataset in the prompt.",
        "Do not invent prices/metrics.",
        "If a metric is missing, say it's missing.",
        "Be concise, structured, and evidence-based.",
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

def ds_to_dataframe(ds: dict) -> pd.DataFrame:
    if not ds or "table" not in ds:
        return pd.DataFrame()
    df = pd.DataFrame(ds["table"]).copy()
    for col in ["price","market_cap","one_week_pct","ret_30d","vol_ann",
                "trailing_pe","score","rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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
                raw_rows    = [get_stock_snapshot(t, period=period, rf_annual=rf) for t in tickers]
                scored_rows = add_scores_and_ranks(raw_rows, weights=weights)

            as_of  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            st.sidebar.success(f"Analysis ready ✅  (run_id={run_id})")

    ds   = st.session_state.dataset
    dkey = dataset_key(ds)
    st.session_state.chat_by_dataset.setdefault(dkey, [])

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("📌 Dataset")
        if not ds:
            st.info("Click ▶ Run to generate a dataset.")
        else:
            st.write(f"**As of:** {ds['as_of']}  |  **Run ID:** {ds.get('run_id','—')}")
            st.write(f"**Tickers:** {', '.join(ds['tickers'])}")
            st.write(f"**Weights:** {ds['weights']}")
            st.markdown(build_table_markdown(ds["table"]))

            st.markdown("### ⬇️ Export CSV")
            if st.button("Export CSV"):
                fname = export_to_csv(ds["table"], as_of=ds["as_of"])
                with open(fname, "rb") as f_:
                    st.download_button(
                        "Download CSV", data=f_, file_name=fname, mime="text/csv"
                    )

            df = ds_to_dataframe(ds)
            if not df.empty and df["score"].notna().any():
                st.markdown("### 📊 Score chart")
                st.bar_chart(
                    df.sort_values("rank")[["ticker","score"]].set_index("ticker"),
                    use_container_width=True
                )

            with st.expander("📘 Glossary"):
                for k, v in GLOSSARY.items():
                    st.markdown(f"**{k}** — {v}")

    with right:
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
                prompt = f"""
Dataset (use ONLY this JSON):
{json.dumps(ds, indent=2, default=str)}

User question:
{q}
Rules:
- Use only dataset fields.
- If missing, say it's missing.
""".strip()
                with st.chat_message("assistant"):
                    resp = agent.run(prompt)
                    text = resp.content if hasattr(resp, "content") else str(resp)
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
                    chart_df = pd.DataFrame({
                        f"Score (Run {run_a})": snap_a.set_index("ticker")["score"],
                        f"Score (Run {run_b})": snap_b.set_index("ticker")["score"],
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
