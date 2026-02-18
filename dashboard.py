"""
Trading Strategy Dashboard
===========================
Comprehensive Streamlit dashboard for visualizing backtest results.
Includes a 3-way head-to-head: Minervini vs Parallel Activity vs Hybrid,
with honest, skepticism-addressing metrics (slippage, holding time,
max drawdown streak, median vs mean PnL).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #a0aec0 !important; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    .warning-box { background: #2d2000; border: 1px solid #b8860b; border-radius: 8px; padding: 12px; margin: 8px 0; }
    .success-box { background: #002d1a; border: 1px solid #00b894; border-radius: 8px; padding: 12px; margin: 8px 0; }
    .hybrid-badge { background: linear-gradient(135deg, #6c5ce7 0%, #00b894 100%); 
                    color: white; padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C = {
    "accent": "#6c5ce7", "green": "#00b894", "red": "#e17055",
    "blue": "#0984e3", "orange": "#fdcb6e", "purple": "#a29bfe",
    "teal": "#00cec9", "pink": "#fd79a8", "hybrid": "#e84393",
}
SETUP_COLORS = {
    "Value Area Rule": "#6c5ce7", "Failed Range Extension": "#e17055",
    "Parallel Activity": "#00b894", "Single Print Retracement": "#0984e3",
    "Go-With Breakout": "#fdcb6e",
}
STRAT_COLORS = {
    "Minervini": C["blue"],
    "Parallel Activity": C["green"],
    "Hybrid": C["hybrid"],
}
TPL = "plotly_dark"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
RESULTS_DIR = "."

def discover_files():
    files = {}
    for f in os.listdir(RESULTS_DIR):
        if not f.endswith(".csv") or "portfolio" in f or f in ("nifty_200_raw.csv", "verification_daily.csv"):
            continue
        if f.startswith("minervini_trades"):
            tf = f.replace("minervini_trades_", "").replace("minervini_trades", "default").replace(".csv", "")
            files[f] = {"label": f"Minervini — {tf.replace('_',' ').title()}", "strategy": "Minervini", "tf": tf}
        elif f.startswith("hybrid_trades"):
            tf = f.replace("hybrid_trades_", "").replace("hybrid_trades", "default").replace(".csv", "")
            files[f] = {"label": f"Hybrid — {tf.replace('_',' ').title()}", "strategy": "Hybrid", "tf": tf}
        elif f.startswith("value_area_trades"):
            tf = f.replace("value_area_trades_", "").replace("value_area_trades", "default").replace(".csv", "")
            files[f] = {"label": f"Value Area — {tf.replace('_',' ').title()}", "strategy": "Value Area", "tf": tf}
    return files

@st.cache_data
def load_trades(filepath):
    df = pd.read_csv(filepath)
    for col in ["Entry Date", "Exit Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data
def load_portfolio(filepath):
    pf_path = filepath.replace(".csv", "_portfolio.csv")
    if os.path.exists(pf_path):
        pf = pd.read_csv(pf_path)
        pf["Date"] = pd.to_datetime(pf["Date"], errors="coerce")
        return pf
    return None

def compute_metrics(df):
    if df.empty:
        return {}
    total = len(df)
    wins = (df["PnL"] > 0).sum()
    losses = (df["PnL"] <= 0).sum()
    wr = wins / total if total else 0
    avg = df["PnL"].mean()
    med = df["PnL"].median()
    pf = abs(df[df["PnL"]>0]["PnL"].sum() / df[df["PnL"]<=0]["PnL"].sum()) if (df["PnL"]<=0).any() and df[df["PnL"]<=0]["PnL"].sum()!=0 else float("inf")
    aw = df[df["PnL"]>0]["PnL"].mean() if wins else 0
    al = df[df["PnL"]<=0]["PnL"].mean() if losses else 0
    return {"Total Trades": total, "Wins": wins, "Losses": losses,
            "Win Rate": wr, "Avg PnL": avg, "Median PnL": med,
            "Best Trade": df["PnL"].max(), "Worst Trade": df["PnL"].min(),
            "Profit Factor": pf, "Avg Win": aw, "Avg Loss": al,
            "Expectancy": (wr * aw) + ((1-wr) * al)}

def max_consecutive_losses(df):
    s = (df.sort_values("Entry Date")["PnL"] <= 0).astype(int)
    groups = s.groupby((s != s.shift()).cumsum()).sum()
    return int(groups.max()) if not groups.empty else 0

def holding_time(df):
    if "Exit Date" in df.columns and "Entry Date" in df.columns:
        delta = (df["Exit Date"] - df["Entry Date"]).dt.total_seconds()
        return delta
    return pd.Series(dtype=float)

# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------
file_map = discover_files()
if not file_map:
    st.error("🚫 No backtest result files found. Run backtests first.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 📊 Trading Dashboard")
st.sidebar.divider()

if st.sidebar.button("🗑️ Clear All Results", type="secondary"):
    count = 0
    for f in list(file_map.keys()):
        try:
            os.remove(f)
            pf = f.replace(".csv", "_portfolio.csv")
            if os.path.exists(pf):
                os.remove(pf)
            count += 1
        except:
            pass
    if count:
        st.sidebar.success(f"Deleted {count} files.")
        st.rerun()

st.sidebar.divider()
view_mode = st.sidebar.radio("View", [
    "⚔️ Head-to-Head: 3 Strategies",
    "📈 All Strategies Overview",
    "📊 Volume Profile",
    "🔬 Trade Charts",
    "🔍 Single Strategy Deep-Dive",
], index=0)

# =====================================================================
# VIEW 1: 3-WAY HEAD-TO-HEAD COMPARISON
# =====================================================================
if view_mode == "⚔️ Head-to-Head: 3 Strategies":
    st.markdown("# ⚔️ Minervini vs Parallel Activity vs Hybrid")
    st.caption("Which strategy actually works after costs, slippage, and real-world constraints?")

    # Find the files
    min_file = None
    va_file = None
    hybrid_file = None
    for f, info in file_map.items():
        if info["strategy"] == "Minervini" and min_file is None:
            min_file = f
        if info["strategy"] == "Value Area" and "30min" in f.lower():
            va_file = f
        if info["strategy"] == "Hybrid" and hybrid_file is None:
            hybrid_file = f

    if not min_file:
        st.warning("No Minervini backtest found. Run `python backtest_minervini.py` first.")
        st.stop()
    if not va_file:
        st.warning("No Value Area 30min backtest found. Run `python backtest_value_area.py` first.")
        st.stop()

    m_df = load_trades(min_file)
    va_df_full = load_trades(va_file)
    pa_df = va_df_full[va_df_full["Setup"] == "Parallel Activity"].copy() if "Setup" in va_df_full.columns else va_df_full.copy()

    # Hybrid is optional (may not have been run yet)
    h_df = load_trades(hybrid_file) if hybrid_file else pd.DataFrame()
    has_hybrid = not h_df.empty

    if m_df.empty or pa_df.empty:
        st.warning("Insufficient trade data.")
        st.stop()

    m_met = compute_metrics(m_df)
    pa_met = compute_metrics(pa_df)
    h_met = compute_metrics(h_df) if has_hybrid else {}

    # --- Side-by-side KPIs ---
    st.markdown("### 📊 Raw Performance")

    if has_hybrid:
        left, mid, right = st.columns(3)
    else:
        left, right = st.columns(2)
        mid = None

    with left:
        st.markdown("#### 🏛️ Minervini (Swing)")
        k1, k2, k3 = st.columns(3)
        k1.metric("Trades", f"{m_met['Total Trades']:,}")
        k2.metric("Win Rate", f"{m_met['Win Rate']:.1%}")
        k3.metric("Avg PnL", f"{m_met['Avg PnL']:.2%}")
        k4, k5, k6 = st.columns(3)
        k4.metric("Median PnL", f"{m_met['Median PnL']:.2%}")
        k5.metric("Profit Factor", f"{m_met['Profit Factor']:.2f}" if m_met['Profit Factor'] < 100 else "∞")
        k6.metric("Best Trade", f"{m_met['Best Trade']:.1%}")

    if mid and has_hybrid:
        with mid:
            st.markdown("#### 🧬 Hybrid (MP-Filtered)")
            k1, k2, k3 = st.columns(3)
            k1.metric("Trades", f"{h_met['Total Trades']:,}")
            k2.metric("Win Rate", f"{h_met['Win Rate']:.1%}",
                       delta=f"+{(h_met['Win Rate']-m_met['Win Rate']):.1%} vs Min")
            k3.metric("Avg PnL", f"{h_met['Avg PnL']:.2%}")
            k4, k5, k6 = st.columns(3)
            k4.metric("Median PnL", f"{h_met['Median PnL']:.2%}",
                       delta=f"+{(h_met['Median PnL']-m_met['Median PnL']):.2%} vs Min")
            k5.metric("Profit Factor", f"{h_met['Profit Factor']:.2f}" if h_met['Profit Factor'] < 100 else "∞")
            k6.metric("Best Trade", f"{h_met['Best Trade']:.1%}")

    with right:
        st.markdown("#### 📐 Parallel Activity (Intraday)")
        k1, k2, k3 = st.columns(3)
        k1.metric("Trades", f"{pa_met['Total Trades']:,}")
        k2.metric("Win Rate", f"{pa_met['Win Rate']:.1%}")
        k3.metric("Avg PnL", f"{pa_met['Avg PnL']:.3%}")
        k4, k5, k6 = st.columns(3)
        k4.metric("Median PnL", f"{pa_met['Median PnL']:.4%}")
        k5.metric("Profit Factor", f"{pa_met['Profit Factor']:.2f}" if pa_met['Profit Factor'] < 100 else "∞")
        k6.metric("Best Trade", f"{pa_met['Best Trade']:.2%}")

    st.divider()

    # --- REALITY CHECK section ---
    st.markdown("### ⚠️ Reality Check — What the Numbers Really Mean")

    m_hold = holding_time(m_df) / 86400  # days
    pa_hold = holding_time(pa_df) / 3600  # hours
    h_hold = holding_time(h_df) / 86400 if has_hybrid else pd.Series(dtype=float)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown("**⏱️ Holding Period**")
        st.metric("Minervini", f"{m_hold.mean():.0f} days")
        if has_hybrid:
            st.metric("Hybrid", f"{h_hold.mean():.0f} days")
        st.metric("PA", f"{pa_hold.mean():.1f} hours")

    with rc2:
        st.markdown("**📉 Max Consecutive Losses**")
        m_cons = max_consecutive_losses(m_df)
        pa_cons = max_consecutive_losses(pa_df)
        st.metric("Minervini", f"{m_cons} in a row")
        if has_hybrid:
            h_cons = max_consecutive_losses(h_df)
            st.metric("Hybrid", f"{h_cons} in a row",
                       delta=f"{h_cons - m_cons} vs Min",
                       delta_color="inverse")
        st.metric("PA", f"{pa_cons} in a row")

    with rc3:
        st.markdown("**🔢 Key Concerns**")
        pa_zero = (pa_df["PnL"] == 0).mean()
        pa_sl = (pa_df["Reason"] == "Stop Loss").mean() if "Reason" in pa_df.columns else 0
        st.metric("PA Breakeven Trades", f"{pa_zero:.1%}")
        st.metric("PA Stop-Loss Exits", f"{pa_sl:.1%}")
        if has_hybrid:
            h_sl = (h_df["Reason"] == "Stop Loss").mean() if "Reason" in h_df.columns else 0
            st.metric("Hybrid Stop-Loss %", f"{h_sl:.1%}",
                       delta=f"{(h_sl - (m_df['Reason']=='Stop Loss').mean()):.1%} vs Min" if "Reason" in m_df.columns else "")

    st.divider()

    # --- HYBRID IMPROVEMENT ANALYSIS (if available) ---
    if has_hybrid:
        st.markdown("### 🧬 What the Hybrid Filter Changed")

        imp1, imp2, imp3, imp4 = st.columns(4)
        with imp1:
            trade_diff = h_met["Total Trades"] - m_met["Total Trades"]
            st.metric("Trades vs Base Minervini",
                       f"{trade_diff:+,}",
                       delta=f"{trade_diff/m_met['Total Trades']*100:+.0f}%")
        with imp2:
            wr_diff = h_met["Win Rate"] - m_met["Win Rate"]
            st.metric("Win Rate Δ", f"{wr_diff:+.1%}",
                       delta="🟢 Improved" if wr_diff > 0 else "🔴 Worse")
        with imp3:
            med_diff = h_met["Median PnL"] - m_met["Median PnL"]
            st.metric("Median PnL Δ",
                       f"{med_diff:+.2%}",
                       delta="Flipped positive!" if h_met["Median PnL"] > 0 > m_met["Median PnL"] else "")
        with imp4:
            exp_diff = h_met["Expectancy"] - m_met["Expectancy"]
            st.metric("Expectancy Δ", f"{exp_diff:+.3%}")

        # MP Context breakdown
        if "MP Context" in h_df.columns:
            st.markdown("#### 🏷️ Performance by Entry Context")
            ctx_data = []
            for ctx, grp in h_df.groupby("MP Context"):
                cm = compute_metrics(grp)
                cm["Context"] = ctx
                ctx_data.append(cm)
            ctx_df = pd.DataFrame(ctx_data)

            ctx1, ctx2 = st.columns(2)
            with ctx1:
                fig_cw = px.bar(ctx_df.sort_values("Win Rate"), x="Win Rate", y="Context",
                                orientation="h", color_discrete_sequence=[C["hybrid"]], template=TPL)
                fig_cw.update_layout(xaxis_tickformat=".0%", height=250, title="Win Rate by Entry Context",
                                     margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_cw, use_container_width=True)
            with ctx2:
                fig_ca = px.bar(ctx_df.sort_values("Avg PnL"), x="Avg PnL", y="Context",
                                orientation="h", color_discrete_sequence=[C["hybrid"]], template=TPL)
                fig_ca.update_layout(xaxis_tickformat=".2%", height=250, title="Avg PnL by Entry Context",
                                     margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_ca, use_container_width=True)

        # Exit reason comparison
        if "Reason" in h_df.columns:
            st.markdown("#### 🚪 Exit Reason Breakdown")
            exit_data = []
            for reason, grp in h_df.groupby("Reason"):
                em = compute_metrics(grp)
                em["Reason"] = reason
                exit_data.append(em)
            exit_df = pd.DataFrame(exit_data)

            ex1, ex2 = st.columns(2)
            with ex1:
                fig_ew = px.bar(exit_df.sort_values("Win Rate"), x="Win Rate", y="Reason",
                                orientation="h", color="Win Rate",
                                color_continuous_scale=["#e17055","#fdcb6e","#00b894"],
                                template=TPL)
                fig_ew.update_layout(xaxis_tickformat=".0%", height=280, title="Win Rate by Exit Type",
                                     showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_ew, use_container_width=True)
            with ex2:
                fig_ea = px.bar(exit_df, x="Total Trades", y="Reason",
                                orientation="h", color_discrete_sequence=[C["teal"]],
                                template=TPL, text="Total Trades")
                fig_ea.update_layout(height=280, title="Trade Count by Exit Type",
                                     margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_ea, use_container_width=True)

        st.divider()

    # --- SLIPPAGE IMPACT ---
    st.markdown("### 💸 Slippage & Commission Impact")
    st.markdown("*This is what kills most intraday strategies. Each round-trip costs real money.*")

    slippage_bps = st.slider("Round-trip slippage + commission (basis points)", 0, 50, 10, 5)
    slip_pct = slippage_bps / 10000

    m_adj = m_met["Avg PnL"] - slip_pct
    pa_adj = pa_met["Avg PnL"] - slip_pct
    h_adj = h_met["Avg PnL"] - slip_pct if has_hybrid else 0

    if has_hybrid:
        sc1, sc2, sc3 = st.columns(3)
    else:
        sc1, sc3 = st.columns(2)
        sc2 = None

    with sc1:
        st.markdown("**Minervini**")
        delta_m = -slip_pct / m_met["Avg PnL"] * 100 if m_met["Avg PnL"] != 0 else 0
        st.metric("Adjusted Avg PnL", f"{m_adj:.3%}", delta=f"{-slippage_bps}bps")
        st.caption(f"Impact: {abs(delta_m):.1f}% of edge lost — **minimal** (long holds)")

    if sc2 and has_hybrid:
        with sc2:
            st.markdown("**Hybrid**")
            delta_h = -slip_pct / h_met["Avg PnL"] * 100 if h_met["Avg PnL"] != 0 else 0
            st.metric("Adjusted Avg PnL", f"{h_adj:.3%}", delta=f"{-slippage_bps}bps")
            if h_adj > 0:
                st.success(f"✅ {abs(delta_h):.0f}% of edge consumed — **robust** (swing holds)")
            else:
                st.warning(f"⚠️ Edge impaired at {slippage_bps}bps.")

    with sc3:
        st.markdown("**Parallel Activity**")
        delta_pa = -slip_pct / pa_met["Avg PnL"] * 100 if pa_met["Avg PnL"] != 0 else 0
        st.metric("Adjusted Avg PnL", f"{pa_adj:.3%}", delta=f"{-slippage_bps}bps")
        if pa_adj <= 0:
            st.error(f"⚠️ Edge wiped out! At {slippage_bps}bps, PA becomes a **losing strategy**.")
        else:
            st.warning(f"⚠️ {abs(delta_pa):.0f}% of edge consumed by costs. Very fragile.")

    if pa_met["Avg PnL"] > 0:
        breakeven_bps = pa_met["Avg PnL"] * 10000
        st.info(f"📌 **Break-even slippage for PA: {breakeven_bps:.0f} bps** — that's only {breakeven_bps/100:.2f}% round-trip. Most brokers + impact cost easily exceed this.")

    st.divider()

    # --- Cumulative PnL comparison ---
    st.markdown("### 📈 Cumulative PnL — Side by Side")

    tab1, tab2 = st.tabs(["Raw (No Slippage)", f"After {slippage_bps}bps Slippage"])

    with tab1:
        fig_raw = go.Figure()
        m_s = m_df.dropna(subset=["Exit Date"]).sort_values("Exit Date")
        m_s["Cum"] = m_s["PnL"].cumsum()
        pa_s = pa_df.dropna(subset=["Exit Date"]).sort_values("Exit Date")
        pa_s["Cum"] = pa_s["PnL"].cumsum()

        fig_raw.add_trace(go.Scatter(x=m_s["Exit Date"], y=m_s["Cum"], name="Minervini",
                                      line=dict(color=C["blue"], width=2)))
        if has_hybrid:
            h_s = h_df.dropna(subset=["Exit Date"]).sort_values("Exit Date")
            h_s["Cum"] = h_s["PnL"].cumsum()
            fig_raw.add_trace(go.Scatter(x=h_s["Exit Date"], y=h_s["Cum"], name="Hybrid",
                                          line=dict(color=C["hybrid"], width=3, dash="dot")))
        fig_raw.add_trace(go.Scatter(x=pa_s["Exit Date"], y=pa_s["Cum"], name="Parallel Activity",
                                      line=dict(color=C["green"], width=2)))
        fig_raw.update_layout(template=TPL, height=500, yaxis_tickformat=".0%",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_raw, use_container_width=True)

    with tab2:
        fig_adj = go.Figure()
        m_s2 = m_s.copy()
        m_s2["Adj"] = (m_s2["PnL"] - slip_pct).cumsum()
        pa_s2 = pa_s.copy()
        pa_s2["Adj"] = (pa_s2["PnL"] - slip_pct).cumsum()

        fig_adj.add_trace(go.Scatter(x=m_s2["Exit Date"], y=m_s2["Adj"], name="Minervini (adj)",
                                      line=dict(color=C["blue"], width=2)))
        if has_hybrid:
            h_s2 = h_s.copy()
            h_s2["Adj"] = (h_s2["PnL"] - slip_pct).cumsum()
            fig_adj.add_trace(go.Scatter(x=h_s2["Exit Date"], y=h_s2["Adj"], name="Hybrid (adj)",
                                          line=dict(color=C["hybrid"], width=3, dash="dot")))
        fig_adj.add_trace(go.Scatter(x=pa_s2["Exit Date"], y=pa_s2["Adj"], name="PA (adj)",
                                      line=dict(color=C["red"], width=2)))
        fig_adj.update_layout(template=TPL, height=500, yaxis_tickformat=".0%",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_adj, use_container_width=True)

    st.divider()

    # --- PnL Distributions ---
    st.markdown("### 📊 PnL Distribution Comparison")

    if has_hybrid:
        d1, d2, d3 = st.columns(3)
    else:
        d1, d3 = st.columns(2)
        d2 = None

    with d1:
        st.markdown("**Minervini** (Long right tail — big winners)")
        fig_mh = px.histogram(m_df, x="PnL", nbins=60, color_discrete_sequence=[C["blue"]], template=TPL)
        fig_mh.update_layout(xaxis_tickformat=".0%", height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_mh, use_container_width=True)

    if d2 and has_hybrid:
        with d2:
            st.markdown("**Hybrid** (Fewer losers, same big winners)")
            fig_hh = px.histogram(h_df, x="PnL", nbins=60, color_discrete_sequence=[C["hybrid"]], template=TPL)
            fig_hh.update_layout(xaxis_tickformat=".0%", height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_hh, use_container_width=True)

    with d3:
        st.markdown("**Parallel Activity** (Tight cluster around zero)")
        fig_ph = px.histogram(pa_df, x="PnL", nbins=100, color_discrete_sequence=[C["green"]], template=TPL)
        fig_ph.update_layout(xaxis_tickformat=".1%", height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_ph, use_container_width=True)

    st.divider()

    # --- Summary comparison table ---
    st.markdown("### 📋 Full Comparison Table")

    comp_data = {
        "Metric": [
            "Total Trades", "Win Rate", "Avg PnL/Trade", "Median PnL/Trade",
            "Profit Factor", "Best Trade", "Worst Trade",
            "Avg Holding", "Max Consecutive Losses",
            "Unique Stocks Traded",
            f"Avg PnL After {slippage_bps}bps", "Survives Slippage?",
        ],
        "Minervini (Swing)": [
            f"{m_met['Total Trades']:,}",
            f"{m_met['Win Rate']:.1%}",
            f"{m_met['Avg PnL']:.2%}",
            f"{m_met['Median PnL']:.2%}",
            f"{m_met['Profit Factor']:.2f}" if m_met['Profit Factor'] < 100 else "∞",
            f"{m_met['Best Trade']:.1%}",
            f"{m_met['Worst Trade']:.1%}",
            f"{m_hold.mean():.0f} days",
            str(max_consecutive_losses(m_df)),
            str(m_df["Symbol"].nunique()),
            f"{m_adj:.3%}",
            "✅ Yes",
        ],
    }

    if has_hybrid:
        comp_data["🧬 Hybrid (MP-Filtered)"] = [
            f"{h_met['Total Trades']:,}",
            f"{h_met['Win Rate']:.1%}",
            f"{h_met['Avg PnL']:.2%}",
            f"{h_met['Median PnL']:.2%}",
            f"{h_met['Profit Factor']:.2f}" if h_met['Profit Factor'] < 100 else "∞",
            f"{h_met['Best Trade']:.1%}",
            f"{h_met['Worst Trade']:.1%}",
            f"{h_hold.mean():.0f} days",
            str(max_consecutive_losses(h_df)),
            str(h_df["Symbol"].nunique()),
            f"{h_adj:.3%}",
            "✅ Yes — robust" if h_adj > 0 else "⚠️ Marginal",
        ]

    comp_data["Parallel Activity (Intraday)"] = [
        f"{pa_met['Total Trades']:,}",
        f"{pa_met['Win Rate']:.1%}",
        f"{pa_met['Avg PnL']:.3%}",
        f"{pa_met['Median PnL']:.4%}",
        f"{pa_met['Profit Factor']:.2f}" if pa_met['Profit Factor'] < 100 else "∞",
        f"{pa_met['Best Trade']:.2%}",
        f"{pa_met['Worst Trade']:.2%}",
        f"{pa_hold.mean():.1f} hours",
        str(max_consecutive_losses(pa_df)),
        str(pa_df["Symbol"].nunique()),
        f"{pa_adj:.3%}",
        "✅ Barely" if pa_adj > 0 else "❌ No",
    ]
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.divider()

    # --- Monthly returns overlay ---
    st.markdown("### 📅 Monthly Returns")
    if has_hybrid:
        mt1, mt2, mt3 = st.columns(3)
    else:
        mt1, mt3 = st.columns(2)
        mt2 = None

    with mt1:
        st.markdown("**Minervini**")
        m_mo = m_df.dropna(subset=["Exit Date"]).copy()
        m_mo["YM"] = m_mo["Exit Date"].dt.to_period("M").dt.to_timestamp()
        m_agg = m_mo.groupby("YM")["PnL"].sum().reset_index()
        fig_mm = px.bar(m_agg, x="YM", y="PnL", color="PnL",
                        color_continuous_scale=["#e17055","#2d3436","#00b894"],
                        color_continuous_midpoint=0, template=TPL)
        fig_mm.update_layout(height=300, yaxis_tickformat=".0%", showlegend=False,
                             margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_mm, use_container_width=True)

    if mt2 and has_hybrid:
        with mt2:
            st.markdown("**Hybrid**")
            h_mo = h_df.dropna(subset=["Exit Date"]).copy()
            h_mo["YM"] = h_mo["Exit Date"].dt.to_period("M").dt.to_timestamp()
            h_agg = h_mo.groupby("YM")["PnL"].sum().reset_index()
            fig_hm = px.bar(h_agg, x="YM", y="PnL", color="PnL",
                            color_continuous_scale=["#e17055","#2d3436","#e84393"],
                            color_continuous_midpoint=0, template=TPL)
            fig_hm.update_layout(height=300, yaxis_tickformat=".0%", showlegend=False,
                                 margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_hm, use_container_width=True)

    with mt3:
        st.markdown("**Parallel Activity**")
        pa_mo = pa_df.dropna(subset=["Exit Date"]).copy()
        pa_mo["YM"] = pa_mo["Exit Date"].dt.to_period("M").dt.to_timestamp()
        pa_agg = pa_mo.groupby("YM")["PnL"].sum().reset_index()
        fig_pm = px.bar(pa_agg, x="YM", y="PnL", color="PnL",
                        color_continuous_scale=["#e17055","#2d3436","#00b894"],
                        color_continuous_midpoint=0, template=TPL)
        fig_pm.update_layout(height=300, yaxis_tickformat=".0%", showlegend=False,
                             margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_pm, use_container_width=True)


# =====================================================================
# VIEW 2: ALL STRATEGIES OVERVIEW
# =====================================================================
elif view_mode == "📈 All Strategies Overview":
    st.markdown("# 📈 Strategy Comparison")
    st.caption("Side-by-side performance of all backtested strategies")

    all_metrics = []
    for fname, info in file_map.items():
        df = load_trades(fname)
        if df.empty:
            continue
        m = compute_metrics(df)
        m["Strategy"] = info["label"]
        m["File"] = fname
        m["Category"] = info["strategy"]
        all_metrics.append(m)

    if not all_metrics:
        st.warning("No trades found.")
        st.stop()

    comp_df = pd.DataFrame(all_metrics)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strategies Loaded", len(comp_df))
    k2.metric("Total Trades (All)", f"{comp_df['Total Trades'].sum():,.0f}")
    k3.metric("Best Avg PnL", f"{comp_df['Avg PnL'].max():.2%}")
    k4.metric("Best Win Rate", f"{comp_df['Win Rate'].max():.1%}")

    st.divider()

    display_df = comp_df[["Strategy","Total Trades","Win Rate","Avg PnL","Median PnL","Profit Factor","Best Trade","Worst Trade","Expectancy"]].copy()
    display_df["Win Rate"] = display_df["Win Rate"].apply(lambda x: f"{x:.1%}")
    display_df["Avg PnL"] = display_df["Avg PnL"].apply(lambda x: f"{x:.3%}")
    display_df["Median PnL"] = display_df["Median PnL"].apply(lambda x: f"{x:.3%}")
    display_df["Best Trade"] = display_df["Best Trade"].apply(lambda x: f"{x:.2%}")
    display_df["Worst Trade"] = display_df["Worst Trade"].apply(lambda x: f"{x:.2%}")
    display_df["Profit Factor"] = display_df["Profit Factor"].apply(lambda x: f"{x:.2f}" if x<100 else "∞")
    display_df["Expectancy"] = display_df["Expectancy"].apply(lambda x: f"{x:.4%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    c1, c2 = st.columns(2)
    cat_colors = {"Minervini": C["blue"], "Value Area": C["accent"], "Hybrid": C["hybrid"]}
    with c1:
        fig_wr = px.bar(comp_df.sort_values("Win Rate",ascending=True), x="Win Rate", y="Strategy",
                        orientation="h", color="Category",
                        color_discrete_map=cat_colors, template=TPL)
        fig_wr.update_layout(xaxis_tickformat=".0%", height=400, title="Win Rate", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_wr, use_container_width=True)
    with c2:
        fig_ap = px.bar(comp_df.sort_values("Avg PnL",ascending=True), x="Avg PnL", y="Strategy",
                        orientation="h", color="Category",
                        color_discrete_map=cat_colors, template=TPL)
        fig_ap.update_layout(xaxis_tickformat=".2%", height=400, title="Avg PnL/Trade", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_ap, use_container_width=True)

    st.divider()
    st.markdown("### Cumulative PnL Overlay")
    overlay_files = st.multiselect("Select strategies", list(file_map.keys()),
                                    default=list(file_map.keys()),
                                    format_func=lambda x: file_map[x]["label"])
    if overlay_files:
        fig_ov = go.Figure()
        cols = [C["accent"],C["green"],C["blue"],C["hybrid"],C["orange"],C["pink"],C["teal"]]
        for i, fname in enumerate(overlay_files):
            df = load_trades(fname)
            if df.empty: continue
            dc = "Exit Date" if "Exit Date" in df.columns else "Entry Date"
            ds = df.dropna(subset=[dc]).sort_values(dc)
            ds["Cum"] = ds["PnL"].cumsum()
            fig_ov.add_trace(go.Scatter(x=ds[dc], y=ds["Cum"], mode="lines",
                                         name=file_map[fname]["label"],
                                         line=dict(color=cols[i%len(cols)], width=2)))
        fig_ov.update_layout(template=TPL, height=500, yaxis_tickformat=".0%",
                              legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                              margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_ov, use_container_width=True)


# =====================================================================
# VIEW 3: VOLUME PROFILE
# =====================================================================
elif view_mode == "📊 Volume Profile":
    st.markdown("# 📊 Volume Profile — Market Profile Visualization")
    st.caption("TPO distribution, POC, Value Area (VAH/VAL), and day classification for any stock")

    # ── Stock selector ────────────────────────────────────────────────
    data_dir_30 = os.path.join("data", "nifty_200_30min")
    if not os.path.isdir(data_dir_30):
        st.error(f"30-min data directory not found: `{data_dir_30}`")
        st.stop()

    stock_files = sorted([f for f in os.listdir(data_dir_30) if f.endswith(".csv")])
    stock_names = [f.replace("_30min.csv", "") for f in stock_files]

    vp_col1, vp_col2 = st.sidebar.columns(2)
    with vp_col1:
        sel_stock = st.selectbox("Stock", stock_names, index=stock_names.index("RELIANCE") if "RELIANCE" in stock_names else 0)
    with vp_col2:
        n_days = st.slider("Days to show", 1, 30, 5)

    stock_csv = os.path.join(data_dir_30, f"{sel_stock}_30min.csv")
    sdf = pd.read_csv(stock_csv, parse_dates=["date"], index_col="date")

    # Build profiles
    try:
        import rust_mp
        dates_list = [str(d) for d in sdf.index]
        vols_list = [min(int(v), 2_000_000_000) for v in sdf["volume"].fillna(0).tolist()]
        rs_profiles = rust_mp.build_profiles(
            dates_list,
            sdf["open"].tolist(), sdf["high"].tolist(),
            sdf["low"].tolist(), sdf["close"].tolist(),
            vols_list, 1.0,
        )
        st.sidebar.success(f"⚡ Rust engine: {len(rs_profiles)} profiles")
        use_rust = True
    except Exception:
        from market_profile import build_daily_profiles
        py_profiles = build_daily_profiles(sdf)
        rs_profiles = py_profiles
        use_rust = False
        st.sidebar.info("Python MP engine (Rust not available)")

    if not rs_profiles:
        st.warning("No profiles for this stock.")
        st.stop()

    # Show last N days
    show_profiles = rs_profiles[-n_days:]

    # ── Day summary table ─────────────────────────────────────────────
    st.markdown("### 📋 Profile Summary")
    summary_rows = []
    for p in show_profiles:
        date_str = p.date if isinstance(p.date, str) else str(p.date)[:10]
        summary_rows.append({
            "Date": date_str,
            "Open": f"{p.open_price:.1f}",
            "High": f"{p.high:.1f}",
            "Low": f"{p.low:.1f}",
            "Close": f"{p.close:.1f}",
            "POC": f"{p.poc:.1f}",
            "VAH": f"{p.vah:.1f}",
            "VAL": f"{p.val:.1f}",
            "IB Range": f"{p.ib_high - p.ib_low:.1f}",
            "Day Type": p.day_type,
            "Open Type": p.open_type,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Volume Profile charts (one per day) ───────────────────────────
    st.markdown("### 📊 TPO Profile Charts")

    for p_idx, p in enumerate(show_profiles):
        date_str = p.date if isinstance(p.date, str) else str(p.date)[:10]

        # Build TPO data for this profile
        if use_rust:
            # Re-build TPO map from raw bars for this date
            day_mask = sdf.index.date == pd.Timestamp(date_str).date()
            day_df = sdf[day_mask].sort_index()
        else:
            day_mask = sdf.index.date == pd.Timestamp(date_str).date()
            day_df = sdf[day_mask].sort_index()

        if day_df.empty:
            continue

        # Build TPO map manually for visualization
        from collections import defaultdict
        tpo_letters = "ABCDEFGHIJKLMN"
        tick_size = 1.0
        tpo_map = defaultdict(list)
        for idx, (ts, row) in enumerate(day_df.iterrows()):
            letter = tpo_letters[idx] if idx < len(tpo_letters) else tpo_letters[-1]
            lo = int(row["low"] / tick_size) * tick_size
            hi = int(row["high"] / tick_size) * tick_size
            level = lo
            while level <= hi + tick_size * 0.1:
                tpo_map[level].append(letter)
                level = round(level + tick_size, 4)

        if not tpo_map:
            continue

        # Sort levels and build horizontal bar chart
        sorted_levels = sorted(tpo_map.keys())
        tpo_counts = [len(tpo_map[lv]) for lv in sorted_levels]
        tpo_strings = ["".join(tpo_map[lv]) for lv in sorted_levels]

        # Color bars: POC = gold, VA = blue, outside = grey
        poc_val = p.poc
        vah_val = p.vah
        val_val = p.val
        colors = []
        for lv in sorted_levels:
            if abs(lv - poc_val) < tick_size * 0.5:
                colors.append("#f39c12")  # gold = POC
            elif val_val <= lv <= vah_val:
                colors.append("#3498db")  # blue = Value Area
            else:
                colors.append("#636e72")  # grey = outside

        # Create subplots: left = TPO profile, right = candlestick
        from plotly.subplots import make_subplots
        fig_vp = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65],
                                shared_yaxes=True,
                                subplot_titles=[f"TPO Profile", f"Price Action"],
                                horizontal_spacing=0.02)

        # Left: horizontal TPO bars
        fig_vp.add_trace(
            go.Bar(
                y=sorted_levels, x=tpo_counts,
                orientation="h",
                marker=dict(color=colors),
                text=tpo_strings,
                textposition="inside",
                textfont=dict(size=9, family="Courier New"),
                hovertemplate="₹%{y:.0f} | TPOs: %{x} | %{text}<extra></extra>",
                showlegend=False,
            ), row=1, col=1
        )

        # Right: candlestick
        fig_vp.add_trace(
            go.Candlestick(
                x=list(range(len(day_df))),
                open=day_df["open"], high=day_df["high"],
                low=day_df["low"], close=day_df["close"],
                increasing_line_color="#00b894",
                decreasing_line_color="#e17055",
                showlegend=False,
            ), row=1, col=2
        )

        # Add POC, VAH, VAL lines on candlestick
        n_bars = len(day_df)
        fig_vp.add_hline(y=poc_val, line_dash="solid", line_color="#f39c12", line_width=2,
                          annotation_text=f"POC {poc_val:.0f}", row=1, col=2)
        fig_vp.add_hline(y=vah_val, line_dash="dash", line_color="#e74c3c", line_width=1,
                          annotation_text=f"VAH {vah_val:.0f}", row=1, col=2)
        fig_vp.add_hline(y=val_val, line_dash="dash", line_color="#2ecc71", line_width=1,
                          annotation_text=f"VAL {val_val:.0f}", row=1, col=2)
        fig_vp.add_hline(y=p.ib_high, line_dash="dot", line_color="#9b59b6", line_width=1,
                          annotation_text=f"IB-H", row=1, col=2)
        fig_vp.add_hline(y=p.ib_low, line_dash="dot", line_color="#9b59b6", line_width=1,
                          annotation_text=f"IB-L", row=1, col=2)

        # Add VA shaded region on candlestick
        fig_vp.add_hrect(y0=val_val, y1=vah_val, fillcolor="rgba(52,152,219,0.1)",
                          line_width=0, row=1, col=2)

        # Layout
        day_type_badge = p.day_type
        open_type_badge = p.open_type
        ext_str = ""
        if p.range_ext_up and p.range_ext_down:
            ext_str = " | ↕ Both Extensions"
        elif p.range_ext_up:
            ext_str = " | ↑ Upside Extension"
        elif p.range_ext_down:
            ext_str = " | ↓ Downside Extension"

        fig_vp.update_layout(
            template=TPL,
            height=420,
            title=dict(
                text=f"<b>{date_str}</b> — {day_type_badge} | {open_type_badge}{ext_str}",
                font=dict(size=14),
            ),
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis=dict(title="TPO Count", autorange="reversed"),
            xaxis2=dict(title="", showticklabels=False),
            yaxis=dict(title="Price (₹)"),
        )
        # Remove rangeslider from candlestick
        fig_vp.update_xaxes(rangeslider_visible=False, row=1, col=2)

        st.plotly_chart(fig_vp, use_container_width=True)

    # ── Day type distribution ─────────────────────────────────
    st.divider()
    st.markdown("### 📊 Day Type Distribution")
    dtype_col1, dtype_col2 = st.columns(2)

    all_day_types = [p.day_type for p in rs_profiles]
    all_open_types = [p.open_type for p in rs_profiles]

    with dtype_col1:
        dt_counts = pd.Series(all_day_types).value_counts()
        fig_dt = go.Figure(go.Pie(
            labels=dt_counts.index, values=dt_counts.values, hole=0.5,
            marker=dict(colors=["#6c5ce7", "#00b894", "#e17055", "#0984e3", "#fdcb6e", "#a29bfe"]),
        ))
        fig_dt.update_layout(template=TPL, height=350, title="Day Types",
                              margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_dt, use_container_width=True)

    with dtype_col2:
        ot_counts = pd.Series(all_open_types).value_counts()
        fig_ot = go.Figure(go.Pie(
            labels=ot_counts.index, values=ot_counts.values, hole=0.5,
            marker=dict(colors=["#e84393", "#00cec9", "#fd79a8", "#636e72"]),
        ))
        fig_ot.update_layout(template=TPL, height=350, title="Open Types",
                              margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_ot, use_container_width=True)


# =====================================================================
# VIEW 4: TRADE CHARTS — VISUAL TRADE INSPECTION
# =====================================================================
elif view_mode == "🔬 Trade Charts":
    st.markdown("# 🔬 Trade Charts — How Trades Were Taken")
    st.caption("Candlestick charts with entry/exit markers for every trade")

    # ── Strategy selector ─────────────────────────────────────
    sorted_files = sorted(file_map.keys(), key=lambda x: file_map[x]["label"])
    sel_strat = st.sidebar.selectbox("Strategy", sorted_files,
                                      format_func=lambda x: file_map[x]["label"])
    trade_df = load_trades(sel_strat)

    if trade_df.empty:
        st.warning("No trades.")
        st.stop()

    # ── Filters ───────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.sidebar.columns(2), st.sidebar.columns(2), None, None
    symbols = sorted(trade_df["Symbol"].unique().tolist())
    sel_sym = st.sidebar.selectbox("Stock", ["All"] + symbols)

    outcomes = st.sidebar.multiselect("Outcome", ["Winner", "Loser", "Breakeven"],
                                       default=["Winner", "Loser", "Breakeven"])

    max_trades = st.sidebar.slider("Max trades to show", 1, 50, 10)

    # Filter
    filt = trade_df.copy()
    if sel_sym != "All":
        filt = filt[filt["Symbol"] == sel_sym]

    outcome_mask = pd.Series(False, index=filt.index)
    if "Winner" in outcomes:
        outcome_mask |= filt["PnL"] > 0
    if "Loser" in outcomes:
        outcome_mask |= filt["PnL"] < 0
    if "Breakeven" in outcomes:
        outcome_mask |= filt["PnL"] == 0
    filt = filt[outcome_mask]

    # Sort by absolute PnL (biggest trades first)
    filt = filt.sort_values("PnL", key=abs, ascending=False).head(max_trades)

    st.markdown(f"**Showing {len(filt)} trades** (sorted by |PnL|)")

    # ── Render each trade ─────────────────────────────────────
    data_dir_30 = os.path.join("data", "nifty_200_30min")
    data_dir_daily = os.path.join("data", "nifty_200_daily")

    for trade_idx, (_, trade) in enumerate(filt.iterrows()):
        sym = trade["Symbol"]
        entry_date = pd.Timestamp(trade["Entry Date"])
        exit_date = pd.Timestamp(trade["Exit Date"]) if pd.notna(trade.get("Exit Date")) else entry_date
        entry_price = trade["Entry Price"]
        exit_price = trade["Exit Price"] if pd.notna(trade.get("Exit Price")) else entry_price
        pnl = trade["PnL"]
        reason = trade.get("Reason", "")

        # Determine if intraday or swing
        is_intraday = (exit_date.date() == entry_date.date()) or ("Direction" in trade.index)

        # Load price data
        if is_intraday:
            csv_path = os.path.join(data_dir_30, f"{sym}_30min.csv")
        else:
            csv_path = os.path.join(data_dir_daily, f"{sym}_daily.csv")

        if not os.path.exists(csv_path):
            # Try the other
            alt_path = os.path.join(data_dir_30, f"{sym}_30min.csv")
            if os.path.exists(alt_path):
                csv_path = alt_path
            else:
                st.warning(f"No data for {sym}")
                continue

        price_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

        # Window: 5 bars before entry to 5 bars after exit
        if is_intraday:
            # Show the full day
            day_mask = price_df.index.date == entry_date.date()
            window = price_df[day_mask]
        else:
            # Show 10 days before entry to 5 days after exit
            start = entry_date - pd.Timedelta(days=15)
            end = exit_date + pd.Timedelta(days=10)
            window = price_df[(price_df.index >= start) & (price_df.index <= end)]

        if window.empty:
            continue

        # Build chart
        pnl_pct = pnl * 100
        color = "#00b894" if pnl > 0 else "#e17055" if pnl < 0 else "#fdcb6e"
        pnl_label = f"+{pnl_pct:.1f}%" if pnl > 0 else f"{pnl_pct:.1f}%"
        direction = trade.get("Direction", "Long")
        setup = trade.get("Setup", trade.get("MP Context", ""))

        fig_tc = go.Figure()

        # Candlestick
        fig_tc.add_trace(go.Candlestick(
            x=window.index,
            open=window["open"], high=window["high"],
            low=window["low"], close=window["close"],
            increasing_line_color="#00b894",
            decreasing_line_color="#e17055",
            showlegend=False,
        ))

        # Entry marker
        fig_tc.add_trace(go.Scatter(
            x=[entry_date], y=[entry_price],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=16, color="#0984e3",
                         line=dict(width=2, color="white")),
            text=[f"BUY ₹{entry_price:.1f}"],
            textposition="bottom center",
            textfont=dict(size=11, color="#0984e3"),
            name="Entry",
            showlegend=False,
        ))

        # Exit marker
        fig_tc.add_trace(go.Scatter(
            x=[exit_date], y=[exit_price],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=16, color=color,
                         line=dict(width=2, color="white")),
            text=[f"EXIT ₹{exit_price:.1f}"],
            textposition="top center",
            textfont=dict(size=11, color=color),
            name="Exit",
            showlegend=False,
        ))

        # Shaded region between entry and exit
        fig_tc.add_vrect(
            x0=entry_date, x1=exit_date,
            fillcolor="rgba(9,132,227,0.08)" if pnl >= 0 else "rgba(225,112,85,0.08)",
            line_width=0,
        )

        # Entry/exit price lines
        fig_tc.add_hline(y=entry_price, line_dash="dot", line_color="#0984e3",
                          line_width=1, opacity=0.5)
        fig_tc.add_hline(y=exit_price, line_dash="dot", line_color=color,
                          line_width=1, opacity=0.5)

        # Title
        hold_str = ""
        if not is_intraday:
            hold_days = (exit_date - entry_date).days
            hold_str = f" | {hold_days}d hold"
        else:
            delta_mins = int((exit_date - entry_date).total_seconds() / 60)
            hold_str = f" | {delta_mins}min hold" if delta_mins > 0 else ""

        title_text = (f"<b>{sym}</b> — <span style='color:{color}'>{pnl_label}</span>"
                      f" | {reason}{hold_str}")
        if setup:
            title_text += f" | {setup}"
        if direction and direction != "Long":
            title_text += f" | {direction}"

        fig_tc.update_layout(
            template=TPL,
            height=380,
            title=dict(text=title_text, font=dict(size=13)),
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_rangeslider_visible=False,
            yaxis_title="Price (₹)",
        )

        st.plotly_chart(fig_tc, use_container_width=True)

        # Trade details expander
        with st.expander(f"📋 Details — {sym} ({entry_date.strftime('%Y-%m-%d')})"):
            det_cols = st.columns(6)
            det_cols[0].metric("Entry", f"₹{entry_price:.1f}")
            det_cols[1].metric("Exit", f"₹{exit_price:.1f}")
            det_cols[2].metric("PnL", pnl_label, delta_color="normal" if pnl >= 0 else "inverse")
            det_cols[3].metric("Reason", reason)
            if setup:
                det_cols[4].metric("Setup", setup)
            if direction:
                det_cols[5].metric("Direction", direction)

    if filt.empty:
        st.info("No trades match the filters. Try selecting 'All' stocks or adjusting outcome filters.")


# =====================================================================
# VIEW 5: SINGLE STRATEGY DEEP-DIVE
# =====================================================================
elif view_mode == "🔍 Single Strategy Deep-Dive":
    sorted_files = sorted(file_map.keys(), key=lambda x: file_map[x]["label"])
    selected = st.sidebar.selectbox("Select Strategy", sorted_files, format_func=lambda x: file_map[x]["label"])
    info = file_map[selected]
    df = load_trades(selected)
    pf = load_portfolio(selected)

    if df.empty:
        st.warning("No trades.")
        st.stop()

    metrics = compute_metrics(df)
    st.markdown(f"# 🔍 {info['label']}")

    # Portfolio curve
    if pf is not None and not pf.empty:
        st.markdown("### 💰 Portfolio Simulation")
        final_eq = pf.iloc[-1]["Equity"]
        start_eq = pf.iloc[0]["Equity"]
        pk1,pk2,pk3 = st.columns(3)
        pk1.metric("Final Equity", f"₹{final_eq:,.0f}")
        pk2.metric("Return", f"{(final_eq-start_eq)/start_eq:.2%}")
        pk3.metric("Peak Positions", int(pf["Positions"].max()))
        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(x=pf["Date"],y=pf["Equity"],fill="tozeroy",
                                     line=dict(color=C["accent"],width=2),fillcolor="rgba(108,92,231,0.15)"))
        fig_pf.update_layout(template=TPL,height=350,yaxis_title="Equity (₹)",margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_pf, use_container_width=True)
        st.divider()

    # KPIs
    st.markdown("### Key Metrics")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Trades", f"{metrics['Total Trades']:,}")
    m2.metric("Win Rate", f"{metrics['Win Rate']:.1%}")
    m3.metric("Avg PnL", f"{metrics['Avg PnL']:.3%}")
    m4.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}" if metrics['Profit Factor']<100 else "∞")
    m5.metric("Best", f"{metrics['Best Trade']:.2%}")
    m6.metric("Worst", f"{metrics['Worst Trade']:.2%}")
    st.divider()

    # Equity + Drawdown
    dc = "Exit Date" if "Exit Date" in df.columns else "Entry Date"
    df_s = df.dropna(subset=[dc]).sort_values(dc).copy()
    df_s["Cum"] = df_s["PnL"].cumsum()
    df_s["Peak"] = df_s["Cum"].cummax()
    df_s["DD"] = df_s["Cum"] - df_s["Peak"]

    fig_eq = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.05)
    fig_eq.add_trace(go.Scatter(x=df_s[dc],y=df_s["Cum"],name="Cumulative PnL",line=dict(color=C["green"],width=2)),row=1,col=1)
    fig_eq.add_trace(go.Scatter(x=df_s[dc],y=df_s["DD"],name="Drawdown",fill="tozeroy",
                                 line=dict(color=C["red"],width=1),fillcolor="rgba(225,112,85,0.3)"),row=2,col=1)
    fig_eq.update_layout(template=TPL,height=500,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                          margin=dict(l=10,r=10,t=40,b=10))
    fig_eq.update_yaxes(tickformat=".0%",row=1,col=1)
    fig_eq.update_yaxes(tickformat=".0%",row=2,col=1)
    st.plotly_chart(fig_eq, use_container_width=True)
    st.divider()

    # Setup breakdown (VA only)
    if "Setup" in df.columns:
        st.markdown("### 🎯 Performance by Setup")
        setup_met = []
        for s, grp in df.groupby("Setup"):
            sm = compute_metrics(grp)
            sm["Setup"] = s
            setup_met.append(sm)
        sdf = pd.DataFrame(setup_met)

        sc1,sc2 = st.columns(2)
        with sc1:
            fig_sw = px.bar(sdf.sort_values("Win Rate"),x="Win Rate",y="Setup",orientation="h",
                            color="Setup",color_discrete_map=SETUP_COLORS,template=TPL)
            fig_sw.update_layout(xaxis_tickformat=".0%",showlegend=False,height=350,title="Win Rate",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_sw, use_container_width=True)
        with sc2:
            fig_sp = px.bar(sdf.sort_values("Avg PnL"),x="Avg PnL",y="Setup",orientation="h",
                            color="Setup",color_discrete_map=SETUP_COLORS,template=TPL)
            fig_sp.update_layout(xaxis_tickformat=".3%",showlegend=False,height=350,title="Avg PnL",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_sp, use_container_width=True)

        fig_sc = go.Figure()
        for s in df["Setup"].unique():
            sgrp = df[df["Setup"]==s].dropna(subset=[dc]).sort_values(dc).copy()
            sgrp["C"] = sgrp["PnL"].cumsum()
            fig_sc.add_trace(go.Scatter(x=sgrp[dc],y=sgrp["C"],name=s,
                                         line=dict(color=SETUP_COLORS.get(s,C["teal"]),width=2)))
        fig_sc.update_layout(template=TPL,height=400,yaxis_tickformat=".0%",
                              legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                              margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.divider()

    # MP Context breakdown (Hybrid only)
    if "MP Context" in df.columns:
        st.markdown("### 🏷️ Performance by MP Entry Context")
        ctx_met = []
        for c, grp in df.groupby("MP Context"):
            cm = compute_metrics(grp)
            cm["Context"] = c
            ctx_met.append(cm)
        cdf = pd.DataFrame(ctx_met)

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_cw = px.bar(cdf.sort_values("Win Rate"), x="Win Rate", y="Context", orientation="h",
                            color_discrete_sequence=[C["hybrid"]], template=TPL)
            fig_cw.update_layout(xaxis_tickformat=".0%", height=300, title="Win Rate", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_cw, use_container_width=True)
        with cc2:
            fig_ca = px.bar(cdf.sort_values("Avg PnL"), x="Avg PnL", y="Context", orientation="h",
                            color_discrete_sequence=[C["hybrid"]], template=TPL)
            fig_ca.update_layout(xaxis_tickformat=".2%", height=300, title="Avg PnL", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_ca, use_container_width=True)
        st.divider()

    # Distribution
    d1,d2 = st.columns(2)
    with d1:
        st.markdown("### PnL Distribution")
        fig_h = px.histogram(df,x="PnL",nbins=80,color_discrete_sequence=[C["accent"]],template=TPL)
        fig_h.update_layout(xaxis_tickformat=".1%",height=350,margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    with d2:
        st.markdown("### Win / Loss / Breakeven")
        w = (df["PnL"]>0).sum(); l = (df["PnL"]<0).sum(); b = (df["PnL"]==0).sum()
        fig_d = go.Figure(go.Pie(labels=["Wins","Losses","Breakeven"],values=[w,l,b],hole=0.55,
                                  marker=dict(colors=[C["green"],C["red"],C["orange"]])))
        fig_d.update_layout(template=TPL,height=350,margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_d, use_container_width=True)

    st.divider()

    # Trade log
    st.markdown("### 📋 Trade Log")
    filter_cols = st.columns(3)
    with filter_cols[0]:
        syms = ["All"] + sorted(df["Symbol"].unique().tolist()) if "Symbol" in df.columns else ["All"]
        sf = st.selectbox("Symbol", syms)
    with filter_cols[1]:
        if "Setup" in df.columns:
            setups = ["All"] + sorted(df["Setup"].unique().tolist())
        elif "MP Context" in df.columns:
            setups = ["All"] + sorted(df["MP Context"].unique().tolist())
        else:
            setups = ["All"]
        stf = st.selectbox("Setup / Context", setups)
    with filter_cols[2]:
        if "Direction" in df.columns:
            dirs = ["All"] + sorted(df["Direction"].unique().tolist())
        elif "Reason" in df.columns:
            dirs = ["All"] + sorted(df["Reason"].unique().tolist())
        else:
            dirs = ["All"]
        df_ = st.selectbox("Direction / Reason", dirs)

    filt = df.copy()
    if sf != "All": filt = filt[filt["Symbol"]==sf]
    if stf != "All":
        col = "Setup" if "Setup" in filt.columns else "MP Context" if "MP Context" in filt.columns else None
        if col: filt = filt[filt[col]==stf]
    if df_ != "All":
        col = "Direction" if "Direction" in filt.columns else "Reason" if "Reason" in filt.columns else None
        if col: filt = filt[filt[col]==df_]
    st.dataframe(filt.sort_values("Entry Date",ascending=False), use_container_width=True, height=500)
