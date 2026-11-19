"""
CoinWeather - a plain-English risk forecast for your crypto portfolio
NEOMA MSc Fintech & DeFi - Final Project

Run with:  streamlit run app.py
Needs prices.csv (columns: date, BTC, ETH, SOL, ...) in the same folder.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# Settings (the same thresholds are shown to the user in the app)
# ----------------------------------------------------------------------------
APP_NAME = "CoinWeather"
DATA_FILE = Path(__file__).parent / "prices.csv"
TRADING_DAYS = 365            # crypto trades every day
LOW_VOL = 0.40                # annualized volatility below 40%  -> Low
HIGH_VOL = 0.80               # annualized volatility above 80%  -> High
CORR_LIMIT = 0.60             # correlation at or above this = "moves closely together"
CONCENTRATION_WARN = 0.60     # warn when this share of the money sits in one correlated group
STABLECOIN = "USDC"
STABLE_SHIFT = 0.20           # summary suggestion: move 20% of the portfolio into the stablecoin
VAR_LEVELS = (0.95, 0.99)     # Value at Risk confidence levels
DEFAULT_TOTAL = 10_000.0      # EUR, used when the user enters percentages

# Each risk level gets a weather word so beginners read it at a glance
FORECAST = {
    "Low":    {"word": "Calm",   "icon": "☀️", "color": "#0ca30c", "tint": "#e7f6e7"},
    "Medium": {"word": "Choppy", "icon": "🌦️", "color": "#c98500", "tint": "#fdf3dc"},
    "High":   {"word": "Stormy", "icon": "⛈️", "color": "#d03b3b", "tint": "#fbe7e7"},
}

# One fixed color per coin (validated categorical palette), so a coin keeps its color everywhere
COIN_COLORS = {
    "BTC": "#2a78d6", "ETH": "#eb6834", "SOL": "#1baf7a", "BNB": "#eda100",
    "XRP": "#e87ba4", "ADA": "#008300", "DOGE": "#e34948", "USDC": "#6250d6",
}
EXTRA_COLORS = ["#256abf", "#d95926", "#199e70", "#c98500"]

INK, MUTED, LINE, SURFACE, ACCENT = "#172437", "#5b6b80", "#dce3ec", "#ffffff", "#1c5cab"

PRESETS = {
    "My own mix": None,
    "Bitcoin only": {"BTC": 5000},
    "Bitcoin + USDC (50/50)": {"BTC": 2500, "USDC": 2500},
    "Equal split, 5 coins": {"BTC": 1000, "ETH": 1000, "SOL": 1000, "BNB": 1000, "XRP": 1000},
    "Dogecoin only": {"DOGE": 3000},
}

st.set_page_config(page_title=f"{APP_NAME} · Crypto risk forecast", page_icon="🌦️", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ background: #f4f7fb; color: {INK}; }}
.block-container {{ padding-top: 2rem; max-width: 1180px; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: #eaf0f7; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color: {INK}; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}

.cw-brand {{ display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; margin-bottom:.2rem; }}
.cw-brand .name {{ font-size:2.1rem; font-weight:800; letter-spacing:-0.03em; color:{INK}; }}
.cw-brand .tag {{ font-size:1rem; color:{MUTED}; }}

.cw-hero {{ border-radius:20px; padding:1.6rem 1.8rem; margin:1rem 0 1.2rem;
  border:1px solid {LINE}; background:{SURFACE}; display:flex; gap:2rem; flex-wrap:wrap; align-items:center; }}
.cw-hero .left {{ flex: 1 1 340px; }}
.cw-hero .right {{ flex: 1 1 300px; }}
.cw-hero .kicker {{ color:{MUTED}; font-size:.95rem; margin-bottom:.2rem; }}
.cw-hero .word {{ font-size:3.4rem; line-height:1; font-weight:800; letter-spacing:-0.04em; }}
.cw-hero .sentence {{ font-size:1.05rem; color:{INK}; margin-top:.7rem; max-width:60ch; line-height:1.55; }}
.cw-pill {{ display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:700; font-size:.85rem;
  color:{INK}; margin-left:.6rem; vertical-align:middle; letter-spacing:0; line-height:1.4; }}
@media (max-width: 640px) {{ .cw-hero {{ padding:1.2rem; }} .cw-hero .word {{ font-size:2.5rem; }} .cw-brand .name {{ font-size:1.7rem; }} .cw-scale {{ font-size:.72rem; }} }}

.cw-meter {{ position:relative; height:14px; border-radius:999px; overflow:hidden; display:flex; }}
.cw-meter span {{ display:block; height:100%; }}
.cw-meter-wrap {{ position:relative; padding-top:28px; }}
.cw-marker {{ position:absolute; top:0; transform:translateX(-50%); text-align:center; font-weight:700; font-size:.9rem; color:{INK}; }}
.cw-marker::after {{ content:""; display:block; width:3px; height:30px; background:{INK}; margin:2px auto 0; border-radius:2px; }}
.cw-scale {{ display:flex; justify-content:space-between; color:{MUTED}; font-size:.8rem; margin-top:.45rem; }}

.cw-tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:.9rem; margin-bottom:1.2rem; }}
.cw-tile {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; padding:1rem 1.1rem; }}
.cw-tile .label {{ color:{MUTED}; font-size:.88rem; font-weight:600; }}
.cw-tile .value {{ font-size:1.9rem; font-weight:800; color:{INK}; letter-spacing:-0.02em; margin:.15rem 0; }}
.cw-tile .hint {{ color:{MUTED}; font-size:.85rem; line-height:1.4; }}

.cw-summary {{ background:{SURFACE}; border:1px solid {LINE}; border-left:6px solid var(--accent); border-radius:14px;
  padding:1.1rem 1.3rem; line-height:1.6; }}
.cw-summary p {{ margin:.35rem 0; max-width:75ch; }}

.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; padding:.55rem 1rem; border-radius:10px 10px 0 0; }}
.stTabs [aria-selected="true"] {{ color:{ACCENT}; }}
div[data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; padding:.8rem 1rem; }}
.stDownloadButton button {{ background:{ACCENT}; color:white; border:none; border-radius:10px; font-weight:700; padding:.55rem 1.1rem; }}
.stDownloadButton button:hover {{ background:#184f95; color:white; }}
.cw-foot {{ color:{MUTED}; font-size:.85rem; margin-top:2rem; padding-top:1rem; border-top:1px solid {LINE}; }}
</style>
""",
    unsafe_allow_html=True,
)


def html(block: str) -> None:
    """Render HTML without Markdown turning indented lines into code blocks."""
    st.markdown("\n".join(line.strip() for line in block.strip().splitlines()), unsafe_allow_html=True)


def md_inline(text: str) -> str:
    """Turn **bold** and _italic_ into HTML so the summary can sit inside one styled box."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)


def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", color=INK, size=13),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#e9eef4", zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED))
    return fig


# ----------------------------------------------------------------------------
# Data and calculations
# ----------------------------------------------------------------------------
@st.cache_data
def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna()


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily portfolio return, assuming the weights are kept constant (daily rebalancing)."""
    return returns[weights.index].mul(weights, axis=1).sum(axis=1)


def annual_volatility(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(r: pd.Series) -> float:
    value = (1 + r).cumprod()
    drawdown = value / value.cummax() - 1
    return float(drawdown.min())


def risk_label(vol: float) -> str:
    if vol < LOW_VOL:
        return "Low"
    if vol <= HIGH_VOL:
        return "Medium"
    return "High"


def concentrated_group(corr: pd.DataFrame, weights: pd.Series):
    """Largest share of the portfolio held in one coin plus the coins that move closely with it."""
    best_share, best_group = 0.0, []
    for coin in weights.index:
        group = [c for c in weights.index if c == coin or corr.loc[coin, c] >= CORR_LIMIT]
        share = float(weights[group].sum())
        if share > best_share:
            best_share, best_group = share, group
    return best_share, best_group


def risk_contributions(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Share of the portfolio's total risk that each coin is responsible for (sums to 100%)."""
    cov = returns[weights.index].cov() * TRADING_DAYS
    port_var = float(weights @ cov @ weights)
    if port_var <= 0:
        return weights * 0
    return weights * (cov @ weights) / port_var


def value_at_risk(r: pd.Series, level: float) -> float:
    """Historical 1-day VaR: loss that was not exceeded on `level` of past days (positive number)."""
    return float(-np.percentile(r, (1 - level) * 100))


def min_risk_mix(returns: pd.DataFrame, coins: list, max_share: float) -> pd.Series:
    """Long-only mix with the lowest volatility, with a cap on each coin."""
    n = len(coins)
    max_share = max(max_share, 1 / n)            # keep the problem feasible
    cov = returns[coins].cov().values * TRADING_DAYS
    start = np.repeat(1 / n, n)
    result = minimize(
        lambda w: float(w @ cov @ w),
        start,
        method="SLSQP",
        bounds=[(0, max_share)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
    )
    w = result.x if result.success else start
    w = np.clip(w, 0, None)
    return pd.Series(w / w.sum(), index=coins)


def with_stablecoin(weights: pd.Series, shift: float) -> pd.Series:
    """Move `shift` of the portfolio into the stablecoin, taken proportionally from every holding."""
    new_w = weights * (1 - shift)
    new_w[STABLECOIN] = new_w.get(STABLECOIN, 0.0) + shift
    return new_w


def pct0(x: float) -> str:
    """Whole-number percent without a '-0%'."""
    return f"{0 if abs(x) < 0.005 else x:.0%}"


def build_summary(weights, vol, mdd, label, corr, returns, data_end):
    lines = [
        f"Your portfolio swings about **{vol:.0%} a year** (annualized volatility), which we rate **{label}** risk. "
        f"In the past, its worst fall from a peak to a low was **{pct0(abs(mdd))}** of its value."
    ]

    share, group = concentrated_group(corr, weights)
    if len(weights) == 1:
        lines.append("You hold a single coin, so you have no diversification.")
    elif share >= CONCENTRATION_WARN and len(group) > 1:
        lines.append(
            f"**{share:.0%}** of your money is in coins that move closely together "
            f"({', '.join(group)}), so you are less diversified than the number of coins suggests."
        )
    else:
        lines.append("Your coins do not all move together, which gives you some diversification.")

    if len(weights) > 1:
        contrib = risk_contributions(returns, weights)
        top = contrib.idxmax()
        lines.append(f"**{top}** is {weights[top]:.0%} of your money but about **{contrib[top]:.0%}** of your total risk.")

    current_stable = float(weights.get(STABLECOIN, 0.0))
    if STABLECOIN in returns.columns and current_stable < STABLE_SHIFT and vol > 0:
        new_vol = annual_volatility(portfolio_returns(returns, with_stablecoin(weights, STABLE_SHIFT)))
        change = (vol - new_vol) / vol
        lines.append(
            f"💡 **Suggestion:** moving {STABLE_SHIFT:.0%} of your portfolio into {STABLECOIN} "
            f"(taken proportionally from the coins you hold now) would lower your volatility from {vol:.0%} "
            f"to about {new_vol:.0%}, around {change:.0%} lower."
        )
    elif weights.max() > 0.5 and len(weights) > 1 and weights.idxmax() != STABLECOIN:
        top = weights.idxmax()
        lines.append(f"💡 **Suggestion:** {top} is {weights.max():.0%} of your portfolio. Reducing it would lower your risk.")
    else:
        lines.append("💡 **Suggestion:** your allocation already has a cushion against swings. Check it again as prices change.")

    lines.append(f"_Based on past prices up to {data_end:%d %b %Y}. Past swings do not predict the future._")
    return lines


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
if not DATA_FILE.exists():
    st.error("prices.csv is missing. Put it in the same folder as app.py, then reload the page.")
    st.stop()

prices = load_prices(str(DATA_FILE))
if len(prices) < 30 or prices.shape[1] < 1:
    st.error("prices.csv needs at least 30 days of prices and one coin column. Check the file, then reload.")
    st.stop()

all_coins = list(prices.columns)
for i, c in enumerate(c for c in all_coins if c not in COIN_COLORS):
    COIN_COLORS[c] = EXTRA_COLORS[i % len(EXTRA_COLORS)]
returns = prices.pct_change().dropna()
data_start, data_end = prices.index.min(), prices.index.max()

# ----------------------------------------------------------------------------
# Sidebar: portfolio input (Feature 1)
# ----------------------------------------------------------------------------
ss = st.session_state
if "coins" not in ss:
    ss.coins = [c for c in ["BTC", "ETH", "SOL"] if c in all_coins] or all_coins[:2]
if "mode" not in ss:
    ss.mode = "Amount (EUR)"


def apply_preset():
    preset = PRESETS.get(ss.preset)
    if not preset:
        return
    ss.coins = [c for c in preset if c in all_coins]
    ss.mode = "Amount (EUR)"
    for coin, amount in preset.items():
        ss[f"eur_{coin}"] = float(amount)


with st.sidebar:
    html(f"<div class='cw-brand'><span class='name' style='font-size:1.5rem'>🌦️ {APP_NAME}</span></div>")
    st.markdown("### Your portfolio")
    st.selectbox("Start from an example", list(PRESETS), key="preset", on_change=apply_preset,
                 help="Pick an example to see the app in action, then change the numbers.")
    chosen = st.multiselect("Coins you hold", all_coins, key="coins")
    mode = st.radio("Enter holdings as", ["Amount (EUR)", "Percentage (%)"], key="mode", horizontal=True)

    raw = {}
    if mode == "Amount (EUR)":
        for coin in chosen:
            key = f"eur_{coin}"
            if key not in ss:
                ss[key] = 1000.0
            raw[coin] = st.number_input(f"{coin} (EUR)", min_value=0.0, step=100.0, key=key)
        total_input = None
    else:
        for coin in chosen:
            key = f"pct_{coin}"
            if key not in ss:
                ss[key] = round(100 / len(chosen), 1)
            raw[coin] = st.number_input(f"{coin} (%)", min_value=0.0, max_value=100.0, step=5.0, key=key)
        total_input = st.number_input("Total portfolio value (EUR)", min_value=0.0, value=DEFAULT_TOTAL, step=500.0,
                                      help="Turns your percentages into euro amounts for the loss estimates.")

    st.caption(f"Prices from {data_start:%d %b %Y} to {data_end:%d %b %Y} ({len(prices)} days).")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
html(f"""
<div class="cw-brand"><span class="name">🌦️ {APP_NAME}</span>
<span class="tag">A plain-English risk forecast for your crypto portfolio</span></div>
""")

if not chosen:
    st.info("👈 Choose the coins you hold in the sidebar, or pick an example to start.")
    st.stop()

raw = pd.Series(raw, dtype=float)
if raw.sum() <= 0:
    st.warning("Every amount is 0. Enter a value above 0 for at least one coin in the sidebar.")
    st.stop()

if mode == "Amount (EUR)":
    total = float(raw.sum())
else:
    pct_sum = float(raw.sum())
    if abs(pct_sum - 100) > 0.5:
        st.sidebar.info(f"Your percentages add up to {pct_sum:.1f}%. We scaled them to 100%.")
    total = float(total_input) if total_input and total_input > 0 else DEFAULT_TOTAL

weights = raw / raw.sum()
weights = weights[weights > 0]
amounts = weights * total

# ----------------------------------------------------------------------------
# Calculations
# ----------------------------------------------------------------------------
port_ret = portfolio_returns(returns, weights)
vol = annual_volatility(port_ret)
mdd = max_drawdown(port_ret)
label = risk_label(vol)
corr = returns[weights.index].corr().fillna(0)
contrib = risk_contributions(returns, weights)
v95 = value_at_risk(port_ret, VAR_LEVELS[0])
v99 = value_at_risk(port_ret, VAR_LEVELS[1])
share, group = concentrated_group(corr, weights)
fc = FORECAST[label]

# ----------------------------------------------------------------------------
# Hero: the forecast (Feature 3 headline)
# ----------------------------------------------------------------------------
meter_max = max(1.5, vol * 1.1)
marker_pos = min(vol / meter_max, 1) * 100
seg_low, seg_mid = LOW_VOL / meter_max * 100, (HIGH_VOL - LOW_VOL) / meter_max * 100
html(f"""
<div class="cw-hero" style="background:linear-gradient(180deg,{fc['tint']} 0%, {SURFACE} 75%);">
<div class="left">
<div class="kicker">Forecast for your {total:,.0f} EUR portfolio</div>
<div class="word" style="color:{fc['color']}">{fc['icon']} {fc['word']}<span class="cw-pill" style="background:{fc['tint']};border:1px solid {fc['color']}">{label} risk</span></div>
<div class="sentence">Its value swings about <b>{vol:.0%} a year</b>. At its worst it fell <b>{pct0(abs(mdd))}</b> from a peak, and on a bad day (1 in 20) it lost about <b>{v95 * total:,.0f} EUR</b>.</div>
</div>
<div class="right">
<div class="cw-meter-wrap">
<div class="cw-marker" style="left:{marker_pos:.1f}%">{vol:.0%}</div>
<div class="cw-meter">
<span style="width:{seg_low:.1f}%;background:#bfe5bf"></span>
<span style="width:{seg_mid:.1f}%;background:#f7dc9c"></span>
<span style="flex:1;background:#f2b9b9"></span>
</div>
</div>
<div class="cw-scale"><span>☀️ Calm, below {LOW_VOL:.0%}</span><span>🌦️ Choppy</span><span>⛈️ Stormy, above {HIGH_VOL:.0%}</span></div>
</div>
</div>
""")

together_value = "100%" if len(weights) == 1 else (f"{share:.0%}" if len(group) > 1 else "None")
together_hint = ("You hold one coin, so there is no diversification." if len(weights) == 1 else
                 f"Share of your money in coins that rise and fall together ({', '.join(group)})." if len(group) > 1 else
                 "None of your coins move closely together.")
html(f"""
<div class="cw-tiles">
<div class="cw-tile" title="Annualized volatility: standard deviation of daily returns x sqrt(365)">
<div class="label">Yearly swings</div><div class="value">{vol:.0%}</div>
<div class="hint">How much the value moves up and down in a year. Higher means a bumpier ride.</div></div>
<div class="cw-tile" title="Maximum drawdown in the data period">
<div class="label">Worst drop</div><div class="value">{pct0(mdd)}</div>
<div class="hint">The biggest fall from a previous high to a low in the data period.</div></div>
<div class="cw-tile" title="Historical 1-day Value at Risk at 95%">
<div class="label">Bad-day loss</div><div class="value">{v95 * total:,.0f} €</div>
<div class="hint">On 19 days out of 20, the daily loss was smaller than this ({v95:.1%}).</div></div>
<div class="cw-tile" title="Correlation of {CORR_LIMIT:.2f} or more counts as moving together">
<div class="label">Moves together</div><div class="value">{together_value}</div>
<div class="hint">{together_hint}</div></div>
</div>
""")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_div, tab_safer, tab_details = st.tabs(
    ["Overview", "Diversification", "Make it safer", "Details & report"]
)

# ---- Overview: summary (Feature 5) + allocation (Feature 2) + history -------
with tab_overview:
    st.markdown("#### What this means")
    summary_lines = build_summary(weights, vol, mdd, label, corr, returns, data_end)
    html(f"<div class='cw-summary' style='--accent:{fc['color']}'>"
         + "".join(f"<p>{md_inline(line)}</p>" for line in summary_lines) + "</div>")

    c_left, c_right = st.columns([1, 1.25], gap="large")
    with c_left:
        st.markdown("#### Your allocation")
        fig_pie = go.Figure(go.Pie(
            labels=list(amounts.index), values=amounts.values, hole=0.58, sort=False,
            marker=dict(colors=[COIN_COLORS[c] for c in amounts.index], line=dict(color="white", width=2)),
            textinfo="label+percent", textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} EUR<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(showlegend=False,
                              annotations=[dict(text=f"<b>{total:,.0f} €</b><br><span style='font-size:12px;color:{MUTED}'>total</span>",
                                                showarrow=False, font=dict(size=18, color=INK))])
        st.plotly_chart(style_fig(fig_pie, 340), config={"displayModeBar": False})
    with c_right:
        st.markdown("#### How it would have moved")
        value = ((1 + port_ret).cumprod() * total)
        hist_df = pd.DataFrame({"Date": value.index, "Value": value.values})
        fig_line = go.Figure(go.Scatter(
            x=hist_df["Date"], y=hist_df["Value"], mode="lines",
            line=dict(color=ACCENT, width=2), fill="tozeroy", fillcolor="rgba(28,92,171,0.08)",
            hovertemplate="%{x|%d %b %Y}<br><b>%{y:,.0f} EUR</b><extra></extra>",
        ))
        fig_line.update_layout(hovermode="x unified", showlegend=False)
        fig_line.update_yaxes(title=None, ticksuffix=" €")
        st.plotly_chart(style_fig(fig_line, 340), config={"displayModeBar": False})
        st.caption("If you had held this mix (kept in the same proportions) over the whole period.")

# ---- Diversification (Feature 4) --------------------------------------------
with tab_div:
    if len(weights) < 2:
        st.info("You hold a single coin, so there is no diversification. Add a second coin in the sidebar to compare how they move.")
    else:
        if share >= CONCENTRATION_WARN and len(group) > 1:
            st.warning(f"**{share:.0%} of your portfolio** is in coins that move closely together ({', '.join(group)}). "
                       f"Owning more of them does not spread your risk much.")
        else:
            st.success("Your coins do not all move together, so you have some diversification.")

        d_left, d_right = st.columns([1.1, 1], gap="large")
        with d_left:
            st.markdown("#### Do your coins move together?")
            fig_heat = go.Figure(go.Heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
                colorscale=[[0, "#1c5cab"], [0.5, "#f0efec"], [1, "#d03b3b"]],
                text=corr.round(2).values, texttemplate="%{text:.2f}", xgap=2, ygap=2,
                hovertemplate="%{y} and %{x}: <b>%{z:.2f}</b><extra></extra>",
                colorbar=dict(thickness=10, tickvals=[-1, 0, 1], ticktext=["opposite", "unrelated", "together"]),
            ))
            fig_heat.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(style_fig(fig_heat, 400), config={"displayModeBar": False})
            st.caption(f"Correlation of daily returns. 1 = always rise and fall together, 0 = unrelated, "
                       f"below 0 = opposite moves. We count {CORR_LIMIT:.2f} or more as moving closely together.")
        with d_right:
            st.markdown("#### Where your risk comes from")
            cmp_df = pd.DataFrame({"Coin": list(weights.index), "Share of money": weights.values,
                                   "Share of risk": contrib.values})
            fig_rc = go.Figure()
            fig_rc.add_bar(x=cmp_df["Coin"], y=cmp_df["Share of money"], name="Share of money", marker_color="#86b6ef",
                           hovertemplate="%{x}: %{y:.0%} of your money<extra></extra>")
            fig_rc.add_bar(x=cmp_df["Coin"], y=cmp_df["Share of risk"], name="Share of risk", marker_color=ACCENT,
                           hovertemplate="%{x}: %{y:.0%} of your risk<extra></extra>")
            fig_rc.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
            fig_rc.update_yaxes(tickformat=".0%")
            st.plotly_chart(style_fig(fig_rc, 400), config={"displayModeBar": False})
            st.caption("When a coin's share of risk is much bigger than its share of money, it is driving your ups and downs.")

# ---- Make it safer: stablecoin what-if + lower-risk mix ---------------------
with tab_safer:
    if STABLECOIN in returns.columns:
        st.markdown(f"#### What if you moved some money into {STABLECOIN}?")
        current_stable = float(weights.get(STABLECOIN, 0.0))
        shift_pct = st.slider(f"Share of your portfolio to move into {STABLECOIN}", 0, 80, 20, step=5, format="%d%%")
        shift = shift_pct / 100
        new_w = with_stablecoin(weights, shift)
        new_ret = portfolio_returns(returns, new_w)
        nv, nm = annual_volatility(new_ret), max_drawdown(new_ret)
        nfc = FORECAST[risk_label(nv)]
        w1, w2, w3 = st.columns(3)
        w1.metric("Yearly swings", f"{nv:.0%}", f"{(nv - vol) * 100:+.0f} pts vs now", delta_color="inverse")
        w2.metric("Worst drop", pct0(nm), f"{(nm - mdd) * 100:+.0f} pts vs now", delta_color="normal")
        w3.metric("Forecast", f"{nfc['icon']} {nfc['word']}", f"{risk_label(nv)} risk", delta_color="off")
        st.caption(f"The {shift_pct}% is taken proportionally from each coin you hold now. "
                   f"{STABLECOIN} would make up {new_w[STABLECOIN]:.0%} of the portfolio.")
        st.divider()

    st.markdown("#### The calmest mix we found")
    o1, o2 = st.columns(2)
    allow_all = o1.checkbox("Allow coins you don't hold yet", value=True)
    cap = o2.slider("Maximum share per coin", 20, 100, 50, step=5, format="%d%%") / 100
    candidates = all_coins if allow_all else list(weights.index)
    if len(candidates) < 2:
        st.info("Hold at least two coins, or tick \"Allow coins you don't hold yet\".")
    else:
        mix = min_risk_mix(returns, candidates, cap)
        mix = mix[mix > 0.005]
        mix = mix / mix.sum()
        mix_ret = portfolio_returns(returns, mix)
        mv, mm = annual_volatility(mix_ret), max_drawdown(mix_ret)
        mfc = FORECAST[risk_label(mv)]
        k1, k2, k3 = st.columns(3)
        k1.metric("Yearly swings", f"{mv:.0%}", f"{(mv - vol) * 100:+.0f} pts vs now", delta_color="inverse")
        k2.metric("Worst drop", pct0(mm), f"{(mm - mdd) * 100:+.0f} pts vs now", delta_color="normal")
        k3.metric("Forecast", f"{mfc['icon']} {mfc['word']}", f"{risk_label(mv)} risk", delta_color="off")

        table = pd.DataFrame({"Now": weights, "Suggested": mix}).fillna(0)
        table["Change (EUR)"] = (table["Suggested"] - table["Now"]) * total
        table.index.name = "Coin"
        st.dataframe(
            table,
            column_config={
                "Now": st.column_config.ProgressColumn("Now", format="percent", min_value=0, max_value=1),
                "Suggested": st.column_config.ProgressColumn("Suggested", format="percent", min_value=0, max_value=1),
                "Change (EUR)": st.column_config.NumberColumn("Change (EUR)", format="%+.0f €"),
            },
        )
        st.caption("This mix had the lowest past swings within your cap. It is an illustration based on history, not advice, "
                   "and lower risk usually also means lower expected return.")

# ---- Details & report --------------------------------------------------------
with tab_details:
    st.markdown("#### Value at Risk (1 day)")
    c1, c2 = st.columns(2)
    c1.metric("95% VaR", f"{v95 * total:,.0f} EUR", f"{v95:.1%} of portfolio", delta_color="off",
              help="On 95% of past days, the one-day loss was smaller than this.")
    c2.metric("99% VaR", f"{v99 * total:,.0f} EUR", f"{v99:.1%} of portfolio", delta_color="off",
              help="On 99% of past days, the one-day loss was smaller than this.")
    st.caption("Historical method, based on the worst past days in the data. A bad day can still be worse than this.")

    st.markdown("#### Latest prices")
    last = prices.iloc[-1]
    change = prices.pct_change().iloc[-1]
    latest = pd.DataFrame({"Last price (USD)": last, "1-day change": change}).loc[list(weights.index)]
    latest.index.name = "Coin"
    st.dataframe(latest, column_config={
        "Last price (USD)": st.column_config.NumberColumn(format="$%.4f"),
        "1-day change": st.column_config.NumberColumn(format="percent"),
    })
    st.caption(f"Data until {data_end:%d %b %Y}, from the saved prices.csv file, so the app works offline.")

    st.markdown("#### How the numbers are calculated")
    st.markdown(
        f"""
- **Daily return** = percentage change of each coin's daily closing price.
- **Portfolio return** = weighted sum of the coins' daily returns (weights kept constant).
- **Yearly swings (annualized volatility)** = standard deviation of daily portfolio returns × √{TRADING_DAYS}.
- **Worst drop (maximum drawdown)** = largest fall of portfolio value from a previous peak.
- **Moves together** = pairwise correlation of daily returns of {CORR_LIMIT:.2f} or more.
- **Forecast / risk score** = ☀️ Calm (Low) below {LOW_VOL:.0%}, 🌦️ Choppy (Medium) {LOW_VOL:.0%} to {HIGH_VOL:.0%}, ⛈️ Stormy (High) above {HIGH_VOL:.0%}.
- **Value at Risk (historical)** = the 5th and 1st percentile of past daily portfolio returns.
- **Data period**: {data_start:%d %b %Y} to {data_end:%d %b %Y}.
"""
    )

    report = pd.DataFrame({"Amount (EUR)": amounts.round(2), "Weight": weights, "Share of risk": contrib})
    summary_rows = pd.DataFrame({
        "Metric": ["Total value (EUR)", "Annualized volatility", "Max drawdown", "Risk score", "Forecast",
                   "95% 1-day VaR (EUR)", "99% 1-day VaR (EUR)", "Data from", "Data until"],
        "Value": [f"{total:.2f}", f"{vol:.2%}", f"{mdd:.2%}", label, fc["word"],
                  f"{v95 * total:.2f}", f"{v99 * total:.2f}", f"{data_start:%Y-%m-%d}", f"{data_end:%Y-%m-%d}"],
    })
    csv_text = summary_rows.to_csv(index=False) + "\n" + report.round(4).to_csv(index_label="Coin")
    st.download_button("Download my risk report (CSV)", csv_text,
                       file_name="coinweather_risk_report.csv", mime="text/csv")

html(f"""
<div class="cw-foot">⚠️ {APP_NAME} is an educational tool, not financial advice. Results use past prices, which do not predict the future.
No login, no wallet connection, and nothing you enter is stored.</div>
""")
