import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# ============================================================
# Quotex-style market structure scanner (Upgraded with Advanced Pre-Open Signals)
# ============================================================
st.set_page_config(page_title="Market Structure Scanner", layout="wide")
st.title("📊 Multi-Asset Predictive Market Structure Scanner")
st.caption(
    "Technical analysis dashboard with pre-open signals. Signals are heuristic, data may differ from Quotex, "
    "and no signal guarantees a profitable outcome."
)
st_autorefresh(interval=1_000, key="scanner_refresh")

AVAILABLE_ASSETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/CAD", "GBP/CAD", "AUD/NZD",
    "BTC/USD", "ETH/USD", "SOL/USD", "XAU/USD", "XAG/USD",
]
TIMEFRAMES = {
    "1 Minute": ("1m", "1d", 60),
    "5 Minutes": ("5m", "5d", 300),
    "15 Minutes": ("15m", "5d", 900),
    "30 Minutes": ("30m", "1mo", 1800),
    "1 Hour": ("60m", "3mo", 3600),
    "4 Hours": ("1h", "2y", 14400),
    "1 Day": ("1d", "5y", 86400),
}
TICKER_MAP = {
    "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "SOL/USD": "SOL-USD",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F",
}

if "market_cache" not in st.session_state:
    st.session_state.market_cache = {}

# --------------------------- Sidebar -------------------------
st.sidebar.header("Scanner settings")
# Master View Mode Configuration Choice
view_mode = st.sidebar.radio(
    "Select App Mode On Startup:",
    ["🎯 Single Asset Focus", "🌐 Complete Market Matrix Scan"],
    index=0,  # Forces Single Asset Mode by default when you manually open the app
    help="Single mode optimizes speed. Matrix mode checks everything at once."
)

if view_mode == "🎯 Single Asset Focus":
    # Drops a clean dropdown menu so you choose exactly ONE asset manually
    manual_choice = st.sidebar.selectbox(
        "Choose focus asset:", 
        AVAILABLE_ASSETS, 
        index=0  # Defaults to EUR/USD on bootup
    )
    selected_assets = [manual_choice]
else:
    # Falls back to your original multi-select matrix dashboard array
    selected_assets = st.sidebar.multiselect(
        "Assets to scan", AVAILABLE_ASSETS, default=AVAILABLE_ASSETS,
        help="All 19 assets are selected by default."
    )

st.sidebar.caption(f"{len(selected_assets)} / {len(AVAILABLE_ASSETS)} assets selected")
selected_timeframe = st.sidebar.selectbox("Chart timeframe", list(TIMEFRAMES), index=0)
chart_interval, chart_period, chart_seconds = TIMEFRAMES[selected_timeframe]
clock_offset = st.sidebar.number_input(
    "Candle clock adjustment (seconds)", min_value=-120, max_value=120, value=-23, step=1,
    help="Keep at 0 unless you have measured a consistent difference..."
)

# Added customization for pre-open evaluation windows
signal_buffer_seconds = st.sidebar.slider(
    "Pre-open Signal Trigger Window", 
    min_value=1, max_value=10, value=5,
    help="Triggers the advanced UP/DOWN prediction signal this many seconds before the candle closes."
)
st.sidebar.caption("Screen refresh: 1 second · Market data cache: 15 seconds")

# -------------------------- Data source ----------------------
def yahoo_symbol(asset: str) -> str:
    return TICKER_MAP.get(asset, asset.replace("/", "") + "=X")


@st.cache_data(ttl=15, show_spinner=False)
def fetch_yahoo_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, timeout=8)
    if data is None or data.empty:
        return pd.DataFrame()
    data = data.reset_index()
    time_col = "Datetime" if "Datetime" in data.columns else "Date"
    data = data.rename(columns={time_col: "Time"})
    keep = [c for c in ["Time", "Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data = data[keep].dropna(subset=["Close"])
    data["Time"] = pd.to_datetime(data["Time"])
    return data


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["BB_Middle"] = df["Close"].rolling(20, min_periods=20).mean()
    std = df["Close"].rolling(20, min_periods=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * std
    df["BB_Lower"] = df["BB_Middle"] - 2 * std
    df["ATR"] = (df["High"] - df["Low"]).rolling(14, min_periods=14).mean()
    return df


def load_asset_data(asset: str, timeframe_label: str):
    interval, period, _ = TIMEFRAMES[timeframe_label]
    key = f"{asset}|{interval}"
    try:
        raw = fetch_yahoo_data(yahoo_symbol(asset), period, interval)
        if raw.empty:
            raise ValueError("Data provider returned no candles")
        st.session_state.market_cache[key] = raw.copy()
        source, error = "Yahoo Finance", None
    except Exception as exc:
        raw = st.session_state.market_cache.get(key, pd.DataFrame())
        if raw.empty:
            return pd.DataFrame(), "Unavailable", str(exc)
        source, error = "Cached real data (last fetch failed)", str(exc)
    return add_indicators(raw), source, error

# ---------------------- Market structure ---------------------
def analyze_market_structure(frame: pd.DataFrame) -> dict:
    result = {
        "trend": "UNKNOWN", "phase": "UNCLASSIFIED", "support": np.nan,
        "resistance": np.nan, "near_support": False, "near_resistance": False,
        "breakout_up": False, "breakout_down": False,
        "higher_highs": False, "higher_lows": False,
        "lower_highs": False, "lower_lows": False,
        "reason": "Insufficient candle history",
    }
    if frame is None or len(frame) < 60:
        return result
    df = frame.iloc[:-1].copy()
    if len(df) < 50:
        return result
    high, low, close = df["High"], df["Low"], df["Close"]
    last, prev = df.iloc[-1], df.iloc[-2]
    lookback = 30
    support = float(low.iloc[-lookback-1:-1].min())
    resistance = float(high.iloc[-lookback-1:-1].max())
    price = float(last["Close"])
    atr = float(df["ATR"].iloc[-1]) if pd.notna(df["ATR"].iloc[-1]) else 0.0
    if not np.isfinite(atr) or atr <= 0:
        atr = max(abs(price) * 0.001, 1e-8)

    recent_high, older_high = float(high.iloc[-10:].max()), float(high.iloc[-25:-10].max())
    recent_low, older_low = float(low.iloc[-10:].min()), float(low.iloc[-25:-10].min())
    hh, hl = recent_high > older_high, recent_low > older_low
    lh, ll = recent_high < older_high, recent_low < older_low
    ema50, ema200 = float(last["EMA_50"]), float(last["EMA_200"])
    ema50_old = float(df["EMA_50"].iloc[-6])
    up = hh and hl and price > ema50 and ema50 > ema200 and ema50 > ema50_old
    down = lh and ll and price < ema50 and ema50 < ema200 and ema50 < ema50_old
    range_width = resistance - support
    sideways = not up and not down and range_width <= max(atr * 10, abs(price) * 0.012)
    near_support = abs(price - support) <= atr * 0.7
    near_resistance = abs(resistance - price) <= atr * 0.7
    breakout_up = price > resistance and float(prev["Close"]) <= resistance
    breakout_down = price < support and float(prev["Close"]) >= support

    result.update({
        "support": support, "resistance": resistance,
        "near_support": near_support, "near_resistance": near_resistance,
        "breakout_up": breakout_up, "breakout_down": breakout_down,
        "higher_highs": hh, "higher_lows": hl,
        "lower_highs": lh, "lower_lows": ll,
    })
    move20 = price / float(close.iloc[-20]) - 1 if float(close.iloc[-20]) else 0
    if up:
        result.update(trend="UPTREND", phase="MARKUP", reason="Higher highs/lows and bullish EMA structure")
    elif down:
        result.update(trend="DOWNTREND", phase="MARKDOWN", reason="Lower highs/lows and bearish EMA structure")
    else:
        result["trend"] = "SIDEWAYS" if sideways else "MIXED"
        if sideways and move20 < -0.01 and near_support:
            result.update(phase="POSSIBLE ACCUMULATION", reason="Range near support after a recent decline; unconfirmed")
        elif sideways and move20 > 0.01 and near_resistance:
            result.update(phase="POSSIBLE DISTRIBUTION", reason="Range near resistance after a recent advance; unconfirmed")
        elif sideways:
            result.update(phase="SIDEWAYS / RANGE", reason="Price is ranging without a confirmed directional trend")
        else:
            result.update(phase="TRANSITION / MIXED", reason="Trend structure is not sufficiently aligned")
    return result

# ---------------------- Predictive Engine ---------------------
def predictive_pre_open_signal(frame: pd.DataFrame, higher_frame: pd.DataFrame, seconds_remaining: int, real_data: bool = True):
    """
    Computes confirmation algorithms. If inside the pre-candle window, 
    upgrades standard suggestions into real-time pre-open direction signals.
    """
    if not real_data:
        return "⚪ CLOCK WAITING", "Real market data unavailable; signal deactivated.", {}, False

    if frame is None or higher_frame is None or len(frame) < 60 or len(higher_frame) < 60:
        return "⚪ COOLDOWN", "Warming up candle sequence historical arrays.", {}, False

    required = ["Close", "EMA_50", "EMA_200", "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
    if any(not all(col in df.columns for col in required) for df in (frame, higher_frame)):
        return "⚪ RE-CALCULATING", "Required mathematical arrays are regenerating.", {}, False

# ---------------------- Predictive Engine ---------------------
def predictive_pre_open_signal(frame: pd.DataFrame, higher_frame: pd.DataFrame, seconds_remaining: int, real_data: bool = True):
    """
    Computes confirmation algorithms. If inside the pre-candle window, 
    upgrades standard suggestions into real-time pre-open direction signals.
    """
    if not real_data:
        return "⚪ CLOCK WAITING", "Real market data unavailable; signal deactivated.", {}, False

    if frame is None or higher_frame is None or len(frame) < 60 or len(higher_frame) < 60:
        return "⚪ COOLDOWN", "Warming up candle sequence historical arrays.", {}, False

    required = ["Close", "EMA_50", "EMA_200", "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
    if any(not all(col in df.columns for col in required) for df in (frame, higher_frame)):
        return "⚪ RE-CALCULATING", "Required mathematical arrays are regenerating.", {}, False

    # During pre-open windows, evaluate the live candle (index -1) instead of waiting for close (index -2)
    is_pre_open_window = (seconds_remaining <= signal_buffer_seconds and seconds_remaining > 0)
    
    if is_pre_open_window:
        entry = frame.iloc[-1]  # Capture volatile live candle momentum right before expiration
    else:
        entry = frame.iloc[-2]  # Default back to verified static data points
        
    trend = higher_frame.iloc[-2]
    structure = analyze_market_structure(frame)
    higher_structure = analyze_market_structure(higher_frame)

    fields = [entry[c] for c in required if c in entry] + [trend[c] for c in required if c in trend]
    if not all(pd.notna(v) and np.isfinite(v) for v in fields):
        return "⚪ CONVERGING", "Dynamic calculations warming up.", structure, False

    # Convergence Condition Rules
    bull_htf = (higher_structure["trend"] == "UPTREND" and trend["Close"] > trend["EMA_50"] and trend["MACD"] > trend["MACD_Signal"])
    bear_htf = (higher_structure["trend"] == "DOWNTREND" and trend["Close"] < trend["EMA_50"] and trend["MACD"] < trend["MACD_Signal"])
    
    bull_momentum = (entry["Close"] > entry["EMA_50"] and entry["MACD"] > entry["MACD_Signal"] and 50 <= entry["RSI"] <= 75)
    bear_momentum = (entry["Close"] < entry["EMA_50"] and entry["MACD"] < entry["MACD_Signal"] and 25 <= entry["RSI"] <= 50)
    
    buy_level_ok = not structure["near_resistance"] or structure["breakout_up"]
    sell_level_ok = not structure["near_support"] or structure["breakout_down"]

    # Trigger logic output processing
    if (structure["trend"] == "UPTREND" or structure["trend"] == "MIXED") and bull_htf and bull_momentum and buy_level_ok:
        if is_pre_open_window:
            return "🚀 PRE-OPEN UP", f"CRITICAL CONVERGENCE: Entering next candle long. Time left: {seconds_remaining}s.", structure, True
        return "🟢 BUY ALERT", "Bullish structure aligned. Waiting for pre-open execution window.", structure, False

    if (structure["trend"] == "DOWNTREND" or structure["trend"] == "MIXED") and bear_htf and bear_momentum and sell_level_ok:
        if is_pre_open_window:
            return "📉 PRE-OPEN DOWN", f"CRITICAL CONVERGENCE: Entering next candle short. Time left: {seconds_remaining}s.", structure, True
        return "🔴 SELL ALERT", "Bearish structure aligned. Waiting for pre-open execution window.", structure, False

    if structure["trend"] == "SIDEWAYS" and entry["Close"] <= structure["support"] * 1.001:
        if is_pre_open_window and entry["RSI"] < 30:
            return "🚀 PRE-OPEN UP", f"RANGE BOUNCE: Oversold baseline rejection at support. Time left: {seconds_remaining}s.", structure, True
        return "🟡 WAIT — SUPPORT", " Ranging near floor support boundary lines.", structure, False

    if structure["trend"] == "SIDEWAYS" and entry["Close"] >= structure["resistance"] * 0.999:
        if is_pre_open_window and entry["RSI"] > 70:
            return "📉 PRE-OPEN DOWN", f"RANGE BOUNCE: Overbought ceiling rejection at resistance. Time left: {seconds_remaining}s.", structure, True
        return "🟡 WAIT — RESISTANCE", " Ranging near ceiling resistance boundary lines.", structure, False

    return "⚪ MONITORING", f"Structure state: {structure['trend']}. Confirmations incomplete.", structure, False


# ----------------------- Clock / formatting ------------------
def candle_clock(seconds: int):
    now = time.time() + float(clock_offset)
    boundary = int(now // seconds) * seconds
    elapsed = int(now - boundary)
    remaining = max(0, int(math.ceil(boundary + seconds - now)))
    return remaining, elapsed, boundary


def fmt_price(value) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "N/A"
        return f"{value:,.5f}" if abs(value) < 100 else f"{value:,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_countdown(seconds: int) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


# ------------------------ Load all assets --------------------
if not selected_assets:
    st.info("Select at least one asset in the sidebar.")
    st.stop()

asset_data = {}
signal_frames = {}
with st.spinner(f"Loading data for {len(selected_assets)} assets…"):
    for asset in selected_assets:
        asset_data[asset] = load_asset_data(asset, selected_timeframe)
        one = asset_data[asset] if selected_timeframe == "1 Minute" else load_asset_data(asset, "1 Minute")
        five = asset_data[asset] if selected_timeframe == "5 Minutes" else load_asset_data(asset, "5 Minutes")
        signal_frames[asset] = {"1m": one, "5m": five}


# ---------------------------- Scanner -------------------------
st.subheader("🌐 Real-Time Scanner Matrix")
st.caption(f"{len(selected_assets)} assets · Timeframe: {selected_timeframe} · System Check Time: {datetime.now().strftime('%H:%M:%S')}")
remaining, elapsed, _ = candle_clock(60)

cols_per_row = 3
for start in range(0, len(selected_assets), cols_per_row):
    cols = st.columns(cols_per_row)
    for col, asset in zip(cols, selected_assets[start:start + cols_per_row]):
        with col:
            chart_frame, chart_source, chart_error = asset_data[asset]
            one_frame, one_source, one_error = signal_frames[asset]["1m"]
            five_frame, five_source, five_error = signal_frames[asset]["5m"]
            
            if chart_frame.empty:
                st.error(f"{asset} data stream interrupted.")
                continue
                
            real_data = (not one_frame.empty and not five_frame.empty and not one_source.startswith("Unavailable"))
            signal, reason, structure, flash_trigger = predictive_pre_open_signal(one_frame, five_frame, remaining, real_data)
            
            if flash_trigger:
                container_title = f"🚨 {asset} - PRE-OPEN CRITICAL"
            else:
                container_title = f"{asset}"
                
            with st.container(border=True):
                st.markdown(f"### {container_title}")
                latest = chart_frame.iloc[-1]
                prev_close = chart_frame.iloc[-2]["Close"] if len(chart_frame) > 1 else latest["Close"]
                change = ((latest["Close"] - prev_close) / prev_close * 100) if prev_close else 0
                st.metric("Latest close", fmt_price(latest["Close"]), f"{change:+.3f}%")
                
                if "PRE-OPEN UP" in signal:
                    st.success(f"Signal: {signal}")
                elif "PRE-OPEN DOWN" in signal:
                    st.error(f"Signal: {signal}")
                elif "BUY" in signal or "SUPPORT" in signal:
                    st.info(f"Signal: {signal}")
                elif "SELL" in signal or "RESISTANCE" in signal:
                    st.warning(f"Signal: {signal}")
                else:
                    st.markdown(f"Signal: {signal}")
                    
                st.caption(reason)
                if structure:
                    st.write(f"Trend / phase: {structure['trend']} · {structure['phase']}")
                
                if remaining <= signal_buffer_seconds:
                    st.metric("⚠️ NEXT CANDLE INITIALIZING", f"{remaining}s", delta="- PRE-OPEN POSITION WINDOW", delta_color="inverse")
                else:
                    st.metric("1m candle countdown", format_countdown(remaining))
                st.caption(f"RSI: {latest['RSI']:.1f} · EMA200: {fmt_price(latest['EMA_200'])}")


# ---------------------- Detailed analysis --------------------
st.divider()
st.subheader("🔎 Deep-Dive Asset Structural Inspector")
focus_asset = st.selectbox("Asset to inspect", selected_assets, key="focus_asset")
frame, source, error = asset_data[focus_asset]
one_frame, one_source, one_error = signal_frames[focus_asset]["1m"]
five_frame, five_source, five_error = signal_frames[focus_asset]["5m"]

real_data = not one_frame.empty and not five_frame.empty and not one_source.startswith("Unavailable")
signal, reason, structure, flash_trigger = predictive_pre_open_signal(one_frame, five_frame, remaining, real_data)

if "PRE-OPEN" in signal:
    st.markdown(f"## 🔥 TARGET LOCK: {signal}")
else:
    st.markdown(f"### Current State: {signal}")
st.info(reason)

if structure:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", structure["trend"])
    c2.metric("Market phase", structure["phase"])
    c3.metric("Support Line", fmt_price(structure["support"]))
    c4.metric("Resistance Line", fmt_price(structure["resistance"]))
    st.caption(structure["reason"])

if not frame.empty:
    chart = frame.tail(150).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.72, 0.28], subplot_titles=("Candles, EMAs and levels", "RSI"))
fig.add_trace(go.Candlestick(x=chart["Time"], open=chart["Open"], high=chart["High"],
low=chart["Low"], close=chart["Close"], name="Candles"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart["Time"], y=chart["EMA_50"], name="EMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart["Time"], y=chart["EMA_200"], name="EMA 200"), row=1, col=1)
if structure and np.isfinite(structure.get("support", np.nan)):
 fig.add_hline(y=structure["resistance"], line_dash="dash", annotation_text="Resistance Ceiling", row=1, col=1)
fig.add_hline(y=30, line_dash="dot", row=2, col=1)
fig.update_layout(height=680, template="plotly_dark", xaxis_rangeslider_visible=False,
margin=dict(l=10, r=10, t=55, b=10), legend=dict(orientation="h"))
fig.update_yaxes(range=[0, 100], row=2, col=1)
st.plotly_chart(fig, use_container_width=True)
display_cols = ["Time", "Open", "High", "Low", "Close", "EMA_50", "EMA_200", "RSI", "MACD", "MACD_Signal"]
st.dataframe(frame[display_cols].tail(10).sort_values("Time", ascending=False), use_container_width=True, hide_index=True)
st.caption(f"Chart source: {source}")
