import math
import time
from datetime import datetime
import base64
import io
import wave
import struct

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Global storage for live OTC price streams
if "otc_live_data" not in st.session_state:
    st.session_state.otc_live_data = {}

# ============================================================
# Quotex-style market structure scanner (Upgraded with Advanced Pre-Open Signals)
# ============================================================
st.set_page_config(page_title="Market Structure Scanner", layout="wide")
st.title("📊 Multi-Asset Predictive Market Structure Scanner")
st.caption(
    "Technical analysis dashboard with pre-open signals. Signals are heuristic, data may differ from Quotex, "
    "and no signal guarantees a profitable outcome."
)

AVAILABLE_ASSETS = [
    # --- Standard Real-Market Assets ---
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/CAD", "GBP/CAD", "AUD/NZD",
    "BTC/USD", "ETH/USD", "SOL/USD", "XAU/USD", "XAG/USD",
    
    # --- Expanded Broker OTC Pairs ---
    "EUR/USD (OTC)", 
    "GBP/USD (OTC)", 
    "USD/JPY (OTC)", 
    "USD/INR (OTC)", 
    "EUR/GBP (OTC)", 
    "USD/BRL (OTC)", 
    "BTC/USD (OTC)", 
    "XAU/USD (OTC)"
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

# ==============================================================================
# INITIALISE ALARM & STATE CACHE TRACKERS
# ==============================================================================
if "market_cache" not in st.session_state:
    st.session_state.market_cache = {}

if "played_alarms" not in st.session_state:
    st.session_state.played_alarms = {}
# ==============================================================================

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
# --- BROWSER AUDIO UNLOCK ENGINE ---
st.sidebar.markdown("### 🔊 Audio System")
if "audio_unlocked" not in st.session_state:
    st.session_state.audio_unlocked = False

if not st.session_state.audio_unlocked:
    if st.sidebar.button(
        "🔓 ENABLE ALARM AUDIO", type="primary", use_container_width=True
    ):
        st.session_state.audio_unlocked = True
        st.sidebar.success("Audio enabled. Your browser may still require interaction before autoplay.")
else:
    st.sidebar.caption("✅ Alarm Sound Enabled")

st.sidebar.caption(f"{len(selected_assets)} / {len(AVAILABLE_ASSETS)} assets selected")
selected_timeframe = st.sidebar.selectbox("Chart timeframe", list(TIMEFRAMES), index=0)
chart_interval, chart_period, chart_seconds = TIMEFRAMES[selected_timeframe]
clock_offset = st.sidebar.number_input(
    "Candle clock adjustment (seconds)", min_value=-120, max_value=120, value=-31, step=1,
    help="Keep at 0 unless you have measured a consistent difference..."
)

# Added customization for pre-open evaluation windows
signal_buffer_seconds = st.sidebar.slider(
    "Pre-open Signal Trigger Window", 
    min_value=1, max_value=10, value=10,
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

# ==============================================================================
# SYNTHETIC OTC MARKET GENERATOR (FOR WEEKENDS AND PRIVATE DATA TESTING)
# ==============================================================================
# ==============================================================================
# UNIFIED LIVE MARKET & HIGH-FIDELITY OTC DATA FEED ENGINE
# ==============================================================================
def load_asset_data(asset: str, timeframe_label: str):
    """
    Unified market engine that matches OTC simulation baselines to 
    actual real-world market pricing at runtime.
    """
    interval, period, _ = TIMEFRAMES[timeframe_label]
    
    # --- 1. SMART OTC REAL-MARKET BASELINE INTERCEPTOR ---
    if "(OTC)" in asset:
        num_candles = 250
        np.random.seed(int(time.time()) // 60 + sum((i + 1) * ord(ch) for i, ch in enumerate(asset)) % 10000)
        
        # Strip the (OTC) tag out to discover the underlying real market ticker symbol
        base_asset = asset.replace(" (OTC)", "")
        symbol = yahoo_symbol(base_asset)
        
        try:
            # Look up the last known actual real market price to use as our anchor point
            real_ticker = fetch_yahoo_data(symbol, "1d", "1m")
            if not real_ticker.empty:
                start_price = float(real_ticker["Close"].iloc[-1])
            else:
                raise ValueError()
        except Exception:
            # Fallback hardcoded baselines if the network request times out
            start_price = 152.40 if "JPY" in asset else (1.1150 if "EUR" in asset or "GBP" in asset else 82.50)
            
        # Determine the appropriate fraction step pip sizes
        pip_size = 0.01 if "JPY" in asset or "INR" in asset or "XAU" in asset else 0.0001

        prices = [start_price]
        for _ in range(num_candles - 1):
            drift = np.random.uniform(-0.01, 0.01) * pip_size
            noise = np.random.normal(0, 1.2) * pip_size
            prices.append(max(pip_size, prices[-1] + drift + noise))
            
        end_time = pd.Timestamp.now()
        times = [end_time - pd.Timedelta(seconds=i * 60) for i in range(num_candles)]
        times.reverse()
        
        df = pd.DataFrame(index=range(num_candles))
        df["Time"] = times
        df["Close"] = prices
        
        volatility = df["Close"].diff().abs().mean() * 1.5
        df["Open"] = df["Close"].shift(1).fillna(start_price)
        df["High"] = df[["Open", "Close"]].max(axis=1) + np.random.uniform(0, volatility, num_candles)
        df["Low"] = df[["Open", "Close"]].min(axis=1) - np.random.uniform(0, volatility, num_candles)
        df["Volume"] = np.random.randint(5000, 85000, num_candles)
        
        return add_indicators(df), "Broker OTC Engine (Market-Matched Simulation)", None

    # --- 2. STANDARD REAL-MARKET ENGINE (YAHOO FINANCE) ---
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
# ==============================================================================
# ADVANCED INSTITUTIONAL LIQUIDITY SWEEP DETECTION MODULE
# ==============================================================================
def detect_liquidity_sweeps(frame: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Scans recent candle history to identify institutional liquidity sweeps.
    Detects when price sweeps past clean highs/lows and aggressively rejects.
    """
    sweep_results = {
        "sweep_type": "NONE",
        "swept_level": np.nan,
        "is_sweep_active": False,
        "volume_confirmed": False
    }
    
    if frame is None or len(frame) < lookback + 5:
        return sweep_results

    # Use closed candles to avoid dynamic array recalculation flickering
    df = frame.copy()
    
    # 1. Identify previous swing highs and lows (Liquidity Pools)
    # Exclude the immediate live candles to find historical resting liquidity
    historical_zone = df.iloc[:-5]
    swing_high = float(historical_zone["High"].max())
    swing_low = float(historical_zone["Low"].min())
    
    # Analyze the most recent completed candle setup
    trigger_candle = df.iloc[-2]
    
    # Calculate average volume for confirmation filters
    avg_volume = df["Volume"].iloc[-lookback:-2].mean()
    high_volume_spike = trigger_candle["Volume"] > (avg_volume * 1.3) if "Volume" in df.columns and avg_volume > 0 else True

    # 2. Check for Buy-Side Liquidity Sweep (BSL)
    # Price breaks above historical swing high, but closes back down below it
    if trigger_candle["High"] > swing_high and trigger_candle["Close"] < swing_high:
        sweep_results.update({
            "sweep_type": "BUY-SIDE SWEEP (BEARISH REVERSAL)",
            "swept_level": swing_high,
            "is_sweep_active": True,
            "volume_confirmed": high_volume_spike
        })
        return sweep_results

    # 3. Check for Sell-Side Liquidity Sweep (SSL)
    # Price breaks below historical swing low, but closes back up above it
    if trigger_candle["Low"] < swing_low and trigger_candle["Close"] > swing_low:
        sweep_results.update({
            "sweep_type": "SELL-SIDE SWEEP (BULLISH REVERSAL)",
            "swept_level": swing_low,
            "is_sweep_active": True,
            "volume_confirmed": high_volume_spike
        })
        return sweep_results

    return sweep_results

# ---------------------- Predictive Engine ---------------------
def predictive_pre_open_signal(frame: pd.DataFrame, higher_frame: pd.DataFrame, seconds_remaining: int, real_data = True):
    """
    Computes confirmation algorithms. Synthesizes market structures 
    and institutional liquidity sweeps to output pre-open directional alerts.
    """
    if not real_data:
         return "⚪ CLOCK WAITING", "Real market data unavailable; signal deactivated.", {}, False

    if frame is None or higher_frame is None or len(frame) < 60 or len(higher_frame) < 60:
        return "⚪ COOLDOWN", "Warming up candle sequence historical arrays.", {}, False

    required = ["Close", "EMA_50", "EMA_200", "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
    if any(not all(col in df.columns for col in required) for df in (frame, higher_frame)):
        return "⚪ RE-CALCULATING", "Required mathematical arrays are regenerating.", {}, False

    # Evaluation windows configuration
    is_pre_open_window = (seconds_remaining <= signal_buffer_seconds and seconds_remaining > 0)
    
    if is_pre_open_window:
        entry = frame.iloc[-1]  
    else:
        entry = frame.iloc[-2]  
        
    trend = higher_frame.iloc[-2]
    structure = analyze_market_structure(frame)
    higher_structure = analyze_market_structure(higher_frame)
    
    # Compute Liquidity Sweeps
    liquidity = detect_liquidity_sweeps(frame, lookback=20)

    fields = [entry[c] for c in required if c in entry] + [trend[c] for c in required if c in trend]
    if not all(pd.notna(v) and np.isfinite(v) for v in fields):
        return "⚪ CONVERGING", "Dynamic calculations warming up.", structure, False

    # 1. PRIORITY EXECUTION: High-Accuracy Institutional Liquidity Sweep Signals
    if liquidity["is_sweep_active"]:
        if "SELL-SIDE" in liquidity["sweep_type"] and entry["RSI"] < 40:
            if is_pre_open_window:
                return "🚀 PRE-OPEN UP", f"LIQUIDITY SWEEP: Swept retail stops at {fmt_price(liquidity['swept_level'])}. Entering Long. Time: {seconds_remaining}s.", structure, True
            return "🟢 BUY ALERT (SWEEP)", f"Institutional demand sweep detected at {fmt_price(liquidity['swept_level'])}. Awaiting execution window.", structure, False
            
        if "BUY-SIDE" in liquidity["sweep_type"] and entry["RSI"] > 60:
            if is_pre_open_window:
                return "📉 PRE-OPEN DOWN", f"LIQUIDITY SWEEP: Swept retail stops at {fmt_price(liquidity['swept_level'])}. Entering Short. Time: {seconds_remaining}s.", structure, True
            return "🔴 SELL ALERT (SWEEP)", f"Institutional supply sweep detected at {fmt_price(liquidity['swept_level'])}. Awaiting execution window.", structure, False

    # 2. FALLBACK EXECUTION: Trend Following & Range Breakouts
    bull_htf = (higher_structure["trend"] == "UPTREND" and trend["Close"] > trend["EMA_50"] and trend["MACD"] > trend["MACD_Signal"])
    bear_htf = (higher_structure["trend"] == "DOWNTREND" and trend["Close"] < trend["EMA_50"] and trend["MACD"] < trend["MACD_Signal"])
    
    bull_momentum = (entry["Close"] > entry["EMA_50"] and entry["MACD"] > entry["MACD_Signal"] and 50 <= entry["RSI"] <= 75)
    bear_momentum = (entry["Close"] < entry["EMA_50"] and entry["MACD"] < entry["MACD_Signal"] and 25 <= entry["RSI"] <= 50)
    
    buy_level_ok = not structure["near_resistance"] or structure["breakout_up"]
    sell_level_ok = not structure["near_support"] or structure["breakout_down"]

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


def play_signal_alarm(pair_name: str, signal_type: str):
    """Show a signal toast and play a short locally generated alarm tone."""
    try:
        sample_rate = 22050
        duration = 0.35
        frequency = 880
        frames = bytearray()
        for i in range(int(sample_rate * duration)):
            envelope = min(1.0, i / 500, (int(sample_rate * duration) - i) / 1500)
            sample = int(9000 * max(0.0, envelope) * math.sin(2 * math.pi * frequency * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(frames))
        encoded = base64.b64encode(audio_buffer.getvalue()).decode("ascii")
        st.markdown(
            f'<audio autoplay><source src="data:audio/wav;base64,{encoded}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass
    st.toast(f"🔥 ALARM TRIGGERED: {pair_name} ({signal_type})!", icon="🔔")




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


# ------------------------ Cached asset bundles ----------------
@st.cache_data(ttl=15, max_entries=256, show_spinner=False)
def get_asset_bundle(asset: str, timeframe_label: str):
    """
    Load chart + signal frames once per 15-second cache window.
    Duplicate timeframes are loaded only once per asset.
    """
    labels = list(dict.fromkeys([timeframe_label, "1 Minute", "5 Minutes"]))
    loaded = {label: load_asset_data(asset, label) for label in labels}
    chart_data = loaded[timeframe_label]
    one_minute = loaded["1 Minute"]
    five_minutes = loaded["5 Minutes"]
    return chart_data, one_minute, five_minutes


def get_signal_result(one_frame, five_frame, remaining):
    real_data = (
        one_frame is not None and five_frame is not None
        and not one_frame.empty and not five_frame.empty
    )
    return predictive_pre_open_signal(one_frame, five_frame, remaining, real_data)


# --------------------- Fast scanner fragment ------------------
@st.fragment(run_every=1)
def render_scanner():
    if not selected_assets:
        st.info("Select at least one asset in the sidebar.")
        return

    remaining, elapsed, boundary = candle_clock(60)
    st.subheader("🌐 Real-Time Scanner Matrix")
    st.caption(
        f"{len(selected_assets)} assets · Timeframe: {selected_timeframe} · "
        f"System Check Time: {datetime.now().strftime('%H:%M:%S')}"
    )

    cols_per_row = 3
    for start in range(0, len(selected_assets), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, asset in zip(cols, selected_assets[start:start + cols_per_row]):
            with col:
                with st.spinner(f"Checking {asset}…"):
                    chart_data, one_data, five_data = get_asset_bundle(asset, selected_timeframe)

                chart_frame, chart_source, chart_error = chart_data
                one_frame, one_source, one_error = one_data
                five_frame, five_source, five_error = five_data

                if chart_frame.empty:
                    st.error(f"{asset} data stream interrupted.")
                    if chart_error:
                        st.caption(str(chart_error))
                    continue

                real_data = (
                    not one_frame.empty and not five_frame.empty
                    and not str(one_source).startswith("Unavailable")
                    and not str(five_source).startswith("Unavailable")
                )
                signal, reason, structure, flash_trigger = predictive_pre_open_signal(
                    one_frame, five_frame, remaining, real_data
                )

                container_title = f"🚨 {asset} - PRE-OPEN CRITICAL" if flash_trigger else asset
                with st.container(border=True):
                    st.markdown(f"### {container_title}")
                    latest = chart_frame.iloc[-1]
                    prev_close = chart_frame.iloc[-2]["Close"] if len(chart_frame) > 1 else latest["Close"]
                    change = ((latest["Close"] - prev_close) / prev_close * 100) if prev_close else 0
                    st.metric("Latest close", fmt_price(latest["Close"]), f"{change:+.3f}%")

                    if "PRE-OPEN UP" in signal:
                        st.success(f"Signal: {signal}")
                        alarm_key = f"{asset}_UP_{boundary}"
                        if alarm_key not in st.session_state.played_alarms:
                            play_signal_alarm(asset, "PRE-OPEN UP")
                            st.session_state.played_alarms[alarm_key] = True
                    elif "PRE-OPEN DOWN" in signal:
                        st.error(f"Signal: {signal}")
                        alarm_key = f"{asset}_DOWN_{boundary}"
                        if alarm_key not in st.session_state.played_alarms:
                            play_signal_alarm(asset, "PRE-OPEN DOWN")
                            st.session_state.played_alarms[alarm_key] = True
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
                        st.metric(
                            "⚠️ NEXT CANDLE INITIALIZING", f"{remaining}s",
                            delta="- PRE-OPEN POSITION WINDOW", delta_color="inverse"
                        )
                    else:
                        st.metric("1m candle countdown", format_countdown(remaining))
                    st.caption(
                        f"RSI: {latest['RSI']:.1f} · EMA200: {fmt_price(latest['EMA_200'])}"
                    )


# ---------------- Detailed analysis fragment -----------------
@st.fragment(run_every=15)
def render_deep_dive():
    if not selected_assets:
        return

    st.divider()
    st.subheader("🔎 Deep-Dive Asset Structural Inspector")
    focus_asset = st.selectbox("Asset to inspect", selected_assets, key="focus_asset")
    chart_data, one_data, five_data = get_asset_bundle(focus_asset, selected_timeframe)
    frame, source, error = chart_data
    one_frame, one_source, one_error = one_data
    five_frame, five_source, five_error = five_data

    remaining, elapsed, boundary = candle_clock(60)
    real_data = (
        not one_frame.empty and not five_frame.empty
        and not str(one_source).startswith("Unavailable")
        and not str(five_source).startswith("Unavailable")
    )
    signal, reason, structure, flash_trigger = predictive_pre_open_signal(
        one_frame, five_frame, remaining, real_data
    )

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
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.72, 0.28], subplot_titles=("Candles, EMAs and levels", "RSI")
        )
        fig.add_trace(
            go.Candlestick(
                x=chart["Time"], open=chart["Open"], high=chart["High"],
                low=chart["Low"], close=chart["Close"], name="Candles"
            ), row=1, col=1
        )
        fig.add_trace(go.Scatter(x=chart["Time"], y=chart["EMA_50"], name="EMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart["Time"], y=chart["EMA_200"], name="EMA 200"), row=1, col=1)

        if structure and np.isfinite(structure.get("support", np.nan)):
            fig.add_hline(
                y=structure["resistance"], line_dash="dash",
                annotation_text="Resistance Ceiling", row=1, col=1
            )
            fig.add_hline(
                y=structure["support"], line_dash="dash",
                annotation_text="Support Floor", row=1, col=1
            )

        fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1)
        fig.update_layout(
            height=680, template="plotly_dark", xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=55, b=10), legend=dict(orientation="h")
        )
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        display_cols = [
            "Time", "Open", "High", "Low", "Close",
            "EMA_50", "EMA_200", "RSI", "MACD", "MACD_Signal"
        ]
        st.dataframe(
            frame[display_cols].tail(10).sort_values("Time", ascending=False),
            use_container_width=True, hide_index=True
        )
        st.caption(f"Chart source: {source}")
    else:
        st.warning("Market data is unavailable. The chart will appear when data is received.")


# Render fragments. Scanner UI refreshes every second; expensive chart refreshes every 15 seconds.
render_scanner()
render_deep_dive()
