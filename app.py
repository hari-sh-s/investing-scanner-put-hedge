import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import BacktestEngine
from portfolio_engine import PortfolioEngine
from scoring import ScoreParser
from nifty_universe import get_all_universe_names, get_universe, get_broad_market_universes
from report_generator import create_excel_with_charts, create_pdf_report, prepare_complete_log_data
from monte_carlo import MonteCarloSimulator, extract_trade_pnls, PortfolioMonteCarloSimulator, extract_monthly_returns
import kite_trader
import strategy_storage
import execution_storage
import datetime
import io
import time
import json
from pathlib import Path

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Investing Scanner",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Bootstrap: copy Streamlit Secrets → os.environ so all modules can use os.getenv()
# This is a no-op on local (where .env is used instead)
try:
    from config import bootstrap_env_from_secrets
    bootstrap_env_from_secrets()
except Exception:
    pass

# Show loading indicator immediately for wake-up
with st.spinner("🔄 App is waking up... Please wait..."):
    pass  # Spinner shows during import time

# Initialize session state for backtest logs
BACKTEST_LOG_FILE = Path("backtest_logs.json")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_backtest_logs_cached():
    """Load backtest logs from file with caching."""
    if BACKTEST_LOG_FILE.exists():
        try:
            with open(BACKTEST_LOG_FILE, 'r') as f:
                logs_data = json.load(f)
                return logs_data
        except Exception as e:
            print(f"Error loading logs: {e}")
            return []
    return []

def load_backtest_logs():
    """Load backtest logs from file."""
    return load_backtest_logs_cached()

def save_backtest_logs(logs):
    """Save backtest logs to file with complete data (no truncation)."""
    try:
        serializable_logs = []
        for log in logs:
            serializable_log = {
                'timestamp': log['timestamp'],
                'name': log['name'],
                'config': log['config'],
                'metrics': log['metrics'],
                # Store complete data without truncation
                'portfolio_values': log.get('portfolio_values', []),
                'trades': log.get('trades', []),
                'monthly_returns': log.get('monthly_returns', {})
            }
            serializable_logs.append(serializable_log)
        
        with open(BACKTEST_LOG_FILE, 'w') as f:
            json.dump(serializable_logs, f, indent=2, default=str)
        # Clear cache after saving
        load_backtest_logs_cached.clear()
    except Exception as e:
        print(f"Error saving logs: {e}")

@st.cache_data(ttl=86400)  # Cache universe names for 24 hours
def get_cached_universe_names():
    """Cache universe names to speed up app loading."""
    try:
        return sorted(get_all_universe_names())
    except Exception:
        return ["NIFTY 50", "NIFTY 100", "NIFTY 200"]  # Fallback

# Initialize session state with error handling
try:
    if 'backtest_logs' not in st.session_state:
        st.session_state.backtest_logs = load_backtest_logs()
    if 'backtest_engines' not in st.session_state:
        st.session_state.backtest_engines = {}
    if 'app_ready' not in st.session_state:
        st.session_state.app_ready = True
except Exception as e:
    st.error(f"Error initializing app: {e}. Please refresh the page.")
    st.session_state.backtest_logs = []
    st.session_state.backtest_engines = {}
    st.session_state.app_ready = True

# CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: white;
        border-radius: 6px;
        padding: 0 20px;
        border: 1px solid #ddd;
        color: #1a1a1a !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #28a745;
        color: white !important;
        font-weight: 600;
    }
    .progress-text {
        font-size: 16px;
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .stock-name {
        color: #00ff88;
        font-weight: 700;
        font-size: 18px;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    .time-remaining {
        font-size: 14px;
        color: #aaaaaa;
        margin-left: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Header - compact with last git commit timestamp
def get_last_update_time():
    """Get last git commit timestamp in IST format"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%cd', '--date=format:%d %b %H:%M'],
            capture_output=True, text=True, cwd='.'
        )
        if result.returncode == 0:
            return result.stdout.strip() + " IST"
    except:
        pass
    return "Unknown"

last_update = get_last_update_time()
st.markdown(f"### 📊 Investing Scanner <span style='font-size: 14px; color: #888;'>Updated: {last_update}</span>", unsafe_allow_html=True)

# Handle Kite OAuth callback at app start (before any tabs)
if kite_trader.is_kite_configured():
    # First, try to restore session from saved HuggingFace token
    if not st.session_state.get('kite_access_token'):
        kite_trader.restore_kite_session()
    
    # Handle OAuth callback if present
    query_params = st.query_params
    if 'request_token' in query_params:
        request_token = query_params['request_token']
        if kite_trader.handle_kite_callback(request_token):
            st.success(f"✅ Logged in to Zerodha as: {st.session_state.get('kite_user_name', 'User')}")
            st.query_params.clear()
        else:
            st.error("❌ Zerodha login failed. Please try again.")

# Main Tabs
main_tabs = st.tabs(["Backtest", "Backtest Logs", "Execute Trades", "Data Download", "🔐 Dhan Auth"])

# ==================== TAB 1: BACKTEST ====================
with main_tabs[0]:
    # Get loaded strategy config if present (shared between columns)
    loaded_config = st.session_state.get('loaded_strategy_config', {})
    
    col_config, col_scoring = st.columns([1, 1.2])
    
    with col_config:
        st.subheader("Configuration")
        
        # ===== BASIC SETTINGS (always visible) =====
        st.markdown("**Universe**")
        
        # Get all available universes
        all_universes = sorted(get_all_universe_names()) + ["Custom"]
        
        # Get default index for universe
        default_universe = loaded_config.get('universe', 'NIFTY 100')
        default_universe_idx = all_universes.index(default_universe) if default_universe in all_universes else 0
        
        selected_universe = st.selectbox(
            "Select", 
            all_universes,
            index=default_universe_idx,
            label_visibility="collapsed"
        )
        
        if selected_universe == "Custom":
            custom_input = st.text_input("Stocks (comma-separated)", "RELIANCE, TCS, INFY")
            universe = [s.strip() for s in custom_input.split(',')]
        else:
            universe = get_universe(selected_universe)
            st.caption(f"{len(universe)} stocks")
        
        # Capital, Stocks, Exit Rank in compact rows
        st.markdown("**Portfolio Settings**")
        cap_col1, cap_col2 = st.columns(2)
        with cap_col1:
            initial_capital = st.number_input("Capital (₹)", 10000, 100000000, 
                                              loaded_config.get('initial_capital', 100000), 10000)
        with cap_col2:
            num_stocks = st.number_input("Stocks", 1, 50, 
                                         loaded_config.get('num_stocks', 5))
        
        exit_col1, exit_col2 = st.columns(2)
        with exit_col1:
            default_exit = loaded_config.get('exit_rank', num_stocks * 2)
            exit_rank = st.number_input("Exit Rank", num_stocks, 200, max(default_exit, num_stocks), 
                                        help="Stocks exit if they fall below this rank")
        with exit_col2:
            reinvest_profits = st.checkbox("Reinvest Profits", 
                                           value=loaded_config.get('reinvest_profits', True))
        
        # Data Source selection
        data_source_options = ["Yahoo Finance", "Broker API (Dhan)"]
        default_ds = loaded_config.get('data_source', 'Yahoo Finance')
        default_ds_idx = data_source_options.index(default_ds) if default_ds in data_source_options else 0
        
        data_source = st.selectbox(
            "Data Source",
            data_source_options,
            index=default_ds_idx,
            help="Yahoo Finance: Free data with potential discrepancies. Broker API: Accurate data from Dhan (requires download first)"
        )
        
        use_historical_universe = st.checkbox("Historical Universe (Beta)", 
                                             value=loaded_config.get('use_historical_universe', False),
                                             help="Use point-in-time index constituents to avoid survivorship bias")
        
        # ===== TIME PERIOD & REBALANCING (in expander) =====
        with st.expander("📅 Time Period & Rebalancing", expanded=False):
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                # Parse loaded dates if available
                default_start = datetime.date(2020, 1, 1)
                if loaded_config.get('start_date'):
                    try:
                        default_start = datetime.datetime.strptime(loaded_config['start_date'], '%Y-%m-%d').date()
                    except:
                        pass
                start_date = st.date_input("Start Date", default_start)
            with date_col2:
                default_end = datetime.date.today()
                if loaded_config.get('end_date'):
                    try:
                        default_end = datetime.datetime.strptime(loaded_config['end_date'], '%Y-%m-%d').date()
                    except:
                        pass
                end_date = st.date_input("End Date", default_end)
            
            rebal_freq_options = ["Weekly", "Every 2 Weeks", "Monthly", "Bi-Monthly", "Quarterly", "Half-Yearly", "Annually"]
            default_rebal = loaded_config.get('rebalance_label', 'Monthly')
            default_rebal_idx = rebal_freq_options.index(default_rebal) if default_rebal in rebal_freq_options else 2
            rebalance_label = st.selectbox("Frequency", rebal_freq_options, index=default_rebal_idx)
            
            if rebalance_label == "Weekly":
                day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                default_day = loaded_config.get('rebal_day', 'Monday')
                default_day_idx = day_options.index(default_day) if default_day in day_options else 0
                rebal_day = st.selectbox("Rebalance Day", day_options, index=default_day_idx)
                rebalance_date = None
            elif rebalance_label == "Every 2 Weeks":
                day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                default_day = loaded_config.get('rebal_day', 'Monday')
                default_day_idx = day_options.index(default_day) if default_day in day_options else 0
                rebal_day = st.selectbox("Rebalance Day", day_options, index=default_day_idx)
                rebalance_date = None
            else:  # Monthly and above
                rebalance_date = st.number_input("Rebalance Date (1-30)", 1, 30, 
                                                loaded_config.get('rebalance_date', 1) or 1,
                                                help="Day of month to rebalance portfolio")
                rebal_day = None
            
            alt_day_options = ["Previous Day", "Next Day"]
            default_alt = loaded_config.get('alt_day_option', 'Next Day')
            default_alt_idx = alt_day_options.index(default_alt) if default_alt in alt_day_options else 1
            alt_day_option = st.selectbox("If Holiday", 
                                         alt_day_options,
                                         index=default_alt_idx,
                                         help="If rebalance day is holiday, use this option")
        
        # ===== POSITION SIZING (in expander) =====
        with st.expander("📊 Position Sizing", expanded=False):
            sizing_options = ["Equal Weight", "Inverse Volatility", "Inverse Downside Vol", "Inverse Max Drawdown", "Score-Weighted", "Risk Parity"]
            default_sizing = loaded_config.get('position_sizing_method', 'Equal Weight')
            default_sizing_idx = sizing_options.index(default_sizing) if default_sizing in sizing_options else 0
            
            position_sizing_method = st.selectbox(
                "Sizing Method",
                sizing_options,
                index=default_sizing_idx,
                help="Equal Weight: Divide equally | Inverse Volatility: Lower vol = higher weight | Inverse Downside Vol: Lower downside risk = higher weight | Inverse Max Drawdown: Lower drawdown = higher weight"
            )
            
            use_max_position_cap = st.checkbox(
                "Apply Max Position Cap",
                value=loaded_config.get('use_max_position_cap', False),
                help="Limit maximum allocation to any single stock"
            )
            
            max_position_pct = loaded_config.get('max_position_pct', 15)
            if use_max_position_cap:
                max_position_pct = st.number_input(
                    "Max Position %",
                    5, 50, max_position_pct,
                    help="Maximum % of portfolio any single stock can hold"
                )
        
        # Create position sizing config
        position_sizing_config = {
            'method': position_sizing_method.lower().replace(' ', '_').replace('-', '_'),
            'use_cap': use_max_position_cap,
            'max_pct': max_position_pct
        }
        
        # ===== REGIME FILTER (in expander) =====
        with st.expander("🛡️ Regime Filter", expanded=False):
            # Get saved regime config if present
            saved_regime = loaded_config.get('regime_config', {}) or {}
            
            use_regime_filter = st.checkbox("Enable Regime Filter", 
                                           value=loaded_config.get('use_regime_filter', False))
            
            regime_config = None
            if use_regime_filter:
                regime_type_options = [
                    "SMA_1D", "SMA_1W", "SMA_1M",
                    "EMA_1D", "EMA_1W", "EMA_1M", 
                    "MACD", 
                    "SUPERTREND_1D", "SUPERTREND_1W", "SUPERTREND_1M", 
                    "EQUITY", "EQUITY_MA", "DONCHIAN", "SWING_ATR", "BREADTH"
                ]
                saved_regime_type = saved_regime.get('type', 'EMA_1D')
                # Handle legacy migrations
                if saved_regime_type == 'SUPERTREND':
                    saved_regime_type = 'SUPERTREND_1D'
                elif saved_regime_type == 'EMA':
                    saved_regime_type = 'EMA_1D'
                regime_type_idx = regime_type_options.index(saved_regime_type) if saved_regime_type in regime_type_options else 0
                
                regime_type = st.selectbox("Regime Filter Type", 
                                          regime_type_options,
                                          index=regime_type_idx,
                                          help="SMA/EMA: 1D=Daily, 1W=Weekly, 1M=Monthly | SuperTrend: trend-following | DONCHIAN: Turtle Trading")
                
                # Initialize defaults
                recovery_dd = None
                ma_period = None
                exit_period = None
                recovery_period = None
                swing_period = None
                atr_buffer = None
                breadth_threshold = None
                breadth_index = None
                
                if regime_type in ["SMA_1D", "SMA_1W", "SMA_1M"]:
                    timeframe_labels = {"SMA_1D": "Daily", "SMA_1W": "Weekly", "SMA_1M": "Monthly"}
                    st.caption(f"📈 SMA on {timeframe_labels[regime_type]} timeframe")
                    sma_options = [20, 50, 100, 150, 200]
                    saved_sma = saved_regime.get('value', 50) if saved_regime.get('type', '').startswith('SMA') else 50
                    if saved_sma not in sma_options:
                        saved_sma = 50
                    sma_idx = sma_options.index(saved_sma)
                    sma_period = st.selectbox("SMA Period", sma_options, index=sma_idx,
                                             help="Trigger when price falls below SMA")
                    regime_value = sma_period
                elif regime_type in ["EMA_1D", "EMA_1W", "EMA_1M"]:
                    timeframe_labels = {"EMA_1D": "Daily", "EMA_1W": "Weekly", "EMA_1M": "Monthly"}
                    st.caption(f"📈 EMA on {timeframe_labels[regime_type]} timeframe")
                    ema_options = [34, 68, 100, 150, 200]
                    saved_ema = saved_regime.get('value', 68) if saved_regime.get('type', '').startswith('EMA') else 68
                    if saved_ema not in ema_options:
                        saved_ema = 68
                    ema_idx = ema_options.index(saved_ema)
                    ema_period = st.selectbox("EMA Period", ema_options, index=ema_idx,
                                             help="Trigger when price falls below EMA")
                    regime_value = ema_period
                elif regime_type == "MACD":
                    macd_options = ["35-70-12", "50-100-15", "75-150-12"]
                    saved_macd = saved_regime.get('value', '35-70-12') if saved_regime.get('type') == 'MACD' else '35-70-12'
                    macd_idx = macd_options.index(saved_macd) if saved_macd in macd_options else 0
                    macd_preset = st.selectbox("MACD Settings", macd_options, index=macd_idx)
                    regime_value = macd_preset
                elif regime_type in ["SUPERTREND_1D", "SUPERTREND_1W", "SUPERTREND_1M"]:
                    timeframe_labels = {"SUPERTREND_1D": "Daily", "SUPERTREND_1W": "Weekly", "SUPERTREND_1M": "Monthly"}
                    st.caption(f"📊 SuperTrend on {timeframe_labels[regime_type]} timeframe")
                    # Period-Multiplier: lower multiplier = more sensitive, higher = smoother
                    st_options = ["7-2", "7-3", "10-2", "10-3"]
                    saved_st = saved_regime.get('value', '7-3') if saved_regime.get('type', '').startswith('SUPERTREND') else '7-3'
                    # Handle migration from old invalid values
                    if saved_st not in st_options:
                        saved_st = '7-3'
                    st_idx = st_options.index(saved_st)
                    st_preset = st.selectbox("SuperTrend (Period-Multiplier)", st_options, index=st_idx,
                                            help="Period=ATR lookback, Multiplier=band width. Lower multiplier = more signals")
                    regime_value = st_preset
                elif regime_type == "EQUITY":
                    eq_col1, eq_col2 = st.columns(2)
                    with eq_col1:
                        saved_sl = saved_regime.get('value', 10) if saved_regime.get('type') == 'EQUITY' else 10
                        realized_sl = st.number_input("DD SL % (Trigger)", 1, 50, saved_sl,
                                                      help="Sell when drawdown exceeds this %")
                    with eq_col2:
                        saved_recovery = saved_regime.get('recovery_dd', 5) or 5
                        recovery_dd = st.number_input("Recovery DD %", 0, 49, saved_recovery,
                                                      help="Re-enter when drawdown below this %")
                    regime_value = realized_sl
                elif regime_type == "EQUITY_MA":
                    ma_options = [20, 30, 50, 100, 200]
                    saved_ma = saved_regime.get('ma_period', 50) if saved_regime.get('type') == 'EQUITY_MA' else 50
                    ma_idx = ma_options.index(saved_ma) if saved_ma in ma_options else 2
                    ma_period = st.selectbox("Equity Curve MA Period", 
                                            ma_options,
                                            index=ma_idx,
                                            help="Reduce exposure when equity falls below this MA")
                    regime_value = ma_period
                elif regime_type == "DONCHIAN":
                    st.caption("📈 Turtle Trading: Exit on N-day low, Recovery on M-day high")
                    don_col1, don_col2 = st.columns(2)
                    with don_col1:
                        exit_options = [40, 50, 55, 60]
                        saved_exit = saved_regime.get('exit_period', 55) if saved_regime.get('type') == 'DONCHIAN' else 55
                        exit_idx = exit_options.index(saved_exit) if saved_exit in exit_options else 2
                        exit_period = st.selectbox("Exit Period (days)", exit_options, index=exit_idx,
                                                   help="Trigger when price breaks N-day low")
                    with don_col2:
                        recov_options = [10, 15, 20, 25]
                        saved_recov = saved_regime.get('recovery_period', 20) if saved_regime.get('type') == 'DONCHIAN' else 20
                        recov_idx = recov_options.index(saved_recov) if saved_recov in recov_options else 2
                        recovery_period = st.selectbox("Recovery Period (days)", recov_options, index=recov_idx,
                                                       help="Recover when price breaks M-day high")
                    regime_value = exit_period
                elif regime_type == "SWING_ATR":
                    st.caption("📊 Swing pivot with ATR buffer to filter noise")
                    swing_col1, swing_col2 = st.columns(2)
                    with swing_col1:
                        saved_swing = saved_regime.get('swing_period', 20) if saved_regime.get('type') == 'SWING_ATR' else 20
                        swing_period = st.number_input("Swing Lookback", 10, 50, saved_swing,
                                                       help="Period for swing high/low detection")
                    with swing_col2:
                        saved_buffer = saved_regime.get('atr_buffer', 1.5) if saved_regime.get('type') == 'SWING_ATR' else 1.5
                        atr_buffer = st.number_input("ATR Buffer", 0.5, 3.0, saved_buffer, step=0.5,
                                                     help="ATR multiplier for exit/recovery buffer")
                    regime_value = swing_period
                elif regime_type == "BREADTH":
                    st.caption("📉 Market health: % of stocks above 200 SMA")
                    saved_threshold = saved_regime.get('breadth_threshold', 60) if saved_regime.get('type') == 'BREADTH' else 60
                    breadth_threshold = st.number_input("Breadth Threshold %", 40, 80, saved_threshold,
                                                        help="Trigger when fewer than X% of stocks above 200 SMA")
                    # Only show indexes with constituent data
                    try:
                        from historical_constituents.store import get_available_indices
                        available_indices = get_available_indices()
                        breadth_index_options = [idx.upper().replace('NIFTY', 'NIFTY ') for idx in available_indices]
                    except:
                        breadth_index_options = ["NIFTY 50", "NIFTY 100"]
                    
                    saved_breadth_idx = saved_regime.get('breadth_index', 'NIFTY50') if saved_regime.get('type') == 'BREADTH' else 'NIFTY50'
                    breadth_idx = 0
                    for i, opt in enumerate(breadth_index_options):
                        if saved_breadth_idx.upper().replace(' ', '') in opt.upper().replace(' ', ''):
                            breadth_idx = i
                            break
                    breadth_index = st.selectbox("Breadth Index", breadth_index_options, index=breadth_idx,
                                                 help="Index whose constituents to use for breadth calculation")
                    regime_value = breadth_threshold
                
                action_options = ["Go Cash", "Half Portfolio", "Nifty Put Hedge"]
                saved_action = saved_regime.get('action', 'Go Cash')
                action_idx = action_options.index(saved_action) if saved_action in action_options else 0
                regime_action = st.selectbox("Regime Filter Action",
                                            action_options,
                                            index=action_idx,
                                            help="Go Cash: Move to cash | Half Portfolio: Keep 50% stocks | Nifty Put Hedge: Buy NIFTY ATM Puts via Dhan API")

                # ── Nifty Put Hedge sub-settings ────────────────────────────
                put_hedge_config = None
                if regime_action == "Nifty Put Hedge":
                    saved_phc = saved_regime.get('put_hedge_config', {})
                    st.markdown("""
                    <div style='background:rgba(255,165,0,0.1);border-left:3px solid orange;
                                padding:8px 12px;border-radius:4px;margin:6px 0;'>
                    🛡️ <b>Nifty Put Hedge</b>: When regime triggers, buys NIFTY ATM <b>Weekly</b> Puts
                    using delta-neutral sizing. Requires Dhan Data API subscription
                    (falls back to VIX/Black-Scholes when unavailable).
                    Strike: ATM &nbsp;|&nbsp; Expiry: Weekly rolling
                    </div>
                    """, unsafe_allow_html=True)

                    ph_col1, ph_col2 = st.columns(2)
                    with ph_col1:
                        hedge_ratio = st.slider(
                            "Hedge Ratio", 0.25, 1.5,
                            float(saved_phc.get('hedge_ratio', 1.0)), step=0.05,
                            key="put_hedge_ratio",
                            help="1.0 = full delta-neutral | 0.5 = half hedge | 1.5 = over-hedge"
                        )
                    with ph_col2:
                        keep_stocks = st.checkbox(
                            "Keep Stocks (Uncheck for Cash+Uncorrelated)",
                            value=saved_phc.get('keep_stocks', True),
                            key="put_hedge_keep_stocks",
                            help="Checked: hold stocks AND buy puts. Unchecked: Go to Cash + Uncorrelated assets AND buy puts."
                        )
                        
                    ph_col3, ph_col4 = st.columns(2)
                    with ph_col3:
                        expiry_type_options = ["WEEKLY", "MONTHLY"]
                        saved_expiry_type = saved_phc.get('expiry_type', 'WEEKLY')
                        expiry_type_idx = expiry_type_options.index(saved_expiry_type) if saved_expiry_type in expiry_type_options else 0
                        expiry_type = st.selectbox(
                            "Option Expiry",
                            expiry_type_options,
                            index=expiry_type_idx,
                            key="put_hedge_expiry_type"
                        )
                    with ph_col4:
                        roll_expiries = st.checkbox(
                            "Roll Expiries (Continuous)",
                            value=saved_phc.get('roll_expiries', True),
                            key="put_hedge_roll_expiries",
                            help="Checked: continually re-buy when options expire. Unchecked: one-time hedge per rebalance."
                        )

                    put_hedge_config = {
                        'hedge_ratio':    hedge_ratio,
                        'keep_stocks':    keep_stocks,
                        'expiry_type':    expiry_type,
                        'roll_expiries':  roll_expiries,
                        'portfolio_beta': 1.0,  # Assumed beta vs NIFTY
                    }

                # Show index selector for regime types that use index data
                if regime_type not in ["EQUITY", "EQUITY_MA", "BREADTH"]:
                    index_options = ["Stock"] + sorted(get_all_universe_names())
                    saved_index = saved_regime.get('index', 'NIFTY 50')
                    index_idx = index_options.index(saved_index) if saved_index in index_options else 0
                    regime_index = st.selectbox("Regime Filter Index", index_options, index=index_idx)
                else:
                    regime_index = None

                # Exit check frequency option (not applicable for EQUITY which always checks daily)
                if regime_type not in ["EQUITY"]:
                    exit_check_options = ["Intraday (Daily Check)", "Rebalance Day Only"]
                    saved_exit_check = saved_regime.get('exit_check', 'Intraday (Daily Check)')
                    exit_check_idx = exit_check_options.index(saved_exit_check) if saved_exit_check in exit_check_options else 0
                    exit_check = st.selectbox("Exit Check Frequency",
                                             exit_check_options,
                                             index=exit_check_idx,
                                             help="Intraday: Exit immediately when regime triggers | Rebalance Day: Only check on rebalance days")
                else:
                    exit_check = "Intraday (Daily Check)"  # EQUITY always uses daily check

                regime_config = {
                    'type': regime_type,
                    'value': regime_value,
                    'action': regime_action,
                    'index': regime_index,
                    'recovery_dd': recovery_dd,
                    'ma_period': ma_period if regime_type == "EQUITY_MA" else None,
                    'exit_period': exit_period if regime_type == "DONCHIAN" else None,
                    'recovery_period': recovery_period if regime_type == "DONCHIAN" else None,
                    'swing_period': swing_period if regime_type == "SWING_ATR" else None,
                    'atr_buffer': atr_buffer if regime_type == "SWING_ATR" else None,
                    'breadth_threshold': breadth_threshold if regime_type == "BREADTH" else None,
                    'breadth_index': breadth_index.replace(' ', '') if regime_type == "BREADTH" and breadth_index else None,
                    'exit_check': exit_check,
                    'put_hedge_config': put_hedge_config,  # None unless action=="Nifty Put Hedge"
                }
                
                # Uncorrelated Asset
                st.markdown("---")
                # Get saved uncorrelated config
                saved_uncorrelated = loaded_config.get('uncorrelated_config', {}) or {}
                use_uncorrelated = st.checkbox("Invest in Uncorrelated Asset", 
                                              value=loaded_config.get('use_uncorrelated', False),
                                              help="Allocate to uncorrelated assets when regime triggers")
                
                uncorrelated_config = None
                if use_uncorrelated:
                    st.caption("Add assets to allocate when regime triggers (allocations should sum to 100%)")
                    
                    # Initialize session state for assets list from loaded config or default
                    if 'uncorrelated_assets' not in st.session_state:
                        saved_assets = saved_uncorrelated.get('assets', [])
                        if saved_assets:
                            st.session_state.uncorrelated_assets = saved_assets.copy()
                        else:
                            st.session_state.uncorrelated_assets = [{'ticker': 'GOLDBEES', 'pct': 100}]
                    
                    available_assets = ["GOLDBEES", "JUNIORBEES", "NIFTYBEES", "SILVERBEES", "BANKBEES", "LIQUIDBEES"]
                    
                    # Display current assets
                    assets_to_remove = []
                    for i, asset in enumerate(st.session_state.uncorrelated_assets):
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            current_idx = available_assets.index(asset['ticker']) if asset['ticker'] in available_assets else 0
                            new_ticker = st.selectbox(f"Asset {i+1}", available_assets, 
                                                      index=current_idx,
                                                      key=f"unc_asset_{i}")
                            st.session_state.uncorrelated_assets[i]['ticker'] = new_ticker
                        with col2:
                            new_pct = st.number_input(f"Alloc %", 1, 100, asset['pct'], key=f"unc_pct_{i}")
                            st.session_state.uncorrelated_assets[i]['pct'] = new_pct
                        with col3:
                            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                            if len(st.session_state.uncorrelated_assets) > 1:
                                if st.button("🗑️", key=f"remove_unc_{i}"):
                                    assets_to_remove.append(i)
                    
                    # Remove marked assets
                    for i in sorted(assets_to_remove, reverse=True):
                        st.session_state.uncorrelated_assets.pop(i)
                        st.rerun()
                    
                    # Add asset button
                    if len(st.session_state.uncorrelated_assets) < 4:
                        if st.button("➕ Add Asset", key="add_unc_asset"):
                            # Default new asset with equal split
                            num_assets = len(st.session_state.uncorrelated_assets) + 1
                            default_pct = 100 // num_assets
                            st.session_state.uncorrelated_assets.append({'ticker': 'SILVERBEES', 'pct': default_pct})
                            st.rerun()
                    
                    # Show total allocation
                    total_pct = sum(a['pct'] for a in st.session_state.uncorrelated_assets)
                    if total_pct != 100:
                        st.warning(f"⚠️ Total allocation: {total_pct}% (should sum to 100%)")
                    else:
                        st.success(f"✅ Total allocation: {total_pct}%")
                    
                    uncorrelated_config = {
                        'assets': [{'ticker': a['ticker'], 'pct': a['pct']} for a in st.session_state.uncorrelated_assets],
                        'total_pct': total_pct
                    }
            else:
                uncorrelated_config = None
        
        # Risk Management section (within col_config)
        with st.expander("⚠️ Risk Management", expanded=False):
            st.caption("Control drawdowns at portfolio and trade level")
            
            # Get saved risk config
            saved_risk = loaded_config.get('risk_config', {}) or {}
            saved_portfolio_risk = saved_risk.get('portfolio', {}) or {}
            saved_trade_risk = saved_risk.get('trade', {}) or {}
            
            # Portfolio Level Risk
            st.markdown("**Portfolio Level**")
            enable_portfolio_risk = st.checkbox("Enable Portfolio Risk", 
                                                 value=saved_portfolio_risk.get('enabled', False),
                                                 help="Exit when total portfolio loss exceeds threshold")
            
            portfolio_risk_config = {'enabled': False}
            if enable_portfolio_risk:
                port_cols = st.columns(2)
                with port_cols[0]:
                    port_type_options = ["Percentage (%)", "Absolute (₹)"]
                    saved_port_type = saved_portfolio_risk.get('type', 'percent')
                    port_type_idx = 0 if saved_port_type == 'percent' else 1
                    port_risk_type = st.selectbox("Risk Type", port_type_options, index=port_type_idx, key="port_risk_type")
                with port_cols[1]:
                    if port_risk_type == "Percentage (%)":
                        saved_port_val = saved_portfolio_risk.get('value', 2.0) if saved_portfolio_risk.get('type') == 'percent' else 2.0
                        port_risk_value = st.number_input("Risk %", 0.5, 20.0, float(saved_port_val), step=0.5, key="port_risk_val")
                    else:
                        saved_port_val = saved_portfolio_risk.get('value', 5000) if saved_portfolio_risk.get('type') == 'absolute' else 5000
                        port_risk_value = st.number_input("Risk ₹", 1000, 100000, int(saved_port_val), step=1000, key="port_risk_val")
                
                port_action_options = ["Exit all positions", "Exit only loss-making positions"]
                saved_port_action = saved_portfolio_risk.get('action', 'exit_losers')
                port_action_idx = 0 if saved_port_action == 'exit_all' else 1
                port_exit_action = st.selectbox("Exit Action", port_action_options, index=port_action_idx, key="port_exit_action",
                                                help="Exit all: nuclear option | Exit losers: keep profitable positions")
                
                portfolio_risk_config = {
                    'enabled': True,
                    'type': 'percent' if port_risk_type == "Percentage (%)" else 'absolute',
                    'value': port_risk_value,
                    'action': 'exit_all' if port_exit_action == "Exit all positions" else 'exit_losers'
                }
            
            st.markdown("---")
            
            # Trade Level Risk
            st.markdown("**Trade Level**")
            enable_trade_risk = st.checkbox("Enable Trade Risk", 
                                            value=saved_trade_risk.get('enabled', False),
                                            help="Exit when individual trade loss exceeds threshold")
            
            trade_risk_config = {'enabled': False}
            if enable_trade_risk:
                trade_cols = st.columns(2)
                with trade_cols[0]:
                    trade_type_options = ["Percentage (%)", "Absolute (₹)"]
                    saved_trade_type = saved_trade_risk.get('type', 'percent')
                    trade_type_idx = 0 if saved_trade_type == 'percent' else 1
                    trade_risk_type = st.selectbox("Risk Type", trade_type_options, index=trade_type_idx, key="trade_risk_type")
                with trade_cols[1]:
                    if trade_risk_type == "Percentage (%)":
                        saved_trade_val = saved_trade_risk.get('value', 2.0) if saved_trade_risk.get('type') == 'percent' else 2.0
                        trade_risk_value = st.number_input("Risk %", 0.5, 20.0, float(saved_trade_val), step=0.5, key="trade_risk_val")
                    else:
                        saved_trade_val = saved_trade_risk.get('value', 5000) if saved_trade_risk.get('type') == 'absolute' else 5000
                        trade_risk_value = st.number_input("Risk ₹", 1000, 100000, int(saved_trade_val), step=1000, key="trade_risk_val")
                
                trade_action_options = ["Exit only breached trade", "Exit all loss-making", "Exit all positions"]
                saved_trade_action = saved_trade_risk.get('action', 'exit_breached')
                trade_action_map = {'exit_breached': 0, 'exit_losers': 1, 'exit_all': 2}
                trade_action_idx = trade_action_map.get(saved_trade_action, 0)
                trade_exit_action = st.selectbox("Exit Action", trade_action_options, index=trade_action_idx, key="trade_exit_action",
                                                 help="Breached only: exit just the stock | All losers: exit all losing positions | All: nuclear")
                
                trade_action_reverse = {0: 'exit_breached', 1: 'exit_losers', 2: 'exit_all'}
                trade_risk_config = {
                    'enabled': True,
                    'type': 'percent' if trade_risk_type == "Percentage (%)" else 'absolute',
                    'value': trade_risk_value,
                    'action': trade_action_reverse[trade_action_options.index(trade_exit_action)]
                }
            
            risk_config = {
                'portfolio': portfolio_risk_config,
                'trade': trade_risk_config
            }
        
        # Default risk_config when expander is collapsed but no config saved
        if 'risk_config' not in dir():
            risk_config = {'portfolio': {'enabled': False}, 'trade': {'enabled': False}}
    
    with col_scoring:
        st.subheader("Scoring Console")
        
        parser = ScoreParser()
        examples = parser.get_example_formulas()
        
        # Get default template from loaded config
        template_options = ["Custom"] + list(examples.keys())
        default_template = loaded_config.get('template', 'Custom')
        default_template_idx = template_options.index(default_template) if default_template in template_options else 0
        
        template = st.selectbox("Template", template_options, index=default_template_idx)
        
        # Get formula - either from loaded config or from template
        if loaded_config.get('formula') and default_template_idx == template_options.index(template):
            default_formula = loaded_config.get('formula', "6 Month Performance")
        else:
            default_formula = examples.get(template, "6 Month Performance")
        
        formula = st.text_area("Scoring Formula", default_formula, height=100)
        
        valid, msg = parser.validate_formula(formula)
        if valid:
            st.success("✅ " + msg)
        else:
            st.error("❌ " + msg)
        
        # Compact metrics reference in collapsible expander
        with st.expander("📖 Available Metrics", expanded=False):
            st.caption("💡 **Tip:** Use any month (1-24) or week (1-52), e.g. `15 Month Performance`, `2 Week Volatility` or `18 Month Sharpe`")
            
            metric_groups = parser.metric_groups if hasattr(parser, 'metric_groups') else {}
            
            # Display metrics in a compact multi-column format
            metrics_text = []
            
            perf = metric_groups.get('Performance', ["1 Month Performance", "3 Month Performance", "6 Month Performance", "12 Month Performance"])
            metrics_text.append("**Performance:** " + " • ".join(perf))
            
            vol = metric_groups.get('Volatility', ["1 Month Volatility", "3 Month Volatility", "6 Month Volatility"])
            metrics_text.append("**Volatility:** " + " • ".join(vol))
            
            dsv = metric_groups.get('Downside Volatility', [])
            if dsv:
                metrics_text.append("**Downside Vol:** " + " • ".join(dsv))
            
            mdd = metric_groups.get('Max Drawdown', [])
            if mdd:
                metrics_text.append("**Max Drawdown:** " + " • ".join(mdd))
            
            sharpe = metric_groups.get('Sharpe Ratio', ["6 Month Sharpe"])
            sortino = metric_groups.get('Sortino Ratio', ["6 Month Sortino"])
            calmar = metric_groups.get('Calmar Ratio', ["6 Month Calmar"])
            risk_adj = sharpe + sortino + calmar
            if risk_adj:
                metrics_text.append("**Risk-Adjusted:** " + " • ".join(risk_adj))
            
            for text in metrics_text:
                st.markdown(text, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== STRATEGY SAVE/LOAD =====
        if strategy_storage.is_strategy_storage_configured():
            st.markdown("**💾 Strategy Templates**")
            
            # Get saved strategies
            saved_strategies = strategy_storage.list_strategies()
            strategy_options = ["-- Select Saved Strategy --"] + saved_strategies
            
            # Strategy dropdown
            selected_strategy = st.selectbox(
                "Load Strategy",
                strategy_options,
                key="strategy_selector",
                label_visibility="collapsed"
            )
            
            # Handle strategy loading
            if selected_strategy != "-- Select Saved Strategy --" and selected_strategy:
                if st.session_state.get('last_loaded_strategy') != selected_strategy:
                    loaded_config = strategy_storage.load_strategy(selected_strategy)
                    if loaded_config:
                        st.session_state['loaded_strategy_config'] = loaded_config
                        st.session_state['last_loaded_strategy'] = selected_strategy
                        st.success(f"✅ Loaded: {selected_strategy}")
                        st.rerun()
            
            # Save and Delete buttons in columns
            save_col, delete_col = st.columns(2)
            
            with save_col:
                with st.popover("💾 Save Strategy", use_container_width=True):
                    strategy_name = st.text_input("Strategy Name", key="new_strategy_name")
                    
                    if st.button("Save", key="save_strategy_btn", type="primary"):
                        if strategy_name:
                            # Collect current configuration
                            current_config = {
                                'universe': selected_universe,
                                'initial_capital': initial_capital,
                                'num_stocks': num_stocks,
                                'exit_rank': exit_rank,
                                'reinvest_profits': reinvest_profits,
                                'data_source': data_source,
                                'use_historical_universe': use_historical_universe,
                                'start_date': str(start_date),
                                'end_date': str(end_date),
                                'rebalance_label': rebalance_label,
                                'rebalance_date': rebalance_date,
                                'rebal_day': rebal_day,
                                'alt_day_option': alt_day_option,
                                'position_sizing_method': position_sizing_method,
                                'use_max_position_cap': use_max_position_cap,
                                'max_position_pct': max_position_pct,
                                'use_regime_filter': use_regime_filter,
                                'regime_config': regime_config,
                                'use_uncorrelated': use_uncorrelated,
                                'uncorrelated_config': uncorrelated_config,
                                'risk_config': risk_config,
                                'formula': formula,
                                'template': template
                            }
                            
                            if strategy_storage.save_strategy(strategy_name, current_config):
                                st.success(f"✅ Saved: {strategy_name}")
                                st.rerun()
                        else:
                            st.warning("Enter a strategy name")
            
            with delete_col:
                if saved_strategies:
                    with st.popover("🗑️ Delete", use_container_width=True):
                        delete_strategy = st.selectbox(
                            "Select to delete",
                            saved_strategies,
                            key="delete_strategy_select"
                        )
                        if st.button("Delete", key="delete_strategy_btn", type="secondary"):
                            if strategy_storage.delete_strategy(delete_strategy):
                                st.success(f"✅ Deleted: {delete_strategy}")
                                st.session_state['last_loaded_strategy'] = None
                                st.rerun()
            
            st.markdown("---")
        
        run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Results Section
    if run_btn:
        if not valid:
            st.error("Fix formula first")
        else:
            # Initialize tracking variables
            start_time = time.time()
            processed_count = [0]  # Use list to avoid nonlocal scope issues
            total_count = len(universe)
            
            def progress_callback(current, total, ticker):
                processed_count[0] = current

                # Calculate time stats
                elapsed = time.time() - start_time
                elapsed_mins = int(elapsed // 60)
                elapsed_secs = int(elapsed % 60)

                if processed_count[0] > 0:
                    avg_time_per_stock = elapsed / processed_count[0]
                    remaining_stocks = total - processed_count[0]
                    time_remaining_sec = avg_time_per_stock * remaining_stocks

                    remaining_mins = int(time_remaining_sec // 60)
                    remaining_secs = int(time_remaining_sec % 60)
                    time_str = f"{remaining_mins:02d}:{remaining_secs:02d}"
                else:
                    time_str = "Calculating..."

                # Update progress bar
                progress = min(processed_count[0] / total, 1.0)
                prog_bar.progress(progress)

                # Update status text with all details
                pct = (processed_count[0] / total * 100) if total > 0 else 0
                status_container.markdown(f"""
                <div style="padding: 10px; background: rgba(0,0,0,0.1); border-radius: 5px;">
                    <div style="font-size: 16px; font-weight: bold;">📊 {ticker}</div>
                    <div style="margin-top: 5px;">
                        Progress: {processed_count[0]}/{total} ({pct:.1f}%)
                    </div>
                    <div style="margin-top: 5px;">
                        ⏱️ Remaining: {time_str} | ⏰ Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("Initializing backtest..."):
                prog_bar = st.progress(0)
                status_container = st.empty()
                
                # Map UI data source to engine parameter
                data_source_map = {"Yahoo Finance": "yahoo", "Broker API (Dhan)": "dhan"}
                engine_data_source = data_source_map.get(data_source, "yahoo")
                
                engine = PortfolioEngine(universe, start_date, end_date, initial_capital, data_source=engine_data_source)
                if engine.fetch_data(progress_callback=progress_callback):
                    prog_bar.empty()
                    status_container.empty()
                    
                    with st.spinner("Running strategy simulation..."):
                        # Build rebalance config
                        rebal_config = {
                            'frequency': rebalance_label,
                            'date': rebalance_date,
                            'day': rebal_day,
                            'alt_day': alt_day_option
                        }
                        
                        # Build historical universe config
                        historical_universe_config = {
                            'enabled': use_historical_universe,
                            'universe_name': selected_universe
                        } if use_historical_universe else None
                        
                        engine.run_rebalance_strategy(
                            formula, 
                            num_stocks,
                            exit_rank,
                            rebal_config,
                            regime_config,
                            uncorrelated_config,
                            reinvest_profits,
                            position_sizing_config,
                            historical_universe_config=historical_universe_config,
                            risk_config=risk_config
                        )
                        metrics = engine.get_metrics()
                        
                        # Store in session_state so results persist across reruns (for benchmark comparison)
                        st.session_state['backtest_engine'] = engine
                        st.session_state['backtest_metrics'] = metrics
                        st.session_state['backtest_start_date'] = start_date
                        st.session_state['backtest_end_date'] = end_date
                    
                    if metrics:
                        # Prepare complete log data (no truncation)
                        complete_log_data = prepare_complete_log_data(
                            {
                                'name': f"Backtest_{datetime.datetime.now().strftime('%m%d_%H%M')}",
                                'initial_capital': initial_capital,
                                'universe_name': selected_universe,
                                'num_stocks': num_stocks,
                                'exit_rank': exit_rank,
                                'rebalance_freq': rebalance_label,
                                'start_date': start_date.strftime('%Y-%m-%d'),
                                'end_date': end_date.strftime('%Y-%m-%d'),
                                'regime_config': regime_config if regime_config else {},
                                'uncorrelated_config': uncorrelated_config if uncorrelated_config else {},
                                'formula': formula
                            },
                            metrics,
                            engine
                        )
                        
                        # Save to logs with complete data
                        backtest_log = {
                            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'name': f"Backtest_{datetime.datetime.now().strftime('%m%d_%H%M')}",
                            'config': complete_log_data['config'],
                            'metrics': metrics,
                            'portfolio_values': complete_log_data['portfolio_values'],
                            'trades': complete_log_data['trades'],
                            'monthly_returns': complete_log_data['monthly_returns']
                        }
                        st.session_state.backtest_logs.append(backtest_log)
                        save_backtest_logs(st.session_state.backtest_logs)  # Save to file
                        
                        # Auto-calculate Monte Carlo Results (10,000 simulations)
                        mc_results = None
                        with st.spinner("Calculating Monte Carlo Analysis & Generating Reports..."):
                            try:
                                # Extract monthly returns from portfolio for robust MC
                                p_values = engine.portfolio_df['Portfolio Value']
                                # Use 'M' for monthly (compatible with older pandas, 'ME' is newer)
                                try:
                                    m_returns = p_values.resample('ME').last().pct_change().dropna()
                                except ValueError:
                                    # Fallback for older pandas versions
                                    m_returns = p_values.resample('M').last().pct_change().dropna()
                                
                                if len(m_returns) >= 6:
                                    # Run Portfolio MC
                                    from monte_carlo import PortfolioMonteCarloSimulator
                                    
                                    # Constructor signature: (monthly_returns, initial_capital, n_simulations)
                                    mc_sim = PortfolioMonteCarloSimulator(m_returns.tolist(), initial_capital, n_simulations=10000)
                                    
                                    # Run simulations
                                    res_reshuffle = mc_sim.run_simulations(method='reshuffle')
                                    res_resample = mc_sim.run_simulations(method='resample')
                                    
                                    mc_results = {
                                        'perm_dd_95': res_reshuffle.get('mc_max_dd_95', 0),
                                        'perm_dd_worst': res_reshuffle.get('mc_max_dd_worst', 0),
                                        'perm_ruin': res_reshuffle.get('ruin_probability', 0),
                                        'perm_cagr_med': res_reshuffle.get('mc_cagr_median', 0),
                                        'boot_dd_95': res_resample.get('mc_max_dd_95', 0),
                                        'boot_dd_worst': res_resample.get('mc_max_dd_worst', 0),
                                        'boot_ruin': res_resample.get('ruin_probability', 0),
                                        'boot_cagr_med': res_resample.get('mc_cagr_median', 0),
                                        'n_simulations': 10000,
                                        'initial_capital': initial_capital,
                                        'monthly_returns': m_returns.tolist()
                                    }
                                else:
                                    st.warning(f"Need ≥6 months for MC. Have {len(m_returns)} months.")
                            except Exception as e:
                                st.error(f"MC Calculation Error: {e}")
                                mc_results = None

                        # Calculate Equity Analysis (Regime)
                        equity_analysis = None
                        if hasattr(engine, 'get_equity_regime_analysis'):
                            equity_analysis = engine.get_equity_regime_analysis()

                        # Store current backtest data in session_state for persistence
                        st.session_state['current_backtest'] = {
                            'engine': engine,
                            'metrics': metrics,
                            'backtest_log': backtest_log,
                            'start_date': start_date,
                            'end_date': end_date,
                            'mc_results': mc_results,
                            'equity_analysis': equity_analysis
                        }
                        # Also assist the MC tab by pre-populating export data
                        st.session_state['mc_results_for_export'] = mc_results
                        
                        st.session_state['current_backtest_active'] = True
                        
                        st.markdown("---")
                        
                        # Prepare Report Config
                        report_config = backtest_log['config'].copy()
                        # Ensure dates are strings
                        report_config['start_date'] = start_date.strftime('%Y-%m-%d')
                        report_config['end_date'] = end_date.strftime('%Y-%m-%d')
                        if regime_config: report_config['regime_config'] = regime_config
                        if uncorrelated_config: report_config['uncorrelated_config'] = uncorrelated_config

                        # Action buttons - Excel and PDF downloads
                        col_h, col_excel, col_pdf = st.columns([3, 1, 1])
                        with col_h:
                            st.subheader("Backtest Results")
                        with col_excel:
                            excel_data = create_excel_with_charts(
                                report_config, metrics, engine, 
                                mc_results=mc_results, 
                                regime_data=equity_analysis
                            )
                            st.download_button(
                                label="📥 Excel",
                                data=excel_data,
                                file_name=f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        with col_pdf:
                            pdf_data = create_pdf_report(
                                report_config, metrics, engine,
                                mc_results=mc_results,
                                regime_data=equity_analysis
                            )
                            st.download_button(
                                label="📄 PDF",
                                data=pdf_data,
                                file_name=f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        
                        # Result tabs - add regime-specific tabs based on filter type
                        
                        # Build tab list dynamically
                        tab_names = ["Performance Metrics", "Charts", "Monthly Breakup", "Monthly Report", "Trade History", "Monte Carlo Analysis"]
                        
                        # Check specific regime filter types
                        is_equity = regime_config and regime_config.get('type') == 'EQUITY'
                        is_equity_ma = regime_config and regime_config.get('type') == 'EQUITY_MA'
                        # Include all MA and trend-based regime types (legacy + new timeframe variants)
                        other_regime_types = [
                            'EMA', 'MACD', 'SUPERTREND', 'DONCHIAN', 'SWING_ATR', 'BREADTH',
                            'SMA_1D', 'SMA_1W', 'SMA_1M',
                            'EMA_1D', 'EMA_1W', 'EMA_1M',
                            'SUPERTREND_1D', 'SUPERTREND_1W', 'SUPERTREND_1M'
                        ]
                        is_other_regime = regime_config and regime_config.get('type') in other_regime_types
                        
                        # Equity Regime Testing tab - only for EQUITY filter
                        if is_equity and equity_analysis:
                            tab_names.append("Equity Regime Testing")
                        
                        # Equity MA Testing tab - only for EQUITY_MA filter
                        if is_equity_ma:
                            tab_names.append("Equity MA Testing")
                        
                        # Regime Filter Analysis tab - for EMA, MACD, SUPERTREND filters
                        if is_other_regime and equity_analysis:
                            tab_names.append("Regime Filter Analysis")
                        
                        result_tabs = st.tabs(tab_names)
                        
                        with result_tabs[0]:
                            st.markdown("### Key Performance Indicators")
                            
                            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                            
                            kpi_col1.metric("Final Value", f"₹{metrics['Final Value']:,.0f}")
                            kpi_col1.metric("Total Return", f"₹{metrics['Total Return']:,.0f}")
                            kpi_col1.metric("Return %", f"{metrics['Return %']:.2f}%")
                            
                            kpi_col2.metric("CAGR %", f"{metrics['CAGR %']:.2f}%")
                            kpi_col2.metric("Max Drawdown %", f"{metrics['Max Drawdown %']:.2f}%")
                            kpi_col2.metric("Volatility %", f"{metrics.get('Volatility %', 0):.2f}%")
                            
                            kpi_col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                            kpi_col3.metric("Win Rate %", f"{metrics['Win Rate %']:.2f}%")
                            kpi_col3.metric("Total Trades", metrics['Total Trades'])
                            
                            kpi_col4.metric("Avg Trade/Year", f"{metrics['Total Trades'] / max(1, (end_date - start_date).days / 365.25):.1f}")
                            kpi_col4.metric("Expectancy", f"₹{metrics.get('Expectancy', 0):,.0f}")
                            
                            # Additional Metrics Row
                            st.markdown("---")
                            st.markdown("**📊 Advanced Metrics**")
                            adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                            
                            adv_col1.metric("Max Consecutive Wins", metrics.get('Max Consecutive Wins', 0))
                            adv_col1.metric("Max Consecutive Losses", metrics.get('Max Consecutive Losses', 0))
                            
                            adv_col2.metric("Avg Win", f"₹{metrics.get('Avg Win', 0):,.0f}")
                            adv_col2.metric("Avg Loss", f"₹{metrics.get('Avg Loss', 0):,.0f}")
                            
                            adv_col3.metric("Days to Recover from DD", metrics.get('Days to Recover from DD', 0))
                            adv_col3.metric("Trades to Recover from DD", metrics.get('Trades to Recover from DD', 0))
                            
                            adv_col4.metric("Total Turnover", f"₹{metrics.get('Total Turnover', 0):,.0f}")
                            adv_col4.metric("Consolidated Charges", f"₹{metrics.get('Total Charges', 0):,.0f}")
                            
                            # Risk Metrics Row
                            st.markdown("---")
                            st.markdown("**⚠️ Risk Analysis**")
                            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                            
                            risk_col1.metric("Median MAE", f"{metrics.get('MAE Median %', 0):.2f}%", help="Typical worst unrealized loss during a trade")
                            risk_col2.metric("95% MAE", f"{metrics.get('MAE 95% %', 0):.2f}%", help="95% of trades never go worse than this drawdown")
                            risk_col3.metric("Max MAE", f"{metrics.get('MAE Max %', 0):.2f}%", help="Worst single trade unrealized drawdown")
                            risk_col4.metric("CVaR (5%)", f"{metrics.get('CVaR 5% %', 0):.2f}%", help="Average loss of the worst 5% of trades (Expected Shortfall)")
                            
                            # Charges Breakdown Expander
                            with st.expander("📋 Zerodha Charges Breakdown"):
                                charges_col1, charges_col2 = st.columns(2)
                                charges_col1.write(f"**STT/CTT (0.1%):** ₹{metrics.get('STT/CTT', 0):,.2f}")
                                charges_col1.write(f"**Transaction Charges:** ₹{metrics.get('Transaction Charges', 0):,.2f}")
                                charges_col1.write(f"**SEBI Charges:** ₹{metrics.get('SEBI Charges', 0):,.2f}")
                                charges_col2.write(f"**Stamp Charges (0.015%):** ₹{metrics.get('Stamp Charges', 0):,.2f}")
                                charges_col2.write(f"**GST (18%):** ₹{metrics.get('GST', 0):,.2f}")
                                charges_col2.write(f"**Total Charges:** ₹{metrics.get('Total Charges', 0):,.2f}")

                            # ── Put Hedge Analysis (only shown when Nifty Put Hedge regime was used) ──
                            if metrics.get('Hedge Events', 0) > 0:
                                st.markdown("---")
                                st.markdown("""
                                <div style='background:rgba(255,165,0,0.12);border-left:4px solid orange;
                                            padding:10px 16px;border-radius:6px;margin-bottom:10px;'>
                                🛡️ <b>Nifty Put Hedge Analysis</b>
                                </div>
                                """, unsafe_allow_html=True)

                                hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns(5)
                                hcol1.metric(
                                    "Hedge Events",
                                    int(metrics.get('Hedge Events', 0)),
                                    help="Number of times put hedge was activated"
                                )
                                hcol2.metric(
                                    "Hedge Cost",
                                    f"₹{metrics.get('Hedge Cost Total', 0):,.0f}",
                                    help="Total premium paid for put options"
                                )
                                hcol3.metric(
                                    "Hedge Proceeds",
                                    f"₹{metrics.get('Hedge Proceeds Total', 0):,.0f}",
                                    help="Total proceeds from closing put positions"
                                )
                                hedge_pnl = metrics.get('Hedge Net PnL', 0)
                                hcol4.metric(
                                    "Hedge Net P&L",
                                    f"₹{hedge_pnl:,.0f}",
                                    delta=f"{hedge_pnl:,.0f}",
                                    delta_color="normal",
                                    help="Net profit/loss from all put hedge positions"
                                )
                                hcol5.metric(
                                    "Hedge Efficiency",
                                    f"{metrics.get('Hedge Efficiency %', 0):.1f}%",
                                    help="(Proceeds - Cost) / Cost × 100"
                                )

                            # Original download buttons removed (consolidated above)
                        with result_tabs[1]:
                            st.markdown("### Performance Charts")
                            
                            # Equity Curve
                            fig_equity = go.Figure()
                            fig_equity.add_trace(go.Scatter(
                                x=engine.portfolio_df.index,
                                y=engine.portfolio_df['Portfolio Value'],
                                fill='tozeroy',
                                line_color='#28a745',
                                name='Portfolio Value'
                            ))
                            fig_equity.update_layout(
                                title="Equity Curve",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value (₹)",
                                height=400,
                                margin=dict(l=0,r=0,t=40,b=0),
                                showlegend=False,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_equity, use_container_width=True)
                            
                            # Drawdown Chart
                            running_max = engine.portfolio_df['Portfolio Value'].cummax()
                            dd = (engine.portfolio_df['Portfolio Value'] - running_max) / running_max * 100
                            
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=dd.index,
                                y=dd,
                                fill='tozeroy',
                                line_color='#dc3545',
                                name='Drawdown'
                            ))
                            fig_dd.update_layout(
                                title="Drawdown Analysis",
                                xaxis_title="Date",
                                yaxis_title="Drawdown %",
                                height=350,
                                margin=dict(l=0,r=0,t=40,b=0),
                                showlegend=False,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_dd, use_container_width=True)

                        with result_tabs[2]:
                            st.markdown("### Monthly Returns Breakup")
                            if not engine.portfolio_df.empty:
                                monthly_returns = engine.get_monthly_returns()
                                if not monthly_returns.empty:
                                    # Format the dataframe for display
                                    display_monthly = monthly_returns.copy()

                                    # Format percentages with color coding
                                    def color_negative_red(val):
                                        if pd.isna(val):
                                            return ''
                                        color = '#28a745' if val > 0 else '#dc3545' if val < 0 else '#6c757d'
                                        return f'color: {color}; font-weight: 600'

                                    # Apply styling (pandas 2.1+: applymap -> map)
                                    styled_df = display_monthly.style.map(color_negative_red)

                                    st.dataframe(styled_df, use_container_width=True, height=400)

                                    # Summary statistics
                                    st.markdown("---")
                                    col1, col2, col3, col4 = st.columns(4)

                                    all_returns = monthly_returns.iloc[:, :-1].values.flatten()
                                    all_returns = all_returns[~pd.isna(all_returns)]

                                    if len(all_returns) > 0:
                                        col1.metric("Positive Months", f"{(all_returns > 0).sum()} ({(all_returns > 0).sum()/len(all_returns)*100:.1f}%)")
                                        col2.metric("Negative Months", f"{(all_returns < 0).sum()} ({(all_returns < 0).sum()/len(all_returns)*100:.1f}%)")
                                        col3.metric("Best Month", f"{all_returns.max():.2f}%")
                                        col4.metric("Worst Month", f"{all_returns.min():.2f}%")
                            else:
                                st.info("No data available for monthly breakdown")

                        with result_tabs[3]:
                            st.markdown("### Monthly Portfolio Report")
                            if not engine.portfolio_df.empty:
                                monthly_df = engine.portfolio_df.copy()
                                monthly_df['Year'] = monthly_df.index.year
                                monthly_df['Month'] = monthly_df.index.month
                                monthly_df['Day'] = monthly_df.index.day
                                
                                display_df = monthly_df[['Year', 'Month', 'Day', 'Portfolio Value', 'Cash', 'Positions']]
                                st.dataframe(display_df, use_container_width=True, height=400)

                        with result_tabs[4]:
                            st.markdown("### Trade History")
                            if not engine.trades_df.empty:
                                # Create consolidated trade view matching BUY with SELL
                                trades_df = engine.trades_df.copy()
                                buy_trades = trades_df[trades_df['Action'].isin(['BUY', 'BUY_HEDGE'])].copy()
                                sell_trades = trades_df[trades_df['Action'].isin(['SELL', 'SELL_HEDGE'])].copy()
                                
                                consolidated_trades = []
                                
                                for _, sell in sell_trades.iterrows():
                                    ticker = sell['Ticker']
                                    sell_date = sell['Date']
                                    
                                    # Find the most recent BUY for this ticker before this SELL
                                    prev_buys = buy_trades[
                                        (buy_trades['Ticker'] == ticker) & 
                                        (buy_trades['Date'] < sell_date)
                                    ]
                                    
                                    if not prev_buys.empty:
                                        buy = prev_buys.iloc[-1]
                                        buy_price = float(buy['Price'])
                                        sell_price = float(sell['Price'])
                                        shares = int(buy['Shares'])
                                        roi = ((sell_price - buy_price) / buy_price) * 100
                                        
                                        consolidated_trades.append({
                                            'Stock': ticker.replace('.NS', ''),
                                            'Buy Date': pd.to_datetime(buy['Date']).strftime('%Y-%m-%d'),
                                            'Buy Price': round(buy_price, 2),
                                            'Exit Date': pd.to_datetime(sell_date).strftime('%Y-%m-%d'),
                                            'Exit Price': round(sell_price, 2),
                                            'Shares': shares,
                                            'ROI %': round(roi, 2)
                                        })
                                
                                if consolidated_trades:
                                    trade_display = pd.DataFrame(consolidated_trades)
                                    
                                    # Color ROI column
                                    def color_roi(val):
                                        if val > 0:
                                            return 'color: #28a745; font-weight: bold'
                                        elif val < 0:
                                            return 'color: #dc3545; font-weight: bold'
                                        return ''
                                    
                                    styled_trades = trade_display.style.map(
                                        color_roi, subset=['ROI %']
                                    )
                                    st.dataframe(styled_trades, use_container_width=True, height=400)
                                    
                                    # Summary stats
                                    st.markdown("---")
                                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                    stat_col1.metric("Total Trades", len(consolidated_trades))
                                    profitable = len([t for t in consolidated_trades if t['ROI %'] > 0])
                                    stat_col2.metric("Profitable", f"{profitable} ({profitable/len(consolidated_trades)*100:.1f}%)")
                                    avg_roi = sum(t['ROI %'] for t in consolidated_trades) / len(consolidated_trades)
                                    stat_col3.metric("Avg ROI", f"{avg_roi:.2f}%")
                                    best_trade = max(consolidated_trades, key=lambda x: x['ROI %'])
                                    stat_col4.metric("Best Trade", f"{best_trade['Stock']} ({best_trade['ROI %']:.1f}%)")
                                else:
                                    st.info("No completed trades to display")
                                
                                # Show Open Positions (BUY trades without matching SELL)
                                # Find buys that don't have a corresponding sell yet
                                sold_tickers_dates = set()
                                for _, sell in sell_trades.iterrows():
                                    # Find the buy this sell matched with
                                    ticker = sell['Ticker']
                                    sell_date = sell['Date']
                                    prev_buys = buy_trades[
                                        (buy_trades['Ticker'] == ticker) & 
                                        (buy_trades['Date'] < sell_date)
                                    ]
                                    if not prev_buys.empty:
                                        buy = prev_buys.iloc[-1]
                                        sold_tickers_dates.add((ticker, buy['Date']))
                                
                                open_positions = []
                                for _, buy in buy_trades.iterrows():
                                    if (buy['Ticker'], buy['Date']) not in sold_tickers_dates:
                                        ticker = buy['Ticker']
                                        buy_date = buy['Date']
                                        buy_price = float(buy['Price'])
                                        shares = int(buy['Shares'])
                                        
                                        # Get current price (last available)
                                        if ticker in engine.data and not engine.data[ticker].empty:
                                            _close = engine.data[ticker]['Close'].iloc[-1]
                                            current_price = float(_close.iloc[0] if hasattr(_close, 'iloc') else _close)
                                            current_date = engine.data[ticker].index[-1]
                                            unrealized_roi = ((current_price - buy_price) / buy_price) * 100
                                        else:
                                            current_price = buy_price
                                            current_date = buy_date
                                            unrealized_roi = 0.0
                                        
                                        open_positions.append({
                                            'Stock': ticker.replace('.NS', ''),
                                            'Buy Date': pd.to_datetime(buy_date).strftime('%Y-%m-%d'),
                                            'Buy Price': round(buy_price, 2),
                                            'Current Price': round(current_price, 2),
                                            'Shares': shares,
                                            'Unrealized ROI %': round(unrealized_roi, 2),
                                            'Status': '🟢 OPEN'
                                        })
                                
                                if open_positions:
                                    # Store in session state for Execute Trades tab
                                    st.session_state['open_positions'] = open_positions
                                    st.session_state['engine_data'] = engine.data
                                    
                                    st.markdown("---")
                                    st.markdown("### 📈 Open Positions (Current Holdings)")
                                    st.caption("These are positions bought but not yet sold at the end of the backtest period. Go to **Execute Trades** tab to place orders.")
                                    
                                    open_df = pd.DataFrame(open_positions)
                                    
                                    def color_unrealized(val):
                                        if val > 0:
                                            return 'color: #28a745; font-weight: bold'
                                        elif val < 0:
                                            return 'color: #dc3545; font-weight: bold'
                                        return ''
                                    
                                    styled_open = open_df.style.map(
                                        color_unrealized, subset=['Unrealized ROI %']
                                    )
                                    st.dataframe(styled_open, use_container_width=True)
                                    
                                    # Prompt to use Execute Trades tab
                                    st.info("👉 Go to the **Execute Trades** tab to place orders on Zerodha with these positions.")
                            else:
                                st.info("No trades executed")
                        
                        # Monte Carlo Analysis Tab (index 5 - always present)
                        with result_tabs[5]:
                            st.markdown("### 🎲 Monte Carlo Analysis")
                            
                            # Determine MC type based on position sizing method
                            use_trade_level = position_sizing_method == "Equal Weight"
                            
                            if use_trade_level:
                                st.caption("**Trade-Level MC** — Valid for equal-weight portfolios where each trade is independent")
                            else:
                                st.caption("**Portfolio-Level MC** — Monthly returns shuffling for vol-weighted portfolios")
                            
                            if not engine.trades_df.empty:
                                # Calculate test duration
                                days = (engine.portfolio_df.index[-1] - engine.portfolio_df.index[0]).days
                                years = days / 365.25
                                
                                if use_trade_level:
                                    # Trade-Level MC (for Equal Weight)
                                    trade_pnls = extract_trade_pnls(engine.trades_df)
                                    
                                    if len(trade_pnls) >= 10:
                                        with st.spinner("Running Trade-Level Monte Carlo (Reshuffle & Resample)..."):
                                            mc = MonteCarloSimulator(
                                                trade_pnls=trade_pnls,
                                                initial_capital=engine.initial_capital,
                                                test_duration_years=years,
                                                n_simulations=10000
                                            )
                                            results_reshuffle = mc.run_simulations(method='reshuffle')
                                            interp_reshuffle = mc.get_interpretation()
                                            
                                            results_resample = mc.run_simulations(method='resample')
                                            interp_resample = mc.get_interpretation()
                                        
                                        st.success(f"✅ Trade-Level MC completed: **10,000 simulations** using {len(trade_pnls)} trades over {years:.1f} years")
                                    else:
                                        st.warning(f"Need at least 10 completed trades for Monte Carlo analysis. Currently have {len(trade_pnls)} trades.")
                                        results_reshuffle = results_resample = None
                                else:
                                    # Portfolio-Level MC (for Inverse Vol, Risk Parity, Score-Weighted)
                                    # Use GPT's exact logic: calculate monthly returns from trade PnLs
                                    
                                    # Build trades DataFrame (same structure as CSV export)
                                    trades_for_mc = []
                                    buy_trades = engine.trades_df[engine.trades_df['Action'].isin(['BUY', 'BUY_HEDGE'])]
                                    sell_trades = engine.trades_df[engine.trades_df['Action'].isin(['SELL', 'SELL_HEDGE'])]
                                    
                                    for ticker in sell_trades['Ticker'].unique():
                                        ticker_sells = sell_trades[sell_trades['Ticker'] == ticker]
                                        for _, sell in ticker_sells.iterrows():
                                            sell_date = sell['Date']
                                            prev_buys = buy_trades[
                                                (buy_trades['Ticker'] == ticker) & 
                                                (buy_trades['Date'] < sell_date)
                                            ]
                                            if not prev_buys.empty:
                                                buy = prev_buys.iloc[-1]
                                                trades_for_mc.append({
                                                    'Stock': ticker.replace('.NS', ''),
                                                    'Buy Date': pd.to_datetime(buy['Date']),
                                                    'Buy Price': float(buy['Price']),
                                                    'Exit Date': pd.to_datetime(sell_date),
                                                    'Exit Price': float(sell['Price']),
                                                    'Shares': int(buy['Shares'])
                                                })
                                    
                                    if trades_for_mc:
                                        trades_df_for_mc = pd.DataFrame(trades_for_mc)
                                        monthly_returns = extract_monthly_returns(trades_df_for_mc, engine.initial_capital)
                                    else:
                                        monthly_returns = []
                                    
                                    if len(monthly_returns) >= 6:
                                        
                                        with st.spinner("Running Portfolio-Level Monte Carlo (Monthly Returns)..."):
                                            mc = PortfolioMonteCarloSimulator(
                                                monthly_returns=monthly_returns,
                                                initial_capital=engine.initial_capital,
                                                n_simulations=10000
                                            )
                                            results_reshuffle = mc.run_simulations(method='reshuffle')
                                            interp_reshuffle = mc.get_interpretation()
                                            
                                            results_resample = mc.run_simulations(method='resample')
                                            interp_resample = mc.get_interpretation()
                                        
                                        st.success(f"✅ Portfolio-Level MC completed: **10,000 simulations** using {len(monthly_returns)} monthly returns over {years:.1f} years")
                                    else:
                                        st.warning(f"Need at least 6 months of data for Portfolio Monte Carlo. Currently have {len(monthly_returns)} months.")
                                        results_reshuffle = results_resample = None
                                
                                # Store MC results in session for export
                                if results_reshuffle is not None and results_resample is not None:
                                    st.session_state['mc_results_for_export'] = {
                                        'perm_dd_95': results_reshuffle.get('mc_max_dd_95', 0),
                                        'perm_dd_worst': results_reshuffle.get('mc_max_dd_worst', 0),
                                        'perm_ruin': results_reshuffle.get('ruin_probability', 0),
                                        'perm_cagr_med': results_reshuffle.get('mc_cagr_median', 0),
                                        'boot_dd_95': results_resample.get('mc_max_dd_95', 0),
                                        'boot_dd_worst': results_resample.get('mc_max_dd_worst', 0),
                                        'boot_ruin': results_resample.get('ruin_probability', 0),
                                        'boot_cagr_med': results_resample.get('mc_cagr_median', 0),
                                        'n_simulations': 10000,
                                        'initial_capital': engine.initial_capital
                                    }
                                
                                # Display results if available
                                if results_reshuffle is not None and results_resample is not None:
                                    # Create two columns for side-by-side comparison
                                    mc_col1, mc_col2 = st.columns(2)
                                    
                                    # Define helper to render results
                                    def render_mc_results(col, title, results, interp):
                                        with col:
                                            st.markdown(f"### {title}")
                                            if 'method_note' in interp:
                                                st.caption(interp['method_note'])
                                            
                                            # Chart
                                            sample_curves = results.get('sample_equity_curves', [])
                                            historical_curve = results.get('historical_equity_curve', [])
                                            
                                            if sample_curves:
                                                fig_mc = go.Figure()
                                                
                                                # Color palette for simulation paths (like GPT's matplotlib colorful fan)
                                                n_curves = min(1000, len(sample_curves))
                                                
                                                # Plot sample simulation curves with varying colors
                                                for i, curve in enumerate(sample_curves[:n_curves]):
                                                    # Create color gradient from light orange to light blue
                                                    hue = (i / n_curves) * 0.6  # Range from 0 to 0.6 (orange to blue)
                                                    color = f'hsla({int(hue * 360)}, 70%, 60%, 0.15)'
                                                    fig_mc.add_trace(go.Scatter(
                                                        x=list(range(len(curve))), y=curve, mode='lines',
                                                        line=dict(color=color, width=0.8),
                                                        showlegend=False, hoverinfo='skip'
                                                    ))
                                                
                                                # Plot historical (thick teal/cyan line like GPT)
                                                if historical_curve:
                                                    fig_mc.add_trace(go.Scatter(
                                                        x=list(range(len(historical_curve))), y=historical_curve,
                                                        mode='lines', name='Historical',
                                                        line=dict(color='#17a2b8', width=4)  # Teal like GPT
                                                    ))
                                                
                                                # Determine X-axis label
                                                mc_level = results.get('level', 'trade')
                                                xaxis_label = "Months" if mc_level == 'portfolio' else "Trades"
                                                
                                                # Chart title with method name
                                                method_name = results.get('method_name', title)
                                                chart_title = f"Monte Carlo {method_name} - Monthly Portfolio Equity Paths"
                                                
                                                fig_mc.update_layout(
                                                    title=chart_title,
                                                    xaxis_title=xaxis_label, yaxis_title="Equity",
                                                    height=500, template='plotly_dark',
                                                    margin=dict(l=50, r=40, t=60, b=50),
                                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                                                )
                                                st.plotly_chart(fig_mc, use_container_width=True)
                                            
                                            # Stats Table
                                            st.markdown("#### Key Statistics")
                                            stats_data = {
                                                'Metric': ['Max Drawdown (95%)', 'Worst Case DD', 'Ruin Probability', 'CAGR (Median)', 'CAGR (5th %ile)'],
                                                'Value': [
                                                    f"{results['mc_max_dd_95']:.1f}%",
                                                    f"{results['mc_max_dd_worst']:.1f}%",
                                                    f"{results['ruin_probability']:.2f}%",
                                                    f"{results['mc_cagr_median']:.1f}%",
                                                    f"{results['mc_cagr_5th']:.1f}%"
                                                ]
                                            }
                                            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                                            
                                            # Interpretation
                                            with st.expander("Detailed Analysis", expanded=False):
                                                st.info(f"**Drawdown:** {interp['max_drawdown']}")
                                                st.info(f"**Streak:** {interp['losing_streak']}")
                                                if results['ruin_probability'] > 0:
                                                    st.warning(f"**Ruin:** {interp['ruin']}")
                                                else:
                                                    st.success(f"**Ruin:** {interp['ruin']}")
                                                    
                                                st.markdown(f"**CAGR Spread:** {results['mc_cagr_5th']:.1f}% to {results['mc_cagr_95th']:.1f}%")
                                    
                                    # Render both
                                    render_mc_results(mc_col1, "Reshuffle (Permutation)", results_reshuffle, interp_reshuffle)
                                    render_mc_results(mc_col2, "Resample (Bootstrap)", results_resample, interp_resample)
                                    
                                    # Comparison insight
                                    st.markdown("---")
                                    st.info("💡 **Comparison:** Reshuffling shows risk assuming the *exact same set* of trades occur in different orders. Resampling (Bootstrap) simulates risk assuming the market conditions could generate *more* of the losing trades or *fewer* of the winning trades, typically showing a wider range of outcomes and risks.")

                                else:
                                    st.warning(f"Need at least 10 completed trades for Monte Carlo analysis. Currently have {len(trade_pnls)} trades.")
                            else:
                                st.info("No trades available for Mone Carlo analysis. Run a backtest first.")
                        
                        # Equity Regime Testing Tab (only shown if EQUITY regime filter was used)
                        if is_equity and equity_analysis:
                            equity_tab_idx = 6  # Base 6 tabs (0-5), this is index 6
                            with result_tabs[equity_tab_idx]:
                                st.markdown("### 📊 Equity Regime Testing")
                                st.warning("⚠️ **DISCLAIMER**: This section is for testing purposes only. The theoretical curve shows what would have happened WITHOUT the EQUITY regime filter.")
                                
                                st.markdown(f"**Stop-Loss Threshold:** {equity_analysis['sl_threshold']}%")
                                
                                # Trigger Events Table
                                trigger_events = equity_analysis.get('trigger_events', [])
                                if trigger_events:
                                    st.markdown("### Regime Trigger Events")
                                    events_data = []
                                    for event in trigger_events:
                                        events_data.append({
                                            'Date': event['date'].strftime('%Y-%m-%d'),
                                            'Event': '🔴 TRIGGERED' if event['type'] == 'trigger' else '🟢 RECOVERED',
                                            'Drawdown %': f"{event['drawdown']:.2f}%",
                                            'Peak Equity': f"₹{event['peak']:,.0f}",
                                            'Current Equity': f"₹{event['current']:,.0f}"
                                        })
                                    st.dataframe(pd.DataFrame(events_data), use_container_width=True)
                                else:
                                    st.info("No regime triggers during this backtest period. The drawdown never exceeded your SL threshold.")
                                
                                # Theoretical vs Actual Equity Curve
                                theoretical_curve = equity_analysis.get('theoretical_curve')
                                if theoretical_curve is not None and not theoretical_curve.empty:
                                    st.markdown("### Theoretical vs Actual Equity Curve")
                                    st.caption("Shows what would have happened WITHOUT the EQUITY regime filter (no mid-drawdown exits)")
                                    
                                    fig_compare = go.Figure()
                                    fig_compare.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=engine.portfolio_df['Portfolio Value'],
                                        name='Actual (With EQUITY Filter)',
                                        line=dict(color='#28a745', width=2)
                                    ))
                                    fig_compare.add_trace(go.Scatter(
                                        x=theoretical_curve.index,
                                        y=theoretical_curve['Theoretical_Equity'],
                                        name='Theoretical (Without Filter)',
                                        line=dict(color='#007bff', width=2, dash='dot')
                                    ))
                                    fig_compare.update_layout(
                                        title="Actual vs Theoretical Equity Curve",
                                        xaxis_title="Date",
                                        yaxis_title="Portfolio Value (₹)",
                                        height=450,
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_compare, use_container_width=True)
                                    
                                    # Summary metrics comparison
                                    actual_final = engine.portfolio_df['Portfolio Value'].iloc[-1]
                                    theoretical_final = theoretical_curve['Theoretical_Equity'].iloc[-1]
                                    actual_return = ((actual_final / engine.initial_capital) - 1) * 100
                                    theoretical_return = ((theoretical_final / engine.initial_capital) - 1) * 100
                                    
                                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                                    comp_col1.metric("Actual Final Value", f"₹{actual_final:,.0f}")
                                    comp_col2.metric("Theoretical Final Value", f"₹{theoretical_final:,.0f}")
                                    
                                    diff = actual_return - theoretical_return
                                    if diff > 0:
                                        comp_col3.metric("Filter Benefit", f"+{diff:.2f}%", delta=f"+{diff:.2f}%")
                                    else:
                                        comp_col3.metric("Filter Impact", f"{diff:.2f}%", delta=f"{diff:.2f}%")
                                
                                st.markdown("---")
                                
                                # Peak Equity and Drawdown Chart
                                st.markdown("### Peak Equity & Drawdown Tracking")
                                if 'Peak_Equity' in engine.portfolio_df.columns:
                                    fig_peak = go.Figure()
                                    fig_peak.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=engine.portfolio_df['Portfolio Value'],
                                        name='Portfolio Value',
                                        line=dict(color='#28a745', width=2)
                                    ))
                                    fig_peak.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=engine.portfolio_df['Peak_Equity'],
                                        name='Peak Equity',
                                        line=dict(color='#ffc107', width=2, dash='dot')
                                    ))
                                    # Add threshold line from peak
                                    threshold_line = engine.portfolio_df['Peak_Equity'] * (1 - equity_analysis['sl_threshold'] / 100)
                                    fig_peak.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=threshold_line,
                                        name=f"SL Threshold ({equity_analysis['sl_threshold']}%)",
                                        line=dict(color='#dc3545', width=1, dash='dash')
                                    ))
                                    
                                    # Add trigger event markers
                                    for event in trigger_events:
                                        color = 'red' if event['type'] == 'trigger' else 'green'
                                        symbol = 'triangle-down' if event['type'] == 'trigger' else 'triangle-up'
                                        fig_peak.add_trace(go.Scatter(
                                            x=[event['date']],
                                            y=[event['current']],
                                            mode='markers',
                                            marker=dict(size=12, color=color, symbol=symbol),
                                            name=f"{'Trigger' if event['type'] == 'trigger' else 'Recovery'} ({event['date'].strftime('%Y-%m-%d')})",
                                            showlegend=True
                                        ))
                                    
                                    fig_peak.update_layout(
                                        title="Portfolio Value vs Peak Equity with SL Threshold",
                                        xaxis_title="Date",
                                        yaxis_title="Value (₹)",
                                        height=450,
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_peak, use_container_width=True)
                                
                                # Drawdown percentage chart
                                st.markdown("### Drawdown from Peak")
                                if 'Drawdown_Pct' in engine.portfolio_df.columns:
                                    fig_dd = go.Figure()
                                    fig_dd.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=-engine.portfolio_df['Drawdown_Pct'],  # Negative to show as positive area below
                                        fill='tozeroy',
                                        line=dict(color='#dc3545', width=1),
                                        name='Drawdown %'
                                    ))
                                    # Add threshold line
                                    fig_dd.add_hline(
                                        y=-equity_analysis['sl_threshold'], 
                                        line_dash="dash", 
                                        line_color="yellow",
                                        annotation_text=f"SL Threshold ({equity_analysis['sl_threshold']}%)"
                                    )
                                    fig_dd.update_layout(
                                        title="Drawdown % from Peak Equity",
                                        xaxis_title="Date",
                                        yaxis_title="Drawdown %",
                                        height=350,
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_dd, use_container_width=True)
                                    
                                    # Answer user's question in the UI
                                    st.markdown("---")
                                    st.info(f"**Note:** With EQUITY regime filter enabled at {equity_analysis['sl_threshold']}% SL, the maximum drawdown should be approximately capped at this threshold. When the drawdown breaches the SL, all positions are sold to prevent further losses.")
                        
                        # EQUITY_MA Testing Tab
                        if is_equity_ma and len(tab_names) > 5:
                            # Find the EQUITY_MA tab index
                            equity_ma_tab_idx = len(tab_names) - 1  # Last tab
                            if equity_analysis:
                                equity_ma_tab_idx = len(tab_names) - 1  # Still last if both exist
                            
                            with result_tabs[equity_ma_tab_idx]:
                                st.markdown("### 📈 Equity Curve MA Analysis")
                                st.markdown("> **Meta-Strategy:** Reduce exposure when portfolio equity falls below its moving average")
                                
                                ma_period = regime_config.get('ma_period', 50)
                                
                                # Check if we have the MA data in portfolio_df
                                if 'Equity_MA' in engine.portfolio_df.columns:
                                    st.markdown("---")
                                    
                                    # Equity vs MA Chart
                                    st.markdown(f"### Equity Curve vs {ma_period}-Day MA")
                                    
                                    fig_ma = go.Figure()
                                    
                                    # Portfolio Value
                                    fig_ma.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=engine.portfolio_df['Portfolio Value'],
                                        name='Portfolio Value',
                                        line=dict(color='#28a745', width=2)
                                    ))
                                    
                                    # MA line (filter out zeros)
                                    ma_data = engine.portfolio_df[engine.portfolio_df['Equity_MA'] > 0]['Equity_MA']
                                    fig_ma.add_trace(go.Scatter(
                                        x=ma_data.index,
                                        y=ma_data,
                                        name=f'{ma_period}-Day MA',
                                        line=dict(color='#ffc107', width=2, dash='dot')
                                    ))
                                    
                                    # Shade triggered periods
                                    if 'Equity_MA_Triggered' in engine.portfolio_df.columns:
                                        triggered = engine.portfolio_df[engine.portfolio_df['Equity_MA_Triggered'] == True]
                                        if len(triggered) > 0:
                                            fig_ma.add_trace(go.Scatter(
                                                x=triggered.index,
                                                y=triggered['Portfolio Value'],
                                                mode='markers',
                                                name='Below MA (Reduced Exposure)',
                                                marker=dict(color='#dc3545', size=4, opacity=0.6)
                                            ))
                                    
                                    fig_ma.update_layout(
                                        title=f"Portfolio Equity vs {ma_period}-Day Moving Average",
                                        xaxis_title="Date",
                                        yaxis_title="Value (₹)",
                                        height=450,
                                        template='plotly_dark',
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    st.plotly_chart(fig_ma, use_container_width=True)
                                    
                                    # Statistics
                                    st.markdown("---")
                                    st.markdown("### 📊 MA Filter Statistics")
                                    
                                    if 'Equity_MA_Triggered' in engine.portfolio_df.columns:
                                        total_days = len(engine.portfolio_df)
                                        triggered_days = engine.portfolio_df['Equity_MA_Triggered'].sum()
                                        pct_triggered = (triggered_days / total_days * 100) if total_days > 0 else 0
                                        
                                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                                        stat_col1.metric("Total Trading Days", f"{total_days:,}")
                                        stat_col2.metric("Days Below MA", f"{int(triggered_days):,}")
                                        stat_col3.metric("% Time in Reduced Exposure", f"{pct_triggered:.1f}%")
                                    
                                    # Theoretical vs Actual Comparison
                                    st.markdown("---")
                                    st.markdown("### 📈 Theoretical vs Actual Equity Curve")
                                    st.markdown("> Compare your actual returns (with MA filter) against theoretical returns (without filter)")
                                    
                                    # Get theoretical data from equity_analysis
                                    if equity_analysis and 'theoretical_curve' in equity_analysis:
                                        theoretical_df = equity_analysis['theoretical_curve']
                                        
                                        fig_compare = go.Figure()
                                        
                                        # Actual equity curve
                                        fig_compare.add_trace(go.Scatter(
                                            x=engine.portfolio_df.index,
                                            y=engine.portfolio_df['Portfolio Value'],
                                            name='Actual (With MA Filter)',
                                            line=dict(color='#28a745', width=2)
                                        ))
                                        
                                        # Theoretical equity curve
                                        fig_compare.add_trace(go.Scatter(
                                            x=theoretical_df.index,
                                            y=theoretical_df['Theoretical_Equity'],
                                            name='Theoretical (Without Filter)',
                                            line=dict(color='#17a2b8', width=2, dash='dot')
                                        ))
                                        
                                        fig_compare.update_layout(
                                            title="Actual vs Theoretical Equity Curve",
                                            xaxis_title="Date",
                                            yaxis_title="Portfolio Value (₹)",
                                            height=450,
                                            template='plotly_dark',
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                        )
                                        st.plotly_chart(fig_compare, use_container_width=True)
                                        
                                        # Summary metrics
                                        actual_final = engine.portfolio_df['Portfolio Value'].iloc[-1]
                                        theoretical_final = theoretical_df['Theoretical_Equity'].iloc[-1]
                                        actual_return = ((actual_final / engine.initial_capital) - 1) * 100
                                        theoretical_return = ((theoretical_final / engine.initial_capital) - 1) * 100
                                        
                                        comp_col1, comp_col2, comp_col3 = st.columns(3)
                                        comp_col1.metric("Actual Final Value", f"₹{actual_final:,.0f}")
                                        comp_col2.metric("Theoretical Final Value", f"₹{theoretical_final:,.0f}")
                                        
                                        diff = actual_return - theoretical_return
                                        if diff > 0:
                                            comp_col3.metric("Filter Benefit", f"+{diff:.2f}%", delta=f"+{diff:.2f}%")
                                        else:
                                            comp_col3.metric("Filter Impact", f"{diff:.2f}%", delta=f"{diff:.2f}%")
                                    
                                    st.markdown("---")
                                    st.info(f"**How it works:** When portfolio equity falls below its {ma_period}-day moving average, exposure is reduced to protect capital. When equity recovers above the MA, full exposure resumes.")
                        
                        # Regime Filter Analysis Tab (for EMA, MACD, SUPERTREND)
                        if is_other_regime and equity_analysis and len(tab_names) > 5:
                            # Find the tab index
                            regime_tab_idx = len(tab_names) - 1  # Last tab
                            
                            with result_tabs[regime_tab_idx]:
                                regime_type = equity_analysis.get('regime_type', 'Unknown')
                                regime_value = equity_analysis.get('regime_value', '')
                                
                                st.markdown(f"### 📊 {regime_type} Regime Filter Analysis")
                                st.markdown(f"> Compare your actual returns (with {regime_type} filter) against theoretical returns (without filter)")
                                
                                if 'theoretical_curve' in equity_analysis:
                                    theoretical_df = equity_analysis['theoretical_curve']
                                    
                                    # Comparison Chart
                                    st.markdown("---")
                                    st.markdown("### 📈 Actual vs Theoretical Equity Curve")
                                    
                                    fig_compare = go.Figure()
                                    
                                    # Actual equity curve
                                    fig_compare.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=engine.portfolio_df['Portfolio Value'],
                                        name=f'Actual (With {regime_type} Filter)',
                                        line=dict(color='#28a745', width=2)
                                    ))
                                    
                                    # Theoretical equity curve
                                    fig_compare.add_trace(go.Scatter(
                                        x=theoretical_df.index,
                                        y=theoretical_df['Theoretical_Equity'],
                                        name='Theoretical (No Filter)',
                                        line=dict(color='#17a2b8', width=2, dash='dot')
                                    ))
                                    
                                    fig_compare.update_layout(
                                        title=f"Equity Curve: With vs Without {regime_type} Filter",
                                        xaxis_title="Date",
                                        yaxis_title="Portfolio Value (₹)",
                                        height=450,
                                        template='plotly_dark',
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    st.plotly_chart(fig_compare, use_container_width=True)
                                    
                                    # Calculate comprehensive metrics for both
                                    st.markdown("---")
                                    st.markdown("### 📋 Metrics Comparison (Before vs After Filter)")
                                    
                                    # === ACTUAL METRICS - Use engine.get_metrics() to match Performance tab ===
                                    actual_metrics = metrics  # Already calculated from engine.get_metrics()
                                    actual_final = actual_metrics['Final Value']
                                    actual_return_pct = actual_metrics['Return %']
                                    actual_cagr = actual_metrics['CAGR %']
                                    actual_max_dd = actual_metrics['Max Drawdown %']
                                    actual_volatility = actual_metrics['Volatility %']
                                    actual_sharpe = actual_metrics['Sharpe Ratio']
                                    actual_win_rate = actual_metrics['Win Rate %']
                                    actual_expectancy = actual_metrics['Expectancy']
                                    actual_total_trades = actual_metrics['Total Trades']
                                    actual_max_wins = actual_metrics['Max Consecutive Wins']
                                    actual_max_losses = actual_metrics['Max Consecutive Losses']
                                    actual_avg_win = actual_metrics['Avg Win']
                                    actual_avg_loss = actual_metrics['Avg Loss']
                                    actual_days_to_recover = actual_metrics['Days to Recover from DD']
                                    
                                    # Common calculations
                                    days = (engine.portfolio_df.index[-1] - engine.portfolio_df.index[0]).days
                                    years = days / 365.25
                                    actual_trades_per_year = actual_total_trades / years if years > 0 else 0
                                    
                                    theoretical_final = theoretical_df['Theoretical_Equity'].iloc[-1]
                                    
                                    # === THEORETICAL METRICS ===
                                    theoretical_return_pct = ((theoretical_final / engine.initial_capital) - 1) * 100
                                    theoretical_cagr = ((theoretical_final / engine.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
                                    theoretical_running_max = theoretical_df['Theoretical_Equity'].cummax()
                                    theoretical_dd_series = ((theoretical_df['Theoretical_Equity'] - theoretical_running_max) / theoretical_running_max * 100)
                                    theoretical_max_dd = abs(theoretical_dd_series.min())
                                    theoretical_daily_returns = theoretical_df['Theoretical_Equity'].pct_change().dropna()
                                    theoretical_volatility = theoretical_daily_returns.std() * (252 ** 0.5) * 100
                                    theoretical_sharpe = (theoretical_cagr - 6) / theoretical_volatility if theoretical_volatility > 0 else 0
                                    
                                    # For theoretical trade metrics, use rebalance-period P/L from theoretical equity
                                    # Get equity values at each rebalance date to calculate period P/Ls
                                    theoretical_total_trades = actual_total_trades
                                    theoretical_trades_per_year = theoretical_total_trades / years if years > 0 else 0
                                    
                                    # Calculate P/L at each rebalance by finding value changes at rebalance dates
                                    if not engine.trades_df.empty and 'Date' in engine.trades_df.columns:
                                        # Get unique rebalance dates (from actual trades)
                                        rebalance_dates = engine.trades_df['Date'].unique()
                                        
                                        # Calculate theoretical P/L between each rebalance period
                                        theoretical_pnls = []
                                        for i in range(1, len(rebalance_dates)):
                                            prev_date = rebalance_dates[i-1]
                                            curr_date = rebalance_dates[i]
                                            
                                            # Find theoretical equity at these dates
                                            if prev_date in theoretical_df.index and curr_date in theoretical_df.index:
                                                prev_equity = theoretical_df.loc[prev_date, 'Theoretical_Equity']
                                                curr_equity = theoretical_df.loc[curr_date, 'Theoretical_Equity']
                                                pnl = curr_equity - prev_equity
                                                theoretical_pnls.append(pnl)
                                            elif len(theoretical_df) > 0:
                                                # Find nearest dates
                                                theo_dates = theoretical_df.index
                                                prev_idx = theo_dates.get_indexer([prev_date], method='nearest')[0]
                                                curr_idx = theo_dates.get_indexer([curr_date], method='nearest')[0]
                                                if prev_idx != curr_idx:
                                                    prev_equity = theoretical_df.iloc[prev_idx]['Theoretical_Equity']
                                                    curr_equity = theoretical_df.iloc[curr_idx]['Theoretical_Equity']
                                                    pnl = curr_equity - prev_equity
                                                    theoretical_pnls.append(pnl)
                                        
                                        # Calculate metrics from period P/Ls
                                        if theoretical_pnls:
                                            theo_wins = [p for p in theoretical_pnls if p > 0]
                                            theo_losses = [abs(p) for p in theoretical_pnls if p < 0]
                                            
                                            theoretical_win_rate = len(theo_wins) / len(theoretical_pnls) * 100 if theoretical_pnls else 0
                                            theoretical_avg_win = sum(theo_wins) / len(theo_wins) if theo_wins else 0
                                            theoretical_avg_loss = sum(theo_losses) / len(theo_losses) if theo_losses else 0
                                            
                                            win_pct = len(theo_wins) / len(theoretical_pnls) if theoretical_pnls else 0
                                            loss_pct = len(theo_losses) / len(theoretical_pnls) if theoretical_pnls else 0
                                            theoretical_expectancy = (win_pct * theoretical_avg_win) - (loss_pct * theoretical_avg_loss)
                                            
                                            # Max consecutive wins/losses from Period P/Ls
                                            theo_wins_streak = theo_losses_streak = theoretical_max_wins = theoretical_max_losses = 0
                                            for pnl in theoretical_pnls:
                                                if pnl > 0:
                                                    theo_wins_streak += 1
                                                    theo_losses_streak = 0
                                                    theoretical_max_wins = max(theoretical_max_wins, theo_wins_streak)
                                                else:
                                                    theo_losses_streak += 1
                                                    theo_wins_streak = 0
                                                    theoretical_max_losses = max(theoretical_max_losses, theo_losses_streak)
                                        else:
                                            theoretical_win_rate = theoretical_avg_win = theoretical_avg_loss = theoretical_expectancy = 0
                                            theoretical_max_wins = theoretical_max_losses = 0
                                    else:
                                        theoretical_win_rate = theoretical_avg_win = theoretical_avg_loss = theoretical_expectancy = 0
                                        theoretical_max_wins = theoretical_max_losses = 0
                                    
                                    # Days to recover for theoretical
                                    theo_dd_min_idx = theoretical_dd_series.idxmin()
                                    theo_recovery_mask = (theoretical_df.index > theo_dd_min_idx) & (theoretical_dd_series >= -0.1)
                                    if theo_recovery_mask.any():
                                        theo_recovery_date = theoretical_df.index[theo_recovery_mask][0]
                                        theoretical_days_to_recover = (theo_recovery_date - theo_dd_min_idx).days
                                    else:
                                        theoretical_days_to_recover = (theoretical_df.index[-1] - theo_dd_min_idx).days
                                    
                                    # Create comprehensive comparison dataframe
                                    comparison_data = {
                                        'Metric': [
                                            'Final Value', 'Total Return %', 'CAGR %', 'Max Drawdown %', 
                                            'Volatility %', 'Sharpe Ratio', 'Win Rate %', 'Expectancy',
                                            'Total Trades', 'Avg Trades/Year', 'Max Consecutive Wins',
                                            'Max Consecutive Losses', 'Avg Win', 'Avg Loss', 'Days to Recover'
                                        ],
                                        'Without Filter': [
                                            f"₹{theoretical_final:,.0f}",
                                            f"{theoretical_return_pct:.2f}%",
                                            f"{theoretical_cagr:.2f}%",
                                            f"{theoretical_max_dd:.2f}%",
                                            f"{theoretical_volatility:.2f}%",
                                            f"{theoretical_sharpe:.2f}",
                                            f"{theoretical_win_rate:.2f}%",
                                            f"₹{theoretical_expectancy:,.0f}",
                                            f"{theoretical_total_trades}",
                                            f"{theoretical_trades_per_year:.1f}",
                                            f"{theoretical_max_wins}",
                                            f"{theoretical_max_losses}",
                                            f"₹{theoretical_avg_win:,.0f}",
                                            f"₹{theoretical_avg_loss:,.0f}",
                                            f"{theoretical_days_to_recover}"
                                        ],
                                        'With Filter': [
                                            f"₹{actual_final:,.0f}",
                                            f"{actual_return_pct:.2f}%",
                                            f"{actual_cagr:.2f}%",
                                            f"{actual_max_dd:.2f}%",
                                            f"{actual_volatility:.2f}%",
                                            f"{actual_sharpe:.2f}",
                                            f"{actual_win_rate:.2f}%",
                                            f"₹{actual_expectancy:,.0f}",
                                            f"{actual_total_trades}",
                                            f"{actual_trades_per_year:.1f}",
                                            f"{actual_max_wins}",
                                            f"{actual_max_losses}",
                                            f"₹{actual_avg_win:,.0f}",
                                            f"₹{actual_avg_loss:,.0f}",
                                            f"{actual_days_to_recover}"
                                        ],
                                        'Better?': [
                                            '✅' if actual_final >= theoretical_final else '❌',
                                            '✅' if actual_return_pct >= theoretical_return_pct else '❌',
                                            '✅' if actual_cagr >= theoretical_cagr else '❌',
                                            '✅' if actual_max_dd <= theoretical_max_dd else '❌',
                                            '✅' if actual_volatility <= theoretical_volatility else '❌',
                                            '✅' if actual_sharpe >= theoretical_sharpe else '❌',
                                            '✅' if actual_win_rate >= theoretical_win_rate else '❌',
                                            '✅' if actual_expectancy >= theoretical_expectancy else '❌',
                                            '➖',  # Total trades neutral
                                            '➖',  # Avg trades/year neutral
                                            '✅' if actual_max_wins >= theoretical_max_wins else '❌',
                                            '✅' if actual_max_losses <= theoretical_max_losses else '❌',
                                            '✅' if actual_avg_win >= theoretical_avg_win else '❌',
                                            '✅' if actual_avg_loss <= theoretical_avg_loss else '❌',
                                            '✅' if actual_days_to_recover <= theoretical_days_to_recover else '❌'
                                        ]
                                    }
                                    
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                                    
                                    # Summary metrics with color
                                    st.markdown("---")
                                    st.markdown("### 🎯 Filter Impact Summary")
                                    
                                    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                                    
                                    return_diff = actual_return_pct - theoretical_return_pct
                                    dd_reduction = theoretical_max_dd - actual_max_dd
                                    vol_reduction = theoretical_volatility - actual_volatility
                                    sharpe_diff = actual_sharpe - theoretical_sharpe
                                    
                                    with sum_col1:
                                        st.metric("Return Impact", f"{return_diff:+.2f}%", 
                                                 delta=f"{return_diff:+.2f}%",
                                                 delta_color="normal" if return_diff >= 0 else "inverse")
                                    with sum_col2:
                                        st.metric("Drawdown Reduced", f"{dd_reduction:+.2f}%", 
                                                 delta=f"{dd_reduction:+.2f}%",
                                                 delta_color="normal" if dd_reduction >= 0 else "inverse")
                                    with sum_col3:
                                        st.metric("Volatility Reduced", f"{vol_reduction:+.2f}%", 
                                                 delta=f"{vol_reduction:+.2f}%",
                                                 delta_color="normal" if vol_reduction >= 0 else "inverse")
                                    with sum_col4:
                                        st.metric("Sharpe Change", f"{sharpe_diff:+.2f}", 
                                                 delta=f"{sharpe_diff:+.2f}",
                                                 delta_color="normal" if sharpe_diff >= 0 else "inverse")
                                    
                                    # Overall assessment
                                    st.markdown("---")
                                    improvements = sum([
                                        1 if actual_max_dd <= theoretical_max_dd else 0,
                                        1 if actual_volatility <= theoretical_volatility else 0,
                                        1 if actual_sharpe >= theoretical_sharpe else 0
                                    ])
                                    
                                    if improvements >= 2 and return_diff >= -5:
                                        st.success(f"✅ **{regime_type} filter improved risk-adjusted returns.** Lower drawdown/volatility with acceptable return trade-off.")
                                    elif return_diff > 0:
                                        st.success(f"✅ **{regime_type} filter improved absolute returns.** Higher returns than without filter.")
                                    else:
                                        st.warning(f"⚠️ **{regime_type} filter reduced returns by {abs(return_diff):.2f}%.** The filter was protective but cost performance in this period.")
                    else:
                        st.warning("No trades generated")
                else:
                    st.error("Data fetch failed")
    
    # STANDALONE BENCHMARK COMPARISON - Persists across reruns using session_state
    if st.session_state.get('current_backtest_active') and 'current_backtest' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Benchmark Comparison")
        
        stored_data = st.session_state['current_backtest']
        stored_engine = stored_data['engine']
        bt_start = stored_data['start_date']
        bt_end = stored_data['end_date']
        
        # Yahoo Finance index mappings - extensive list of available NSE indices
        yahoo_index_map = {
            # Major Indices
            "NIFTY 50": "^NSEI",
            "NIFTY NEXT 50": "^NSMIDCP",
            "NIFTY 100": "^CNX100",
            "NIFTY 200": "^CNX200",
            "NIFTY 500": "^CRSLDX",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY FIN SERVICE": "^CNXFIN",
            "NIFTY IT": "^CNXIT",
            # Midcap & Smallcap
            "NIFTY MIDCAP 50": "^NIFTYMIDCAP50",
            "NIFTY MIDCAP 100": "^CNXMDCP",
            "NIFTY SMLCAP 50": "^NSMALLCAP50",
            "NIFTY SMLCAP 100": "^CNXSC",
            "NIFTY SMLCAP 250": "^NSMALLCAP250",
            # Sectoral
            "NIFTY AUTO": "^CNXAUTO",
            "NIFTY PHARMA": "^CNXPHARMA",
            "NIFTY PSE": "^CNXPSE",
            "NIFTY REALTY": "^CNXREALTY",
            "NIFTY INFRA": "^CNXINFRA",
            "NIFTY ENERGY": "^CNXENERGY",
            "NIFTY FMCG": "^CNXFMCG",
            "NIFTY METAL": "^CNXMETAL",
            "NIFTY COMMODITIES": "^CNXCMDT",
            "NIFTY CONSUMPTION": "^CNXCONSUMD",
            "NIFTY CPSE": "^CNXCPSE",
            "NIFTY MEDIA": "^CNXMEDIA",
            "NIFTY PRIVATE BANK": "^NIFTYPVTBANK",
            "NIFTY PSU BANK": "^CNXPSUBANK",
            # Thematic
            "NIFTY MNC": "^CNXMNC",
            "NIFTY SERV SECTOR": "^CNXSERVICE",
            "NIFTY GROWSECT 15": "^NIFTYGROWSECT15",
            "NIFTY100 QUALITY 30": "^NIFTYQUALLV30",
            "NIFTY50 VALUE 20": "^NIFTY50VALUE20",
            "NIFTY DIVIDEND OPPS 50": "^CNXDIVIDEND",
            # Strategy
            "NIFTY ALPHA 50": "^NIFTYALPHA50",
            "NIFTY HIGH BETA 50": "^NIFTYHIGHBETA50",
            "NIFTY LOW VOLATILITY 50": "^NIFTYLOWVOL50",
        }
        
        benchmark_options = list(yahoo_index_map.keys())
        
        # Get stored selection or default
        stored_benchmark = st.session_state.get('benchmark_selection', 'NIFTY 50')
        try:
            default_idx = benchmark_options.index(stored_benchmark)
        except ValueError:
            default_idx = 0
        
        selected_benchmark = st.selectbox(
            "Select Benchmark Index", 
            benchmark_options,
            index=default_idx,
            key="standalone_benchmark_selector"
        )
        st.session_state['benchmark_selection'] = selected_benchmark
        
        try:
            import yfinance as yf
            benchmark_ticker = yahoo_index_map.get(selected_benchmark, "^NSEI")
            benchmark_data = yf.download(benchmark_ticker, start=bt_start, end=bt_end, progress=False)
            
            if not benchmark_data.empty:
                portfolio_values = stored_engine.portfolio_df['Portfolio Value']
                portfolio_norm = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
                
                benchmark_close = benchmark_data['Close']
                if isinstance(benchmark_close, pd.DataFrame):
                    benchmark_close = benchmark_close.iloc[:, 0]
                benchmark_norm = (benchmark_close / benchmark_close.iloc[0] - 1) * 100
                
                # Calculate drawdowns
                portfolio_cummax = portfolio_values.cummax()
                portfolio_dd = ((portfolio_values - portfolio_cummax) / portfolio_cummax) * 100
                benchmark_cummax = benchmark_close.cummax()
                benchmark_dd = ((benchmark_close - benchmark_cummax) / benchmark_cummax) * 100
                
                # PnL Comparison Chart
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(x=portfolio_norm.index, y=portfolio_norm, name="Portfolio", line=dict(color="#28a745", width=2)))
                fig_pnl.add_trace(go.Scatter(x=benchmark_norm.index, y=benchmark_norm, name=selected_benchmark, line=dict(color="#007bff", width=2)))
                fig_pnl.update_layout(title=f"Cumulative Returns: Portfolio vs {selected_benchmark}", xaxis_title="Date", yaxis_title="Return (%)", height=400, template="plotly_dark")
                st.plotly_chart(fig_pnl, use_container_width=True)
                
                # Drawdown Comparison
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=portfolio_dd.index, y=portfolio_dd, name="Portfolio DD", line=dict(color="#28a745", width=2), fill='tozeroy', fillcolor='rgba(40, 167, 69, 0.2)'))
                fig_dd.add_trace(go.Scatter(x=benchmark_dd.index, y=benchmark_dd, name=f"{selected_benchmark} DD", line=dict(color="#007bff", width=2), fill='tozeroy', fillcolor='rgba(0, 123, 255, 0.2)'))
                fig_dd.update_layout(title=f"Drawdown Comparison", xaxis_title="Date", yaxis_title="Drawdown (%)", height=400, template="plotly_dark")
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Portfolio Return", f"{portfolio_norm.iloc[-1]:.1f}%")
                col2.metric(f"{selected_benchmark} Return", f"{benchmark_norm.iloc[-1]:.1f}%")
                col3.metric("Portfolio Max DD", f"{portfolio_dd.min():.1f}%")
                col4.metric(f"{selected_benchmark} Max DD", f"{benchmark_dd.min():.1f}%")
                
                alpha = portfolio_norm.iloc[-1] - benchmark_norm.iloc[-1]
                if alpha > 0:
                    st.success(f"🎯 **Alpha Generated: +{alpha:.1f}%**")
                else:
                    st.warning(f"📉 **Alpha: {alpha:.1f}%**")
            else:
                st.warning(f"Could not fetch data for {selected_benchmark}")
        except Exception as e:
            st.error(f"Error loading benchmark: {e}")

# ==================== TAB 2: BACKTEST LOGS ====================
with main_tabs[1]:
    st.subheader("Backtest History")
    
    if not st.session_state.backtest_logs:
        st.info("No backtest logs yet. Run a backtest to see results here.")
    else:
        st.markdown(f"**Total Backtests:** {len(st.session_state.backtest_logs)}")
        
        # Display logs in reverse chronological order
        for idx, log in enumerate(reversed(st.session_state.backtest_logs)):
            with st.expander(f"📊 {log['name']} - {log['timestamp']}"):
                st.markdown(f"**Universe:** {log['config']['universe_name']}")
                st.markdown(f"**Period:** {log['config']['start_date']} to {log['config']['end_date']}")
                st.markdown(f"**Formula:** `{log['config']['formula']}`")
                
                # Key metrics - display vertically to avoid nested columns
                metrics = log['metrics']
                st.markdown(f"**Final Value:** ₹{metrics['Final Value']:,.0f} | **CAGR:** {metrics['CAGR %']:.2f}% | **Sharpe:** {metrics['Sharpe Ratio']:.2f} | **Win Rate:** {metrics['Win Rate %']:.1f}%")
                
                # Show additional log info if available
                if log.get('trades'):
                    st.caption(f"📈 {len(log['trades'])} trades recorded")
                if log.get('portfolio_values'):
                    st.caption(f"📊 {len(log['portfolio_values'])} daily values stored")
                
                # Download buttons - Excel and PDF
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    excel_data = create_excel_with_charts(log['config'], metrics)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"{log['name']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{idx}"
                    )
                with dl_col2:
                    pdf_data = create_pdf_report(log['config'], metrics)
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_data,
                        file_name=f"{log['name']}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{idx}"
                    )
        
        # Clear all logs button
        if st.button("🗑️ Clear All Logs"):
            st.session_state.backtest_logs = []
            st.session_state.backtest_engines = {}
            save_backtest_logs([])  # Save empty list to file
            st.experimental_rerun()

# ==================== TAB 3: EXECUTE TRADES ====================
with main_tabs[2]:
    st.subheader("🚀 Execute Trades on Zerodha")
    st.markdown("Execute strategy positions live or paper trade from saved strategy templates.")
    
    # Check if Kite is configured
    if not kite_trader.is_kite_configured():
        st.warning("""
        **Zerodha Trading not configured.**
        
        To enable live trading, add these to your Streamlit secrets:
        ```
        KITE_API_KEY = "your_api_key"
        KITE_API_SECRET = "your_api_secret"
        ```
        
        Get your API credentials from [Kite Connect](https://kite.trade/).
        """)
    else:
        # Check if authenticated
        is_authenticated = st.session_state.get('kite_access_token') is not None
        
        # Login/Logout section
        auth_col1, auth_col2 = st.columns([2, 1])
        with auth_col1:
            if is_authenticated:
                st.success(f"✅ Connected to Zerodha as: **{st.session_state.get('kite_user_name', 'User')}**")
            else:
                st.warning("⚠️ Not logged in to Zerodha. Click below to authenticate.")
        
        with auth_col2:
            if is_authenticated:
                if st.button("🚪 Logout", key="kite_logout_main", use_container_width=True):
                    kite_trader.clear_kite_token_from_hf()
                    st.session_state.kite_access_token = None
                    st.session_state.kite_user_id = None
                    st.session_state.kite_user_name = None
                    st.rerun()
            else:
                login_url = kite_trader.get_login_url()
                st.link_button("🔐 Login to Zerodha", login_url, type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Strategy Mode Selector
        strategy_mode = st.radio(
            "Select Mode",
            ["📈 New Strategy", "📊 View Past Executions"],
            horizontal=True,
            key="exec_strategy_mode"
        )
        
        # ==================== NEW STRATEGY ====================
        if strategy_mode == "📈 New Strategy":
            st.markdown("### 📈 Execute New Strategy")
            
            # Get saved strategy templates
            saved_strategies = strategy_storage.list_strategies() if strategy_storage.is_strategy_storage_configured() else []
            
            if not saved_strategies:
                st.info("📋 No saved strategy templates. Go to **Backtest** tab and save a strategy first.")
            else:
                # Strategy selection
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    selected_template = st.selectbox(
                        "Select Strategy Template",
                        saved_strategies,
                        key="exec_template_select"
                    )
                
                with col2:
                    execution_name = st.text_input(
                        "Execution Name",
                        placeholder="e.g., My Momentum Jan 2026",
                        help="Give a unique name to this execution for tracking",
                        key="exec_name_input"
                    )
                
                # Capital and Run Backtest
                cap_col, btn_col = st.columns([1, 1])
                
                with cap_col:
                    exec_capital = st.number_input(
                        "Capital to Deploy (₹)",
                        min_value=10000,
                        max_value=10000000,
                        value=100000,
                        step=10000,
                        key="exec_capital"
                    )
                
                with btn_col:
                    st.write("")
                    st.write("")
                    run_backtest_clicked = st.button(
                        "🔄 Run Backtest",
                        type="secondary",
                        use_container_width=True,
                        key="exec_run_backtest"
                    )
                
                # Run backtest and show positions
                if run_backtest_clicked and selected_template:
                    # Load strategy config
                    strategy_config = strategy_storage.load_strategy(selected_template)
                    
                    if strategy_config:
                        with st.spinner("Running backtest..."):
                            try:
                                # Get universe
                                universe_name = strategy_config.get('universe', 'NIFTY 100')
                                if universe_name == "Custom":
                                    universe = strategy_config.get('custom_stocks', [])
                                else:
                                    universe = get_universe(universe_name)
                                
                                # Parse dates
                                start_date = datetime.datetime.strptime(
                                    strategy_config.get('start_date', '2020-01-01'), '%Y-%m-%d'
                                ).date()
                                end_date = datetime.datetime.strptime(
                                    strategy_config.get('end_date', datetime.datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d'
                                ).date()
                                
                                # Run engine
                                engine = PortfolioEngine(
                                    universe, start_date, end_date, 
                                    exec_capital, 
                                    data_source="yahoo"
                                )
                                
                                if engine.fetch_data():
                                    # Build configs
                                    rebal_config = {
                                        'frequency': strategy_config.get('rebalance_label', 'Monthly'),
                                        'date': strategy_config.get('rebalance_date', 1),
                                        'day': strategy_config.get('rebal_day'),
                                        'alt_day': strategy_config.get('alt_day_option', 'Next Day')
                                    }
                                    
                                    position_sizing_config = {
                                        'method': strategy_config.get('position_sizing_method', 'equal_weight').lower().replace(' ', '_'),
                                        'use_cap': strategy_config.get('use_max_position_cap', False),
                                        'max_pct': strategy_config.get('max_position_pct', 15)
                                    }
                                    
                                    regime_config = strategy_config.get('regime_config')
                                    uncorrelated_config = strategy_config.get('uncorrelated_config')
                                    
                                    engine.run_rebalance_strategy(
                                        strategy_config.get('formula', '6 Month Performance'),
                                        strategy_config.get('num_stocks', 5),
                                        strategy_config.get('exit_rank', 10),
                                        rebal_config,
                                        regime_config,
                                        uncorrelated_config,
                                        strategy_config.get('reinvest_profits', True),
                                        position_sizing_config
                                    )
                                    
                                    metrics = engine.get_metrics()
                                    
                                    # Store in session state
                                    st.session_state['exec_engine'] = engine
                                    st.session_state['exec_metrics'] = metrics
                                    st.session_state['exec_strategy_config'] = strategy_config
                                    st.session_state['exec_template_name'] = selected_template
                                    
                                    st.success("✅ Backtest complete!")
                                else:
                                    st.error("Failed to fetch data for backtest.")
                            except Exception as e:
                                st.error(f"Backtest error: {e}")
                    else:
                        st.error("Failed to load strategy template.")
                
                # Display positions and execute buttons if backtest was run
                if 'exec_engine' in st.session_state and st.session_state.get('exec_engine'):
                    engine = st.session_state['exec_engine']
                    
                    st.markdown("---")
                    
                    # Get open positions
                    open_positions = engine.get_open_positions() if hasattr(engine, 'get_open_positions') else []
                    
                    if open_positions:
                        st.markdown("### 📊 Current Open Positions")
                        open_df = pd.DataFrame(open_positions)
                        
                        def color_pnl(val):
                            if isinstance(val, (int, float)):
                                if val > 0:
                                    return 'color: #28a745; font-weight: bold'
                                elif val < 0:
                                    return 'color: #dc3545; font-weight: bold'
                            return ''
                        
                        if 'Unrealized ROI %' in open_df.columns:
                            styled_df = open_df.style.map(color_pnl, subset=['Unrealized ROI %'])
                        else:
                            styled_df = open_df
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Calculate orders
                        engine_data = st.session_state.get('engine_data', {})
                        calculated_orders = kite_trader.calculate_order_quantities(
                            open_positions, exec_capital, engine_data, max_position_pct=25.0
                        )
                        
                        if calculated_orders:
                            st.markdown("#### 📋 Order Preview")
                            preview_df = pd.DataFrame(calculated_orders)
                            if 'note' in preview_df.columns:
                                preview_df = preview_df[['tradingsymbol', 'quantity', 'price', 'weight_pct', 'estimated_value', 'note']]
                                preview_df.columns = ['Stock', 'Qty', 'Price (₹)', 'Weight %', 'Est. Value (₹)', 'Note']
                            else:
                                preview_df = preview_df[['tradingsymbol', 'quantity', 'price', 'weight_pct', 'estimated_value']]
                                preview_df.columns = ['Stock', 'Qty', 'Price (₹)', 'Weight %', 'Est. Value (₹)']
                            st.dataframe(preview_df, use_container_width=True, hide_index=True)
                            
                            total_value = sum(o['estimated_value'] for o in calculated_orders)
                            st.caption(f"**Total:** ₹{total_value:,.2f} | **Unused:** ₹{exec_capital - total_value:,.2f}")
                        
                        st.markdown("---")
                        
                        # Execution buttons
                        if not execution_name:
                            st.warning("⚠️ Please enter an Execution Name above before executing.")
                        else:
                            exec_col1, exec_col2 = st.columns(2)
                            
                            with exec_col1:
                                paper_clicked = st.button(
                                    "📄 Paper Execute",
                                    use_container_width=True,
                                    help="Simulate execution without placing real orders",
                                    key="paper_execute_btn"
                                )
                            
                            with exec_col2:
                                if is_authenticated:
                                    live_clicked = st.button(
                                        "🚀 Live Execute",
                                        type="primary",
                                        use_container_width=True,
                                        help="Place real orders on Zerodha",
                                        key="live_execute_btn"
                                    )
                                else:
                                    st.info("🔐 Login to execute live trades")
                                    live_clicked = False
                            
                            # Handle Paper Execute
                            if paper_clicked:
                                with st.spinner("Saving paper execution..."):
                                    # Build trades from orders
                                    trades = []
                                    for order in calculated_orders:
                                        if order['quantity'] > 0:
                                            trades.append({
                                                'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                                                'stock': order['tradingsymbol'],
                                                'action': 'BUY',
                                                'quantity': order['quantity'],
                                                'price': order['price'],
                                                'value': order['estimated_value'],
                                                'status': 'PAPER'
                                            })
                                    
                                    # Start with execution date value only (not full backtest history)
                                    portfolio_values = [{
                                        'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                                        'value': exec_capital
                                    }]
                                    
                                    # Save execution
                                    success = execution_storage.save_execution(
                                        name=execution_name,
                                        strategy_template=st.session_state.get('exec_template_name', selected_template),
                                        mode='paper',
                                        capital=exec_capital,
                                        trades=trades,
                                        portfolio_values=portfolio_values,
                                        open_positions=open_positions
                                    )
                                    
                                    if success:
                                        st.success(f"✅ Paper execution saved: **{execution_name}**")
                                        st.balloons()
                                    else:
                                        st.error("Failed to save paper execution.")
                            
                            # Handle Live Execute
                            if live_clicked:
                                with st.spinner("Placing orders on Zerodha..."):
                                    result = kite_trader.execute_orders_on_kite(calculated_orders, dry_run=False)
                                    
                                    # Build trades from result
                                    trades = []
                                    for order in result.get('orders_placed', []):
                                        trades.append({
                                            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                                            'stock': order['tradingsymbol'],
                                            'action': 'BUY',
                                            'quantity': order['quantity'],
                                            'order_id': order.get('order_id', ''),
                                            'status': 'PLACED'
                                        })
                                    
                                    for order in result.get('orders_failed', []):
                                        trades.append({
                                            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                                            'stock': order['tradingsymbol'],
                                            'action': 'BUY',
                                            'quantity': order['quantity'],
                                            'status': 'FAILED',
                                            'error': order.get('message', '')
                                        })
                                    
                                    # Start with execution date value only (not full backtest history)
                                    portfolio_values = [{
                                        'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                                        'value': exec_capital
                                    }]
                                    
                                    # Save execution
                                    execution_storage.save_execution(
                                        name=execution_name,
                                        strategy_template=st.session_state.get('exec_template_name', selected_template),
                                        mode='live',
                                        capital=exec_capital,
                                        trades=trades,
                                        portfolio_values=portfolio_values,
                                        open_positions=open_positions
                                    )
                                    
                                    if result['success']:
                                        st.success(f"✅ {result['message']} - Execution saved as **{execution_name}**")
                                        if result['orders_placed']:
                                            st.markdown("**Orders Placed:**")
                                            for order in result['orders_placed']:
                                                st.write(f"• {order['tradingsymbol']}: {order['quantity']} shares - ID: {order.get('order_id', 'N/A')}")
                                    else:
                                        st.error(f"❌ {result['message']}")
                                    
                                    if result['orders_failed']:
                                        st.markdown("**Failed Orders:**")
                                        for order in result['orders_failed']:
                                            st.error(f"• {order['tradingsymbol']}: {order['message']}")
                    else:
                        st.info("No open positions from backtest.")
        
        # ==================== VIEW PAST EXECUTIONS ====================
        else:
            st.markdown("### 📊 Past Executions")
            
            # Get saved executions
            executions = execution_storage.list_executions() if execution_storage.is_execution_storage_configured() else []
            
            if not executions:
                st.info("📋 No past executions found. Execute a strategy first.")
            else:
                # Build dropdown options
                exec_options = [f"{e['name']} ({e['mode'].upper()}, {e['created_at'][:10]})" for e in executions]
                exec_names = [e['name'] for e in executions]
                
                selected_exec_idx = st.selectbox(
                    "Select Execution",
                    range(len(exec_options)),
                    format_func=lambda i: exec_options[i],
                    key="past_exec_select"
                )
                
                if selected_exec_idx is not None:
                    selected_exec_name = exec_names[selected_exec_idx]
                    execution_data = execution_storage.load_execution(selected_exec_name)
                    
                    if execution_data:
                        # Execution summary
                        st.markdown("---")
                        info_cols = st.columns(4)
                        info_cols[0].metric("Mode", execution_data.get('mode', '').upper())
                        info_cols[1].metric("Capital", f"₹{execution_data.get('capital', 0):,.0f}")
                        info_cols[2].metric("Template", execution_data.get('strategy_template', 'N/A'))
                        info_cols[3].metric("Trades", len(execution_data.get('trades', [])))
                        
                        st.caption(f"Created: {execution_data.get('created_at', '')[:19]} | Last Updated: {execution_data.get('last_updated', '')[:19]}")
                        
                        # Trade log
                        trades = execution_data.get('trades', [])
                        if trades:
                            st.markdown("### 📝 Trade Log")
                            trades_df = pd.DataFrame(trades)
                            st.dataframe(trades_df, use_container_width=True, hide_index=True)
                        
                        # Portfolio values / charts
                        portfolio_values = execution_data.get('portfolio_values', [])
                        if portfolio_values and len(portfolio_values) > 1:
                            st.markdown("### 📈 Performance Charts")
                            
                            pv_df = pd.DataFrame(portfolio_values)
                            pv_df['date'] = pd.to_datetime(pv_df['date'])
                            pv_df = pv_df.sort_values('date')
                            
                            chart_tabs = st.tabs(["Equity Curve", "Drawdown"])
                            
                            with chart_tabs[0]:
                                fig_equity = go.Figure()
                                fig_equity.add_trace(go.Scatter(
                                    x=pv_df['date'],
                                    y=pv_df['value'],
                                    fill='tozeroy',
                                    line_color='#28a745',
                                    name='Portfolio Value'
                                ))
                                fig_equity.update_layout(
                                    title="Equity Curve",
                                    xaxis_title="Date",
                                    yaxis_title="Portfolio Value (₹)",
                                    height=350,
                                    margin=dict(l=0,r=0,t=40,b=0),
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig_equity, use_container_width=True)
                            
                            with chart_tabs[1]:
                                # Calculate drawdown
                                pv_df['running_max'] = pv_df['value'].cummax()
                                pv_df['drawdown'] = (pv_df['value'] - pv_df['running_max']) / pv_df['running_max'] * 100
                                
                                fig_dd = go.Figure()
                                fig_dd.add_trace(go.Scatter(
                                    x=pv_df['date'],
                                    y=pv_df['drawdown'],
                                    fill='tozeroy',
                                    line_color='#dc3545',
                                    name='Drawdown'
                                ))
                                fig_dd.update_layout(
                                    title="Drawdown",
                                    xaxis_title="Date",
                                    yaxis_title="Drawdown %",
                                    height=350,
                                    margin=dict(l=0,r=0,t=40,b=0),
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig_dd, use_container_width=True)
                        
                        # Delete button
                        st.markdown("---")
                        if st.button("🗑️ Delete Execution", key="delete_exec_btn"):
                            if execution_storage.delete_execution(selected_exec_name):
                                st.success(f"✅ Deleted: {selected_exec_name}")
                                st.rerun()
                    else:
                        st.error("Failed to load execution data.")

# ==================== TAB 4: DATA DOWNLOAD ====================
with main_tabs[3]:
    st.subheader("📥 Data Download")
    st.markdown("Download historical data for all universes. This is a one-time setup - data will be cached for fast backtests.")

    from portfolio_engine import DataCache
    cache = DataCache()
    cache_info = cache.get_cache_info()

    # Show cache status
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("Cached Stocks", cache_info['total_files'])
    col2.metric("Storage Used", f"{cache_info['total_size_mb']:.2f} MB")
    
    with col3:
        st.write("")  # Spacer
        if st.button("🗑️ Clear All Cache", type="secondary", key="clear_cache"):
            cache.clear()
            st.success("✅ Cache cleared! Please refresh the page.")
            st.rerun()

    st.markdown("---")
    
    # Refresh Universes from NSE
    st.markdown("### 🔄 Refresh Universe Constituents")
    st.info("Fetch live index constituents from NSE India. This updates the stock lists for all universes.")
    
    refresh_col1, refresh_col2 = st.columns([1, 3])
    
    with refresh_col1:
        if st.button("🔄 Refresh Universes", type="secondary", key="refresh_universes"):
            try:
                from nse_fetcher import refresh_universes, load_from_cache
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                success, message = refresh_universes(progress_callback)
                
                progress_bar.progress(1.0)
                status_text.text("Complete!")
                
                if success:
                    cached, timestamp = load_from_cache()
                    st.success(f"✅ {message}. Cached at: {timestamp}")
                    
                    # Show summary - only for our specified indexes
                    from nifty_universe import INDEX_NAMES
                    with st.expander("Universe Summary"):
                        for name in INDEX_NAMES:
                            if name in cached:
                                st.write(f"**{name}**: {len(cached[name])} stocks")
                            else:
                                st.write(f"**{name}**: Not in cache")
                else:
                    # Show cached data info even if refresh failed
                    cached, timestamp = load_from_cache()
                    if cached:
                        st.warning(f"⚠️ Live refresh failed (NSE blocks cloud servers). Using cached data from: {timestamp} ({len(cached)} universes)")
                    else:
                        st.error(f"❌ {message}")
            except Exception as e:
                # Check if we have cached data to fall back to
                try:
                    from nse_fetcher import load_from_cache
                    cached, timestamp = load_from_cache()
                    if cached:
                        st.warning(f"⚠️ NSE blocks cloud requests. Using pre-loaded cache: {timestamp} ({len(cached)} universes)")
                    else:
                        st.error(f"❌ Error refreshing: {e}. Run locally: python nse_fetcher.py")
                except:
                    st.error(f"❌ Error: {e}")
    
    with refresh_col2:
        # Show current cache status
        try:
            from nifty_universe import INDEX_NAMES
            from nse_fetcher import load_from_cache
            cached, timestamp = load_from_cache()
            st.write(f"📊 Active indexes: **{len(INDEX_NAMES)}**")
            if cached and timestamp:
                st.write(f"📅 Cache updated: **{timestamp}**")
        except:
            st.write("⚠️ Universe data not initialized.")
    
    st.markdown("---")

    # Download All Data Button
    st.markdown("### 🔽 Download All Universe Data")
    st.info("This will download and cache data for ALL stocks across ALL universes. Takes ~10-15 minutes.")
    
    # Clear cache option
    col_clear, col_download = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Cache First", key="clear_data_cache_btn"):
            from portfolio_engine import DataCache
            cache = DataCache()
            cache.clear()
            st.success("✅ Cache cleared! Now click 'Download All Data' to get fresh data.")
            st.rerun()
    
    with col_download:
        download_clicked = st.button("📥 Download All Data", type="primary", key="download_all_data_btn")

    if download_clicked:
        # Get all unique tickers from all universes
        all_tickers = set()
        all_universe_names = get_all_universe_names()

        for universe_name in all_universe_names:
            universe = get_universe(universe_name)
            all_tickers.update(universe)

        all_tickers = sorted(list(all_tickers))

        st.markdown(f"### Downloading {len(all_tickers)} unique stocks...")

        # Progress display
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        def download_progress(current, total, ticker, remaining_seconds):
            pct = (current / total) if total > 0 else 0
            progress_bar.progress(min(pct, 1.0))

            mins = int(remaining_seconds // 60)
            secs = int(remaining_seconds % 60)
            elapsed = time.time() - start_time
            elapsed_mins = int(elapsed // 60)
            elapsed_secs = int(elapsed % 60)

            status_text.markdown(f"""
            <div style="padding: 10px; background: rgba(0,255,136,0.1); border-radius: 5px;">
                <div style="font-size: 16px; font-weight: bold;">📊 {ticker}</div>
                <div>Progress: {current}/{total} ({pct*100:.1f}%)</div>
                <div>⏱️ Remaining: {mins:02d}:{secs:02d} | ⏰ Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}</div>
            </div>
            """, unsafe_allow_html=True)

        temp_engine = PortfolioEngine(all_tickers, datetime.date(2020, 1, 1), datetime.date.today())
        success_count = temp_engine.download_and_cache_universe(all_tickers, download_progress, None)

        progress_bar.empty()
        status_text.empty()

        total_time = time.time() - start_time
        st.success(f"✅ Downloaded {success_count}/{len(all_tickers)} stocks in {int(total_time)}s!")
        st.balloons()

    # ===== BROKER API DATA DOWNLOAD =====
    st.markdown("---")
    st.markdown("### 📊 Broker API Data (Dhan)")
    st.info("Download historical data from Dhan API and store in Hugging Face for use in backtests. This provides more accurate data than Yahoo Finance.")
    
    # Dhan API Authentication Check Button
    if st.button("🔐 Check Dhan API Access", key="check_dhan_api_btn"):
        with st.spinner("Checking Dhan API credentials..."):
            try:
                from config import validate_credentials, get_dhan_client, DHAN_CLIENT_ID
                
                # Step 1: Check credentials are configured
                validate_credentials()
                st.success(f"✅ Credentials configured (Client ID: {DHAN_CLIENT_ID[:4]}...{DHAN_CLIENT_ID[-4:]})")
                
                # Step 2: Try to create client and fetch test data
                dhan = get_dhan_client()
                st.success("✅ Dhan client created successfully")
                
                # Step 3: Try a test API call (historical data for RELIANCE)
                from datetime import date, timedelta
                test_date = date.today() - timedelta(days=7)
                response = dhan.historical_daily_data(
                    security_id="1333",  # RELIANCE security ID
                    exchange_segment='NSE_EQ',
                    instrument_type='EQUITY',
                    from_date=test_date.strftime('%Y-%m-%d'),
                    to_date=date.today().strftime('%Y-%m-%d')
                )
                
                if response.get('status') == 'success':
                    data = response.get('data', {})
                    data_points = len(data.get('timestamp', []))
                    st.success(f"✅ API test passed! Fetched {data_points} data points for RELIANCE")
                    st.balloons()
                else:
                    st.error(f"❌ API test failed: {response.get('remarks', 'Unknown error')}")
                    
            except ValueError as e:
                st.error(f"❌ Credentials not configured: {e}")
                st.info("Add DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN to your .env file")
            except ImportError as e:
                st.error(f"❌ dhanhq SDK not installed: {e}")
                st.code("pip install dhanhq", language="bash")
            except Exception as e:
                st.error(f"❌ API Error: {e}")
    
    st.markdown("")  # Spacing
    
    # Check HF configuration
    from huggingface_manager import is_hf_configured
    hf_configured = is_hf_configured()
    
    if not hf_configured:
        st.warning("""
        ⚠️ **Hugging Face not configured.** To use Broker API data:
        1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
        2. Get a write access token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
        3. Create a new dataset repository
        4. Add to your `.env` file:
           ```
           HF_TOKEN=your_token_here
           HF_DATASET_REPO=your-username/nse-dhan-ohlc
           ```
        """)
    else:
        # Show current HF dataset status
        from huggingface_manager import HuggingFaceManager
        try:
            hf = HuggingFaceManager()
            available_symbols = hf.list_available_symbols()
            st.success(f"✅ Hugging Face connected. **{len(available_symbols)}** symbols available.")
        except Exception as e:
            st.error(f"HuggingFace connection error: {e}")
            available_symbols = []
        
        # Download settings
        dhan_col1, dhan_col2 = st.columns(2)
        with dhan_col1:
            dhan_from_date = st.date_input("From Date", datetime.date(2020, 1, 1), key="dhan_from")
        with dhan_col2:
            dhan_to_date = st.date_input("To Date", datetime.date.today(), key="dhan_to")
        
        if st.button("📥 Download Broker API Data", type="primary", key="download_dhan_data_btn"):
            # Get all unique tickers from all universes
            all_tickers = set()
            all_universe_names = get_all_universe_names()
            for universe_name in all_universe_names:
                uni = get_universe(universe_name)
                all_tickers.update(uni)
            all_tickers = sorted(list(all_tickers))
            
            st.markdown(f"### Syncing {len(all_tickers)} stocks with Hugging Face...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            def dhan_progress(current, total, symbol, status):
                pct = current / total if total > 0 else 0
                progress_bar.progress(min(pct, 1.0))
                
                elapsed = time.time() - start_time
                elapsed_mins = int(elapsed // 60)
                elapsed_secs = int(elapsed % 60)
                
                status_text.markdown(f"""
                <div style="padding: 10px; background: rgba(0,255,136,0.1); border-radius: 5px;">
                    <div style="font-size: 16px; font-weight: bold;">📊 {symbol}</div>
                    <div>Progress: {current}/{total} ({pct*100:.1f}%)</div>
                    <div>Status: {status}</div>
                    <div>⏰ Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}</div>
                </div>
                """, unsafe_allow_html=True)
            
            try:
                hf = HuggingFaceManager()
                success_count = hf.sync_all_symbols(
                    symbols=all_tickers,
                    from_date=dhan_from_date,
                    to_date=dhan_to_date,
                    progress_callback=dhan_progress
                )
                
                progress_bar.empty()
                status_text.empty()
                
                total_time = time.time() - start_time
                st.success(f"✅ Synced {success_count}/{len(all_tickers)} stocks in {int(total_time)}s!")
                st.balloons()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Download failed: {e}")


# ==================== TAB 5: DHAN AUTH ====================
with main_tabs[4]:
    st.markdown("### 🔐 Dhan API Authentication")

    from config import (
        get_saved_credentials, save_credentials_to_env,
        authenticate_dhan, _is_streamlit_cloud,
    )

    _on_cloud = _is_streamlit_cloud()
    creds     = get_saved_credentials()

    # ── Environment banner ───────────────────────────────────────────
    if _on_cloud:
        st.info(
            "🌥️ **Running on Streamlit Cloud**  \n"
            "Client ID and PIN are read from **Streamlit Secrets** (permanent).  \n"
            "Access Token is stored in the **session** only — re-enter TOTP after each restart."
        )
    else:
        st.info(
            "💻 **Running locally**  \n"
            "Credentials are read from your **.env file** (permanent for Client ID/PIN).  \n"
            "Access Token is also saved to .env and persists until Dhan invalidates it (~24h)."
        )

    # ── Step 1: Credentials Setup ──────────────────────────────────────
    st.markdown("---")

    if _on_cloud:
        # On Streamlit Cloud: show what's configured in secrets, no save UI
        st.markdown("#### ⚙️ Step 1 — Streamlit Secrets Configuration")
        st.markdown("""
        Client ID and PIN are configured in your app's **Streamlit Secrets** — not entered here.
        To update them:
        1. Go to your Streamlit Cloud dashboard
        2. Click your app → **⋮ → Settings → Secrets**
        3. Add / edit the values in TOML format (see below)
        4. Click **Save** — the app restarts with new values
        """)

        with st.expander("📋 Required Streamlit Secrets format", expanded=not bool(creds.get('client_id'))):
            st.code("""
# Paste this into Streamlit Cloud → App Settings → Secrets

DHAN_CLIENT_ID = "1000000001"
DHAN_PIN       = "12345"

# Optional: set a long-lived token here so you don't need TOTP every restart
# Leave blank to always use TOTP to generate a fresh token
DHAN_ACCESS_TOKEN = ""
""", language="toml")

        # Status
        if creds.get('client_id'):
            cid     = creds['client_id']
            masked  = cid[:3] + '*' * max(0, len(cid) - 6) + cid[-3:] if len(cid) > 6 else '***'
            st.success(f"✅ Client ID found in Secrets: `{masked}`")
        else:
            st.error("❌ DHAN_CLIENT_ID not found in Streamlit Secrets. Add it via the dashboard.")

        if creds.get('pin'):
            st.success("✅ PIN found in Secrets")
        else:
            st.error("❌ DHAN_PIN not found in Streamlit Secrets.")

    else:
        # Local: show editable fields that save to .env
        st.markdown("#### ⚙️ Step 1 — Save Client ID & PIN (one-time)")
        st.caption("Saved permanently to your .env file. You never need to enter these again.")

        c1, c2 = st.columns(2)
        with c1:
            input_client_id = st.text_input(
                "Dhan Client ID",
                value=creds.get('client_id', ''),
                key="dhan_auth_client_id",
                placeholder="e.g. 1000000001",
                help="Your Dhan trading account client ID (Profile → Account Info)"
            )
        with c2:
            input_pin = st.text_input(
                "PIN",
                value=creds.get('pin', ''),
                type="password",
                key="dhan_auth_pin",
                placeholder="Your 5-digit Dhan PIN",
                help="Your Dhan account login PIN (stored in .env, never in git)"
            )

        if st.button("💾 Save to .env", key="save_dhan_creds_btn"):
            if input_client_id.strip() and input_pin.strip():
                save_credentials_to_env(
                    client_id=input_client_id.strip(),
                    pin=input_pin.strip()
                )
                st.success("✅ Saved to .env file permanently.")
                st.rerun()
            else:
                st.error("❌ Enter both Client ID and PIN.")

        if creds.get('client_id'):
            cid    = creds['client_id']
            masked = cid[:3] + '*' * max(0, len(cid) - 6) + cid[-3:] if len(cid) > 6 else '***'
            st.success(f"✅ Client ID saved: `{masked}`")
        else:
            st.warning("⚠️ Client ID not set yet")

    # ── Step 2: TOTP Authentication ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔢 Step 2 — Authenticate with TOTP")

    # Show token status
    fresh_creds = get_saved_credentials()
    cur_token   = fresh_creds.get('access_token', '')
    if cur_token:
        masked_tok = cur_token[:8] + '...' + cur_token[-4:]
        if _on_cloud:
            st.success(f"✅ Token active this session: `{masked_tok}` ({len(cur_token)} chars)")
        else:
            st.success(f"✅ Token saved in .env: `{masked_tok}` ({len(cur_token)} chars)")
    else:
        st.warning("⚠️ No active token. Enter TOTP below to generate one.")

    if _on_cloud:
        st.caption(
            "🔄 On Streamlit Cloud, tokens are stored **in your session only** and lost on restart. "
            "Open your authenticator app and enter a fresh TOTP to re-authenticate after each restart."
        )
    else:
        st.caption(
            "💾 Locally, the token is saved to .env after auth and reloaded automatically until Dhan expires it (~24h)."
        )

    totp_col, btn_col = st.columns([2, 1])
    with totp_col:
        totp_input = st.text_input(
            "TOTP",
            key="dhan_totp_input",
            max_chars=6,
            placeholder="6-digit code from Google Authenticator",
            label_visibility="collapsed"
        )
    with btn_col:
        auth_btn = st.button(
            "🔑 Authenticate", type="primary",
            key="dhan_auth_btn", use_container_width=True
        )

    if auth_btn:
        if not totp_input or len(totp_input.strip()) != 6 or not totp_input.strip().isdigit():
            st.error("❌ Enter a valid 6-digit TOTP.")
        else:
            with st.spinner("Authenticating with Dhan..."):
                result = authenticate_dhan(totp_input.strip())
            if result['success']:
                st.success(result['message'])
                st.balloons()
                # Sync the new token into os.environ immediately
                bootstrap_env_from_secrets()
            else:
                st.error(f"❌ {result['message']}")

    # ── Step 3: Connection Test ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📡 Step 3 — Test Connection")

    if st.button("🔍 Test Dhan API", key="dhan_test_conn_btn"):
        with st.spinner("Testing..."):
            try:
                from config import get_dhan_client, validate_credentials
                validate_credentials()
                dhan_test = get_dhan_client()
                st.success("✅ Dhan client created.")

                from datetime import date as _date, timedelta
                resp = dhan_test.historical_daily_data(
                    security_id="1333",
                    exchange_segment="NSE_EQ",
                    instrument_type="EQUITY",
                    from_date=(_date.today() - timedelta(days=10)).strftime('%Y-%m-%d'),
                    to_date=_date.today().strftime('%Y-%m-%d'),
                )
                if isinstance(resp, dict) and resp.get('status') == 'success':
                    pts = len(resp.get('data', {}).get('timestamp', []))
                    st.success(f"✅ Equity API: {pts} data points (RELIANCE)")
                else:
                    st.warning(f"⚠️ Equity API: {str(resp)[:200]}")

                st.markdown("**🛡️ Rolling Options (Put Hedge)**")
                try:
                    o_resp = dhan_test.rolling_options_data(
                        exchange_segment="NSE_FNO", instrument="OPTIDX",
                        drvOptionType="PUT",
                        fromDate=(_date.today() - timedelta(days=7)).strftime('%Y-%m-%d'),
                        toDate=_date.today().strftime('%Y-%m-%d'),
                        strike="ATM", expiryType="WEEKLY",
                    )
                    if isinstance(o_resp, dict) and o_resp.get('status') == 'success':
                        n = len(o_resp.get('data', {}).get('timestamp', []))
                        st.success(f"✅ Rolling Options API: {n} data points")
                    else:
                        st.warning(f"⚠️ Rolling Options: {str(o_resp)[:300]}")
                except AttributeError:
                    st.info("⚠️ SDK missing rolling_options_data — direct REST will be used during backtest.")
                except Exception as oe:
                    st.warning(f"⚠️ Rolling Options: {oe}")

            except ValueError as ve:
                st.error(f"❌ {ve}")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
