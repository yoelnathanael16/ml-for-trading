import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import requests
from datetime import datetime
from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import prepare_features_and_labels, walk_forward_split
from src.models.model_wrappers import ModelWrapper
from src.models.backtester import run_advanced_backtest
from src.models.portfolio_sizing import calculate_equal_weights, calculate_risk_parity_weights, calculate_mvo_weights
from src.data_ingestion import fetch_stock_data
from src.models.regime_hmm import HMMRegimeDetector
from src.models.market_regime import MarketRegimeDetector
from src.models.volatility_garch import GARCHVolatilityModel
from src.models.risk_model import TailRiskModel
from src.models.anomaly_detector import MarketAnomalyDetector
from src.models.mean_reversion import MeanReversionDetector
from src.models.explainability import ModelExplainer

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

def api_get(endpoint: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def api_post(endpoint: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=body, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Dark background style for matplotlib
plt.style.use('dark_background')

st.set_page_config(page_title="ML Research: Advanced Trading & Portfolio Dashboard", layout="wide")

# Custom Premium Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f141c
    }
    .metric-card {
        background-color: #1a2332;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3d54;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00d2d3;
    }
    .metric-val {
        font-size: 28px;
        font-weight: bold;
        color: #00d2d3;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Task-Oriented Benchmarking: Traditional ML in Stock Markets")
st.markdown("""
Dashboard interaktif ini menyajikan evaluasi model Machine Learning berdasarkan **Kinerja Finansial Nyata** (Next Steps: Market Regime, Position Sizing, Entry/Exit, dan Portfolio Sizing) di atas metrik akurasi statistik dasar.
""")

# Sidebar
st.sidebar.header("Configuration Panel")
ticker = st.sidebar.selectbox("Select Core Ticker", ["AAPL", "GOOGL", "MSFT", "TSLA"])

# Helper function to get or download stock data
def get_or_download_data(t, start_date="2020-01-01", end_date="2026-05-11"):
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    df = None
    # Try to find existing file
    files = [f for f in os.listdir(raw_dir) if f.startswith(f"{t}_") and f.endswith(".parquet")]
    if files:
        df = pd.read_parquet(os.path.join(raw_dir, files[0]))
    else:
        # If not found, download it
        filepath = fetch_stock_data(t, start_date, end_date, output_dir=raw_dir, allow_synthetic_fallback=True)
        if filepath and os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            
    if df is not None:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            new_cols = []
            for col in df.columns:
                if col[0] in standard_cols:
                    new_cols.append(col[0])
                elif len(col) > 1 and col[1] in standard_cols:
                    new_cols.append(col[1])
                else:
                    new_cols.append(col[0])
            df.columns = new_cols
        return df
    return None


# ── Columns to drop when building the technical-indicator feature matrix ──────
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]


# ── Local-compute helpers (no API required) ───────────────────────────────────
# Each is @st.cache_data so expensive fits (GARCH, HMM, SHAP) run only once
# per (ticker, df) pair and are retrieved instantly on subsequent reruns.
# Every helper returns a plain dict and never raises — failures return {}.


@st.cache_data(show_spinner=False)
def compute_hmm_local(ticker: str, df: pd.DataFrame) -> dict:
    """Fit HMM + GMM regimes locally from df. Returns {} on failure."""
    try:
        X = df[["Log_Returns", "Volatility"]].dropna().values
        if len(X) < 30:
            return {}
        # HMM — always fit fresh from price data
        det = HMMRegimeDetector()
        det.fit(X)
        preds = det.predict(X)
        regime_hmm = det.get_regime_names()[int(preds[-1])]
        tm = det.get_transition_matrix().tolist()
        # GMM — use pre-trained artifact if available, fall back to local fit
        regime_gmm = None
        gmm_path = os.path.join("models", f"regime_detector_{ticker}.joblib")
        if os.path.exists(gmm_path):
            try:
                gmm_det = joblib.load(gmm_path)
                feat = df[["Log_Returns", "Volatility"]].dropna().iloc[[-1]].values
                regime_gmm = gmm_det.predict_regime_name(feat)[0]
            except Exception:
                pass
        if regime_gmm is None:
            gmm_det_local = MarketRegimeDetector()
            gmm_det_local.fit(X)
            feat = df[["Log_Returns", "Volatility"]].dropna().iloc[[-1]].values
            regime_gmm = gmm_det_local.predict_regime_name(feat)[0]
        return {
            "regime_gmm": regime_gmm,
            "regime_hmm": regime_hmm,
            "hmm_transition_matrix": tm,
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def compute_volatility_local(ticker: str, df: pd.DataFrame) -> dict:
    """Fit GARCH and extract rolling vol locally. Returns {} on failure."""
    try:
        returns = df["Log_Returns"].dropna()
        if len(returns) < 30:
            return {}
        m = GARCHVolatilityModel()
        m.fit(returns)
        garch_vol = m.forecast(horizon=1)
        # Annualize the 20-day rolling daily std stored in the Volatility column
        rolling_vol = float(df["Volatility"].dropna().iloc[-1]) * np.sqrt(252)
        return {"garch_vol": garch_vol, "rolling_vol": rolling_vol}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def compute_risk_local(ticker: str, df: pd.DataFrame) -> dict:
    """Compute CVaR/VaR tail-risk metrics locally. Returns {} on failure."""
    try:
        returns = df["Log_Returns"].dropna()
        if len(returns) < 30:
            return {}
        m = TailRiskModel()
        m.fit(returns)
        return m.compute()
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def compute_anomaly_local(ticker: str, df: pd.DataFrame) -> dict:
    """Fit IsolationForest and score the latest row. Returns {} on failure."""
    try:
        feat_df = df.drop(columns=[c for c in _OHLCV_COLS if c in df.columns]).dropna()
        if len(feat_df) < 30:
            return {}
        X = feat_df.values
        det = MarketAnomalyDetector()
        det.fit(X)
        flag = det.is_anomaly(X[-1])
        score = float(det.score_samples(X[-1:])[0])
        return {"anomaly_flag": flag, "anomaly_score": score}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def compute_mean_reversion_local(ticker: str, df: pd.DataFrame) -> dict:
    """Run MeanReversionDetector on Close prices. Returns {} on failure."""
    try:
        prices = df["Close"].dropna()
        if len(prices) < 30:
            return {}
        return MeanReversionDetector().predict(prices)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def compute_explain_local(ticker: str, model_choice: str, df: pd.DataFrame) -> dict:
    """Compute SHAP importances using on-disk scaler + model. Returns {} if unavailable."""
    try:
        scaler_path = os.path.join("models", f"scaler_{ticker}.joblib")
        model_path = os.path.join("models", f"{model_choice}_{ticker}.joblib")
        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            return {}
        scaler = joblib.load(scaler_path)
        raw_model = joblib.load(model_path)
        feat_df = df.drop(columns=[c for c in _OHLCV_COLS if c in df.columns]).dropna()
        if feat_df.empty:
            return {}
        # SVM KernelExplainer is expensive — cap rows to stay responsive
        n_rows = 50 if model_choice == "SVM" else 250
        X_scaled = scaler.transform(feat_df.values[-n_rows:])
        feature_names = list(feat_df.columns)
        explainer = ModelExplainer(model_choice, raw_model, feature_names)
        importances = explainer.get_feature_importance(X_scaled)
        return {"feature_importances": importances}
    except Exception:
        return {}


# Load Core Data
df_core = get_or_download_data(ticker)

if df_core is None:
    st.error(f"Gagal memuat data saham untuk {ticker}.")
    st.stop()

# Ensure indicators exist
df_core_indicators = add_technical_indicators(df_core)

# Status Banner
status_data = api_get("/status")
if status_data:
    ticker_status = status_data.get(ticker, {})
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        last_refresh = ticker_status.get("last_refresh") or "Never"
        st.metric("Last Data Refresh", last_refresh)
    with col_s2:
        last_retrain = ticker_status.get("last_retrain") or "Never"
        st.metric("Last Model Retrain", last_retrain)
    with col_s3:
        anomaly = ticker_status.get("anomaly_flag")
        if anomaly is True:
            st.metric("Market Status", "⚠️ ANOMALY DETECTED")
        elif anomaly is False:
            st.metric("Market Status", "✅ Normal")
        else:
            st.metric("Market Status", "—")
    st.divider()

# Setup tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Model Benchmarks",
    "📈 Market Regimes (GMM)",
    "⚙️ Trading Simulator",
    "💼 Portfolio Allocation",
    "🔮 Regime (HMM)",
    "📉 Risk & Volatility",
    "🔄 Mean Reversion",
    "🔍 Explainability"
])

# ==========================================
# TAB 1: MODEL BENCHMARKS
# ==========================================
with tab1:
    st.header(f"Model Benchmarking Results: {ticker}")
    benchmarking_file = f"data/processed/{ticker}_benchmarking_results.csv"
    
    if os.path.exists(benchmarking_file):
        results_df = pd.read_csv(benchmarking_file, index_col=0)
        
        # Display table
        st.subheader("Base vs. Advanced Strategy Performance Matrix")
        st.markdown("""
        * **Base (Baseline):** Strategi dasar buy/sell konstan 100% tanpa Stop Loss / Sizing.
        * **Adv (Advanced):** Strategi lanjutan dengan Kelly Sizing, Stop Loss (1.5%), Profit Target (3.0%), Trailing Stop (2.0%), dan Trend Filter (SMA 50).
        """)
        st.dataframe(results_df.style.highlight_max(axis=0, subset=[
            'Total Return (Base)', 'Total Return (Adv)', 
            'Sharpe Ratio (Base)', 'Sharpe Ratio (Adv)'
        ]))
        
        # Barplot comparisons
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total Return Comparison")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_data = results_df[['Total Return (Base)', 'Total Return (Adv)']].reset_index()
            plot_data_melt = plot_data.melt(id_vars='index', var_name='Strategy', value_name='Return')
            sns.barplot(x='index', y='Return', hue='Strategy', data=plot_data_melt, ax=ax, palette="viridis")
            plt.xticks(rotation=0)
            plt.xlabel("Model")
            plt.ylabel("Total Return")
            st.pyplot(fig)
            plt.close(fig)
            
        with col2:
            st.subheader("Sharpe Ratio Comparison")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_data = results_df[['Sharpe Ratio (Base)', 'Sharpe Ratio (Adv)']].reset_index()
            plot_data_melt = plot_data.melt(id_vars='index', var_name='Strategy', value_name='Sharpe')
            sns.barplot(x='index', y='Sharpe', hue='Strategy', data=plot_data_melt, ax=ax, palette="magma")
            plt.xticks(rotation=0)
            plt.xlabel("Model")
            plt.ylabel("Sharpe Ratio")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning(f"Hasil benchmarking untuk {ticker} belum tersedia. Silakan jalankan pipeline pelatihan terlebih dahulu.")
        if st.button("Jalankan Pipeline Pelatihan Sekarang"):
            with st.spinner("Melatih model dan mengevaluasi... (Ini memerlukan waktu beberapa menit untuk ARIMA)"):
                try:
                    from src.train_benchmark import run_benchmarking
                    run_benchmarking(ticker, "2020-01-01", "2026-05-11")
                    st.success("Pelatihan selesai! Silakan muat ulang halaman.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melatih model: {e}")

# ==========================================
# TAB 2: MARKET REGIMES (GMM)
# ==========================================
with tab2:
    st.header("Gaussian Mixture Model (GMM) Market Regime Detection")
    regime_model_path = f"models/regime_detector_{ticker}.joblib"
    
    if os.path.exists(regime_model_path):
        regime_detector = joblib.load(regime_model_path)
        
        # Predict regime for the whole core data
        regime_features = df_core_indicators[['Log_Returns', 'Volatility']].dropna()
        regimes = regime_detector.predict(regime_features)
        regime_names = regime_detector.predict_regime_name(regime_features)
        
        # Add to df
        df_regime = df_core_indicators.loc[regime_features.index].copy()
        df_regime['Regime'] = regimes
        df_regime['Regime_Name'] = regime_names
        
        # Plots
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Price Chart Colored by Detected Market Regime")
            fig, ax = plt.subplots(figsize=(12, 5.5))
            ax.plot(df_regime.index, df_regime['Close'], color='#a4b0be', alpha=0.5, label='Price Curve')
            
            # Map colors: 0 (Low Vol) -> Green, 1 (Med Vol) -> Yellow, 2 (High Vol) -> Red
            colors = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c'}
            scatter = ax.scatter(
                df_regime.index, 
                df_regime['Close'], 
                c=df_regime['Regime'].map(colors), 
                s=8, 
                zorder=3, 
                label='Regime Marker'
            )
            
            # Custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Low Volatility (Steady/Trend)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#f1c40f', markersize=8, label='Moderate Volatility (Sideways/Range)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='High Volatility (Turbulent/Stress)')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
            ax.set_ylabel("Stock Price ($)")
            st.pyplot(fig)
            plt.close(fig)
            
        with col2:
            st.subheader("Regime Stats Summary")
            stats = []
            for r_id in range(3):
                r_df = df_regime[df_regime['Regime'] == r_id]
                stats.append({
                    "Regime ID": r_id,
                    "Total Days": len(r_df),
                    "Mean Daily Return": f"{r_df['Log_Returns'].mean()*100:.3f}%",
                    "Daily Volatility": f"{r_df['Volatility'].mean()*100:.3f}%",
                })
            st.dataframe(pd.DataFrame(stats).set_index("Regime ID"))
            
            st.info("""
            **GMM Regime Classifier** bekerja dengan mengidentifikasi pola distribusi gabungan dari tingkat return dan volatilitas aset secara statistik, mendeteksi pergeseran perilaku pasar tanpa perlu label manual.
            """)
    else:
        st.warning("Model Market Regime GMM belum dilatih. Silakan jalankan pipeline pelatihan terlebih dahulu di Tab 1.")

# ==========================================
# TAB 3: TRADING SIMULATOR (ENTRY/EXIT & SIZING)
# ==========================================
with tab3:
    st.header("Interactive Trading Simulator & Backtester")
    st.markdown("Uji pengaruh aturan **Entry/Exit** dan **Position Sizing** secara dinamis pada data uji (*test period*).")
    
    # Check if models exist
    model_names = ['SVM', 'RandomForest', 'XGBoost', 'LightGBM']
    available_models = [m for m in model_names if os.path.exists(f"models/{m}_{ticker}.joblib")]
    
    if available_models:
        # Prepare inputs
        col_inputs1, col_inputs2 = st.columns(2)
        with col_inputs1:
            selected_model = st.selectbox("Select ML Model", available_models)
            sizing_method = st.selectbox("Position Sizing Method", ["constant", "volatility", "kelly"])
            use_trend = st.checkbox("Enable Trend Filter (Buy only when Close > SMA 50)", value=True)
            time_barrier = st.slider("Time Barrier (Max trade duration in days)", 0, 20, 5)
            
        with col_inputs2:
            stop_loss = st.slider("Stop Loss (%)", 0.0, 5.0, 1.5, step=0.1) / 100.0
            profit_taking = st.slider("Profit Target (%)", 0.0, 10.0, 3.0, step=0.1) / 100.0
            trailing_stop = st.slider("Trailing Stop (%)", 0.0, 5.0, 2.0, step=0.1) / 100.0
            target_vol = st.slider("Volatility Target (for Vol Sizing method, %)", 0.5, 3.0, 1.5, step=0.1) / 100.0
            
        # Re-run predictions on test set
        scaler = joblib.load(f"models/scaler_{ticker}.joblib")
        df_feat = add_technical_indicators(df_core)
        df_feat['Label'] = 0 # Dummy
        X, y, _ = prepare_features_and_labels(df_feat)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        
        # Test split
        X_train, X_test, _, _ = walk_forward_split(X_scaled, y)
        test_prices = df_feat.loc[X_test.index, 'Close']
        test_vols = df_feat.loc[X_test.index, 'Volatility'].values
        test_sma_50 = df_feat.loc[X_test.index, 'SMA_50'].values
        test_trend = test_prices.values > test_sma_50 if use_trend else None
        
        # Load model and predict
        wrapper = ModelWrapper(selected_model)
        wrapper.model = joblib.load(f"models/{selected_model}_{ticker}.joblib")
        if selected_model == "XGBoost":
            wrapper._class_to_label = {0: -1, 1: 0, 2: 1}
            wrapper._label_to_class = {-1: 0, 0: 1, 1: 2}
            
        preds = wrapper.predict(X_test)
        probas = wrapper.predict_proba(X_test)
        
        # Run advanced backtester
        sim_results = run_advanced_backtest(
            prices=test_prices,
            signals=preds,
            probabilities=probas,
            volatilities=test_vols,
            trend_filter=test_trend,
            sizing_method=sizing_method,
            target_vol=target_vol,
            stop_loss=stop_loss,
            profit_taking=profit_taking,
            trailing_stop=trailing_stop,
            time_barrier=time_barrier
        )
        
        # Display Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{sim_results["Total Return"]*100:.2f}%</div><div class="metric-label">Total Return</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{sim_results["Sharpe Ratio"]:.2f}</div><div class="metric-label">Sharpe Ratio</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{sim_results["Max Drawdown"]*100:.2f}%</div><div class="metric-label">Max Drawdown</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(sim_results["trades"])}</div><div class="metric-label">Total Trades</div></div>', unsafe_allow_html=True)
            
        # Plot Equity Curve vs Buy & Hold
        st.subheader("Equity Curve vs. Buy-and-Hold Strategy")
        fig, ax = plt.subplots(figsize=(12, 5))
        bh_curve = test_prices / test_prices.iloc[0]
        ax.plot(bh_curve.index, bh_curve, label='Buy & Hold (Stock)', color='#8b9bb4', linestyle='--')
        ax.plot(sim_results["equity_curve"].index, sim_results["equity_curve"], label='Strategy Equity Curve', color='#00d2d3', linewidth=2)
        ax.set_ylabel("Normalized Value")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
        # Display completed trades
        if sim_results["trades"]:
            st.subheader("Recent Completed Trades (Last 10)")
            trades_df = pd.DataFrame(sim_results["trades"]).tail(10)
            # Format numbers
            trades_df['entry_price'] = trades_df['entry_price'].map(lambda x: f"${x:.2f}")
            trades_df['exit_price'] = trades_df['exit_price'].map(lambda x: f"${x:.2f}")
            trades_df['size'] = trades_df['size'].map(lambda x: f"{x*100:.1f}%")
            trades_df['pnl'] = trades_df['pnl'].map(lambda x: f"{x*100:.2f}%")
            st.dataframe(trades_df.iloc[::-1])
        else:
            st.info("Tidak ada transaksi yang tercatat dalam periode data uji dengan konfigurasi parameter saat ini.")
            
    else:
        st.warning("Belum ada model machine learning yang siap. Silakan jalankan pelatihan model terlebih dahulu di Tab 1.")

# ==========================================
# TAB 4: PORTFOLIO ALLOCATION
# ==========================================
with tab4:
    st.header("Multi-Asset Portfolio Allocation & Sizing")
    st.markdown("Optimalkan pembagian modal lintas aset berdasarkan teori portofolio modern (MVO & Risk Parity).")
    
    selected_tickers = st.multiselect("Select Tickers for Portfolio", ["AAPL", "GOOGL", "MSFT", "TSLA"], default=["AAPL", "MSFT", "GOOGL"])
    
    if len(selected_tickers) >= 2:
        if st.button("Optimize Weights"):
            with st.spinner("Mengambil data historis dan menghitung bobot optimal..."):
                # Load prices
                prices_dict = {}
                for t in selected_tickers:
                    df_t = get_or_download_data(t)
                    if df_t is not None:
                        prices_dict[t] = df_t['Close']
                        
                # Merge into one DataFrame
                df_portfolio_prices = pd.DataFrame(prices_dict).dropna()
                
                # Daily Returns
                daily_returns = df_portfolio_prices.pct_change().dropna()
                
                # Annualized expected returns and covariance matrix
                expected_returns = daily_returns.mean() * 252
                cov_matrix = daily_returns.cov() * 252
                
                # Calculate weights
                weights_ew = calculate_equal_weights(len(selected_tickers))
                weights_rp = calculate_risk_parity_weights(cov_matrix)
                weights_mvo = calculate_mvo_weights(expected_returns, cov_matrix)

                # HRP weights via API
                hrp_response = api_post("/portfolio", {"tickers": selected_tickers, "method": "hrp"})
                if hrp_response:
                    hrp_weights_dict = hrp_response["weights"]
                    weights_hrp = np.array([hrp_weights_dict.get(t, 0.0) for t in selected_tickers])
                else:
                    weights_hrp = weights_ew.copy()  # fallback

                # Display Weight comparison in table
                weights_df = pd.DataFrame({
                    "Equal Weight (EW)": weights_ew,
                    "Risk Parity (RP)": weights_rp,
                    "Mean-Variance (MVO)": weights_mvo,
                    "Hierarchical RP (HRP)": weights_hrp,
                }, index=selected_tickers)
                
                # Format to percentage
                formatted_df = weights_df.map(lambda x: f"{x*100:.2f}%")
                st.subheader("Optimal Weight Allocation Matrix")
                st.dataframe(formatted_df)
                
                # Visual Bar Chart Comparison
                st.subheader("Allocation Weight Comparison")
                fig, ax = plt.subplots(figsize=(10, 4.5))
                weights_df.plot(kind='bar', ax=ax, width=0.8, colormap="viridis")
                plt.ylabel("Portfolio Weight Allocation")
                plt.xlabel("Ticker")
                plt.xticks(rotation=0)
                plt.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                
                # Portfolio expected metrics
                st.subheader("Portfolio Expected Performance Characteristics")
                perf_metrics = []
                for name, w in [("Equal Weighting", weights_ew), ("Risk Parity", weights_rp), ("MVO (Max Sharpe)", weights_mvo), ("HRP", weights_hrp)]:
                    port_return = np.dot(w, expected_returns)
                    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                    port_sharpe = port_return / port_vol if port_vol > 0 else 0.0
                    perf_metrics.append({
                        "Allocation Method": name,
                        "Expected Annual Return": f"{port_return*100:.2f}%",
                        "Expected Annual Volatility": f"{port_vol*100:.2f}%",
                        "Portfolio Sharpe Ratio": f"{port_sharpe:.2f}"
                    })
                st.dataframe(pd.DataFrame(perf_metrics).set_index("Allocation Method"))
                
    else:
        st.info("Pilih minimal 2 ticker saham untuk menghitung alokasi portofolio optimal.")

# ==========================================
# TAB 5: HMM REGIME DETECTION
# ==========================================
with tab5:
    st.header("Hidden Markov Model (HMM) Market Regime Detection")
    bundle = compute_hmm_local(ticker, df_core_indicators)

    if bundle:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Regime Comparison")
            st.metric("GMM Regime", bundle.get("regime_gmm", "N/A"))
            st.metric("HMM Regime", bundle.get("regime_hmm", "N/A"))

        with col2:
            st.subheader("HMM State Transition Matrix")
            tm = bundle.get("hmm_transition_matrix")
            if tm:
                tm_df = pd.DataFrame(tm,
                    index=["Bull→", "Sideways→", "Bear→"],
                    columns=["→Bull", "→Sideways", "→Bear"])
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(tm_df.astype(float), annot=True, fmt=".2f", cmap="Blues",
                           vmin=0, vmax=1, ax=ax, linewidths=0.5)
                ax.set_title("Transition Probabilities")
                st.pyplot(fig)
                plt.close(fig)

        st.info("""
        **HMM Regime Detector** uses a Gaussian Hidden Markov Model with 3 states (Bull, Sideways, Bear),
        ordered by mean log-return. The transition matrix shows the probability of moving from one regime to another.
        """)
    else:
        st.warning("HMM regime data could not be computed. Ensure sufficient price history is available.")

# ==========================================
# TAB 6: RISK & VOLATILITY
# ==========================================
with tab6:
    st.header("GARCH Volatility Forecast & Tail Risk (CVaR)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Volatility Forecast")
        vol_data = compute_volatility_local(ticker, df_core_indicators)
        if vol_data:
            garch = vol_data.get("garch_vol")
            roll = vol_data.get("rolling_vol")
            st.metric("GARCH Forecast Vol (Annualized)", f"{garch*100:.2f}%" if garch is not None else "N/A")
            st.metric("Rolling 20d Vol (Annualized)", f"{roll*100:.2f}%" if roll is not None else "N/A")
            if garch and roll and garch > 0 and roll > 0:
                if garch > roll * 1.1:
                    st.warning("GARCH forecasts HIGHER volatility than recent history — consider reducing position size.")
                elif garch < roll * 0.9:
                    st.success("GARCH forecasts LOWER volatility — market may be calming.")
        else:
            st.warning("Volatility data unavailable.")

    with col2:
        st.subheader("Tail Risk (CVaR)")
        risk_data = compute_risk_local(ticker, df_core_indicators)
        if risk_data:
            c1, c2 = st.columns(2)
            cvar_95 = risk_data.get("cvar_95")
            cvar_99 = risk_data.get("cvar_99")
            var_95 = risk_data.get("var_95")
            var_99 = risk_data.get("var_99")
            scale = risk_data.get("position_scale")
            c1.metric("CVaR 95%", f"{cvar_95*100:.2f}%" if cvar_95 is not None else "N/A")
            c2.metric("CVaR 99%", f"{cvar_99*100:.2f}%" if cvar_99 is not None else "N/A")
            c1.metric("VaR 95%", f"{var_95*100:.2f}%" if var_95 is not None else "N/A")
            c2.metric("VaR 99%", f"{var_99*100:.2f}%" if var_99 is not None else "N/A")
            if scale is not None:
                st.metric("CVaR Position Scale Factor", f"{scale:.2f}x",
                         delta=f"{'Reduce' if scale < 1 else 'Increase'} size by {abs(1-scale)*100:.0f}%")
        else:
            st.warning("Risk data unavailable.")

    # Anomaly detection info
    st.subheader("Anomaly Detection (Isolation Forest)")
    anomaly_data = compute_anomaly_local(ticker, df_core_indicators)
    if anomaly_data:
        flag = anomaly_data.get("anomaly_flag", False)
        score = anomaly_data.get("anomaly_score", 0.0)
        if flag:
            st.error(f"⚠️ MARKET ANOMALY DETECTED — Score: {score:.4f}. Exercise caution.")
        else:
            st.success(f"✅ Market conditions appear normal — Anomaly score: {score:.4f}")
    else:
        st.warning("Anomaly data unavailable.")

# ==========================================
# TAB 7: MEAN REVERSION
# ==========================================
with tab7:
    st.header("Mean Reversion Analysis (Z-Score & Ornstein-Uhlenbeck)")
    mr_data = compute_mean_reversion_local(ticker, df_core_indicators)

    if mr_data:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Z-score chart from historical data
            if df_core_indicators is not None and 'Close' in df_core_indicators.columns:
                prices_s = df_core_indicators['Close']
                rolling_mean = prices_s.rolling(20).mean()
                rolling_std = prices_s.rolling(20).std()
                zscore_series = ((prices_s - rolling_mean) / rolling_std).dropna()

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(zscore_series.index, zscore_series, label='Z-Score', color='#00d2d3', linewidth=1.5)
                ax.axhline(y=2.0, color='#e74c3c', linestyle='--', alpha=0.7, label='Entry Threshold (+2σ)')
                ax.axhline(y=-2.0, color='#2ecc71', linestyle='--', alpha=0.7, label='Entry Threshold (-2σ)')
                ax.axhline(y=0.5, color='#f1c40f', linestyle=':', alpha=0.5, label='Exit Threshold (+0.5σ)')
                ax.axhline(y=-0.5, color='#f1c40f', linestyle=':', alpha=0.5, label='Exit Threshold (-0.5σ)')
                ax.axhline(y=0, color='white', alpha=0.2)
                ax.set_ylabel("Z-Score")
                ax.legend(loc='upper left', fontsize=8)
                st.pyplot(fig)
                plt.close(fig)

        with col2:
            st.subheader("Mean Reversion Metrics")
            halflife = mr_data.get("halflife", float('inf'))
            is_mr = mr_data.get("is_mean_reverting", False)
            signal = mr_data.get("signal", 0)  # key is "signal" in MeanReversionDetector
            zscore_val = mr_data.get("zscore")

            if zscore_val is not None and not (isinstance(zscore_val, float) and np.isnan(zscore_val)):
                st.metric("Current Z-Score", f"{zscore_val:.3f}")
            else:
                st.metric("Current Z-Score", "N/A")

            if halflife is None or halflife == float('inf') or halflife > 252:
                st.metric("OU Half-Life", "Non-mean-reverting")
            else:
                st.metric("OU Half-Life", f"{halflife:.1f} days")

            signal_map = {1: "📈 Long (MR)", -1: "📉 Short (MR)", 0: "➡️ Neutral"}
            st.metric("MR Signal", signal_map.get(signal, "Neutral"))

            if is_mr:
                st.success("✅ Mean Reverting: YES")
            else:
                st.warning("❌ Mean Reverting: NO (trending)")
    else:
        st.warning("Mean reversion data unavailable.")

# ==========================================
# TAB 8: EXPLAINABILITY (SHAP)
# ==========================================
with tab8:
    st.header("Model Explainability (SHAP Feature Importance)")

    model_choice = st.selectbox("Select Model to Explain", ["SVM", "RandomForest", "XGBoost", "LightGBM"])

    explain_data = compute_explain_local(ticker, model_choice, df_core_indicators)

    if explain_data:
        importances = explain_data.get("feature_importances", {})
        if importances:
            shap_df = pd.DataFrame(list(importances.items()), columns=["Feature", "SHAP Importance"])
            shap_df = shap_df.sort_values("SHAP Importance", ascending=True).tail(15)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#00d2d3' if v > 0 else '#e74c3c' for v in shap_df["SHAP Importance"]]
            ax.barh(shap_df["Feature"], shap_df["SHAP Importance"], color=colors)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title(f"Feature Importance — {model_choice} ({ticker})")
            st.pyplot(fig)
            plt.close(fig)

            # Top 5 table
            st.subheader("Top 5 Most Important Features")
            top5 = pd.DataFrame(list(importances.items()), columns=["Feature", "Mean |SHAP|"])
            top5 = top5.nlargest(5, "Mean |SHAP|").reset_index(drop=True)
            st.dataframe(top5)
        else:
            st.warning("No SHAP importance data available.")
    else:
        _scaler_path = os.path.join("models", f"scaler_{ticker}.joblib")
        _model_path = os.path.join("models", f"{model_choice}_{ticker}.joblib")
        if not os.path.exists(_scaler_path) or not os.path.exists(_model_path):
            st.info(
                f"📋 Model artifacts for **{model_choice} / {ticker}** are not yet trained. "
                "SHAP explainability will be available automatically once training completes."
            )
        else:
            st.warning("SHAP computation failed. Check that the model and scaler are valid.")

    st.info("""
    **SHAP (SHapley Additive exPlanations)** values show how much each feature contributes to the model's prediction.
    Tree-based models (RF, XGB, LGBM) use TreeExplainer for fast computation. SVM uses KernelExplainer.
    """)
