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

# Load Core Data
df_core = get_or_download_data(ticker)

if df_core is None:
    st.error(f"Gagal memuat data saham untuk {ticker}.")
    st.stop()

# Ensure indicators exist
df_core_indicators = add_technical_indicators(df_core)

# Setup tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Benchmarks", 
    "📈 Market Regimes (GMM)", 
    "⚙️ Trading Simulator (Entry/Exit & Sizing)", 
    "💼 Portfolio Allocation (MVO & Risk Parity)"
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
                
                # Display Weight comparison in table
                weights_df = pd.DataFrame({
                    "Equal Weight (EW)": weights_ew,
                    "Risk Parity (RP)": weights_rp,
                    "Mean-Variance (MVO)": weights_mvo
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
                for name, w in [("Equal Weighting", weights_ew), ("Risk Parity", weights_rp), ("MVO (Max Sharpe)", weights_mvo)]:
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
