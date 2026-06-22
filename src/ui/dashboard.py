import sys
import os

# Ensure repo root is in sys.path so both `src.*` and `scripts.*` resolve
# regardless of which directory Streamlit Cloud picks as cwd.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
import base64
import requests
import joblib
from datetime import datetime
import streamlit.components.v1 as components

from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import (
    prepare_features_and_labels,
    walk_forward_split,
    triple_barrier_labeling,
)
from src.models.model_wrappers import ModelWrapper
from src.models.backtester import run_advanced_backtest
from src.models.portfolio_sizing import (
    calculate_equal_weights,
    calculate_risk_parity_weights,
    calculate_mvo_weights,
)
from src.data_ingestion import fetch_stock_data
from src.models.regime_hmm import HMMRegimeDetector
from src.models.market_regime import MarketRegimeDetector
from src.models.volatility_garch import GARCHVolatilityModel
from src.models.risk_model import TailRiskModel
from src.models.anomaly_detector import MarketAnomalyDetector
from src.models.mean_reversion import MeanReversionDetector
from src.models.explainability import ModelExplainer
# ponytail: lazy import — keeps app loading even if scripts/ path is unusual
def _load_eda_fns():
    from scripts.eda_preprocessing import (
        plot_raw_ohlcv, plot_technical_indicators, plot_labeling,
        plot_feature_distributions, plot_scaling_impact, plot_train_test_split,
    )
    return (plot_raw_ohlcv, plot_technical_indicators, plot_labeling,
            plot_feature_distributions, plot_scaling_impact, plot_train_test_split)

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


# Dark background for all matplotlib figures
plt.style.use("dark_background")

st.set_page_config(
    page_title="ML Research: Advanced Trading & Portfolio Dashboard",
    layout="wide",
)

st.markdown("""
<style>
    .reportview-container { background: #0f141c }
    .metric-card {
        background-color: #1a2332;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3d54;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: #00d2d3; }
    .metric-val { font-size: 28px; font-weight: bold; color: #00d2d3; margin-bottom: 5px; }
    .metric-label {
        font-size: 14px; color: #8b9bb4;
        text-transform: uppercase; letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Task-Oriented Benchmarking: Traditional ML in Stock Markets")
st.markdown("""
Dashboard interaktif ini menyajikan evaluasi model Machine Learning berdasarkan **Kinerja Finansial Nyata**
(Market Regime, Position Sizing, Entry/Exit, dan Portfolio Sizing) di atas metrik akurasi statistik dasar.
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration Panel")

PRESET_TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]

preset_ticker = st.sidebar.selectbox("Select Core Ticker (preset)", PRESET_TICKERS)
custom_ticker = st.sidebar.text_input(
    "Or enter any ticker symbol",
    value="",
    placeholder="e.g. AMD, NFLX, SPY",
    help="Type any symbol to analyze it dynamically. Overrides the preset above.",
).strip().upper()

ticker = custom_ticker or preset_ticker

if not ticker:
    st.warning("Masukkan simbol ticker yang valid untuk memulai analisis.")
    st.stop()

st.sidebar.caption(
    "Untrained tickers download price data live. Model-dependent tabs "
    "(Benchmarks, GMM, Simulator, Explainability) will prompt you to train "
    "via the button in the Model Benchmarks tab."
)

_TODAY = datetime.now().strftime("%Y-%m-%d")


def get_or_download_data(t, start_date="2020-01-01", end_date=None):
    if end_date is None:
        end_date = _TODAY
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    df = None
    files = [f for f in os.listdir(raw_dir) if f.startswith(f"{t}_") and f.endswith(".parquet")]
    if files:
        df = pd.read_parquet(os.path.join(raw_dir, files[0]))
    else:
        filepath = fetch_stock_data(t, start_date, end_date, output_dir=raw_dir, allow_synthetic_fallback=True)
        if filepath and os.path.exists(filepath):
            df = pd.read_parquet(filepath)

    if df is not None:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            standard_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
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


_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]


# ── Viz helpers ───────────────────────────────────────────────────────────────

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()


def _dl(fig, fname: str) -> None:
    st.download_button("💾 Download PNG", fig_to_png_bytes(fig), fname, "image/png")


def render_mermaid(graph_def: str, height: int = 650) -> None:
    safe = graph_def.replace("`", "&#96;")
    html = (
        '<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>'
        '<script>mermaid.initialize({startOnLoad:true,theme:"dark"});</script>'
        f'<div class="mermaid">\n{safe}\n</div>'
    )
    components.html(html, height=height, scrolling=True)


@st.cache_data(show_spinner=False)
def mermaid_to_image(graph_def: str) -> bytes | None:
    # ponytail: external render via mermaid.ink; returns PNG bytes or None if offline
    encoded = base64.urlsafe_b64encode(graph_def.encode()).decode()
    try:
        resp = requests.get(
            f"https://mermaid.ink/img/{encoded}?bgColor=0f141c", timeout=10
        )
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


# ── Local-compute helpers ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_hmm_local(ticker: str, df: pd.DataFrame) -> dict:
    try:
        X = df[["Log_Returns", "Volatility"]].dropna().values
        if len(X) < 30:
            return {}
        det = HMMRegimeDetector()
        det.fit(X)
        preds = det.predict(X)
        regime_hmm = det.get_regime_names()[int(preds[-1])]
        tm = det.get_transition_matrix().tolist()
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
        return {"regime_gmm": regime_gmm, "regime_hmm": regime_hmm, "hmm_transition_matrix": tm}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def compute_volatility_local(ticker: str, df: pd.DataFrame) -> dict:
    try:
        returns = df["Log_Returns"].dropna()
        if len(returns) < 30:
            return {}
        m = GARCHVolatilityModel()
        m.fit(returns)
        garch_vol = m.forecast(horizon=1)
        rolling_vol = float(df["Volatility"].dropna().iloc[-1]) * np.sqrt(252)
        return {"garch_vol": garch_vol, "rolling_vol": rolling_vol}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def compute_risk_local(ticker: str, df: pd.DataFrame) -> dict:
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
    try:
        prices = df["Close"].dropna()
        if len(prices) < 30:
            return {}
        return MeanReversionDetector().predict(prices)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def compute_explain_local(ticker: str, model_choice: str, df: pd.DataFrame) -> dict:
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
        n_rows = 50 if model_choice == "SVM" else 250
        X_scaled = scaler.transform(feat_df.values[-n_rows:])
        feature_names = list(feat_df.columns)
        explainer = ModelExplainer(model_choice, raw_model, feature_names)
        importances = explainer.get_feature_importance(X_scaled)
        return {"feature_importances": importances}
    except Exception:
        return {}


# ── Load core data ────────────────────────────────────────────────────────────
df_core = get_or_download_data(ticker)

if df_core is None:
    st.error(f"Gagal memuat data saham untuk {ticker}.")
    st.stop()

df_core_indicators = add_technical_indicators(df_core)

# Status banner (FastAPI only)
status_data = api_get("/status")
if status_data:
    ticker_status = status_data.get(ticker, {})
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Last Data Refresh", ticker_status.get("last_refresh") or "Never")
    with col_s2:
        st.metric("Last Model Retrain", ticker_status.get("last_retrain") or "Never")
    with col_s3:
        anomaly = ticker_status.get("anomaly_flag")
        if anomaly is True:
            st.metric("Market Status", "⚠️ ANOMALY DETECTED")
        elif anomaly is False:
            st.metric("Market Status", "✅ Normal")
        else:
            st.metric("Market Status", "—")
    st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab1, tab2, tab3, tab4, tab5,
 tab6, tab7, tab8, tab9, tab10) = st.tabs([
    "📊 Model Benchmarks",
    "📈 Market Regimes (GMM)",
    "⚙️ Trading Simulator",
    "💼 Portfolio Allocation",
    "🔮 Regime (HMM)",
    "📉 Risk & Volatility",
    "🔄 Mean Reversion",
    "🔍 Explainability",
    "🔁 ML Pipeline",
    "🖼️ EDA Gallery",
])

# ==========================================
# TAB 1: MODEL BENCHMARKS
# ==========================================
with tab1:
    st.header(f"Model Benchmarking Results: {ticker}")
    benchmarking_file = f"data/processed/{ticker}_benchmarking_results.csv"

    if os.path.exists(benchmarking_file):
        results_df = pd.read_csv(benchmarking_file, index_col=0)

        st.subheader("Base vs. Advanced Strategy Performance Matrix")
        st.markdown("""
        * **Base (Baseline):** Strategi dasar buy/sell konstan 100% tanpa Stop Loss / Sizing.
        * **Adv (Advanced):** Strategi lanjutan dengan Kelly Sizing, Stop Loss (1.5%), Profit Target (3.0%), Trailing Stop (2.0%), dan Trend Filter (SMA 50).
        """)
        st.dataframe(results_df.style.highlight_max(axis=0, subset=[
            "Total Return (Base)", "Total Return (Adv)",
            "Sharpe Ratio (Base)", "Sharpe Ratio (Adv)",
        ]))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total Return Comparison")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_data = results_df[["Total Return (Base)", "Total Return (Adv)"]].reset_index()
            sns.barplot(
                x="index", y="Return", hue="Strategy",
                data=plot_data.melt(id_vars="index", var_name="Strategy", value_name="Return"),
                ax=ax, palette="viridis",
            )
            plt.xticks(rotation=0)
            plt.xlabel("Model")
            plt.ylabel("Total Return")
            st.pyplot(fig)
            _dl(fig, f"total_return_{ticker}.png")
            plt.close(fig)

        with col2:
            st.subheader("Sharpe Ratio Comparison")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_data = results_df[["Sharpe Ratio (Base)", "Sharpe Ratio (Adv)"]].reset_index()
            sns.barplot(
                x="index", y="Sharpe", hue="Strategy",
                data=plot_data.melt(id_vars="index", var_name="Strategy", value_name="Sharpe"),
                ax=ax, palette="magma",
            )
            plt.xticks(rotation=0)
            plt.xlabel("Model")
            plt.ylabel("Sharpe Ratio")
            st.pyplot(fig)
            _dl(fig, f"sharpe_ratio_{ticker}.png")
            plt.close(fig)

    else:
        st.warning(f"Hasil benchmarking untuk {ticker} belum tersedia.")
        if st.button("🚀 Train & Benchmark Models"):
            with st.spinner("Melatih model... (beberapa menit untuk ARIMA)"):
                try:
                    from src.train_benchmark import run_benchmarking
                    run_benchmarking(ticker, "2020-01-01", _TODAY)
                    st.success("Pelatihan selesai!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

# ==========================================
# TAB 2: MARKET REGIMES (GMM)
# ==========================================
with tab2:
    st.header("Gaussian Mixture Model (GMM) Market Regime Detection")
    regime_model_path = f"models/regime_detector_{ticker}.joblib"

    if os.path.exists(regime_model_path):
        regime_detector = joblib.load(regime_model_path)
        regime_features = df_core_indicators[["Log_Returns", "Volatility"]].dropna()
        regimes = regime_detector.predict(regime_features)
        regime_names = regime_detector.predict_regime_name(regime_features)

        df_regime = df_core_indicators.loc[regime_features.index].copy()
        df_regime["Regime"] = regimes
        df_regime["Regime_Name"] = regime_names

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Price Chart Colored by Detected Market Regime")
            fig, ax = plt.subplots(figsize=(12, 5.5))
            ax.plot(df_regime.index, df_regime["Close"], color="#a4b0be", alpha=0.5)
            colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}
            ax.scatter(df_regime.index, df_regime["Close"],
                       c=df_regime["Regime"].map(colors), s=8, zorder=3)
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8, label="Low Volatility"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#f1c40f", markersize=8, label="Moderate Volatility"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="High Volatility"),
            ], loc="upper left")
            ax.set_ylabel("Stock Price ($)")
            st.pyplot(fig)
            _dl(fig, f"gmm_regimes_{ticker}.png")
            plt.close(fig)

        with col2:
            st.subheader("Regime Stats")
            stats = []
            for r_id in range(3):
                r_df = df_regime[df_regime["Regime"] == r_id]
                stats.append({
                    "Regime ID": r_id,
                    "Days": len(r_df),
                    "Mean Return": f"{r_df['Log_Returns'].mean()*100:.3f}%",
                    "Volatility": f"{r_df['Volatility'].mean()*100:.3f}%",
                })
            st.dataframe(pd.DataFrame(stats).set_index("Regime ID"))
            st.info("GMM identifies 3 regimes by joint return+volatility distribution.")
    else:
        st.warning("GMM model not trained yet. Run training in Tab 1.")

# ==========================================
# TAB 3: TRADING SIMULATOR
# ==========================================
with tab3:
    st.header("Interactive Trading Simulator & Backtester")
    st.markdown("Uji pengaruh aturan **Entry/Exit** dan **Position Sizing** secara dinamis pada data uji (*test period*).")

    model_names = ["SVM", "RandomForest", "XGBoost", "LightGBM"]
    available_models = [m for m in model_names if os.path.exists(f"models/{m}_{ticker}.joblib")]

    if available_models:
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
            target_vol = st.slider("Volatility Target (for Vol Sizing, %)", 0.5, 3.0, 1.5, step=0.1) / 100.0

        scaler = joblib.load(f"models/scaler_{ticker}.joblib")
        df_feat = add_technical_indicators(df_core)
        df_feat["Label"] = 0  # dummy label for feature prep
        X, y, _ = prepare_features_and_labels(df_feat)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        X_train, X_test, _, _ = walk_forward_split(X_scaled, y)
        test_prices = df_feat.loc[X_test.index, "Close"]
        test_vols = df_feat.loc[X_test.index, "Volatility"].values
        test_sma_50 = df_feat.loc[X_test.index, "SMA_50"].values
        test_trend = test_prices.values > test_sma_50 if use_trend else None

        wrapper = ModelWrapper(selected_model)
        wrapper.model = joblib.load(f"models/{selected_model}_{ticker}.joblib")
        if selected_model == "XGBoost":
            wrapper._class_to_label = ModelWrapper.XGB_CLASS_TO_LABEL
            wrapper._label_to_class = ModelWrapper.XGB_LABEL_TO_CLASS

        preds = wrapper.predict(X_test)
        probas = wrapper.predict_proba(X_test)

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
            time_barrier=time_barrier,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{sim_results["Total Return"]*100:.2f}%</div><div class="metric-label">Total Return</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{sim_results["Sharpe Ratio"]:.2f}</div><div class="metric-label">Sharpe Ratio</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{sim_results["Max Drawdown"]*100:.2f}%</div><div class="metric-label">Max Drawdown</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(sim_results["trades"])}</div><div class="metric-label">Total Trades</div></div>', unsafe_allow_html=True)

        st.subheader("Equity Curve vs. Buy-and-Hold Strategy")
        fig, ax = plt.subplots(figsize=(12, 5))
        bh_curve = test_prices / test_prices.iloc[0]
        ax.plot(bh_curve.index, bh_curve, label="Buy & Hold", color="#8b9bb4", linestyle="--")
        ax.plot(sim_results["equity_curve"].index, sim_results["equity_curve"],
                label="Strategy", color="#00d2d3", linewidth=2)
        ax.set_ylabel("Normalized Value")
        ax.legend()
        st.pyplot(fig)
        _dl(fig, f"equity_curve_{ticker}.png")
        plt.close(fig)

        if sim_results["trades"]:
            st.subheader("Recent Completed Trades (Last 10)")
            trades_df = pd.DataFrame(sim_results["trades"]).tail(10)
            trades_df["entry_price"] = trades_df["entry_price"].map(lambda x: f"${x:.2f}")
            trades_df["exit_price"] = trades_df["exit_price"].map(lambda x: f"${x:.2f}")
            trades_df["size"] = trades_df["size"].map(lambda x: f"{x*100:.1f}%")
            trades_df["pnl"] = trades_df["pnl"].map(lambda x: f"{x*100:.2f}%")
            st.dataframe(trades_df.iloc[::-1])
        else:
            st.info("Tidak ada transaksi dalam periode data uji dengan konfigurasi ini.")
    else:
        st.warning("No trained models found. Run training in Tab 1.")

# ==========================================
# TAB 4: PORTFOLIO ALLOCATION
# ==========================================
with tab4:
    st.header("Multi-Asset Portfolio Allocation & Sizing")
    st.markdown("Optimalkan pembagian modal lintas aset berdasarkan teori portofolio modern.")

    preset_selected = st.multiselect(
        "Select Tickers for Portfolio",
        PRESET_TICKERS,
        default=["AAPL", "MSFT", "GOOGL"],
    )
    extra_raw = st.text_input(
        "Add custom tickers (comma-separated)",
        value="",
        placeholder="e.g. AMD, NFLX, SPY",
    )
    extra_tickers = [t.strip().upper() for t in extra_raw.split(",") if t.strip()]

    selected_tickers = list(dict.fromkeys([*preset_selected, *extra_tickers]))

    if len(selected_tickers) >= 2:
        if st.button("Optimize Weights"):
            with st.spinner("Mengambil data historis dan menghitung bobot optimal..."):
                prices_dict = {}
                for t in selected_tickers:
                    df_t = get_or_download_data(t)
                    if df_t is not None:
                        prices_dict[t] = df_t["Close"]

                df_portfolio_prices = pd.DataFrame(prices_dict).dropna()
                daily_returns = df_portfolio_prices.pct_change().dropna()
                expected_returns = daily_returns.mean() * 252
                cov_matrix = daily_returns.cov() * 252

                weights_ew = calculate_equal_weights(len(selected_tickers))
                weights_rp = calculate_risk_parity_weights(cov_matrix)
                weights_mvo = calculate_mvo_weights(expected_returns, cov_matrix)

                hrp_response = api_post("/portfolio", {"tickers": selected_tickers, "method": "hrp"})
                if hrp_response:
                    hrp_w = hrp_response["weights"]
                    weights_hrp = np.array([hrp_w.get(t, 0.0) for t in selected_tickers])
                else:
                    weights_hrp = weights_ew.copy()

                weights_df = pd.DataFrame({
                    "Equal Weight (EW)": weights_ew,
                    "Risk Parity (RP)": weights_rp,
                    "Mean-Variance (MVO)": weights_mvo,
                    "HRP": weights_hrp,
                }, index=selected_tickers)

                st.subheader("Optimal Weight Allocation Matrix")
                st.dataframe(weights_df.map(lambda x: f"{x*100:.2f}%"))

                st.subheader("Allocation Weight Comparison")
                fig, ax = plt.subplots(figsize=(10, 4.5))
                weights_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
                plt.ylabel("Portfolio Weight")
                plt.xlabel("Ticker")
                plt.xticks(rotation=0)
                plt.grid(axis="y", alpha=0.3)
                st.pyplot(fig)
                _dl(fig, "portfolio_weights.png")
                plt.close(fig)

                st.subheader("Portfolio Expected Performance")
                perf_metrics = []
                for name, w in [
                    ("Equal Weighting", weights_ew),
                    ("Risk Parity", weights_rp),
                    ("MVO (Max Sharpe)", weights_mvo),
                    ("HRP", weights_hrp),
                ]:
                    port_return = np.dot(w, expected_returns)
                    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                    port_sharpe = port_return / port_vol if port_vol > 0 else 0.0
                    perf_metrics.append({
                        "Method": name,
                        "Expected Annual Return": f"{port_return*100:.2f}%",
                        "Expected Annual Vol": f"{port_vol*100:.2f}%",
                        "Sharpe Ratio": f"{port_sharpe:.2f}",
                    })
                st.dataframe(pd.DataFrame(perf_metrics).set_index("Method"))
    else:
        st.info("Pilih minimal 2 ticker untuk menghitung alokasi portofolio.")

# ==========================================
# TAB 5: HMM REGIME
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
                tm_df = pd.DataFrame(
                    tm,
                    index=["Bull→", "Sideways→", "Bear→"],
                    columns=["→Bull", "→Sideways", "→Bear"],
                )
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(tm_df.astype(float), annot=True, fmt=".2f",
                            cmap="Blues", vmin=0, vmax=1, ax=ax, linewidths=0.5)
                ax.set_title("Transition Probabilities")
                st.pyplot(fig)
                _dl(fig, f"hmm_transition_{ticker}.png")
                plt.close(fig)

        st.info("GaussianHMM with 3 states (Bull/Sideways/Bear). Transition matrix shows regime shift probabilities.")
    else:
        st.warning("HMM regime data unavailable.")

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
                    st.warning("GARCH forecasts HIGHER volatility — consider reducing size.")
                elif garch < roll * 0.9:
                    st.success("GARCH forecasts LOWER volatility — market calming.")
        else:
            st.warning("Volatility data unavailable.")

    with col2:
        st.subheader("Tail Risk (CVaR)")
        risk_data = compute_risk_local(ticker, df_core_indicators)
        if risk_data:
            c1, c2 = st.columns(2)
            c1.metric("CVaR 95%", f"{risk_data.get('cvar_95')*100:.2f}%" if risk_data.get("cvar_95") is not None else "N/A")
            c2.metric("CVaR 99%", f"{risk_data.get('cvar_99')*100:.2f}%" if risk_data.get("cvar_99") is not None else "N/A")
            c1.metric("VaR 95%", f"{risk_data.get('var_95')*100:.2f}%" if risk_data.get("var_95") is not None else "N/A")
            c2.metric("VaR 99%", f"{risk_data.get('var_99')*100:.2f}%" if risk_data.get("var_99") is not None else "N/A")
            scale = risk_data.get("position_scale")
            if scale is not None:
                st.metric("CVaR Position Scale", f"{scale:.2f}x",
                          delta=f"{'Reduce' if scale < 1 else 'Increase'} by {abs(1-scale)*100:.0f}%")
        else:
            st.warning("Risk data unavailable.")

    st.subheader("Anomaly Detection (Isolation Forest)")
    anomaly_data = compute_anomaly_local(ticker, df_core_indicators)
    if anomaly_data:
        flag = anomaly_data.get("anomaly_flag", False)
        score = anomaly_data.get("anomaly_score", 0.0)
        if flag:
            st.error(f"⚠️ MARKET ANOMALY DETECTED — Score: {score:.4f}. Exercise caution.")
        else:
            st.success(f"✅ Market conditions appear normal — Score: {score:.4f}")
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
            prices_s = df_core_indicators["Close"]
            rolling_mean = prices_s.rolling(20).mean()
            rolling_std = prices_s.rolling(20).std()
            zscore_series = ((prices_s - rolling_mean) / rolling_std).dropna()

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(zscore_series.index, zscore_series, color="#00d2d3", linewidth=1.5, label="Z-Score")
            ax.axhline(y=2.0, color="#e74c3c", linestyle="--", alpha=0.7, label="+2σ entry")
            ax.axhline(y=-2.0, color="#2ecc71", linestyle="--", alpha=0.7, label="-2σ entry")
            ax.axhline(y=0.5, color="#f1c40f", linestyle=":", alpha=0.5, label="+0.5σ exit")
            ax.axhline(y=-0.5, color="#f1c40f", linestyle=":", alpha=0.5, label="-0.5σ exit")
            ax.axhline(y=0, color="white", alpha=0.2)
            ax.set_ylabel("Z-Score")
            ax.legend(loc="upper left", fontsize=8)
            st.pyplot(fig)
            _dl(fig, f"mean_reversion_{ticker}.png")
            plt.close(fig)

        with col2:
            st.subheader("Mean Reversion Metrics")
            halflife = mr_data.get("halflife", float("inf"))
            is_mr = mr_data.get("is_mean_reverting", False)
            signal = mr_data.get("signal", 0)
            zscore_val = mr_data.get("zscore")

            if zscore_val is not None and not (isinstance(zscore_val, float) and np.isnan(zscore_val)):
                st.metric("Current Z-Score", f"{zscore_val:.3f}")
            else:
                st.metric("Current Z-Score", "N/A")

            if halflife is None or halflife == float("inf") or halflife > 252:
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
            colors = ["#00d2d3" if v > 0 else "#e74c3c" for v in shap_df["SHAP Importance"]]
            ax.barh(shap_df["Feature"], shap_df["SHAP Importance"], color=colors)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title(f"Feature Importance — {model_choice} ({ticker})")
            st.pyplot(fig)
            _dl(fig, f"shap_{model_choice}_{ticker}.png")
            plt.close(fig)

            st.subheader("Top 5 Most Important Features")
            top5 = pd.DataFrame(list(importances.items()), columns=["Feature", "Mean |SHAP|"])
            top5 = top5.nlargest(5, "Mean |SHAP|").reset_index(drop=True)
            st.dataframe(top5)
        else:
            st.warning("No SHAP importance data available.")
    else:
        _sp = os.path.join("models", f"scaler_{ticker}.joblib")
        _mp = os.path.join("models", f"{model_choice}_{ticker}.joblib")
        if not os.path.exists(_sp) or not os.path.exists(_mp):
            st.info(f"Model **{model_choice}/{ticker}** not trained yet. Train in Tab 1 to enable SHAP.")
        else:
            st.warning("SHAP computation failed. Check model and scaler.")

    st.info("""
    **SHAP** values quantify each feature's contribution. Tree models (RF/XGB/LGBM) use TreeExplainer.
    SVM uses KernelExplainer (slower, ~30s on first load).
    """)

# ==========================================
# TAB 9: ML PIPELINE (MERMAID)
# ==========================================
with tab9:
    st.header("ML Pipeline Architecture")
    st.markdown(
        "Interactive diagrams of the end-to-end ML pipeline — from data ingestion through training, "
        "artifact storage, and serving."
    )

    pipeline_md_path = os.path.join("docs", "pipeline.md")
    readme_path = "README.md"

    def _extract_mermaid(md_text: str) -> list[str]:
        return re.findall(r"```mermaid\s*(.*?)```", md_text, re.DOTALL)

    diagrams = []
    diagram_labels = []
    if os.path.exists(pipeline_md_path):
        with open(pipeline_md_path) as f:
            blocks = _extract_mermaid(f.read())
        if blocks:
            diagrams.append(blocks[0])
            diagram_labels.append("End-to-End Pipeline Flow")
        if len(blocks) > 1:
            diagrams.append(blocks[1])
            diagram_labels.append("Standalone Compute (Cache Flow)")

    if os.path.exists(readme_path):
        with open(readme_path) as f:
            readme_blocks = _extract_mermaid(f.read())
        for i, block in enumerate(readme_blocks[:4]):
            diagrams.append(block)
            diagram_labels.append(f"Architecture Diagram {i+1}")

    if not diagrams:
        st.warning("No Mermaid diagrams found. Ensure `docs/pipeline.md` or `README.md` exists.")
    else:
        view_mode = st.radio(
            "Diagram source",
            ["Mermaid (live render)", "Generated visualizations (offline-safe)"],
            horizontal=True,
        )

        if view_mode == "Generated visualizations (offline-safe)":
            _gen_dir = os.path.join("docs", "diagrams")
            _gen_map = {
                "Stage 1 — EDA: Raw Data Exploration": "eda_raw_data_exploration.png",
                "Stage 2 — Feature Extraction (Technical Indicators)": "feature_extraction_indicators.png",
                "Stages 3–6 — Preprocessing Pipeline": "preprocessing_pipeline.png",
                "Stage 7 — Model Selection & Benchmarking": "model_selection_benchmarking.png",
                "Full Pipeline Overview": "ml_pipeline_end_to_end.png",
            }
            gen_label = st.selectbox("Select visualization", list(_gen_map.keys()))
            gen_path = os.path.join(_gen_dir, _gen_map[gen_label])
            if os.path.exists(gen_path):
                st.image(gen_path, use_container_width=True)
                with open(gen_path, "rb") as f:
                    st.download_button(
                        "💾 Download PNG",
                        f.read(),
                        _gen_map[gen_label],
                        "image/png",
                        key=f"dl_gen_{gen_label}",
                    )
            else:
                st.warning(f"Generated image not found: {gen_path}")
        else:
            selected_label = st.selectbox("Select diagram", diagram_labels)
            idx = diagram_labels.index(selected_label)
            graph_def = diagrams[idx].strip()

            png_bytes = mermaid_to_image(graph_def)
            if png_bytes:
                st.image(png_bytes, use_container_width=True)
                st.download_button(
                    "💾 Download diagram PNG",
                    png_bytes,
                    f"pipeline_diagram_{idx+1}.png",
                    "image/png",
                )
            else:
                st.caption("Image service unavailable; rendering via mermaid.js fallback.")
                render_mermaid(graph_def, height=700)

            with st.expander("View raw Mermaid source"):
                st.code(graph_def, language="")

# ==========================================
# TAB 10: EDA GALLERY
# ==========================================
with tab10:
    st.header("Exploratory Data Analysis Gallery")
    st.markdown(
        "Full preprocessing pipeline visualization for the selected ticker. "
        "Shows each stage from raw OHLCV through feature engineering, labeling, and train/test split."
    )

    eda_dir = os.path.join("docs", "eda_plots", ticker)
    fallback_dir = os.path.join("docs", "eda_plots", "AAPL")

    col_left, col_right = st.columns([3, 1])
    with col_left:
        generate_btn = st.button(f"🔬 Generate EDA Plots for {ticker}")
    with col_right:
        show_cached = st.checkbox("Show cached plots", value=True)

    if generate_btn:
        with st.spinner("Running preprocessing pipeline and building visualizations..."):
            try:
                (plot_raw_ohlcv, plot_technical_indicators, plot_labeling,
                 plot_feature_distributions, plot_scaling_impact,
                 plot_train_test_split) = _load_eda_fns()

                df_raw_eda = get_or_download_data(ticker)
                df_eda = add_technical_indicators(df_raw_eda.copy())
                df_eda["Label"] = triple_barrier_labeling(df_eda)

                ohlcv_drop = [c for c in _OHLCV_COLS if c in df_eda.columns]
                X_raw_eda = df_eda.dropna(subset=["Label"]).drop(columns=ohlcv_drop + ["Label"])
                X_scaled_eda, y_eda, _ = prepare_features_and_labels(df_eda)
                X_train_eda, X_test_eda, y_train_eda, y_test_eda = walk_forward_split(X_scaled_eda, y_eda)

                _specs = [
                    ("Stage 1 — Raw OHLCV", lambda: plot_raw_ohlcv(df_raw_eda, None), "01_raw_ohlcv.png"),
                    ("Stage 2 — Technical Indicators", lambda: plot_technical_indicators(df_eda, None), "02_technical_indicators.png"),
                    ("Stage 3 — Triple-Barrier Labels", lambda: plot_labeling(df_eda, None), "03_labels.png"),
                    ("Stage 5 — RobustScaler Impact", lambda: plot_scaling_impact(X_raw_eda, X_scaled_eda, None), "05_scaling.png"),
                    ("Stage 6 — Train/Test Split", lambda: plot_train_test_split(df_eda, X_train_eda, X_test_eda, y_train_eda, y_test_eda, None), "06_split.png"),
                ]

                for title, fn, fname in _specs:
                    st.subheader(title)
                    fig = fn()
                    st.pyplot(fig)
                    _dl(fig, f"{ticker}_{fname}")
                    plt.close(fig)

                st.subheader("Stage 4 — Feature Correlations & Distributions")
                fig_corr, fig_hist = plot_feature_distributions(X_raw_eda, None)
                st.pyplot(fig_corr)
                _dl(fig_corr, f"{ticker}_04a_correlations.png")
                plt.close(fig_corr)
                st.pyplot(fig_hist)
                _dl(fig_hist, f"{ticker}_04b_distributions.png")
                plt.close(fig_hist)

            except Exception as e:
                st.error(f"EDA generation failed: {e}")

    elif show_cached:
        display_dir = eda_dir if os.path.isdir(eda_dir) else (fallback_dir if os.path.isdir(fallback_dir) else None)
        if display_dir:
            if display_dir == fallback_dir and ticker != "AAPL":
                st.info(f"Showing AAPL sample plots. Click 'Generate EDA Plots' for {ticker}.")
            pngs = sorted(f for f in os.listdir(display_dir) if f.endswith(".png"))
            if pngs:
                for fname in pngs:
                    img_path = os.path.join(display_dir, fname)
                    label = fname.replace(".png", "").replace("_", " ").title()
                    st.image(img_path, caption=label, use_container_width=True)
                    with open(img_path, "rb") as f:
                        st.download_button(f"💾 Download {fname}", f.read(), fname, "image/png", key=f"dl_{fname}")
            else:
                st.info("No pre-generated plots found. Click 'Generate EDA Plots' above.")
        else:
            st.info("No cached plots available. Click 'Generate EDA Plots' above.")
