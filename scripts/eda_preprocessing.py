"""
EDA & Preprocessing Visualization
Usage:
    python scripts/eda_preprocessing.py --parquet data/raw/AAPL_2020-01-01_2024-01-01.parquet
    python scripts/eda_preprocessing.py --parquet data/raw/AAPL_2020-01-01_2024-01-01.parquet --no-save
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Allow running from project root: python scripts/eda_preprocessing.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import (
    triple_barrier_labeling,
    prepare_features_and_labels,
    walk_forward_split,
)

LABEL_COLORS = {1: "#2ecc71", 0: "#f39c12", -1: "#e74c3c"}
LABEL_NAMES  = {1: "Profit-take (+1)", 0: "Time barrier (0)", -1: "Stop-loss (−1)"}
STYLE = "seaborn-v0_8-darkgrid"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")


def _ticker_from_path(parquet_path):
    """Best-effort: extract ticker from filename like AAPL_2020-01-01_2024-01-01.parquet"""
    stem = os.path.splitext(os.path.basename(parquet_path))[0]
    return stem.split("_")[0]


# ─── Figure 1: Raw OHLCV overview ─────────────────────────────────────────────

def plot_raw_ohlcv(df_raw, out_dir):
    print("[1/6] Raw OHLCV overview")
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                                 gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.suptitle("Stage 1 · Raw OHLCV Data", fontsize=14, fontweight="bold")

        ax_price, ax_vol, ax_ret = axes

        # Price + High/Low band
        ax_price.fill_between(df_raw.index, df_raw["Low"], df_raw["High"],
                              alpha=0.15, color="#3498db", label="High–Low range")
        ax_price.plot(df_raw.index, df_raw["Close"], color="#2c3e50", lw=1.2, label="Close")
        ax_price.plot(df_raw.index, df_raw["Open"],  color="#7f8c8d", lw=0.6,
                      alpha=0.7, label="Open")
        ax_price.set_ylabel("Price (USD)")
        ax_price.legend(fontsize=8)
        ax_price.set_title("Price", fontsize=10)

        # Volume
        ax_vol.bar(df_raw.index, df_raw["Volume"], width=1, color="#2980b9", alpha=0.7)
        ax_vol.set_ylabel("Volume")
        ax_vol.set_title("Volume", fontsize=10)

        # Daily simple returns
        daily_ret = df_raw["Close"].pct_change() * 100
        pos = daily_ret.clip(lower=0)
        neg = daily_ret.clip(upper=0)
        ax_ret.bar(df_raw.index, pos, width=1, color="#27ae60", alpha=0.8)
        ax_ret.bar(df_raw.index, neg, width=1, color="#c0392b", alpha=0.8)
        ax_ret.axhline(0, color="black", lw=0.5)
        ax_ret.set_ylabel("Daily Return (%)")
        ax_ret.set_title("Daily Returns", fontsize=10)

        for ax in axes:
            ax.set_xlabel("")
        plt.tight_layout()
        if out_dir is not None:
            _save(fig, out_dir, "01_raw_ohlcv.png")
        return fig


# ─── Figure 2: Technical indicators ──────────────────────────────────────────

def plot_technical_indicators(df, out_dir):
    print("[2/6] Technical indicators")
    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(14, 16))
        gs  = gridspec.GridSpec(5, 1, hspace=0.45)
        fig.suptitle("Stage 2 · Feature Engineering — Technical Indicators",
                     fontsize=14, fontweight="bold")

        # 2a: Price + SMA + Bollinger Bands
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df["Close"],     color="#2c3e50", lw=1.2, label="Close")
        ax1.plot(df.index, df["SMA_20"],    color="#e67e22", lw=1,   label="SMA 20")
        ax1.plot(df.index, df["SMA_50"],    color="#8e44ad", lw=1,   label="SMA 50")
        ax1.plot(df.index, df["BB_Upper"],  color="#2980b9", lw=0.8, ls="--", label="BB Upper")
        ax1.plot(df.index, df["BB_Lower"],  color="#2980b9", lw=0.8, ls="--", label="BB Lower")
        ax1.fill_between(df.index, df["BB_Lower"], df["BB_Upper"],
                         alpha=0.07, color="#2980b9")
        ax1.set_title("Close + SMA (20/50) + Bollinger Bands", fontsize=10)
        ax1.set_ylabel("Price (USD)")
        ax1.legend(fontsize=7, ncol=3)

        # 2b: RSI
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df.index, df["RSI"], color="#16a085", lw=1)
        ax2.axhline(70, color="#e74c3c", lw=0.8, ls="--", label="Overbought (70)")
        ax2.axhline(30, color="#27ae60", lw=0.8, ls="--", label="Oversold (30)")
        ax2.axhline(50, color="gray",    lw=0.5, ls=":")
        ax2.fill_between(df.index, df["RSI"], 70,
                         where=(df["RSI"] >= 70), alpha=0.3, color="#e74c3c")
        ax2.fill_between(df.index, df["RSI"], 30,
                         where=(df["RSI"] <= 30), alpha=0.3, color="#27ae60")
        ax2.set_ylim(0, 100)
        ax2.set_title("RSI (14-period)", fontsize=10)
        ax2.set_ylabel("RSI")
        ax2.legend(fontsize=7)

        # 2c: MACD
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(df.index, df["MACD"],        color="#2980b9", lw=1,   label="MACD")
        ax3.plot(df.index, df["MACD_Signal"], color="#e74c3c", lw=1,   label="Signal")
        hist_pos = df["MACD_Hist"].clip(lower=0)
        hist_neg = df["MACD_Hist"].clip(upper=0)
        ax3.bar(df.index, hist_pos, width=1, color="#27ae60", alpha=0.6, label="Histogram +")
        ax3.bar(df.index, hist_neg, width=1, color="#e74c3c", alpha=0.6, label="Histogram −")
        ax3.axhline(0, color="black", lw=0.5)
        ax3.set_title("MACD (12/26/9)", fontsize=10)
        ax3.set_ylabel("MACD")
        ax3.legend(fontsize=7, ncol=4)

        # 2d: Momentum
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(df.index, df["Momentum"], color="#8e44ad", lw=1)
        ax4.axhline(0, color="black", lw=0.5)
        ax4.fill_between(df.index, df["Momentum"], 0,
                         where=(df["Momentum"] >= 0), alpha=0.3, color="#27ae60")
        ax4.fill_between(df.index, df["Momentum"], 0,
                         where=(df["Momentum"] < 0),  alpha=0.3, color="#e74c3c")
        ax4.set_title("Momentum (10-day diff)", fontsize=10)
        ax4.set_ylabel("Momentum")

        # 2e: Volatility + Log Returns
        ax5 = fig.add_subplot(gs[4])
        ax5_twin = ax5.twinx()
        ax5.bar(df.index, df["Log_Returns"], width=1, color="#7f8c8d",
                alpha=0.5, label="Log Returns")
        ax5_twin.plot(df.index, df["Volatility"], color="#c0392b", lw=1.2,
                      label="Volatility (20d)")
        ax5.set_title("Log Returns & Rolling Volatility (20-day std)", fontsize=10)
        ax5.set_ylabel("Log Returns", color="#7f8c8d")
        ax5_twin.set_ylabel("Volatility", color="#c0392b")
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

        if out_dir is not None:
            _save(fig, out_dir, "02_technical_indicators.png")
        return fig


# ─── Figure 3: Triple-barrier labeling ───────────────────────────────────────

def plot_labeling(df, out_dir):
    print("[3/6] Triple-barrier labeling")
    df_labeled = df.dropna(subset=["Label"])

    with plt.style.context(STYLE):
        fig, (ax_price, ax_bar) = plt.subplots(1, 2, figsize=(14, 5),
                                                gridspec_kw={"width_ratios": [3, 1]})
        fig.suptitle("Stage 3 · Triple-Barrier Labeling", fontsize=14, fontweight="bold")

        # Price scatter colored by label
        for lbl in [1, 0, -1]:
            mask = df_labeled["Label"] == lbl
            ax_price.scatter(df_labeled.index[mask], df_labeled["Close"][mask],
                             c=LABEL_COLORS[lbl], s=4, alpha=0.7,
                             label=LABEL_NAMES[lbl], rasterized=True)
        ax_price.plot(df_labeled.index, df_labeled["Close"],
                      color="#bdc3c7", lw=0.6, alpha=0.5, zorder=0)
        ax_price.set_title("Close price colored by label", fontsize=10)
        ax_price.set_ylabel("Price (USD)")
        ax_price.legend(fontsize=8, markerscale=3)

        # Label distribution bar
        counts = df_labeled["Label"].value_counts().sort_index()
        colors = [LABEL_COLORS[l] for l in counts.index]
        bars = ax_bar.bar([LABEL_NAMES[l] for l in counts.index], counts.values,
                          color=colors, edgecolor="white", width=0.5)
        ax_bar.bar_label(bars, fmt="%d", padding=3, fontsize=9)
        total = counts.sum()
        ax_bar.set_title(f"Label distribution (n={total:,})", fontsize=10)
        ax_bar.set_ylabel("Count")
        ax_bar.tick_params(axis="x", labelsize=7)

        plt.tight_layout()
        if out_dir is not None:
            _save(fig, out_dir, "03_triple_barrier_labels.png")
        return fig


# ─── Figure 4: Feature correlations & distributions (pre-scale) ──────────────

def plot_feature_distributions(X_raw, out_dir):
    print("[4/6] Feature distributions & correlations")
    features = X_raw.columns.tolist()

    with plt.style.context(STYLE):
        # 4a: correlation heatmap
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr = X_raw.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax_corr, annot_kws={"size": 7})
        ax_corr.set_title("Stage 4 · Feature Correlation Matrix (pre-scaling)",
                          fontsize=13, fontweight="bold")
        plt.tight_layout()
        if out_dir is not None:
            _save(fig_corr, out_dir, "04a_feature_correlations.png")

        # 4b: individual histograms
        ncols = 4
        nrows = int(np.ceil(len(features) / ncols))
        fig_hist, axes = plt.subplots(nrows, ncols,
                                      figsize=(ncols * 3.5, nrows * 3))
        fig_hist.suptitle("Stage 4 · Feature Distributions (pre-scaling)",
                          fontsize=13, fontweight="bold")
        for i, feat in enumerate(features):
            ax = axes.flat[i]
            ax.hist(X_raw[feat].dropna(), bins=50, color="#2980b9",
                    edgecolor="white", alpha=0.8)
            ax.axvline(X_raw[feat].median(), color="#e74c3c", lw=1.2,
                       ls="--", label=f"median={X_raw[feat].median():.3g}")
            ax.set_title(feat, fontsize=9)
            ax.legend(fontsize=7)
        for j in range(i + 1, len(axes.flat)):
            axes.flat[j].set_visible(False)
        plt.tight_layout()
        if out_dir is not None:
            _save(fig_hist, out_dir, "04b_feature_distributions.png")
        return fig_corr, fig_hist


# ─── Figure 5: Scaling impact (before vs after RobustScaler) ─────────────────

def plot_scaling_impact(X_raw, X_scaled, out_dir):
    print("[5/6] Scaling impact (RobustScaler before vs after)")
    # Pick 6 representative features
    showcase = ["RSI", "MACD", "Volatility", "Log_Returns", "Momentum", "BB_Upper"]
    showcase = [f for f in showcase if f in X_raw.columns]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(len(showcase), 2,
                                 figsize=(12, len(showcase) * 2.2))
        fig.suptitle("Stage 5 · RobustScaler: Before vs After",
                     fontsize=13, fontweight="bold")
        for row, feat in enumerate(showcase):
            for col, (data, label, color) in enumerate([
                (X_raw[feat],    "Before scaling", "#e67e22"),
                (X_scaled[feat], "After RobustScaler", "#2980b9"),
            ]):
                ax = axes[row][col]
                ax.hist(data.dropna(), bins=60, color=color,
                        edgecolor="white", alpha=0.8)
                ax.set_title(f"{feat} — {label}", fontsize=8)
                ax.axvline(data.median(), color="black", lw=1, ls="--")
                stats = f"μ={data.mean():.3g}  σ={data.std():.3g}"
                ax.set_xlabel(stats, fontsize=7)
        plt.tight_layout()
        if out_dir is not None:
            _save(fig, out_dir, "05_scaling_impact.png")
        return fig


# ─── Figure 6: Train/test split ───────────────────────────────────────────────

def plot_train_test_split(df, X_train, X_test, y_train, y_test, out_dir):
    print("[6/6] Train/test split")
    split_date = X_test.index[0]
    label_counts = {
        "Train": y_train.value_counts().sort_index(),
        "Test":  y_test.value_counts().sort_index(),
    }

    with plt.style.context(STYLE):
        fig, (ax_price, ax_bar) = plt.subplots(1, 2, figsize=(14, 5),
                                                gridspec_kw={"width_ratios": [3, 1]})
        fig.suptitle("Stage 6 · Walk-Forward Train / Test Split (80 / 20)",
                     fontsize=14, fontweight="bold")

        # Price with train/test shading
        close = df.loc[X_train.index.union(X_test.index), "Close"]
        ax_price.plot(close.index, close, color="#2c3e50", lw=1)
        ax_price.axvspan(X_train.index[0], split_date,
                         alpha=0.12, color="#27ae60", label=f"Train  (n={len(X_train):,})")
        ax_price.axvspan(split_date, X_test.index[-1],
                         alpha=0.15, color="#e74c3c",  label=f"Test   (n={len(X_test):,})")
        ax_price.axvline(split_date, color="#e74c3c", lw=1.5, ls="--")
        ax_price.set_title("Close price — train / test regions", fontsize=10)
        ax_price.set_ylabel("Price (USD)")
        ax_price.legend(fontsize=9)

        # Stacked label counts
        x = np.arange(2)
        width = 0.55
        bottoms = np.zeros(2)
        split_names = ["Train", "Test"]
        for lbl in [-1, 0, 1]:
            vals = [label_counts[s].get(lbl, 0) for s in split_names]
            ax_bar.bar(x, vals, width, bottom=bottoms,
                       color=LABEL_COLORS[lbl], label=LABEL_NAMES[lbl],
                       edgecolor="white")
            bottoms += np.array(vals, dtype=float)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(split_names)
        ax_bar.set_title("Label distribution per split", fontsize=10)
        ax_bar.set_ylabel("Count")
        ax_bar.legend(fontsize=7)

        plt.tight_layout()
        if out_dir is not None:
            _save(fig, out_dir, "06_train_test_split.png")
        return fig


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA & preprocessing visualizations")
    parser.add_argument("--parquet", required=True,
                        help="Path to raw parquet file, e.g. data/raw/AAPL_2020-01-01_2024-01-01.parquet")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for plots (default: docs/eda_plots/<ticker>)")
    parser.add_argument("--no-save", action="store_true",
                        help="Show plots interactively instead of saving to disk")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        print(f"ERROR: file not found: {args.parquet}")
        print("Run data ingestion first:")
        print("  python -c \"from src.data_ingestion import fetch_stock_data; "
              "fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')\"")
        sys.exit(1)

    ticker  = _ticker_from_path(args.parquet)
    out_dir = args.out_dir or os.path.join("docs", "eda_plots", ticker)

    print(f"\nLoading {args.parquet} ...")
    df_raw = pd.read_parquet(args.parquet)

    # Flatten yfinance MultiIndex columns e.g. ('Close', 'AAPL') → 'Close'
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = [col[0] for col in df_raw.columns]

    print(f"  {len(df_raw):,} rows  |  columns: {list(df_raw.columns)}\n")

    # ── pipeline ──────────────────────────────────────────────────────────────
    print("Running preprocessing pipeline...")
    df = add_technical_indicators(df_raw.copy())
    df["Label"] = triple_barrier_labeling(df)

    # Raw feature matrix (before scaling) for distribution plots
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    X_raw = df.dropna(subset=["Label"]).drop(
        columns=[c for c in ohlcv_cols if c in df.columns] + ["Label"]
    )

    X_scaled, y, scaler = prepare_features_and_labels(df)
    X_train, X_test, y_train, y_test = walk_forward_split(X_scaled, y)
    print(f"  features: {list(X_scaled.columns)}")
    print(f"  train: {len(X_train):,}  test: {len(X_test):,}\n")

    # ── plots ─────────────────────────────────────────────────────────────────
    if args.no_save:
        out_dir = None  # suppress saving; plt.show() called at end instead

    figs = [
        plot_raw_ohlcv(df_raw, out_dir or "."),
        plot_technical_indicators(df, out_dir or "."),
        plot_labeling(df, out_dir or "."),
        *plot_feature_distributions(X_raw, out_dir or "."),
        plot_scaling_impact(X_raw, X_scaled, out_dir or "."),
        plot_train_test_split(df, X_train, X_test, y_train, y_test, out_dir or "."),
    ]

    if args.no_save:
        plt.show()
    else:
        print(f"\nAll plots saved to: {out_dir}/")

    for f in figs:
        plt.close(f)


if __name__ == "__main__":
    main()
