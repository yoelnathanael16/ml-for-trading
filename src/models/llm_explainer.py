"""
LLM explainability via Alibaba Qwen (DashScope, OpenAI-compatible endpoint).

Streamlit-free: import and call from the dashboard or any other context.
"""

import json
import requests

BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-flash"

_SYSTEM = (
    "You are a clear, concise financial-data analyst. "
    "When given quantitative trading-model output, explain what the numbers mean "
    "in 3-4 plain sentences aimed at a retail investor. "
    "Highlight what is notable or unusual. "
    "Do not give investment advice or add disclaimer boilerplate."
)

_USER_TEMPLATES = {
    "shap": (
        "Model: {model} | Ticker: {ticker}\n"
        "Top SHAP feature importances (mean |SHAP value|):\n{features}\n"
        "Explain which features drive this model's predictions and why they matter."
    ),
    "risk": (
        "Ticker: {ticker}\n"
        "GARCH annualised vol forecast: {garch_vol}\n"
        "Rolling 20-day annualised vol: {rolling_vol}\n"
        "CVaR 95%: {cvar_95} | CVaR 99%: {cvar_99}\n"
        "VaR 95%: {var_95} | VaR 99%: {var_99}\n"
        "CVaR position scale: {position_scale}\n"
        "Anomaly detected: {anomaly_flag} (score {anomaly_score})\n"
        "Interpret this risk profile and note anything traders should watch."
    ),
    "benchmark": (
        "Ticker: {ticker}\n"
        "Model benchmark results (Base = no risk controls, Adv = Kelly sizing + stops):\n"
        "{table}\n"
        "Summarise which model performed best and what the base-vs-advanced gap reveals."
    ),
}


def build_messages(kind: str, facts: dict) -> list[dict]:
    """Pure function — no network. Returns OpenAI-style messages list."""
    if kind == "shap":
        features_txt = "\n".join(
            f"  {f}: {v:.4f}" for f, v in
            sorted(facts.get("top_features", {}).items(), key=lambda x: -x[1])
        )
        user_msg = _USER_TEMPLATES["shap"].format(
            model=facts.get("model", "?"),
            ticker=facts.get("ticker", "?"),
            features=features_txt,
        )
    elif kind == "risk":
        def _pct(v):
            return f"{v*100:.2f}%" if v is not None else "N/A"
        user_msg = _USER_TEMPLATES["risk"].format(
            ticker=facts.get("ticker", "?"),
            garch_vol=_pct(facts.get("garch_vol")),
            rolling_vol=_pct(facts.get("rolling_vol")),
            cvar_95=_pct(facts.get("cvar_95")),
            cvar_99=_pct(facts.get("cvar_99")),
            var_95=_pct(facts.get("var_95")),
            var_99=_pct(facts.get("var_99")),
            position_scale=f"{facts.get('position_scale', 1.0):.2f}x",
            anomaly_flag="YES" if facts.get("anomaly_flag") else "NO",
            anomaly_score=f"{facts.get('anomaly_score', 0.0):.4f}",
        )
    elif kind == "benchmark":
        table = facts.get("table", {})
        rows = []
        for model, metrics in table.items():
            row = f"  {model}: " + ", ".join(f"{k}={v}" for k, v in metrics.items())
            rows.append(row)
        user_msg = _USER_TEMPLATES["benchmark"].format(
            ticker=facts.get("ticker", "?"),
            table="\n".join(rows) if rows else str(table),
        )
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def call_qwen(messages: list[dict], api_key: str, timeout: int = 20) -> str | None:
    """POST to DashScope; return text or None on any failure (mirrors dashboard api_get style)."""
    try:
        resp = requests.post(
            BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "temperature": 0.3},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        pass
    return None


def explain(kind: str, facts: dict, api_key: str) -> str | None:
    """Build prompt for `kind`, call Qwen, return narrative or None."""
    return call_qwen(build_messages(kind, facts), api_key)


if __name__ == "__main__":
    # ponytail: self-check — no network, just verify message structure
    for k, sample in [
        ("shap", {"model": "RandomForest", "ticker": "AAPL", "top_features": {"RSI": 0.12, "MACD": 0.08}}),
        ("risk", {"ticker": "AAPL", "garch_vol": 0.28, "rolling_vol": 0.25, "cvar_95": -0.032,
                  "cvar_99": -0.055, "var_95": -0.028, "var_99": -0.048,
                  "position_scale": 0.85, "anomaly_flag": False, "anomaly_score": -0.12}),
        ("benchmark", {"ticker": "AAPL", "table": {
            "XGBoost": {"Total Return (Adv)": 0.34, "Sharpe Ratio (Adv)": 1.2},
            "LightGBM": {"Total Return (Adv)": 0.29, "Sharpe Ratio (Adv)": 1.05},
        }}),
    ]:
        msgs = build_messages(k, sample)
        assert len(msgs) == 2, f"{k}: expected 2 messages"
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        content = msgs[1]["content"]
        # check at least one fact value lands in user message
        assert any(str(v) in content or str(v)[:4] in content
                   for v in sample.values() if isinstance(v, (str, float, int))), \
            f"{k}: fact value not found in user message"
    print("All self-checks passed.")
