#!/usr/bin/env python3
"""
答辩导向基线实验脚本（兼容 Python 3.7）

用途：
1) 在不依赖深度学习模块的前提下，快速跑出可答辩的 baseline 结果；
2) 生成可直接放入中期/答辩材料的关键图表与指标；
3) 作为后续模型（LSTM/GNN/混合）对比基线。

输出：
- results/defense/<timestamp>/metrics_summary.csv
- results/defense/<timestamp>/portfolio_curves.png
- results/defense/<timestamp>/drawdown_curves.png
- results/defense/<timestamp>/rolling_sharpe.png
- results/defense/<timestamp>/weights_heatmap.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


@dataclass
class StrategyResult:
    name: str
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    weights: pd.DataFrame


def annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(daily_returns) == 0:
        return 0.0
    total = (1 + daily_returns).prod()
    years = len(daily_returns) / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def annualized_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(daily_returns) == 0:
        return 0.0
    return float(daily_returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    vol = annualized_volatility(daily_returns, periods_per_year)
    if vol < 1e-12:
        return 0.0
    ann_ret = annualized_return(daily_returns, periods_per_year)
    return float((ann_ret - risk_free_rate) / vol)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    if len(cumulative_returns) == 0:
        return 0.0
    wealth = (1 + cumulative_returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return float(dd.min())


def compute_metrics(daily_returns: pd.Series, cumulative_returns: pd.Series) -> Dict[str, float]:
    return {
        "annual_return": annualized_return(daily_returns),
        "annual_volatility": annualized_volatility(daily_returns),
        "sharpe": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(cumulative_returns),
        "total_return": float((1 + daily_returns).prod() - 1),
    }


def load_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> List[pd.Timestamp]:
    if len(index) == 0:
        return []
    s = pd.Series(index=index, data=1)
    return list(s.resample(freq).last().index.intersection(index))


def _long_only_weight_opt(mean_ret: np.ndarray, cov: np.ndarray, max_weight: float = 0.15) -> np.ndarray:
    n = len(mean_ret)
    if n == 0:
        return np.array([])

    def neg_sharpe(w: np.ndarray) -> float:
        p_ret = float(np.dot(w, mean_ret))
        p_vol = float(np.sqrt(np.dot(w, np.dot(cov, w))))
        if p_vol < 1e-10:
            return 1e6
        return -p_ret / p_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(n)]
    w0 = np.ones(n) / n

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        return res.x
    return w0


def build_equal_weight(returns: pd.DataFrame, rebalance_freq: str = "M") -> StrategyResult:
    dates = rebalance_dates(returns.index, rebalance_freq)
    cols = returns.columns.tolist()
    n = len(cols)

    weights = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
    current_w = np.ones(n) / n

    for dt in returns.index:
        if dt in dates:
            current_w = np.ones(n) / n
        weights.loc[dt] = current_w

    daily = (weights.shift(1).fillna(method="bfill") * returns).sum(axis=1)
    cum = (1 + daily).cumprod() - 1
    return StrategyResult("equal_weight", daily, cum, weights)


def build_momentum_topk(
    returns: pd.DataFrame,
    lookback: int = 20,
    topk: int = 20,
    rebalance_freq: str = "M",
) -> StrategyResult:
    dates = rebalance_dates(returns.index, rebalance_freq)
    cols = returns.columns.tolist()
    weights = pd.DataFrame(0.0, index=returns.index, columns=cols)

    current_w = np.zeros(len(cols))
    for i, dt in enumerate(returns.index):
        if dt in dates and i >= lookback:
            hist = returns.iloc[i - lookback:i]
            score = hist.mean().sort_values(ascending=False)
            selected = score.head(topk).index
            current_w = np.zeros(len(cols))
            idx = [cols.index(c) for c in selected]
            current_w[idx] = 1.0 / len(idx)
        weights.loc[dt] = current_w

    daily = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    cum = (1 + daily).cumprod() - 1
    return StrategyResult("momentum_topk", daily, cum, weights)


def build_markowitz(
    returns: pd.DataFrame,
    train_window: int = 120,
    max_weight: float = 0.15,
    rebalance_freq: str = "M",
) -> StrategyResult:
    dates = rebalance_dates(returns.index, rebalance_freq)
    cols = returns.columns.tolist()
    weights = pd.DataFrame(0.0, index=returns.index, columns=cols)

    current_w = np.ones(len(cols)) / len(cols)
    for i, dt in enumerate(returns.index):
        if dt in dates and i >= train_window:
            hist = returns.iloc[i - train_window:i]
            mean_ret = hist.mean().values
            cov = hist.cov().values + np.eye(len(cols)) * 1e-6
            current_w = _long_only_weight_opt(mean_ret, cov, max_weight=max_weight)
        weights.loc[dt] = current_w

    daily = (weights.shift(1).fillna(method="bfill") * returns).sum(axis=1)
    cum = (1 + daily).cumprod() - 1
    return StrategyResult("markowitz", daily, cum, weights)


def plot_curves(results: List[StrategyResult], out_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(r.cumulative_returns.index, r.cumulative_returns.values, label=r.name, linewidth=1.8)
    plt.title("Portfolio Cumulative Return Curves")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_drawdowns(results: List[StrategyResult], out_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    for r in results:
        wealth = (1 + r.daily_returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        plt.plot(drawdown.index, drawdown.values, label=r.name, linewidth=1.5)
    plt.title("Drawdown Curves")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_rolling_sharpe(results: List[StrategyResult], out_path: Path, window: int = 60) -> None:
    plt.figure(figsize=(12, 6))
    for r in results:
        rolling_mean = r.daily_returns.rolling(window).mean()
        rolling_std = r.daily_returns.rolling(window).std().replace(0, np.nan)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        plt.plot(rolling_sharpe.index, rolling_sharpe.values, label=r.name, linewidth=1.5)
    plt.title("Rolling Sharpe (window=60)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_weight_heatmap(weights: pd.DataFrame, out_path: Path, top_n_assets: int = 30) -> None:
    avg_w = weights.mean().sort_values(ascending=False)
    selected = avg_w.head(top_n_assets).index
    mat = weights[selected].T.values

    plt.figure(figsize=(13, 7))
    plt.imshow(mat, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Weight")
    plt.title("Weight Heatmap (Top 30 assets by avg weight)")
    plt.xlabel("Time Index")
    plt.ylabel("Assets")
    plt.yticks(range(len(selected)), selected)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run(args: argparse.Namespace) -> Path:
    returns = load_returns(Path(args.returns_csv))
    if args.max_assets > 0:
        returns = returns.iloc[:, : args.max_assets]

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    ew = build_equal_weight(returns, args.rebalance_freq)
    mom = build_momentum_topk(returns, args.lookback, args.topk, args.rebalance_freq)
    mkv = build_markowitz(returns, args.train_window, args.max_weight, args.rebalance_freq)

    results = [ew, mom, mkv]

    metric_rows = []
    for r in results:
        m = compute_metrics(r.daily_returns, r.cumulative_returns)
        m["strategy"] = r.name
        metric_rows.append(m)

    metrics_df = pd.DataFrame(metric_rows).set_index("strategy")
    metrics_df.to_csv(output_dir / "metrics_summary.csv")

    plot_curves(results, output_dir / "portfolio_curves.png")
    plot_drawdowns(results, output_dir / "drawdown_curves.png")
    plot_rolling_sharpe(results, output_dir / "rolling_sharpe.png")
    plot_weight_heatmap(mkv.weights, output_dir / "weights_heatmap.png")

    print("✅ 基线实验完成，结果目录:", output_dir)
    print(metrics_df)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Defense baseline backtest")
    parser.add_argument("--returns-csv", default="data/processed/stock_returns.csv")
    parser.add_argument("--output-dir", default="results/defense")
    parser.add_argument("--rebalance-freq", default="M")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--max-weight", type=float, default=0.15)
    parser.add_argument("--max-assets", type=int, default=120, help="控制运行时长；0 表示使用全部资产")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
