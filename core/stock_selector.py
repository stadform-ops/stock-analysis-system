# core/stock_selector.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class StockSelector:
    """智能选股器 - 选择表现最好的10只股票"""

    def __init__(self, n_stocks: int = 10, lookback_days: int = 60):
        self.n_stocks = n_stocks
        self.lookback_days = lookback_days

    def calculate_stock_metrics(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """计算股票表现指标"""
        metrics = []

        for stock in returns_df.columns:
            returns = returns_df[stock]

            # 基础指标
            mean_return = returns.mean() * 252
            std_return = returns.std() * np.sqrt(252)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0

            # 回撤计算
            cumulative = (1 + returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()

            # 胜率
            win_rate = (returns > 0).mean()

            # 动量指标
            momentum_1m = (1 + returns.tail(20)).prod() - 1
            momentum_3m = (1 + returns.tail(60)).prod() - 1

            metrics.append({
                'stock': stock,
                'annual_return': mean_return,
                'annual_vol': std_return,
                'sharpe': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m
            })

        return pd.DataFrame(metrics)

    def composite_score(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """计算综合得分"""
        # 归一化指标
        metrics_df['return_score'] = self._normalize(metrics_df['annual_return'], positive=True)
        metrics_df['sharpe_score'] = self._normalize(metrics_df['sharpe'], positive=True)
        metrics_df['winrate_score'] = self._normalize(metrics_df['win_rate'], positive=True)
        metrics_df['momentum_score'] = self._normalize(metrics_df['momentum_3m'], positive=True)
        metrics_df['drawdown_score'] = self._normalize(metrics_df['max_drawdown'], positive=False)  # 回撤越小越好

        # 计算综合得分
        metrics_df['composite_score'] = (
                metrics_df['return_score'] * 0.25 +
                metrics_df['sharpe_score'] * 0.30 +
                metrics_df['winrate_score'] * 0.20 +
                metrics_df['momentum_score'] * 0.15 +
                metrics_df['drawdown_score'] * 0.10
        )

        return metrics_df.sort_values('composite_score', ascending=False)

    def _normalize(self, series: pd.Series, positive: bool = True) -> pd.Series:
        """归一化处理"""
        if positive:
            return (series - series.min()) / (series.max() - series.min() + 1e-8)
        else:
            # 对于越小越好的指标（如回撤）
            return 1 - (series - series.min()) / (series.max() - series.min() + 1e-8)

    def select_top_stocks(self, returns_df: pd.DataFrame) -> List[str]:
        """选择Top N股票"""
        print(f"📊 开始选股分析...")
        print(f"   总股票数: {len(returns_df.columns)}")
        print(f"   目标选择数: {self.n_stocks}")

        # 计算指标
        metrics_df = self.calculate_stock_metrics(returns_df)
        scored_df = self.composite_score(metrics_df)

        # 选择Top N
        top_stocks = scored_df.head(self.n_stocks)['stock'].tolist()

        # 显示选股结果
        print(f"\n🏆 Top{self.n_stocks} 选股结果:")
        print("-" * 80)
        for i, (_, row) in enumerate(scored_df.head(self.n_stocks).iterrows(), 1):
            print(f"{i:2d}. {row['stock']}: "
                  f"年化收益={row['annual_return']:7.2%}, "
                  f"夏普={row['sharpe']:6.3f}, "
                  f"胜率={row['win_rate']:6.2%}, "
                  f"综合得分={row['composite_score']:.4f}")

        # 显示关键统计数据
        print(f"\n📈 选股统计:")
        selected_metrics = scored_df.head(self.n_stocks)
        print(f"   平均年化收益: {selected_metrics['annual_return'].mean():.2%}")
        print(f"   平均夏普比率: {selected_metrics['sharpe'].mean():.4f}")
        print(f"   平均胜率: {selected_metrics['win_rate'].mean():.2%}")

        return top_stocks