# utils/evaluator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class PortfolioEvaluator:
    """投资组合评估器"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
    
    def backtest(self, 
                returns_df: pd.DataFrame,
                weights: np.ndarray,
                rebalance_freq: str = 'M') -> Dict:
        """
        简单回测
        returns_df: 收益率DataFrame
        weights: 投资组合权重
        """
        # 如果权重是1D，扩展为2D（所有时间点相同权重）
        if weights.ndim == 1:
            weights_array = np.tile(weights, (len(returns_df), 1))
        else:
            weights_array = weights
        
        # 确保权重和为1
        weights_array = weights_array / weights_array.sum(axis=1, keepdims=True)
        
        # 计算每日组合收益率
        portfolio_returns = (returns_df.values * weights_array).sum(axis=1)
        
        # 计算累计净值
        portfolio_values = self.initial_capital * (1 + portfolio_returns).cumprod()
        
        # 计算回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        
        # 计算各项指标
        metrics = self.calculate_metrics(portfolio_returns, portfolio_values, drawdown)
        
        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'drawdown': drawdown,
            'metrics': metrics
        }
    
    def calculate_metrics(self, 
                         returns: np.ndarray,
                         values: np.ndarray,
                         drawdown: np.ndarray) -> Dict:
        """计算绩效指标"""
        n_periods = len(returns)
        
        # 总收益率
        total_return = values[-1] / self.initial_capital - 1
        
        # 年化收益率（假设252个交易日）
        if n_periods > 252:
            annual_return = (1 + total_return) ** (252 / n_periods) - 1
        else:
            annual_return = total_return * 252 / n_periods if n_periods > 0 else 0
        
        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率（无风险利率2%）
        sharpe_ratio = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        max_drawdown = abs(drawdown.min())
        
        # 胜率
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        return {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '年化波动率': annual_vol,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '胜率': win_rate,
            '交易天数': n_periods
        }
    
    def print_report(self, metrics: Dict):
        """打印评估报告"""
        print("\n" + "="*60)
        print("投资组合绩效报告")
        print("="*60)
        
        for key, value in metrics.items():
            if '率' in key or '收益' in key or '撤' in key:
                print(f"{key:10s}: {value:.2%}")
            elif key == '夏普比率':
                print(f"{key:10s}: {value:.3f}")
            else:
                print(f"{key:10s}: {value}")