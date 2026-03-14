"""
投资组合优化器
包含基本和高级优化方法
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    基本投资组合优化器
    实现马科维茨优化和相关变体
    """

    def __init__(self, risk_free_rate: float = 0.02, max_allocation: float = 0.3):
        """
        初始化优化器

        Args:
            risk_free_rate: 无风险利率
            max_allocation: 单只股票最大权重
        """
        self.risk_free_rate = risk_free_rate
        self.max_allocation = max_allocation

    def markowitz_optimization(self, returns_df: pd.DataFrame) -> Dict:
        """
        基本马科维茨优化

        Args:
            returns_df: 收益率数据

        Returns:
            优化结果
        """
        print("⚡ 马科维茨优化...")

        if returns_df.empty:
            return {}

        try:
            n_assets = len(returns_df.columns)

            # 计算预期收益率和协方差
            expected_returns = returns_df.mean().values
            cov_matrix = returns_df.cov().values

            # 目标函数：最大化夏普比率
            def negative_sharpe(weights: np.ndarray) -> float:
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                if portfolio_vol < 1e-8:
                    return 1e6

                sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol
                return -sharpe

            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]

            # 边界条件
            bounds = [(0, self.max_allocation) for _ in range(n_assets)]

            # 初始权重
            initial_weights = np.ones(n_assets) / n_assets

            # 优化
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )

            if result.success:
                weights = result.x

                # 计算绩效
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (
                                           portfolio_return - self.risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0

                return {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'num_assets': n_assets
                }
            else:
                print(f"    优化失败: {result.message}")
                return {}

        except Exception as e:
            print(f"    马科维茨优化失败: {e}")
            return {}

    def embedding_enhanced_optimization(self, returns_df: pd.DataFrame,
                                        embeddings: np.ndarray = None) -> Dict:
        """
        嵌入增强优化
        """
        print("⚡ 嵌入增强优化...")

        if returns_df.empty:
            return {}

        try:
            n_assets = len(returns_df.columns)

            # 计算历史收益和协方差
            expected_returns = returns_df.mean().values
            historical_cov = returns_df.cov().values

            # 合成协方差矩阵
            if embeddings is not None and embeddings.shape[0] == n_assets:
                # 计算嵌入相似度
                embedding_similarity = np.dot(embeddings, embeddings.T)
                embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embedding_similarity = embedding_similarity / (embedding_norms @ embedding_norms.T + 1e-8)

                # 转换为协方差
                embedding_cov = 1 - embedding_similarity
                embedding_cov = (embedding_cov - embedding_cov.min()) / (
                            embedding_cov.max() - embedding_cov.min() + 1e-8)

                # 合成
                combined_cov = 0.7 * historical_cov + 0.3 * embedding_cov
            else:
                combined_cov = historical_cov

            # 优化
            def negative_sharpe(weights: np.ndarray) -> float:
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(combined_cov, weights)))

                if portfolio_vol < 1e-8:
                    return 1e6

                return -(portfolio_return - self.risk_free_rate / 252) / portfolio_vol

            # 约束
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, self.max_allocation) for _ in range(n_assets)]
            initial_weights = np.ones(n_assets) / n_assets

            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )

            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(combined_cov, weights)))
                sharpe_ratio = (
                                           portfolio_return - self.risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0

                return {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'num_assets': n_assets
                }
            else:
                print(f"    嵌入增强优化失败: {result.message}")
                return {}

        except Exception as e:
            print(f"    嵌入增强优化失败: {e}")
            return {}

    def markowitz_optimization_with_custom_returns(self, returns_df: pd.DataFrame,
                                                   custom_returns: np.ndarray,
                                                   embeddings: np.ndarray = None) -> Dict:
        """
        使用自定义收益率的优化
        """
        print("⚡ 使用自定义收益率的优化...")

        if returns_df.empty:
            return {}

        try:
            n_assets = len(returns_df.columns)

            # 处理自定义收益率
            if len(custom_returns) != n_assets:
                print(f"⚠️  收益率维度不匹配: {len(custom_returns)} != {n_assets}")
                expected_returns = returns_df.mean().values
            else:
                # 处理异常值
                mean_pred = np.mean(custom_returns)
                std_pred = np.std(custom_returns)

                if std_pred < 1e-6:
                    # 波动率太小，增加随机性
                    noise = np.random.randn(n_assets) * 1e-4
                    custom_returns = custom_returns + noise
                    std_pred = np.std(custom_returns)

                if std_pred > 1e-6:
                    # 裁剪异常值
                    z_scores = np.abs((custom_returns - mean_pred) / std_pred)
                    extreme_mask = z_scores > 3
                    if np.any(extreme_mask):
                        custom_returns[extreme_mask] = mean_pred + np.sign(
                            custom_returns[extreme_mask] - mean_pred) * 3 * std_pred

                expected_returns = custom_returns

            # 协方差矩阵
            historical_cov = returns_df.cov().values

            if embeddings is not None and embeddings.shape[0] == n_assets:
                # 计算嵌入相似度
                embedding_similarity = np.dot(embeddings, embeddings.T)
                embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embedding_similarity = embedding_similarity / (embedding_norms @ embedding_norms.T + 1e-8)

                embedding_cov = 1 - embedding_similarity
                embedding_cov = (embedding_cov - embedding_cov.min()) / (
                            embedding_cov.max() - embedding_cov.min() + 1e-8)

                combined_cov = 0.7 * historical_cov + 0.3 * embedding_cov
            else:
                combined_cov = historical_cov

            # 优化
            def negative_sharpe(weights: np.ndarray) -> float:
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(combined_cov, weights)))

                if portfolio_vol < 1e-8:
                    return 1e6

                return -(portfolio_return - self.risk_free_rate / 252) / portfolio_vol

            # 约束
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, self.max_allocation) for _ in range(n_assets)]
            initial_weights = np.ones(n_assets) / n_assets

            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )

            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(combined_cov, weights)))
                sharpe_ratio = (
                                           portfolio_return - self.risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0

                print(f"    预测收益率统计: 均值={portfolio_return:.4%}, 波动率={portfolio_vol:.4%}")

                return {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'num_assets': n_assets,
                    'used_custom_returns': True
                }
            else:
                print(f"    优化失败: {result.message}")
                return {}

        except Exception as e:
            print(f"    自定义收益率优化失败: {e}")
            return {}


class AdvancedPortfolioOptimizer:
    """
    高级投资组合优化器
    实现智能选股和CoCVaR优化
    """

    def __init__(self, risk_free_rate: float = 0.02, max_weight: float = 0.3,
                 optimization_method: str = 'markowitz'):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.optimization_method = optimization_method

    def markowitz_optimization_with_embeddings(self, returns_df: pd.DataFrame,
                                               expected_returns: Optional[np.ndarray] = None,
                                               embeddings: Optional[np.ndarray] = None,
                                               alpha: float = 0.7) -> Dict:
        """
        嵌入增强的Markowitz优化
        """
        print("⚡ 嵌入增强的Markowitz优化...")

        try:
            n_assets = len(returns_df.columns)

            # 计算历史协方差
            historical_cov = returns_df.cov().values

            # 计算嵌入协方差
            if embeddings is not None and len(embeddings) == n_assets:
                embedding_similarity = np.dot(embeddings, embeddings.T)
                embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embedding_similarity = embedding_similarity / (embedding_norms @ embedding_norms.T + 1e-8)

                embedding_cov = 1 - embedding_similarity
                embedding_cov = (embedding_cov - embedding_cov.min()) / (
                            embedding_cov.max() - embedding_cov.min() + 1e-8)
            else:
                embedding_cov = np.eye(n_assets)

            # 合成协方差矩阵
            combined_cov = alpha * historical_cov + (1 - alpha) * embedding_cov

            # 预期收益率
            if expected_returns is not None and len(expected_returns) == n_assets:
                mu = expected_returns
            else:
                mu = returns_df.mean().values

            # 优化问题
            def negative_sharpe_ratio(weights: np.ndarray) -> float:
                portfolio_return = np.dot(weights, mu)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(combined_cov, weights)))

                if portfolio_vol < 1e-8:
                    return 1e6

                sharpe_ratio = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol
                return -sharpe_ratio

            # 约束
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, self.max_weight) for _ in range(n_assets)]
            initial_weights = np.ones(n_assets) / n_assets

            result = minimize(
                negative_sharpe_ratio,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )

            if result.success:
                optimal_weights = result.x
                portfolio_return = np.dot(optimal_weights, mu)
                portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(combined_cov, optimal_weights)))
                sharpe_ratio = (
                                           portfolio_return - self.risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0

                annual_return = (1 + portfolio_return) ** 252 - 1
                annual_vol = portfolio_vol * np.sqrt(252)
                annual_sharpe = sharpe_ratio * np.sqrt(252) if sharpe_ratio != 0 else 0

                return {
                    'weights': optimal_weights,
                    'expected_daily_return': portfolio_return,
                    'daily_volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'expected_annual_return': annual_return,
                    'annual_volatility': annual_vol,
                    'annual_sharpe_ratio': annual_sharpe,
                    'covariance_matrix': combined_cov,
                    'optimization_success': True
                }
            else:
                print(f"    优化失败: {result.message}")
                return {}

        except Exception as e:
            print(f"    嵌入增强优化失败: {e}")
            return {}

    def select_top_stocks(self, returns_df: pd.DataFrame,
                          expected_returns: Optional[np.ndarray] = None,
                          n_stocks: int = 20) -> List[str]:
        """
        智能选股
        """
        print(f"🔍 智能选股 (选择前{n_stocks}只股票)...")

        stock_codes = returns_df.columns.tolist()
        n_assets = len(stock_codes)

        if expected_returns is None or len(expected_returns) != n_assets:
            expected_returns = returns_df.mean().values

        # 计算指标
        metrics = {}

        for i, code in enumerate(stock_codes):
            stock_returns = returns_df[code]

            # 计算各项指标
            sharpe_ratio = 0
            if stock_returns.std() > 0:
                sharpe_ratio = (stock_returns.mean() - self.risk_free_rate / 252) / stock_returns.std()

            annual_return = (1 + stock_returns.mean()) ** 252 - 1
            win_rate = (stock_returns > 0).mean()

            momentum = 0
            if len(stock_returns) >= 20:
                momentum = (1 + stock_returns[-20:]).prod() - 1

            # 最大回撤
            cum_returns = (1 + stock_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            pred_return = expected_returns[i] if expected_returns is not None else 0

            # 综合评分
            score = (
                    0.3 * sharpe_ratio +
                    0.2 * annual_return +
                    0.1 * win_rate +
                    0.1 * momentum +
                    0.1 * (-max_drawdown) +
                    0.2 * pred_return
            )

            metrics[code] = score

        # 排序
        sorted_stocks = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = [code for code, _ in sorted_stocks[:n_stocks]]

        print(f"    ✅ 选中股票: {', '.join(selected_stocks[:5])}...")
        return selected_stocks


# 导出
__all__ = ['PortfolioOptimizer', 'AdvancedPortfolioOptimizer']