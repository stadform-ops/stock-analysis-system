"""
可视化工具模块
提供多种图表生成功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class Visualizer:
    """可视化类"""

    def __init__(self, output_dir: str = "results/visualizations"):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_returns_distribution(self, returns_df: pd.DataFrame,
                                  n_stocks: int = 5,
                                  save_name: Optional[str] = None) -> plt.Figure:
        """
        绘制收益率分布图

        Args:
            returns_df: 收益率DataFrame
            n_stocks: 显示的股票数量
            save_name: 保存文件名

        Returns:
            Matplotlib图表对象
        """
        print("   生成收益率分布图...")

        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. 整体收益率分布
            flat_returns = returns_df.values.flatten()
            axes[0, 0].hist(flat_returns, bins=100, alpha=0.7,
                            edgecolor='black', density=True)
            axes[0, 0].axvline(x=np.mean(flat_returns), color='red',
                               linestyle='--', linewidth=2,
                               label=f'均值: {np.mean(flat_returns):.4%}')
            axes[0, 0].set_title(f'全体股票收益率分布 (共{len(returns_df.columns)}只)')
            axes[0, 0].set_xlabel('日收益率')
            axes[0, 0].set_ylabel('密度')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 累积收益率曲线（前n_stocks只）
            n_stocks = min(n_stocks, len(returns_df.columns))
            for i in range(n_stocks):
                stock_returns = returns_df.iloc[:, i]
                cumulative = (1 + stock_returns).cumprod()
                axes[0, 1].plot(cumulative.index, cumulative.values,
                                label=returns_df.columns[i], alpha=0.7, linewidth=1.5)
            axes[0, 1].set_title(f'累积收益率曲线 (前{n_stocks}只)')
            axes[0, 1].set_xlabel('日期')
            axes[0, 1].set_ylabel('累积收益率')
            axes[0, 1].legend(loc='upper left', fontsize='small')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. 波动率分布
            volatilities = returns_df.std() * np.sqrt(252)
            axes[1, 0].hist(volatilities, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=np.mean(volatilities), color='red',
                               linestyle='--', linewidth=2,
                               label=f'均值: {np.mean(volatilities):.2%}')
            axes[1, 0].set_title('年化波动率分布')
            axes[1, 0].set_xlabel('年化波动率')
            axes[1, 0].set_ylabel('股票数量')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 收益率统计
            stats_data = {
                '平均值 (%)': returns_df.mean() * 100,
                '标准差 (%)': returns_df.std() * 100,
                '夏普比率': returns_df.mean() / returns_df.std() * np.sqrt(252)
            }
            stats_df = pd.DataFrame(stats_data)
            top_stats = stats_df.nlargest(10, '夏普比率')

            x_pos = np.arange(len(top_stats))
            width = 0.25

            axes[1, 1].bar(x_pos - width, top_stats['平均值 (%)'], width,
                           label='平均收益率', alpha=0.7)
            axes[1, 1].bar(x_pos, top_stats['标准差 (%)'], width,
                           label='波动率', alpha=0.7)
            axes[1, 1].bar(x_pos + width, top_stats['夏普比率'], width,
                           label='夏普比率', alpha=0.7)

            axes[1, 1].set_title('前10名股票绩效指标')
            axes[1, 1].set_xlabel('股票代码')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(top_stats.index, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            if save_name:
                save_path = self.output_dir / f"{save_name}_returns_distribution.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   图表已保存: {save_path}")

            return fig

        except Exception as e:
            print(f"   生成收益率分布图失败: {e}")
            # 创建一个简单的图表作为回退
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '收益率分布图生成失败',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16)
            return fig

    def plot_portfolio_values(self, returns_df: pd.DataFrame, weights: np.ndarray,
                              initial_capital: float = 100000,
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        绘制投资组合净值曲线

        Args:
            returns_df: 收益率DataFrame
            weights: 投资组合权重
            initial_capital: 初始资金
            save_name: 保存文件名

        Returns:
            Matplotlib图表对象
        """
        print("   生成投资组合净值曲线...")

        try:
            # 计算投资组合日收益率
            portfolio_returns = returns_df @ weights

            # 计算净值曲线
            portfolio_values = (1 + portfolio_returns).cumprod() * initial_capital

            fig, ax = plt.subplots(figsize=(12, 6))

            # 绘制净值曲线
            ax.plot(portfolio_values.index, portfolio_values.values,
                    'b-', linewidth=2, label='投资组合净值')

            # 计算基准（等权重组合）
            n_assets = len(returns_df.columns)
            equal_weights = np.ones(n_assets) / n_assets
            benchmark_returns = returns_df @ equal_weights
            benchmark_values = (1 + benchmark_returns).cumprod() * initial_capital
            ax.plot(benchmark_values.index, benchmark_values.values,
                    'r--', linewidth=1.5, alpha=0.7, label='等权重基准')

            # 标注关键点
            max_value = portfolio_values.max()
            max_date = portfolio_values.idxmax()
            final_value = portfolio_values.iloc[-1]

            ax.annotate(f'最高: ¥{max_value:,.0f}',
                        xy=(max_date, max_value),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')

            ax.annotate(f'最终: ¥{final_value:,.0f}',
                        xy=(portfolio_values.index[-1], final_value),
                        xytext=(-100, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green')

            ax.set_title('投资组合净值曲线')
            ax.set_xlabel('日期')
            ax.set_ylabel('净值 (元)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            if save_name:
                save_path = self.output_dir / f"{save_name}_portfolio_values.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   图表已保存: {save_path}")

            return fig

        except Exception as e:
            print(f"   生成净值曲线图失败: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '净值曲线图生成失败',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16)
            return fig

    def plot_stock_weights(self, weights: np.ndarray, stock_codes: List[str],
                           top_n: int = 10, save_name: Optional[str] = None) -> plt.Figure:
        """
        绘制股票权重分布图

        Args:
            weights: 权重数组
            stock_codes: 股票代码列表
            top_n: 显示前n个权重最大的股票
            save_name: 保存文件名

        Returns:
            Matplotlib图表对象
        """
        print("   生成股票权重分布图...")

        try:
            # 创建权重DataFrame
            weight_df = pd.DataFrame({
                'stock_code': stock_codes,
                'weight': weights
            })

            # 按权重排序
            weight_df = weight_df.sort_values('weight', ascending=False)

            # 只显示前top_n个
            top_weights = weight_df.head(top_n)

            fig, ax = plt.subplots(figsize=(10, 6))

            # 水平条形图
            bars = ax.barh(range(len(top_weights)), top_weights['weight'] * 100)

            # 添加权重值标签
            for i, (idx, row) in enumerate(top_weights.iterrows()):
                weight_pct = row['weight'] * 100
                ax.text(weight_pct + 0.1, i, f'{weight_pct:.2f}%',
                        va='center', fontsize=9)

            ax.set_yticks(range(len(top_weights)))
            ax.set_yticklabels(top_weights['stock_code'])
            ax.invert_yaxis()  # 权重最大的在顶部
            ax.set_xlabel('权重 (%)')
            ax.set_title(f'前{top_n}大权重股票分布')
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()

            if save_name:
                save_path = self.output_dir / f"{save_name}_stock_weights.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   图表已保存: {save_path}")

            return fig

        except Exception as e:
            print(f"   生成权重分布图失败: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '权重分布图生成失败',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16)
            return fig

    def plot_correlation_heatmap(self, returns_df: pd.DataFrame,
                                 n_stocks: int = 20,
                                 save_name: Optional[str] = None) -> plt.Figure:
        """
        绘制相关性热图

        Args:
            returns_df: 收益率DataFrame
            n_stocks: 显示的股票数量
            save_name: 保存文件名

        Returns:
            Matplotlib图表对象
        """
        print("   生成相关性热图...")

        try:
            n_stocks = min(n_stocks, len(returns_df.columns))

            # 只取前n_stocks只股票
            returns_subset = returns_df.iloc[:, :n_stocks]

            # 计算相关性矩阵
            corr_matrix = returns_subset.corr()

            fig, ax = plt.subplots(figsize=(12, 10))

            # 创建热图
            im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)

            # 添加颜色条
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('相关系数', rotation=270, labelpad=15)

            # 设置刻度
            ax.set_xticks(range(n_stocks))
            ax.set_yticks(range(n_stocks))
            ax.set_xticklabels(returns_subset.columns, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(returns_subset.columns, fontsize=8)

            # 添加相关系数值
            for i in range(n_stocks):
                for j in range(n_stocks):
                    corr_value = corr_matrix.iloc[i, j]
                    color = 'white' if abs(corr_value) > 0.5 else 'black'
                    ax.text(j, i, f'{corr_value:.2f}',
                            ha='center', va='center', color=color, fontsize=7)

            ax.set_title(f'股票相关性热图 (前{n_stocks}只)')

            plt.tight_layout()

            if save_name:
                save_path = self.output_dir / f"{save_name}_correlation_heatmap.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   图表已保存: {save_path}")

            return fig

        except Exception as e:
            print(f"   生成相关性热图失败: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '相关性热图生成失败',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16)
            return fig

    def save_all_charts(self, returns_df: pd.DataFrame, weights: np.ndarray,
                        stock_codes: List[str], prefix: str = "analysis"):
        """
        保存所有图表

        Args:
            returns_df: 收益率DataFrame
            weights: 投资组合权重
            stock_codes: 股票代码列表
            prefix: 文件名前缀
        """
        print(f"   生成并保存所有图表...")

        try:
            # 1. 收益率分布图
            self.plot_returns_distribution(returns_df, save_name=prefix)

            # 2. 净值曲线
            self.plot_portfolio_values(returns_df, weights, save_name=prefix)

            # 3. 权重分布图
            self.plot_stock_weights(weights, stock_codes, save_name=prefix)

            # 4. 相关性热图
            self.plot_correlation_heatmap(returns_df, save_name=prefix)

            print(f"   ✅ 所有图表已保存到: {self.output_dir}")

        except Exception as e:
            print(f"   保存图表失败: {e}")


# 简化版可视化器（用于回退）
class SimpleVisualizer:
    """简化版可视化器"""

    def plot_returns_distribution(self, returns_df: pd.DataFrame):
        """绘制收益率分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 简单的实现...
        return fig


# 测试函数
def test_visualizer():
    """测试可视化器"""
    print("🧪 测试可视化模块...")

    # 创建模拟数据
    np.random.seed(42)
    n_stocks = 20
    n_days = 100

    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    returns_data = np.random.randn(n_days, n_stocks) * 0.01
    returns_df = pd.DataFrame(
        returns_data,
        index=dates,
        columns=[f"stock_{i:03d}" for i in range(n_stocks)]
    )

    weights = np.random.rand(n_stocks)
    weights = weights / weights.sum()

    # 创建可视化器
    visualizer = Visualizer(output_dir="test_visualizations")

    # 生成图表
    visualizer.plot_returns_distribution(returns_df, save_name="test")
    visualizer.plot_portfolio_values(returns_df, weights, save_name="test")
    visualizer.plot_stock_weights(weights, returns_df.columns.tolist(), save_name="test")
    visualizer.plot_correlation_heatmap(returns_df, save_name="test")

    print("✅ 可视化模块测试完成")


if __name__ == "__main__":
    test_visualizer()