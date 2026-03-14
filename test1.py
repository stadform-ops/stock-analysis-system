"""
股票分析系统主程序 - 最简单版本
确保能运行完整流程
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback

import numpy as np
import pandas as pd
import torch

# 抑制警告
warnings.filterwarnings('ignore')

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def print_section(title: str, emoji: str = "📊"):
    """打印章节标题"""
    print(f"\n{'='*80}")
    print(f"{emoji} {title}")
    print(f"{'='*80}")


def print_progress(step: int, total: int, description: str):
    """打印进度"""
    print(f"\n[{step}/{total}] {description}")


class SimpleStockAnalysisSystem:
    """简化版股票分析系统"""

    def __init__(self):
        self.start_time = datetime.now()

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {self.device}")

        # 创建目录
        self.setup_directories()

    def setup_directories(self):
        """创建目录"""
        directories = ['results', 'results/visualizations', 'results/reports']
        for dir_name in directories:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        print("📁 目录结构已创建")

    def load_data(self) -> Optional[pd.DataFrame]:
        """加载数据"""
        data_path = Path("data/processed/stock_returns.csv")
        if data_path.exists():
            returns_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"📂 加载数据: {data_path}")
            print(f"   形状: {returns_df.shape}")
            print(f"   时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
            return returns_df
        else:
            print("❌ 未找到数据文件")
            return None

    def create_features(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """创建简单特征"""
        print("🔧 创建特征...")

        n_stocks = len(returns_df.columns)
        features_list = []

        for stock_code in returns_df.columns:
            stock_returns = returns_df[stock_code]
            features = {
                'mean': stock_returns.mean(),
                'std': stock_returns.std(),
                'skew': stock_returns.skew(),
                'kurt': stock_returns.kurtosis(),
                'max': stock_returns.max(),
                'min': stock_returns.min(),
            }
            features_list.append(pd.Series(features, name=stock_code))

        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)

        print(f"✅ 特征创建完成: {features_df.shape}")
        return features_df

    def apply_pca(self, features_df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """应用PCA"""
        print("🔧 应用PCA降维...")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)

        # PCA
        pca = PCA(n_components=min(n_components, features_df.shape[1]))
        features_pca = pca.fit_transform(features_scaled)

        # 转换为DataFrame
        feature_names = [f'pca_{i+1}' for i in range(features_pca.shape[1])]
        features_pca_df = pd.DataFrame(
            features_pca,
            index=features_df.index,
            columns=feature_names
        )

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        print(f"✅ PCA降维完成")
        print(f"   原始维度: {features_df.shape[1]}")
        print(f"   降维后: {features_pca.shape[1]}")
        print(f"   累计解释方差: {cumulative_variance[-1]:.3%}")

        return features_pca_df

    def run_analysis(self):
        """运行分析"""
        print_section("🚀 股票分析系统 - 简化版", "🎯")

        try:
            # 1. 数据加载
            print_progress(1, 5, "数据加载")
            returns_df = self.load_data()
            if returns_df is None:
                return

            # 2. 特征工程
            print_progress(2, 5, "特征工程")
            features_df = self.create_features(returns_df)
            features_pca = self.apply_pca(features_df, n_components=10)

            # 3. 简单预测
            print_progress(3, 5, "收益预测")
            predicted_returns = returns_df.mean()  # 简单使用历史平均

            # 4. 投资组合优化
            print_progress(4, 5, "组合优化")
            n_stocks = len(returns_df.columns)
            weights = np.ones(n_stocks) / n_stocks  # 等权重
            expected_return = predicted_returns.mean()

            # 5. 回测
            print_progress(5, 5, "回测评估")
            portfolio_returns = returns_df @ weights
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252/len(portfolio_returns)) - 1

            # 打印结果
            duration = (datetime.now() - self.start_time).total_seconds()

            print_section("📊 分析结果", "✅")
            print(f"📈 股票数量: {n_stocks}")
            print(f"📅 交易日数: {len(returns_df)}")
            print(f"💰 预期年化收益: {annual_return:.2%}")
            print(f"📉 投资组合总收益: {total_return:.2%}")
            print(f"⏱️  运行时间: {duration:.2f}秒")
            print_section("🎉 分析完成", "🎯")

        except Exception as e:
            print(f"\n❌ 分析失败: {e}")
            traceback.print_exc()


def main():
    """主函数"""
    system = SimpleStockAnalysisSystem()
    system.run_analysis()


if __name__ == "__main__":
    main()