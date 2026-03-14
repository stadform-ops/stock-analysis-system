"""
股票关联性分析模块 - 完整实现
实现动态相关性计算和对比学习嵌入
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class DynamicCorrelationAnalyzer:
    """
    动态相关性分析器
    计算股票间的滚动窗口动态相关性
    """

    def __init__(self, window_size: int = 20, min_periods: int = 10):
        """
        初始化动态相关性分析器

        Args:
            window_size: 滚动窗口大小
            min_periods: 最小计算周期
        """
        self.window_size = window_size
        self.min_periods = min_periods

    def compute_dynamic_correlations(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        计算动态相关性矩阵 - 完整实现

        公式: C_t(i,j) = cov(R_i, R_j) / (std(R_i) * std(R_j))
        其中 R_i, R_j 是窗口内的收益率序列

        Args:
            returns_df: 收益率DataFrame，形状 (T, N)

        Returns:
            动态相关性张量，形状 (T-window_size, N, N)
        """
        print("    计算动态相关性矩阵...")

        try:
            n_days, n_stocks = returns_df.shape
            print(f"    数据维度: {n_days} 天, {n_stocks} 只股票")

            if n_days < self.window_size:
                print(f"    数据不足，返回静态相关性矩阵")
                static_corr = returns_df.corr().values
                return np.tile(static_corr, (n_days, 1, 1))

            # 计算滚动窗口相关性
            dynamic_corrs = []

            for i in range(self.window_size, n_days):
                window_returns = returns_df.iloc[i - self.window_size:i]

                # 确保有足够的数据
                valid_stocks = window_returns.columns[
                    window_returns.notna().sum() >= self.min_periods
                    ]

                if len(valid_stocks) < 2:
                    # 如果没有足够股票，使用单位矩阵
                    corr_matrix = np.eye(n_stocks)
                else:
                    # 计算相关系数矩阵
                    corr_matrix = np.eye(n_stocks)
                    valid_returns = window_returns[valid_stocks]

                    # 使用numpy计算相关性，避免pandas的版本问题
                    corr_values = np.corrcoef(valid_returns.values.T)

                    # 处理NaN
                    corr_values = np.nan_to_num(corr_values, nan=0.0)

                    # 确保数值稳定性
                    corr_values = np.clip(corr_values, -1, 1)

                    # 将计算的相关性填充到矩阵中
                    stock_indices = {code: idx for idx, code in enumerate(returns_df.columns)}
                    for i_idx, stock_i in enumerate(valid_stocks):
                        for j_idx, stock_j in enumerate(valid_stocks):
                            idx_i = stock_indices[stock_i]
                            idx_j = stock_indices[stock_j]
                            corr_matrix[idx_i, idx_j] = corr_values[i_idx, j_idx]

                dynamic_corrs.append(corr_matrix)

                if i % 100 == 0:
                    print(f"      已处理 {i}/{n_days} 天")

            if not dynamic_corrs:
                print("    未生成动态相关性矩阵，使用静态相关性")
                static_corr = returns_df.corr().values
                return np.tile(static_corr, (1, n_stocks, n_stocks))

            dynamic_corr_array = np.array(dynamic_corrs)
            print(f"    动态相关性矩阵形状: {dynamic_corr_array.shape}")

            return dynamic_corr_array

        except Exception as e:
            print(f"    动态相关性计算失败: {e}")
            import traceback
            traceback.print_exc()

            # 返回简单的相关性矩阵
            n_stocks = len(returns_df.columns)
            static_corr = returns_df.corr().fillna(0).values
            return np.tile(static_corr, (n_days, 1, 1))

    def compute_rolling_correlation(self, returns_series_i: pd.Series,
                                    returns_series_j: pd.Series) -> np.ndarray:
        """
        计算两个序列的滚动相关性
        """
        # 对齐数据
        aligned_series = pd.concat([returns_series_i, returns_series_j], axis=1).dropna()

        if len(aligned_series) < self.window_size:
            return np.array([])

        correlations = []
        for i in range(self.window_size, len(aligned_series)):
            window = aligned_series.iloc[i - self.window_size:i]
            corr = window.iloc[:, 0].corr(window.iloc[:, 1])
            correlations.append(corr if not np.isnan(corr) else 0.0)

        return np.array(correlations)


class ContrastiveEmbeddingModel(nn.Module):
    """
    对比学习嵌入模型
    使用InfoNCE损失学习股票嵌入
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 embedding_dim: int = 16, temperature: float = 0.5):
        super(ContrastiveEmbeddingModel, self).__init__()

        self.temperature = temperature

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # 投影头（用于对比学习）
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            嵌入向量 [batch_size, embedding_dim]
        """
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)  # L2归一化
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取编码表示
        """
        return self.encoder(x)


class ContrastiveEmbeddingTrainer:
    """
    对比学习嵌入训练器
    """

    def __init__(self, sequence_length: int = 20, embedding_dim: int = 16,
                 device: str = 'cpu', learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.learning_rate = learning_rate

        self.model = None
        self.optimizer = None

    def prepare_contrastive_data(self, returns_df: pd.DataFrame,
                                 correlation_threshold: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        准备对比学习数据

        Args:
            returns_df: 收益率DataFrame
            correlation_threshold: 相关性阈值，高于此值为正样本

        Returns:
            anchor_samples: 锚点样本
            positive_samples: 正样本
            negative_samples: 负样本
        """
        print("    准备对比学习数据...")

        try:
            n_stocks = len(returns_df.columns)

            # 计算平均相关性矩阵
            corr_matrix = returns_df.corr().fillna(0).values

            # 生成样本
            anchor_indices = []
            positive_indices = []
            negative_indices = []

            for i in range(n_stocks):
                for j in range(n_stocks):
                    if i == j:
                        continue

                    if corr_matrix[i, j] > correlation_threshold:
                        # 正样本对
                        anchor_indices.append(i)
                        positive_indices.append(j)

                        # 为每个正样本对添加负样本
                        # 选择相关性最低的股票作为负样本
                        negative_candidates = np.where(corr_matrix[i, :] < 0)[0]
                        if len(negative_candidates) > 0:
                            negative_idx = np.random.choice(negative_candidates)
                            negative_indices.append(negative_idx)
                        else:
                            # 如果没有负相关的，选择相关性最低的
                            sorted_indices = np.argsort(corr_matrix[i, :])
                            negative_idx = sorted_indices[0]
                            negative_indices.append(negative_idx)

            if len(anchor_indices) == 0:
                print("    未生成足够的正样本，使用随机样本")
                # 创建一些随机样本
                n_samples = min(1000, n_stocks * 10)
                anchor_indices = np.random.randint(0, n_stocks, n_samples)
                positive_indices = np.random.randint(0, n_stocks, n_samples)
                negative_indices = np.random.randint(0, n_stocks, n_samples)

            # 转换为Tensor
            anchor_tensor = torch.tensor(anchor_indices, dtype=torch.long)
            positive_tensor = torch.tensor(positive_indices, dtype=torch.long)
            negative_tensor = torch.tensor(negative_indices, dtype=torch.long)

            print(f"    生成 {len(anchor_indices)} 个样本对")
            return anchor_tensor, positive_tensor, negative_tensor

        except Exception as e:
            print(f"    准备对比学习数据失败: {e}")
            # 返回空样本
            return (torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long))

    def train(self, returns_df: pd.DataFrame, n_epochs: int = 50) -> Dict[str, List[float]]:
        """
        训练对比学习模型

        Args:
            returns_df: 收益率数据
            n_epochs: 训练轮数

        Returns:
            训练历史
        """
        print(f"    训练对比学习嵌入模型 ({n_epochs} epochs)...")

        try:
            n_stocks = len(returns_df.columns)

            # 准备数据
            anchor_idx, positive_idx, negative_idx = self.prepare_contrastive_data(returns_df)

            if len(anchor_idx) == 0:
                print("    没有训练数据，跳过训练")
                return {}

            # 创建模型
            self.model = ContrastiveEmbeddingModel(
                input_dim=self.sequence_length,
                hidden_dim=64,
                embedding_dim=self.embedding_dim
            ).to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # 获取收益率序列
            returns_tensor = torch.FloatTensor(returns_df.values.T).to(self.device)  # [n_stocks, n_days]

            # 训练循环
            history = {'loss': []}

            for epoch in range(n_epochs):
                self.model.train()
                self.optimizer.zero_grad()

                # 获取序列
                seq_start = torch.randint(0, returns_tensor.size(1) - self.sequence_length, (1,)).item()
                anchor_seq = returns_tensor[anchor_idx, seq_start:seq_start + self.sequence_length]
                positive_seq = returns_tensor[positive_idx, seq_start:seq_start + self.sequence_length]
                negative_seq = returns_tensor[negative_idx, seq_start:seq_start + self.sequence_length]

                # 计算嵌入
                anchor_embed = self.model(anchor_seq)
                positive_embed = self.model(positive_seq)
                negative_embed = self.model(negative_seq)

                # InfoNCE损失
                pos_sim = F.cosine_similarity(anchor_embed, positive_embed, dim=1)
                neg_sim = F.cosine_similarity(anchor_embed, negative_embed, dim=1)

                numerator = torch.exp(pos_sim / 0.5)
                denominator = torch.exp(pos_sim / 0.5) + torch.exp(neg_sim / 0.5)
                loss = -torch.log(numerator / denominator).mean()

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                history['loss'].append(loss.item())

                if (epoch + 1) % 10 == 0:
                    print(f"      Epoch {epoch + 1}/{n_epochs}: loss={loss.item():.6f}")

            print(f"    训练完成，最终损失: {history['loss'][-1]:.6f}")
            return history

        except Exception as e:
            print(f"    训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_stock_embeddings(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        获取所有股票的嵌入表示

        Args:
            returns_df: 收益率数据

        Returns:
            股票嵌入矩阵 [n_stocks, embedding_dim]
        """
        if self.model is None:
            print("    模型未训练，返回随机嵌入")
            n_stocks = len(returns_df.columns)
            np.random.seed(42)
            return np.random.randn(n_stocks, self.embedding_dim)

        try:
            self.model.eval()

            returns_tensor = torch.FloatTensor(returns_df.values.T).to(self.device)
            n_stocks, n_days = returns_tensor.shape

            # 使用最后sequence_length天的数据
            if n_days < self.sequence_length:
                sequence = returns_tensor[:, -n_days:]
                # 填充
                padding = torch.zeros(n_stocks, self.sequence_length - n_days).to(self.device)
                sequence = torch.cat([padding, sequence], dim=1)
            else:
                sequence = returns_tensor[:, -self.sequence_length:]

            with torch.no_grad():
                embeddings = self.model.encode(sequence)

            embeddings_np = embeddings.cpu().numpy()

            # 标准化
            embeddings_np = (embeddings_np - embeddings_np.mean(axis=0)) / (embeddings_np.std(axis=0) + 1e-8)

            print(f"    生成嵌入矩阵: {embeddings_np.shape}")
            return embeddings_np

        except Exception as e:
            print(f"    获取嵌入失败: {e}")
            n_stocks = len(returns_df.columns)
            np.random.seed(42)
            return np.random.randn(n_stocks, self.embedding_dim)


# 主类
class CorrelationAnalyzer:
    """
    股票关联性分析主类
    整合动态相关性计算和对比学习嵌入
    """

    def __init__(self, window_size: int = 20, embedding_dim: int = 16,
                 device: str = 'cpu', enable_contrastive: bool = True):
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.enable_contrastive = enable_contrastive

        self.dynamic_corr_analyzer = DynamicCorrelationAnalyzer(window_size=window_size)
        self.contrastive_trainer = None

        if enable_contrastive:
            self.contrastive_trainer = ContrastiveEmbeddingTrainer(
                sequence_length=window_size,
                embedding_dim=embedding_dim,
                device=device
            )

        self.dynamic_correlations = None
        self.stock_embeddings = None

    def analyze(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行完整的关联性分析

        Args:
            returns_df: 收益率数据

        Returns:
            分析结果字典
        """
        print("🔍 执行股票关联性分析...")

        results = {}

        try:
            # 1. 计算动态相关性
            print("  1. 计算动态相关性...")
            self.dynamic_correlations = self.dynamic_corr_analyzer.compute_dynamic_correlations(returns_df)
            results['dynamic_correlations'] = self.dynamic_correlations
            print(f"    动态相关性矩阵形状: {self.dynamic_correlations.shape}")

            # 2. 训练对比学习嵌入
            if self.enable_contrastive and self.contrastive_trainer is not None:
                print("  2. 训练对比学习嵌入...")
                history = self.contrastive_trainer.train(returns_df, n_epochs=30)

                if history:
                    self.stock_embeddings = self.contrastive_trainer.get_stock_embeddings(returns_df)
                    results['stock_embeddings'] = self.stock_embeddings
                    results['training_history'] = history
                    print(f"    股票嵌入矩阵形状: {self.stock_embeddings.shape}")

            # 3. 计算嵌入相似性矩阵
            if self.stock_embeddings is not None:
                print("  3. 计算嵌入相似性矩阵...")
                similarity_matrix = np.dot(self.stock_embeddings, self.stock_embeddings.T)

                # 标准化到[-1, 1]范围
                norm = np.linalg.norm(self.stock_embeddings, axis=1, keepdims=True)
                similarity_matrix = similarity_matrix / (norm @ norm.T + 1e-8)

                results['embedding_similarity'] = similarity_matrix
                print(f"    嵌入相似性矩阵形状: {similarity_matrix.shape}")

            # 4. 计算平均相关性统计
            print("  4. 计算相关性统计...")
            if self.dynamic_correlations is not None and len(self.dynamic_correlations) > 0:
                avg_corr = np.mean(self.dynamic_correlations, axis=0)
                avg_corr_value = np.mean(avg_corr[np.triu_indices_from(avg_corr, k=1)])

                results['avg_correlation_matrix'] = avg_corr
                results['avg_correlation'] = avg_corr_value

                print(f"    平均股票相关性: {avg_corr_value:.4f}")

                # 分析相关性分布
                flat_corr = avg_corr[np.triu_indices_from(avg_corr, k=1)]
                results['corr_distribution'] = {
                    'mean': float(np.mean(flat_corr)),
                    'std': float(np.std(flat_corr)),
                    'min': float(np.min(flat_corr)),
                    'max': float(np.max(flat_corr)),
                    'positive_ratio': float(np.mean(flat_corr > 0.1)),
                    'negative_ratio': float(np.mean(flat_corr < -0.1))
                }

            print("✅ 关联性分析完成")
            return results

        except Exception as e:
            print(f"❌ 关联性分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """获取相关性矩阵"""
        if self.dynamic_correlations is not None and len(self.dynamic_correlations) > 0:
            return np.mean(self.dynamic_correlations, axis=0)
        return None

    def get_embeddings(self) -> Optional[np.ndarray]:
        """获取股票嵌入"""
        return self.stock_embeddings


# 导出
__all__ = [
    'DynamicCorrelationAnalyzer',
    'ContrastiveEmbeddingModel',
    'ContrastiveEmbeddingTrainer',
    'CorrelationAnalyzer'
]