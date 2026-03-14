"""
动态图神经网络 - 修复版
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Tuple, List, Any
import warnings

warnings.filterwarnings('ignore')

# 检查是否有torch_geometric
try:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.nn import GATConv, global_mean_pool

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class DynamicGNN(nn.Module):
    """
    动态图神经网络
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_heads: int = 4,
                 dropout: float = 0.2):
        super(DynamicGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        if HAS_PYG:
            # 图注意力层
            self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
            self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)

            # LayerNorm
            self.ln1 = nn.LayerNorm(hidden_dim * num_heads)
            self.ln2 = nn.LayerNorm(hidden_dim)
        else:
            # 回退到MLP
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, output_dim) if HAS_PYG else nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        """
        if HAS_PYG:
            x = self.gat1(x, edge_index)
            x = self.ln1(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.gat2(x, edge_index)
            x = self.ln2(x)
            x = self.relu(x)

            if batch is not None:
                x = global_mean_pool(x, batch)

            x = self.fc_out(x)
        else:
            # 回退模式
            if len(x.shape) == 3:
                batch_size, num_nodes, _ = x.shape
                x = x.view(batch_size, -1)
            else:
                num_nodes = x.shape[0]
                x = x.view(1, -1)

            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.fc_out(x)

        return x


class DynamicGNNTrainer:
    """
    动态GNN训练器
    """

    def __init__(self, sequence_length: int = 20, batch_size: int = 32,
                 device: str = 'cpu', learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.learning_rate = learning_rate

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def prepare_data(self, features_df: pd.DataFrame,
                     returns_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        准备数据
        """
        print("    准备动态GNN数据...")

        try:
            stock_codes = [c for c in returns_df.columns if c in features_df.index]
            if len(stock_codes) < 2:
                return None, None

            returns_subset = returns_df[stock_codes]
            features_subset = features_df.loc[stock_codes]

            if len(returns_subset) < self.sequence_length + 1:
                return None, None

            # 创建简单数据
            X, y = [], []
            n_samples = len(returns_subset) - self.sequence_length

            for i in range(min(50, n_samples)):
                start_idx = i
                end_idx = i + self.sequence_length

                returns_window = returns_subset.iloc[start_idx:end_idx].values

                node_features = []
                for stock_idx in range(len(stock_codes)):
                    time_series = returns_window[:, stock_idx]
                    static_features = features_subset.iloc[stock_idx].values
                    combined = np.concatenate([time_series, static_features])
                    node_features.append(combined)

                node_features = np.array(node_features, dtype=np.float32)

                X.append(node_features)
                y.append(returns_subset.iloc[end_idx].values)

            if not X:
                return None, None

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            print(f"    数据形状: X={X.shape}, y={y.shape}")
            return X, y

        except Exception as e:
            print(f"    数据准备失败: {e}")
            return None, None

    def train(self, features_df: pd.DataFrame, returns_df: pd.DataFrame,
              n_epochs: int = 10) -> Optional[Dict]:
        """
        训练模型
        """
        print(f"    训练动态GNN模型 ({n_epochs} epochs)...")

        try:
            X, y = self.prepare_data(features_df, returns_df)
            if X is None or y is None:
                print("    数据准备失败，跳过训练")
                return None

            # 划分数据
            n_samples = len(X)
            split_idx = int(n_samples * 0.8)

            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # 转换为Tensor
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device) if len(X_val) > 0 else None
            y_val = torch.FloatTensor(y_val).to(self.device) if len(y_val) > 0 else None

            # 创建模型
            input_dim = X.shape[2]
            n_stocks = y.shape[1]

            self.model = DynamicGNN(
                input_dim=input_dim,
                hidden_dim=64,
                output_dim=n_stocks,  # 输出维度等于股票数量
                num_heads=4,
                dropout=0.2
            ).to(self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # 训练循环
            history = {'train_loss': [], 'val_loss': []}

            for epoch in range(n_epochs):
                self.model.train()
                self.optimizer.zero_grad()

                # 创建简单的全连接图
                batch_size, num_nodes, _ = X_train.shape
                edge_list = []
                for j in range(num_nodes):
                    for k in range(num_nodes):
                        if j != k:
                            edge_list.append([j, k])

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)

                # 前向传播
                outputs_list = []
                for i in range(batch_size):
                    x_i = X_train[i]
                    output_i = self.model(x_i, edge_index)
                    outputs_list.append(output_i)

                outputs = torch.stack(outputs_list, dim=0)
                loss = self.criterion(outputs, y_train)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # 验证
                val_loss = 0
                if X_val is not None and len(X_val) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs_list = []
                        for i in range(len(X_val)):
                            x_i = X_val[i]
                            val_output_i = self.model(x_i, edge_index)
                            val_outputs_list.append(val_output_i)

                        val_outputs = torch.stack(val_outputs_list, dim=0)
                        val_loss = self.criterion(val_outputs, y_val).item()

                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss)

                if (epoch + 1) % 2 == 0 or epoch == 0:
                    print(f"      Epoch {epoch + 1}/{n_epochs}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

            print(f"    训练完成，最终训练损失: {history['train_loss'][-1]:.6f}")
            return history

        except Exception as e:
            print(f"    训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_future_returns(self, features_df: pd.DataFrame,
                               returns_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        预测未来收益率
        """
        if self.model is None:
            print("    模型未训练，无法预测")
            return None

        try:
            self.model.eval()

            X, _ = self.prepare_data(features_df, returns_df)
            if X is None or len(X) == 0:
                print("    无法准备预测数据")
                return None

            # 使用最后一个样本
            last_X = torch.FloatTensor(X[-1:]).to(self.device)

            # 创建边
            num_nodes = last_X.shape[1]
            edge_list = []
            for j in range(num_nodes):
                for k in range(num_nodes):
                    if j != k:
                        edge_list.append([j, k])

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)

            with torch.no_grad():
                prediction = self.model(last_X[0], edge_index)

            pred_np = prediction.cpu().numpy().flatten()
            stock_codes = returns_df.columns.tolist()

            if len(pred_np) == len(stock_codes):
                return pd.Series(pred_np, index=stock_codes)
            else:
                n = min(len(pred_np), len(stock_codes))
                return pd.Series(pred_np[:n], index=stock_codes[:n])

        except Exception as e:
            print(f"    预测失败: {e}")
            return None


__all__ = ['DynamicGNN', 'DynamicGNNTrainer']