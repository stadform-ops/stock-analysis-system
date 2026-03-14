"""
图神经网络模型
基于股票关联性分析构建的GNN模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, List, Dict, Tuple, Any, Union
import warnings

warnings.filterwarnings('ignore')

# 检查是否有torch_geometric
try:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️  PyTorch Geometric 未安装，GNN功能将受限")


class GNNModel(nn.Module):
    """
    简化版图神经网络模型
    用于股票关系建模
    """

    # 修改 GNNModel 类的 __init__ 方法中LayerNorm的维度
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_heads: int = 4,
                 dropout: float = 0.2):
        super(GNNModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # 如果PyG可用，使用GAT层
        if HAS_PYG:
            # 图注意力层
            self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
            self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)

            # 修复：LayerNorm维度要与GAT层输出匹配
            self.ln1 = nn.LayerNorm(hidden_dim * num_heads)  # 第一层GAT输出维度
            self.ln2 = nn.LayerNorm(hidden_dim)  # 第二层GAT输出维度

            # 全连接层
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        else:
            # 回退到简单的全连接网络
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    # 修改 forward 方法
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if HAS_PYG:
            # 图注意力网络
            x = self.gat1(x, edge_index)
            x = self.ln1(x)  # 使用修复后的LayerNorm
            x = self.relu(x)
            x = self.dropout_layer(x)

            x = self.gat2(x, edge_index)
            x = self.ln2(x)  # 使用修复后的LayerNorm
            x = self.relu(x)

            # 如果提供了批次信息，进行全局池化
            if batch is not None:
                x = global_mean_pool(x, batch)

            # 全连接层
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout_layer(x)

            x = self.fc2(x)
        else:
            # 回退模式...
            pass

        return x


class DynamicGNN(nn.Module):
    """
    动态图神经网络
    处理时变股票关系
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_temporal_layers: int = 2, num_gnn_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2):
        super(DynamicGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_temporal_layers = num_temporal_layers
        self.num_gnn_layers = num_gnn_layers
        self.output_dim = output_dim

        # 时间编码层（处理时间序列）
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_temporal_layers,
            batch_first=True,
            dropout=dropout if num_temporal_layers > 1 else 0,
            bidirectional=True
        )

        # 图神经网络层
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_channels = hidden_dim * 2 if i == 0 else hidden_dim
            out_channels = hidden_dim

            if HAS_PYG:
                self.gnn_layers.append(
                    GATConv(in_channels, out_channels, heads=4, dropout=dropout)
                )
            else:
                # 回退到MLP
                self.gnn_layers.append(
                    nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )

        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # 辅助层
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq: torch.Tensor, edge_index_seq: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            x_seq: 时间序列特征 [batch_size, seq_len, num_nodes, input_dim]
            edge_index_seq: 时间序列的边索引列表，每个元素为 [2, num_edges]

        Returns:
            输出 [batch_size, output_dim]
        """
        batch_size, seq_len, num_nodes, input_dim = x_seq.shape

        # 重新组织维度以便LSTM处理
        x_reshaped = x_seq.permute(0, 2, 1, 3)  # [batch, nodes, seq_len, input_dim]
        x_reshaped = x_reshaped.reshape(batch_size * num_nodes, seq_len, input_dim)

        # 时间编码
        temporal_out, (h_n, c_n) = self.temporal_encoder(x_reshaped)  # [batch*nodes, seq_len, hidden*2]
        last_hidden = temporal_out[:, -1, :]  # 取最后一个时间步 [batch*nodes, hidden*2]
        last_hidden = last_hidden.reshape(batch_size, num_nodes, -1)  # [batch, nodes, hidden*2]

        # 如果没有PyG，使用简化处理
        if not HAS_PYG:
            # 对节点特征进行平均池化
            node_features = last_hidden.mean(dim=1)  # [batch, hidden*2]
            output = self.fc_out(node_features)
            return output

        # 图神经网络处理
        x = last_hidden

        for i, gnn_layer in enumerate(self.gnn_layers):
            # 将批次维度与节点维度合并以适应PyG
            x_reshaped = x.reshape(batch_size * num_nodes, -1)

            # 处理每个时间步的图结构
            # 这里简化处理：使用第一个时间步的图结构
            if i == 0 and edge_index_seq and len(edge_index_seq) > 0:
                edge_index = edge_index_seq[0]
                # 扩展边索引以处理批次
                edge_index_batch = []
                for batch_idx in range(batch_size):
                    edge_index_batch.append(edge_index + batch_idx * num_nodes)
                edge_index = torch.cat(edge_index_batch, dim=1)

                x_reshaped = gnn_layer(x_reshaped, edge_index)
            elif i > 0:
                # 后续层使用相同的图结构
                x_reshaped = gnn_layer(x_reshaped, edge_index)
            else:
                # 回退到MLP
                x_reshaped = gnn_layer(x_reshaped)

            x_reshaped = F.relu(x_reshaped)
            x_reshaped = self.dropout(x_reshaped)

            # 重新组织维度
            x = x_reshaped.reshape(batch_size, num_nodes, -1)

        # 全局池化
        x_pooled = x.mean(dim=1)  # [batch, hidden_dim]

        # 输出层
        output = self.fc_out(x_pooled)

        return output


class GNNTrainer:
    """
    GNN模型训练器
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

    def prepare_gnn_data(self, features_df: pd.DataFrame,
                         returns_df: pd.DataFrame) -> Optional[List[Any]]:
        """
        准备GNN训练数据（修复版）

        Args:
            features_df: 特征DataFrame [n_stocks, n_features]
            returns_df: 收益率DataFrame [n_days, n_stocks]

        Returns:
            Data对象列表
        """
        print("  🔧 准备GNN数据（修复版）...")

        try:
            # 确保特征和收益率数据对齐
            stock_codes = [c for c in returns_df.columns if c in features_df.index]
            if len(stock_codes) < 2:
                print(f"❌ 共同股票数量不足: {len(stock_codes)}")
                return None

            # 获取对齐后的数据
            returns_subset = returns_df[stock_codes]
            features_subset = features_df.loc[stock_codes]

            # 计算最大可能的序列长度
            max_sequence_length = len(returns_subset) - 1
            sequence_length = min(self.sequence_length, max_sequence_length)

            if sequence_length < 5:
                print(f"❌ 序列长度过短: {sequence_length}，最小需要5")
                return None

            print(f"   股票数量: {len(stock_codes)}, 序列长度: {sequence_length}, "
                  f"总样本数: {len(returns_subset)}")

            # 简化处理：使用最后一段数据
            n_samples = min(50, len(returns_subset) - sequence_length)

            if n_samples <= 0:
                print(f"❌ 样本数不足: {n_samples}")
                return None

            data_list = []

            for i in range(n_samples):
                start_idx = i
                end_idx = i + sequence_length

                # 获取时间窗口数据
                returns_window = returns_subset.iloc[start_idx:end_idx].values

                # 构建节点特征：组合时序特征和静态特征
                n_stocks = len(stock_codes)
                node_features = []

                for stock_idx in range(n_stocks):
                    # 时序特征：该股票在时间窗口内的收益率
                    time_series = returns_window[:, stock_idx]

                    # 静态特征：该股票的PCA特征
                    static_features = features_subset.iloc[stock_idx].values

                    # 组合特征
                    combined = np.concatenate([time_series, static_features])
                    node_features.append(combined)

                node_features = np.array(node_features, dtype=np.float32)

                # 构建简单的全连接图
                edge_list = []
                for j in range(n_stocks):
                    for k in range(n_stocks):
                        if j != k:
                            edge_list.append([j, k])

                if edge_list:
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

                    # 创建Data对象
                    if HAS_PYG:
                        data = Data(
                            x=torch.FloatTensor(node_features),
                            edge_index=edge_index,
                            y=torch.FloatTensor(returns_subset.iloc[end_idx:end_idx + 1].values.T)
                        )
                        data_list.append(data)
                    else:
                        # 回退模式：使用字典存储
                        data_dict = {
                            'x': torch.FloatTensor(node_features),
                            'edge_index': edge_index,
                            'y': torch.FloatTensor(returns_subset.iloc[end_idx:end_idx + 1].values.T)
                        }
                        data_list.append(data_dict)

            if not data_list:
                print("❌ 没有生成有效数据")
                return None

            print(f"✅ 成功生成 {len(data_list)} 个训练样本")
            return data_list

        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train(self, features_df: pd.DataFrame, returns_df: pd.DataFrame,
              n_epochs: int = 10, val_ratio: float = 0.2) -> Optional[Dict[str, List[float]]]:
        """
        训练模型

        Args:
            features_df: 特征DataFrame
            returns_df: 收益率DataFrame
            n_epochs: 训练轮数
            val_ratio: 验证集比例

        Returns:
            训练历史记录
        """
        print(f"    训练GNN模型 ({n_epochs} epochs)...")

        try:
            # 准备数据
            data_list = self.prepare_gnn_data(features_df, returns_df)
            if data_list is None or len(data_list) == 0:
                print("    数据准备失败，跳过训练")
                return None

            # 划分训练验证集
            n_samples = len(data_list)
            split_idx = int(n_samples * (1 - val_ratio))

            train_data = data_list[:split_idx]
            val_data = data_list[split_idx:]

            # 创建模型
            if HAS_PYG and train_data and isinstance(train_data[0], Data):
                sample_data = train_data[0]
                input_dim = sample_data.x.shape[1]
            else:
                # 估计输入维度
                input_dim = self.sequence_length + features_df.shape[1]

            output_dim = 1  # 预测收益率

            self.model = GNNModel(
                input_dim=input_dim,
                hidden_dim=64,
                output_dim=output_dim,
                num_heads=4,
                dropout=0.2
            ).to(self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # 训练循环
            history = {'train_loss': [], 'val_loss': []}

            for epoch in range(n_epochs):
                # 训练模式
                self.model.train()
                train_loss_accum = 0

                for data in train_data:
                    self.optimizer.zero_grad()

                    if HAS_PYG and isinstance(data, Data):
                        # PyG Data对象
                        data = data.to(self.device)
                        out = self.model(data.x, data.edge_index)
                        loss = self.criterion(out, data.y)
                    else:
                        # 回退模式
                        x = data['x'].to(self.device)
                        edge_index = data['edge_index'].to(self.device)
                        y = data['y'].to(self.device)
                        out = self.model(x, edge_index)
                        loss = self.criterion(out, y)

                    loss.backward()
                    self.optimizer.step()
                    train_loss_accum += loss.item()

                # 验证
                val_loss = 0
                if val_data:
                    self.model.eval()
                    with torch.no_grad():
                        for data in val_data:
                            if HAS_PYG and isinstance(data, Data):
                                data = data.to(self.device)
                                out = self.model(data.x, data.edge_index)
                                val_loss += self.criterion(out, data.y).item()
                            else:
                                x = data['x'].to(self.device)
                                edge_index = data['edge_index'].to(self.device)
                                y = data['y'].to(self.device)
                                out = self.model(x, edge_index)
                                val_loss += self.criterion(out, y).item()

                    if val_data:
                        val_loss /= len(val_data)

                avg_train_loss = train_loss_accum / len(train_data) if train_data else 0
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(val_loss)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"      Epoch {epoch + 1}/{n_epochs}: "
                          f"train_loss={avg_train_loss:.6f}, "
                          f"val_loss={val_loss:.6f}")

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

        Args:
            features_df: 特征DataFrame
            returns_df: 收益率DataFrame

        Returns:
            预测的收益率序列
        """
        if self.model is None:
            print("    模型未训练，无法预测")
            return None

        try:
            self.model.eval()

            # 准备数据
            data_list = self.prepare_gnn_data(features_df, returns_df)
            if data_list is None or len(data_list) == 0:
                print("    无法准备预测数据")
                return None

            # 使用最后一个数据样本进行预测
            data = data_list[-1]

            with torch.no_grad():
                if HAS_PYG and isinstance(data, Data):
                    data = data.to(self.device)
                    prediction = self.model(data.x, data.edge_index)
                else:
                    x = data['x'].to(self.device)
                    edge_index = data['edge_index'].to(self.device)
                    prediction = self.model(x, edge_index)

                # 转换为numpy
                pred_np = prediction.cpu().numpy().flatten()

            # 获取股票代码
            stock_codes = [c for c in returns_df.columns if c in features_df.index]

            if len(pred_np) == len(stock_codes):
                return pd.Series(pred_np, index=stock_codes)
            else:
                # 如果维度不匹配，返回前n个
                n = min(len(pred_np), len(stock_codes))
                return pd.Series(pred_np[:n], index=stock_codes[:n])

        except Exception as e:
            print(f"    预测失败: {e}")
            return None

    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            }, filepath)
            print(f"    模型已保存: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # 需要先创建模型结构
        if self.model is None:
            # 这里需要知道模型的结构参数
            # 在实际使用中，应该保存模型参数
            pass

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"    模型已加载: {filepath}")


# 测试函数
def test_gnn_model():
    """测试GNN模型"""
    print("🧪 测试GNN模型...")

    # 创建模拟数据
    np.random.seed(42)
    n_stocks = 10
    n_days = 100
    n_features = 8

    # 模拟特征数据
    features_data = np.random.randn(n_stocks, n_features)
    features_df = pd.DataFrame(
        features_data,
        index=[f"stock_{i}" for i in range(n_stocks)],
        columns=[f"feature_{j}" for j in range(n_features)]
    )

    # 模拟收益率数据
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    returns_data = np.random.randn(n_days, n_stocks) * 0.01
    returns_df = pd.DataFrame(
        returns_data,
        index=dates,
        columns=[f"stock_{i}" for i in range(n_stocks)]
    )

    # 创建训练器
    trainer = GNNTrainer(
        sequence_length=20,
        batch_size=16,
        device='cpu',
        learning_rate=0.001
    )

    # 训练模型
    history = trainer.train(features_df, returns_df, n_epochs=5)

    if history:
        print(f"✅ 模型训练成功")
        print(f"   最终训练损失: {history['train_loss'][-1]:.6f}")

        # 测试预测
        predictions = trainer.predict_future_returns(features_df, returns_df)
        if predictions is not None:
            print(f"✅ 预测成功")
            print(f"   预测形状: {predictions.shape}")
            print(f"   预测示例:\n{predictions.head()}")

    return trainer


if __name__ == "__main__":
    trainer = test_gnn_model()