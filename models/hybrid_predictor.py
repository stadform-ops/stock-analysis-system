"""
LSTM-Transformer混合预测模型
完全兼容PyTorch 1.8.0的版本
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class LSTMTransformerHybrid(nn.Module):
    """
    LSTM-Transformer混合模型
    兼容PyTorch 1.8.0版本
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4,
                 output_dim: int = 1, sequence_length: int = 20,
                 dropout: float = 0.1):
        """
        初始化混合模型（兼容PyTorch 1.8.0）
        """
        super(LSTMTransformerHybrid, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # LSTM层 - 捕捉时序依赖
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 层归一化
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Transformer编码器层（兼容PyTorch 1.8.0）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
            # 注意：PyTorch 1.8.0没有batch_first参数
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # 自注意力池化层（不使用MultiheadAttention，避免batch_first问题）
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # 输出层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（兼容PyTorch 1.8.0）
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 1. LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, seq_len, hidden_dim]
        lstm_out = self.ln1(lstm_out)

        # 2. Transformer编码 - 需要调整维度
        # 从 [batch, seq_len, hidden_dim] 调整为 [seq_len, batch, hidden_dim]
        transformer_input = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden_dim]
        transformer_out = self.transformer_encoder(transformer_input)  # [seq_len, batch, hidden_dim]

        # 3. 调整回原始维度
        transformer_out = transformer_out.transpose(0, 1)  # [batch, seq_len, hidden_dim]

        # 4. 自注意力池化
        attn_weights = self.attention_pool(transformer_out)  # [batch, seq_len, 1]
        attn_out = torch.sum(transformer_out * attn_weights, dim=1)  # [batch, hidden_dim]

        # 5. 全连接层
        out = self.fc1(attn_out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out


class LSTMTransformerTrainer:
    """
    LSTM-Transformer混合模型训练器
    修复版
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

    def prepare_data(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        准备训练数据（简化版）
        """
        print("    准备混合模型数据（简化版）...")

        try:
            # 对齐股票
            stock_codes = [c for c in returns_df.columns if c in features_df.index]
            if len(stock_codes) < 1:
                print("    没有共同的股票")
                return None, None

            returns_subset = returns_df[stock_codes]
            features_subset = features_df.loc[stock_codes]

            # 检查数据长度
            if len(returns_subset) < self.sequence_length + 1:
                print(f"    数据长度不足: {len(returns_subset)} < {self.sequence_length + 1}")
                return None, None

            # 简化：只取前几只股票的特征
            n_stocks_to_use = min(10, len(stock_codes))
            stock_codes = stock_codes[:n_stocks_to_use]

            returns_subset = returns_subset[stock_codes]
            features_subset = features_subset.loc[stock_codes]

            # 创建特征和标签
            X, y = [], []
            n_samples = len(returns_subset) - self.sequence_length

            for i in range(min(50, n_samples)):  # 限制样本数量
                # 获取时间窗口
                start_idx = i
                end_idx = i + self.sequence_length

                # 时间序列特征
                time_series = returns_subset.iloc[start_idx:end_idx].values  # [seq_len, n_stocks]

                # 股票特征
                stock_features = features_subset.values  # [n_stocks, n_features]

                # 简化：对股票特征取平均
                avg_stock_features = stock_features.mean(axis=0)  # [n_features]

                # 组合特征：将时间序列和股票特征拼接
                seq_features = []
                for t in range(self.sequence_length):
                    # 每个时间点的特征 = 该时间点的收益率 + 平均股票特征
                    combined = np.concatenate([
                        time_series[t],  # 当前时间点各股票收益率
                        avg_stock_features  # 平均股票特征
                    ])
                    seq_features.append(combined)

                if seq_features:
                    X.append(seq_features)
                    y.append(returns_subset.iloc[end_idx].values)  # 下一时刻的收益率

            if not X:
                print("    没有生成有效的训练样本")
                return None, None

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            print(f"    数据形状: X={X.shape}, y={y.shape}")
            return X, y

        except Exception as e:
            print(f"    数据准备失败: {e}")
            return None, None

    def train(self, features_df: pd.DataFrame, returns_df: pd.DataFrame,
              n_epochs: int = 10, val_ratio: float = 0.2) -> Optional[Dict]:
        """
        训练模型（简化版）
        """
        print(f"    训练混合模型 ({n_epochs} epochs)...")

        try:
            # 准备数据
            X, y = self.prepare_data(features_df, returns_df)
            if X is None or y is None:
                print("    数据准备失败，跳过训练")
                return None

            # 划分训练验证集
            n_samples = len(X)
            split_idx = int(n_samples * (1 - val_ratio))

            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # 转换为Tensor
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device) if len(X_val) > 0 else None
            y_val = torch.FloatTensor(y_val).to(self.device) if len(y_val) > 0 else None

            # 创建模型
            n_stocks = y.shape[1]
            n_features = X.shape[2]

            self.model = LSTMTransformerHybrid(
                input_dim=n_features,
                hidden_dim=32,  # 减小维度
                num_layers=1,  # 减少层数
                num_heads=2,  # 减少注意力头
                output_dim=n_stocks,
                sequence_length=self.sequence_length,
                dropout=0.1
            ).to(self.device)

            # 优化器
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # 学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )

            # 训练循环
            history = {'train_loss': [], 'val_loss': []}

            for epoch in range(n_epochs):
                # 训练模式
                self.model.train()
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # 验证
                val_loss = 0
                if X_val is not None and len(X_val) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_val)
                        val_loss = self.criterion(val_outputs, y_val).item()

                    # 更新学习率
                    scheduler.step(val_loss)

                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss)

                if (epoch + 1) % 2 == 0 or epoch == 0:
                    print(f"      Epoch {epoch+1}/{n_epochs}: "
                          f"train_loss={loss.item():.6f}, "
                          f"val_loss={val_loss:.6f}")

            print(f"    训练完成，最终训练损失: {history['train_loss'][-1]:.6f}")
            return history

        except Exception as e:
            print(f"    训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_future_returns(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        预测未来收益率
        """
        if self.model is None:
            print("    模型未训练，无法预测")
            return None

        try:
            self.model.eval()

            # 使用最近的数据进行预测
            X, _ = self.prepare_data(features_df, returns_df)
            if X is None or len(X) == 0:
                print("    无法准备预测数据")
                return None

            # 使用最后一个时间窗口
            last_window = torch.FloatTensor(X[-1:]).to(self.device)

            with torch.no_grad():
                prediction = self.model(last_window)

            # 转换为numpy数组
            pred_np = prediction.cpu().numpy().flatten()

            # 对齐股票代码
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


# 测试函数
def test_hybrid_predictor_fixed():
    """测试修复后的混合预测器"""
    print("🧪 测试修复后的LSTM-Transformer混合模型...")

    # 创建模拟数据
    np.random.seed(42)
    n_stocks = 5
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
    trainer = LSTMTransformerTrainer(
        sequence_length=10,
        batch_size=8,
        device='cpu',
        learning_rate=0.001
    )

    # 训练模型
    history = trainer.train(features_df, returns_df, n_epochs=3)

    if history:
        print(f"✅ 模型训练成功")
        print(f"   最终训练损失: {history['train_loss'][-1]:.6f}")

        # 测试预测
        predictions = trainer.predict_future_returns(features_df, returns_df)
        if predictions is not None:
            print(f"✅ 预测成功")
            print(f"   预测形状: {predictions.shape}")
            print(f"   预测示例: {predictions.head()}")

    return trainer


if __name__ == "__main__":
    test_hybrid_predictor_fixed()