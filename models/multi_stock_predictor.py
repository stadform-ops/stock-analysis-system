"""
多股票预测模型 - 修复版
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')


class MultiStockLSTM(nn.Module):
    """多股票LSTM模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_layers: int = 2,
                 dropout: float = 0.2):
        super(MultiStockLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]

        # 全连接层
        output = self.fc(last_hidden)

        return output


class MultiStockTrainer:
    """多股票模型训练器"""

    def __init__(self, sequence_length: int = 20, batch_size: int = 32,
                 device: str = 'cpu', learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.learning_rate = learning_rate

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def prepare_data(self, returns_df: pd.DataFrame,
                    features_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """准备训练数据"""
        print("    准备多股票数据...")

        try:
            stock_codes = returns_df.columns.tolist()
            n_stocks = len(stock_codes)
            n_days = len(returns_df)

            if n_days < self.sequence_length + 1:
                print(f"    数据不足: {n_days} < {self.sequence_length + 1}")
                return None, None

            # 创建序列
            X, y = [], []
            n_samples = n_days - self.sequence_length

            for i in range(min(100, n_samples)):
                # 输入序列
                X_seq = returns_df.iloc[i:i+self.sequence_length].values

                # 输出（下一个时间步）
                y_seq = returns_df.iloc[i+self.sequence_length].values

                X.append(X_seq)
                y.append(y_seq)

            if not X:
                return None, None

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            print(f"    数据形状: X={X.shape}, y={y.shape}")
            return X, y

        except Exception as e:
            print(f"    数据准备失败: {e}")
            return None, None

    def train(self, returns_df: pd.DataFrame, n_epochs: int = 10,
             features_df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """训练模型"""
        print(f"    训练多股票模型 ({n_epochs} epochs)...")

        try:
            # 准备数据
            X, y = self.prepare_data(returns_df, features_df)
            if X is None or y is None:
                print("    数据准备失败，跳过训练")
                return None

            # 划分训练验证集
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
            n_stocks = y.shape[1]
            input_dim = X.shape[2]

            self.model = MultiStockLSTM(
                input_dim=input_dim,
                hidden_dim=32,
                output_dim=n_stocks,
                num_layers=1,
                dropout=0.1
            ).to(self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # 训练循环
            history = {'train_loss': [], 'val_loss': []}

            for epoch in range(n_epochs):
                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                val_loss = 0
                if X_val is not None and len(X_val) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_val)
                        val_loss = self.criterion(val_outputs, y_val).item()

                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss)

                if (epoch + 1) % 2 == 0 or epoch == 0:
                    print(f"      Epoch {epoch+1}/{n_epochs}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

            print(f"    训练完成，最终训练损失: {history['train_loss'][-1]:.6f}")
            return history

        except Exception as e:
            print(f"    训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_future_returns(self, features_df: pd.DataFrame,
                             returns_df: pd.DataFrame) -> Optional[pd.Series]:
        """预测未来收益率"""
        if self.model is None:
            print("    模型未训练，无法预测")
            return None

        try:
            self.model.eval()

            # 准备数据
            X, _ = self.prepare_data(returns_df, features_df)
            if X is None or len(X) == 0:
                print("    无法准备预测数据")
                return None

            # 使用最后一个时间窗口
            last_window = torch.FloatTensor(X[-1:]).to(self.device)

            with torch.no_grad():
                prediction = self.model(last_window)

            pred_np = prediction.cpu().numpy().flatten()
            stock_codes = returns_df.columns.tolist()
            n_stocks = len(stock_codes)

            if len(pred_np) == n_stocks:
                return pd.Series(pred_np, index=stock_codes)
            else:
                # 填充
                padded_pred = np.zeros(n_stocks)
                n = min(len(pred_np), n_stocks)
                padded_pred[:n] = pred_np[:n]
                return pd.Series(padded_pred, index=stock_codes)

        except Exception as e:
            print(f"    预测失败: {e}")
            return None


# 导出
__all__ = ['MultiStockLSTM', 'MultiStockTrainer']