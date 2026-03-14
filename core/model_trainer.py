# core/model_trainer.py
"""
模型训练模块
训练和保存预测模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')
import json
from pathlib import Path
import time


class ModelTrainer:
    """模型训练器"""

    def __init__(self,
                 model_type: str = 'lstm',
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 device: str = None):
        """
        初始化模型训练器

        Args:
            model_type: 模型类型 ('lstm', 'transformer_lstm')
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: dropout比例
            learning_rate: 学习率
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 模型
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

        print(f"模型训练器初始化:")
        print(f"  模型类型: {model_type}")
        print(f"  隐藏层大小: {hidden_size}")
        print(f"  层数: {num_layers}")
        print(f"  Dropout: {dropout}")
        print(f"  学习率: {learning_rate}")
        print(f"  设备: {self.device}")

    def create_model(self, input_size: int, output_size: int = 1):
        """创建模型"""
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                output_size=output_size
            )
        elif self.model_type == 'transformer_lstm':
            self.model = TransformerLSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                output_size=output_size
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"创建 {self.model_type} 模型:")
        print(f"  输入大小: {input_size}")
        print(f"  输出大小: {output_size}")
        print(f"  总参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self,
              train_data: Dict,
              val_data: Dict,
              epochs: int = 50,
              batch_size: int = 32,
              patience: int = 10,
              save_dir: str = './models/saved_models') -> Dict:
        """
        训练模型

        Args:
            train_data: 训练数据 {'features': ..., 'targets': ...}
            val_data: 验证数据 {'features': ..., 'targets': ...}
            epochs: 训练轮数
            batch_size: 批大小
            patience: 早停耐心值
            save_dir: 保存目录

        Returns:
            训练历史
        """
        print("=" * 60)
        print("开始模型训练")
        print("=" * 60)

        # 创建数据加载器
        train_loader = self._create_data_loader(train_data, batch_size, shuffle=True)
        val_loader = self._create_data_loader(val_data, batch_size, shuffle=False)

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        # 早停
        patience_counter = 0
        best_model_state = None

        start_time = time.time()

        for epoch in range(epochs):
            # 训练
            train_loss = self._train_epoch(train_loader)

            # 验证
            val_loss = self._validate(val_loader)

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # 打印进度
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")

            # 检查是否最佳模型
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0

                # 保存最佳模型
                self.save_model(save_dir, f"best_{self.model_type}_model.pth")
                print(f"  ✓ 保存最佳模型，验证损失: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⏹️ 早停，在epoch {epoch + 1}")
                    break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        history['training_time'] = training_time

        print(f"\n训练完成:")
        print(f"  总时间: {training_time:.1f}秒")
        print(f"  最佳验证损失: {history['best_val_loss']:.6f} (epoch {history['best_epoch'] + 1})")

        return history

    def _create_data_loader(self, data: Dict, batch_size: int, shuffle: bool) -> DataLoader:
        """创建数据加载器"""
        features = torch.FloatTensor(data['features'])
        targets = torch.FloatTensor(data['targets'])

        dataset = TensorDataset(features, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_features, batch_targets in dataloader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_features)
            loss = self.criterion(predictions, batch_targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                predictions = self.model(batch_features)
                loss = self.criterion(predictions, batch_targets)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            predictions = self.model(features_tensor)
            predictions = predictions.cpu().numpy()

        return predictions

    def save_model(self, save_dir: str, filename: str):
        """保存模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        model_path = save_path / filename

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }, model_path)

        print(f"模型已保存: {model_path}")

    def load_model(self, model_path: str, input_size: int):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model_type = checkpoint['model_type']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']

        # 创建模型
        self.create_model(input_size=input_size)

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        print(f"模型已加载: {model_path}")


class LSTMModel(nn.Module):
    """LSTM模型"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步
        lstm_out = lstm_out[:, -1, :]

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 全连接
        output = self.fc(lstm_out)

        return output


class TransformerLSTMModel(nn.Module):
    """Transformer-LSTM混合模型"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(TransformerLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        # Transformer编码
        transformer_out = self.transformer(x)

        # LSTM
        lstm_out, _ = self.lstm(transformer_out)

        # 取最后一个时间步
        lstm_out = lstm_out[:, -1, :]

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 全连接
        output = self.fc(lstm_out)

        return output


def test_model_trainer():
    """测试模型训练器"""
    print("=" * 60)
    print("测试模型训练器")
    print("=" * 60)

    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    seq_len = 20
    n_features = 10

    X_train = np.random.randn(n_samples, seq_len, n_features)
    y_train = np.random.randn(n_samples, 1)

    X_val = np.random.randn(200, seq_len, n_features)
    y_val = np.random.randn(200, 1)

    train_data = {'features': X_train, 'targets': y_train}
    val_data = {'features': X_val, 'targets': y_val}

    # 训练LSTM模型
    print("训练LSTM模型...")
    lstm_trainer = ModelTrainer(model_type='lstm', hidden_size=32, num_layers=1)
    lstm_trainer.create_model(input_size=n_features)
    history_lstm = lstm_trainer.train(train_data, val_data, epochs=5, batch_size=32)

    # 训练Transformer-LSTM模型
    print("\n训练Transformer-LSTM模型...")
    transformer_trainer = ModelTrainer(model_type='transformer_lstm', hidden_size=32, num_layers=1)
    transformer_trainer.create_model(input_size=n_features)
    history_transformer = transformer_trainer.train(train_data, val_data, epochs=5, batch_size=32)

    return lstm_trainer, transformer_trainer


if __name__ == "__main__":
    test_model_trainer()