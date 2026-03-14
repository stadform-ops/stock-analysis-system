# core/hyperparameter_search.py
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """超参数优化器"""

    def __init__(self, model_class: Callable, trainer_class: Callable):
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.best_params = None
        self.best_score = float('inf')
        self.results = []

    def grid_search(self, param_grid: Dict, features_df: pd.DataFrame,
                    returns_df: pd.DataFrame, n_trials: int = 20) -> Dict:
        """
        网格搜索超参数优化
        Args:
            param_grid: 超参数网格
            features_df: 特征数据
            returns_df: 收益率数据
            n_trials: 最大试验次数
        Returns:
            最佳超参数配置
        """
        print(f"🔍 开始超参数网格搜索")
        print(f"   参数空间大小: {len(ParameterGrid(param_grid))}")
        print(f"   最大试验次数: {n_trials}")

        # 生成参数组合
        param_combinations = list(ParameterGrid(param_grid))
        if len(param_combinations) > n_trials:
            # 随机选择n_trials个组合
            np.random.seed(42)
            selected_indices = np.random.choice(len(param_combinations), n_trials, replace=False)
            param_combinations = [param_combinations[i] for i in selected_indices]

        for i, params in enumerate(param_combinations):
            print(f"\n📊 试验 {i + 1}/{len(param_combinations)}")
            print(f"   参数: {params}")

            try:
                # 训练模型
                trainer = self.trainer_class(**params)
                history = trainer.train(features_df, returns_df, n_epochs=30)

                # 评估
                val_loss = min(history['val_loss'])

                # 记录结果
                trial_result = {
                    'params': params,
                    'best_val_loss': val_loss,
                    'final_train_loss': history['train_loss'][-1],
                    'epochs': len(history['train_loss'])
                }
                self.results.append(trial_result)

                print(f"   验证损失: {val_loss:.6f}")

                # 更新最佳参数
                if val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_params = params
                    print(f"   ✅ 新的最佳参数!")

            except Exception as e:
                print(f"   ❌ 试验失败: {e}")
                continue

        # 输出总结
        print(f"\n🎯 超参数搜索完成")
        print(f"   最佳验证损失: {self.best_score:.6f}")
        print(f"   最佳参数: {self.best_params}")

        return self.best_params

    def get_results_dataframe(self) -> pd.DataFrame:
        """获取试验结果DataFrame"""
        if not self.results:
            return pd.DataFrame()

        # 转换结果为DataFrame
        results_list = []
        for result in self.results:
            row = result['params'].copy()
            row['best_val_loss'] = result['best_val_loss']
            row['final_train_loss'] = result['final_train_loss']
            row['epochs'] = result['epochs']
            results_list.append(row)

        return pd.DataFrame(results_list).sort_values('best_val_loss')

    def visualize_results(self, top_n: int = 10):
        """可视化搜索结果"""
        if not self.results:
            return

        df_results = self.get_results_dataframe()

        print(f"\n📈 前{top_n}个最佳配置:")
        print(df_results.head(top_n).to_string())

        # 可以添加可视化图表
        import matplotlib.pyplot as plt

        # 损失分布
        plt.figure(figsize=(10, 6))
        plt.hist(df_results['best_val_loss'].values, bins=20, alpha=0.7)
        plt.axvline(self.best_score, color='red', linestyle='--', label=f'最佳: {self.best_score:.6f}')
        plt.xlabel('验证损失')
        plt.ylabel('频率')
        plt.title('超参数搜索结果分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./results/hyperparameter_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   图表已保存: ./results/hyperparameter_distribution.png")


# 定义LSTM-Transformer的超参数网格
LSTM_TRANSFORMER_PARAM_GRID = {
    'sequence_length': [10, 20, 30],
    'batch_size': [16, 32, 64],
    'lstm_hidden': [32, 64, 128],
    'd_model': [16, 32, 64],
    'nhead': [2, 4, 8],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005, 0.0001]
}

# 定义GNN的超参数网格
GNN_PARAM_GRID = {
    'sequence_length': [10, 20, 30],
    'window_size': [20, 40, 60],
    'batch_size': [8, 16, 32],
    'hidden_dim': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'heads': [2, 4, 8],
    'dropout': [0.1, 0.2, 0.3],
    'use_gat': [True, False],
    'threshold': [0.2, 0.3, 0.4]
}