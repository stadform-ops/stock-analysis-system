"""
模型模块初始化文件
确保所有模型模块能正确导入
"""

import importlib
import sys
from typing import Dict, Any

# 动态导入所有模型模块
def load_all_models() -> Dict[str, Any]:
    """动态加载所有模型模块"""
    modules = {}

    # 尝试导入contrastive_model
    try:
        from .contrastive_model import ContrastiveModel, ContrastiveTrainer
        modules['ContrastiveModel'] = ContrastiveModel
        modules['ContrastiveTrainer'] = ContrastiveTrainer
    except ImportError as e:
        print(f"⚠️  contrastive_model 导入失败: {e}")
    except Exception as e:
        print(f"⚠️  contrastive_model 导入异常: {e}")

    # 尝试导入dynamic_gnn
    try:
        from .dynamic_gnn import DynamicGNN, DynamicGNNTrainer
        modules['DynamicGNN'] = DynamicGNN
        modules['DynamicGNNTrainer'] = DynamicGNNTrainer
    except ImportError as e:
        print(f"⚠️  dynamic_gnn 导入失败: {e}")
    except Exception as e:
        print(f"⚠️  dynamic_gnn 导入异常: {e}")

    # 尝试导入hybrid_predictor
    try:
        from .hybrid_predictor import LSTMTransformerHybrid, LSTMTransformerTrainer
        modules['LSTMTransformerHybrid'] = LSTMTransformerHybrid
        modules['LSTMTransformerTrainer'] = LSTMTransformerTrainer
    except ImportError as e:
        print(f"⚠️  hybrid_predictor 导入失败: {e}")
    except Exception as e:
        print(f"⚠️  hybrid_predictor 导入异常: {e}")

    # 尝试导入multi_stock_predictor
    try:
        from .multi_stock_predictor import MultiStockLSTM, MultiStockTrainer
        modules['MultiStockLSTM'] = MultiStockLSTM
        modules['MultiStockTrainer'] = MultiStockTrainer
    except ImportError as e:
        print(f"⚠️  multi_stock_predictor 导入失败: {e}")
    except Exception as e:
        print(f"⚠️  multi_stock_predictor 导入异常: {e}")

    print(f"✅ 模型模块加载完成: {', '.join(modules.keys())}")
    return modules

# 自动加载所有模块
_model_modules = load_all_models()

# 导出到全局命名空间
if 'ContrastiveModel' in _model_modules:
    ContrastiveModel = _model_modules['ContrastiveModel']
if 'ContrastiveTrainer' in _model_modules:
    ContrastiveTrainer = _model_modules['ContrastiveTrainer']
if 'DynamicGNN' in _model_modules:
    DynamicGNN = _model_modules['DynamicGNN']
if 'DynamicGNNTrainer' in _model_modules:
    DynamicGNNTrainer = _model_modules['DynamicGNNTrainer']
if 'LSTMTransformerHybrid' in _model_modules:
    LSTMTransformerHybrid = _model_modules['LSTMTransformerHybrid']
if 'LSTMTransformerTrainer' in _model_modules:
    LSTMTransformerTrainer = _model_modules['LSTMTransformerTrainer']
if 'MultiStockLSTM' in _model_modules:
    MultiStockLSTM = _model_modules['MultiStockLSTM']
if 'MultiStockTrainer' in _model_modules:
    MultiStockTrainer = _model_modules['MultiStockTrainer']

__all__ = list(_model_modules.keys())