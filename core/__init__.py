"""
股票分析系统 - 核心模块
简化版本，直接导入可用的模块
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "股票分析系统"
__description__ = "核心数据分析模块"

# 定义可导出的模块列表
__all__ = []

# 检查哪些模块可用
available_modules = []

# 尝试导入核心模块
try:
    from .data_pipeline import StockDataPipeline
    __all__.append('StockDataPipeline')
    available_modules.append('data_pipeline')
except ImportError:
    print("⚠️  data_pipeline 模块不可用")

try:
    from .feature_engineer import FeatureEngineer
    __all__.append('FeatureEngineer')
    available_modules.append('feature_engineer')
except ImportError:
    print("⚠️  feature_engineer 模块不可用")

try:
    from .stock_selector import StockSelector
    __all__.append('StockSelector')
    available_modules.append('stock_selector')
except ImportError:
    print("⚠️  stock_selector 模块不可用")

try:
    from .correlation_analyzer import CorrelationAnalyzer
    __all__.append('CorrelationAnalyzer')
    available_modules.append('correlation_analyzer')
except ImportError:
    print("⚠️  correlation_analyzer 模块不可用")

try:
    from .model_trainer import ModelTrainer
    __all__.append('ModelTrainer')
    available_modules.append('model_trainer')
except ImportError:
    print("⚠️  model_trainer 模块不可用")

try:
    from .portfolio_optimizer import PortfolioOptimizer
    __all__.append('PortfolioOptimizer')
    available_modules.append('portfolio_optimizer')
except ImportError:
    print("⚠️  portfolio_optimizer 模块不可用")

try:
    from .gnn_model import GNNTrainer, GNNStockPredictor
    __all__.extend(['GNNTrainer', 'GNNStockPredictor'])
    available_modules.append('gnn_model')
except ImportError:
    # 不显示警告，因为GNN是可选模块
    pass

try:
    from .hyperparameter_search import HyperparameterOptimizer
    __all__.append('HyperparameterOptimizer')
    available_modules.append('hyperparameter_search')
except ImportError:
    print("⚠️  hyperparameter_search 模块不可用")

# 打印可用模块信息
if available_modules:
    print(f"✅ 核心模块加载完成: {', '.join(available_modules)}")
else:
    print("❌ 没有可用的核心模块")
