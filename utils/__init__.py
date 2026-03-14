"""
工具模块
简化版本
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "股票分析系统"
__description__ = "工具函数模块"

# 定义可导出的模块列表
__all__ = []

# 尝试导入工具模块
try:
    from .data_loader import StockDataLoader
    __all__.append('StockDataLoader')
except ImportError:
    print("⚠️  data_loader 模块不可用")

try:
    from .evaluator import PortfolioEvaluator
    __all__.append('PortfolioEvaluator')
except ImportError:
    print("⚠️  evaluator 模块不可用")

try:
    from .technical_indicators import TechnicalIndicators
    __all__.append('TechnicalIndicators')
except ImportError:
    print("⚠️  technical_indicators 模块不可用")

try:
    from .visualization import Visualizer
    __all__.append('Visualizer')
except ImportError:
    print("⚠️  visualization 模块不可用")

try:
    from .logger import setup_logger
    __all__.append('setup_logger')
except ImportError:
    print("⚠️  logger 模块不可用")

# 打印加载结果
if __all__:
    print(f"✅ 工具模块加载完成: {len(__all__)} 个模块")
else:
    print("❌ 没有可用的工具模块")
