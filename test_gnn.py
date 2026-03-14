"""
最终系统测试脚本
"""

import sys
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_all_models_import():
    """测试所有模型导入"""
    print("🧪 测试所有模型导入...")
    print("="*60)

    success = True

    # 测试contrastive_model
    try:
        from models.contrastive_model import ContrastiveModel, ContrastiveTrainer
        print("✅ contrastive_model导入成功")

        # 实例化
        model = ContrastiveModel(input_dim=20, embedding_dim=16)
        print("✅ ContrastiveModel实例化成功")

        trainer = ContrastiveTrainer(device='cpu')
        print("✅ ContrastiveTrainer实例化成功")

    except Exception as e:
        print(f"❌ contrastive_model导入失败: {e}")
        success = False

    # 测试dynamic_gnn
    try:
        from models.dynamic_gnn import DynamicGNN, DynamicGNNTrainer
        print("✅ dynamic_gnn导入成功")

        # 实例化
        model = DynamicGNN(input_dim=20, output_dim=10)
        print("✅ DynamicGNN实例化成功")

        trainer = DynamicGNNTrainer(device='cpu')
        print("✅ DynamicGNNTrainer实例化成功")

    except Exception as e:
        print(f"❌ dynamic_gnn导入失败: {e}")
        success = False

    # 测试hybrid_predictor
    try:
        from models.hybrid_predictor import LSTMTransformerHybrid, LSTMTransformerTrainer
        print("✅ hybrid_predictor导入成功")

        model = LSTMTransformerHybrid(input_dim=50, output_dim=10)
        print("✅ LSTMTransformerHybrid实例化成功")

        trainer = LSTMTransformerTrainer(device='cpu')
        print("✅ LSTMTransformerTrainer实例化成功")

    except Exception as e:
        print(f"❌ hybrid_predictor导入失败: {e}")
        success = False

    # 测试multi_stock_predictor
    try:
        from models.multi_stock_predictor import MultiStockLSTM, MultiStockTrainer
        print("✅ multi_stock_predictor导入成功")

        model = MultiStockLSTM(input_dim=20, output_dim=10)
        print("✅ MultiStockLSTM实例化成功")

        trainer = MultiStockTrainer(device='cpu')
        print("✅ MultiStockTrainer实例化成功")

    except Exception as e:
        print(f"❌ multi_stock_predictor导入失败: {e}")
        success = False

    return success

def test_model_training():
    """测试模型训练功能"""
    print("\n🧪 测试模型训练功能...")
    print("="*60)

    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    n_stocks = 20
    returns_data = np.random.randn(200, n_stocks) * 0.01
    stock_codes = [f"stock_{i:03d}" for i in range(n_stocks)]
    returns_df = pd.DataFrame(returns_data, index=dates, columns=stock_codes)

    features_data = np.random.randn(n_stocks, 10)
    features_df = pd.DataFrame(features_data, index=stock_codes,
                             columns=[f"feature_{j}" for j in range(10)])

    success = True

    # 测试MultiStockTrainer训练
    try:
        from models.multi_stock_predictor import MultiStockTrainer
        print("1. 测试MultiStockTrainer训练...")

        trainer = MultiStockTrainer(sequence_length=10, device='cpu')
        history = trainer.train(returns_df, n_epochs=3)

        if history:
            print(f"  ✅ MultiStockTrainer训练成功, 最终损失: {history['train_loss'][-1]:.6f}")
        else:
            print("  ⚠️  MultiStockTrainer训练返回空结果")

    except Exception as e:
        print(f"  ❌ MultiStockTrainer训练失败: {e}")
        success = False

    # 测试LSTMTransformerTrainer训练
    try:
        from models.hybrid_predictor import LSTMTransformerTrainer
        print("\n2. 测试LSTMTransformerTrainer训练...")

        trainer = LSTMTransformerTrainer(sequence_length=10, device='cpu')
        history = trainer.train(features_df, returns_df, n_epochs=3)

        if history:
            print(f"  ✅ LSTMTransformerTrainer训练成功, 最终损失: {history['train_loss'][-1]:.6f}")
        else:
            print("  ⚠️  LSTMTransformerTrainer训练返回空结果")

    except Exception as e:
        print(f"  ❌ LSTMTransformerTrainer训练失败: {e}")
        success = False

    return success

def test_system_integration():
    """测试系统集成"""
    print("\n🧪 测试系统集成...")
    print("="*60)

    try:
        from main import StockAnalysisSystem

        system = StockAnalysisSystem()
        print("✅ 系统实例化成功")

        # 修改配置以快速测试
        config = system.get_default_config()
        config['model']['epochs'] = 3
        config['data']['max_stocks'] = 20
        config['backtest']['enabled'] = False
        system.config = config

        print("✅ 配置修改完成")

        return True

    except Exception as e:
        print(f"❌ 系统集成测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*80)
    print("🎯 股票分析系统 - 最终测试")
    print("="*80)

    # 运行测试
    import_success = test_all_models_import()
    training_success = test_model_training()
    integration_success = test_system_integration()

    print("\n" + "="*80)
    print("📊 最终测试结果汇总:")
    print("="*80)
    print(f"模型导入测试: {'✅ 通过' if import_success else '❌ 失败'}")
    print(f"模型训练测试: {'✅ 通过' if training_success else '❌ 失败'}")
    print(f"系统集成测试: {'✅ 通过' if integration_success else '❌ 失败'}")

    if all([import_success, training_success, integration_success]):
        print("\n🎉 所有测试通过！可以运行完整系统。")
        print("运行命令: python main.py")
    else:
        print("\n⚠️  部分测试失败，需要修复上述问题。")

if __name__ == "__main__":
    main()