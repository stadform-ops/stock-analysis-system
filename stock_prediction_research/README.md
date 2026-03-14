# Stock Prediction Research Workspace

该目录基于你提供的理想结构建立，用于在不破坏现有代码的情况下进行渐进式重构。

## 目标
- 提供统一的研究目录结构，减少模块分散。
- 为数据、实验、评估、可视化提供标准落位。
- 通过兼容层逐步迁移旧实现（`core/`, `experiments/`, `models/`, `utils/`）。

## 建议迁移顺序
1. 数据与特征工程（`core/data_loader.py`, `core/feature_engineer.py`）
2. 训练与评估（`core/trainer.py`, `core/evaluator.py`）
3. 模型实现（`models/*.py`, `baseline_models/*.py`）
4. 实验脚本（`experiments/*.py`）
5. 回测、指标与可视化（`evaluation/*`, `visualization/*`）

## 当前优化动作
- 新增统一配置模板：`configs/*.yaml`
- 新增实验统一入口：`run_experiment.py`
- 新增代码分析文档：`docs/codebase_analysis.md`

