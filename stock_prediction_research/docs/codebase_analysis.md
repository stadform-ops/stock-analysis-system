# 代码分析与优化建议

## 现状分析
当前仓库已经具备模型、实验、特征工程等核心功能，但目录职责有重叠：
- `core/` 与 `utils/` 中均存在数据/评估相关能力。
- `experiments/models/` 与顶层 `models/` 并存，模型来源不统一。
- 实验入口分散（多个 `exp_*.py`），缺乏统一调度层。

## 优化策略
1. **结构规范化**：采用 `stock_prediction_research/` 作为研究主目录。
2. **配置中心化**：用 `configs/*.yaml` 管理数据、模型、实验参数。
3. **统一运行入口**：通过 `run_experiment.py` 按配置启动实验。
4. **兼容迁移**：新模块初期作为桥接层，逐步替换旧目录中的具体实现。

## 下一步建议
- 将数据读取逻辑集中到 `core/data_loader.py`，并统一缓存策略。
- 在 `evaluation/metrics.py` 增加回归与方向性指标并输出到 `results/metrics/`。
- 在 `experiments/exp_model_compare.py` 实现同数据切分下的可重复模型对比。
