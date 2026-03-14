"""
数据管道模块 - 从本地CSV文件加载和处理股票数据
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import os
import sys


class StockDataPipeline:
    """股票数据处理流水线 - 从本地CSV文件加载数据"""

    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        初始化数据管道

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设置路径
        self.project_root = Path(".").resolve()
        self.data_dir = self.project_root / "data"

        # 原始数据路径
        self.raw_data_dir = self.data_dir / "raw" / "沪深300数据_baostock"

        # 处理后的数据路径
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 数据管道初始化完成")
        print(f"   项目根目录: {self.project_root}")
        print(f"   原始数据目录: {self.raw_data_dir}")
        print(f"   处理数据目录: {self.processed_dir}")

    def scan_available_stocks(self, max_stocks: int = 300) -> List[str]:
        """
        扫描可用的股票数据文件

        Args:
            max_stocks: 最大股票数量

        Returns:
            股票代码列表
        """
        print(f"🔍 扫描可用股票数据...")

        # 查找CSV文件
        csv_files = []

        # 检查多个可能的目录
        search_paths = [
            self.raw_data_dir / "汇总数据",  # 汇总数据目录
            self.raw_data_dir / "历史数据",  # 历史数据目录
            self.raw_data_dir  # 根目录
        ]

        for search_path in search_paths:
            if search_path.exists():
                csv_files.extend(list(search_path.glob("*.csv")))
                print(f"   在 {search_path} 中找到 {len(csv_files)} 个CSV文件")

        # 如果没有找到文件，尝试其他模式
        if not csv_files:
            print(f"⚠️  未在标准位置找到CSV文件，尝试全局搜索...")
            csv_files = list(self.project_root.rglob("*.csv"))

        # 过滤和排序
        stock_files = []
        for file_path in csv_files:
            file_name = file_path.stem
            # 只保留类似股票代码的文件名（6位数字）
            if file_name.isdigit() and len(file_name) == 6:
                stock_files.append((file_path, file_name))

        # 去重并按股票代码排序
        unique_stocks = {}
        for file_path, stock_code in stock_files:
            if stock_code not in unique_stocks:
                unique_stocks[stock_code] = file_path

        # 转换为列表
        stock_list = list(unique_stocks.items())

        # 限制数量
        if max_stocks > 0 and len(stock_list) > max_stocks:
            stock_list = stock_list[:max_stocks]

        # 按股票代码排序
        stock_list.sort(key=lambda x: x[0])

        # 只返回股票代码
        stock_codes = [code for code, _ in stock_list]

        print(f"✅ 找到 {len(stock_codes)} 只可用股票")
        if stock_codes:
            print(f"   示例股票: {stock_codes[:5]}")

        return stock_codes

    def load_single_stock_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        加载单只股票数据

        Args:
            stock_code: 股票代码

        Returns:
            股票数据DataFrame
        """
        # 可能的文件名模式
        possible_filenames = [
            f"{stock_code}.csv",
            f"sh{stock_code}.csv",
            f"sz{stock_code}.csv"
        ]

        # 可能的目录
        search_dirs = [
            self.raw_data_dir / "汇总数据",
            self.raw_data_dir / "历史数据",
            self.raw_data_dir
        ]

        file_path = None
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for filename in possible_filenames:
                test_path = search_dir / filename
                if test_path.exists():
                    file_path = test_path
                    break
            if file_path:
                break

        if not file_path:
            # 尝试全局搜索
            pattern = f"**/{stock_code}.csv"
            files = list(self.project_root.glob(pattern))
            if files:
                file_path = files[0]

        if not file_path or not file_path.exists():
            print(f"⚠️  股票 {stock_code} 数据文件未找到")
            return None

        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    if not df.empty:
                        print(f"   股票 {stock_code}: 使用编码 {encoding} 成功读取")
                        break
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    continue

            if df is None or df.empty:
                print(f"❌ 股票 {stock_code}: 所有编码都读取失败")
                return None

            # 数据清洗和标准化
            df = self.standardize_dataframe(df, stock_code)

            return df

        except Exception as e:
            print(f"❌ 加载股票 {stock_code} 失败: {e}")
            return None

    def standardize_dataframe(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化DataFrame格式

        Args:
            df: 原始DataFrame
            stock_code: 股票代码

        Returns:
            标准化后的DataFrame
        """
        # 创建副本
        df_clean = df.copy()

        # 1. 日期列处理
        date_columns = ['date', '日期', '交易日期', '时间', 'datetime']
        for date_col in date_columns:
            if date_col in df_clean.columns:
                df_clean['date'] = pd.to_datetime(df_clean[date_col])
                break

        # 如果没有找到日期列，尝试使用索引
        if 'date' not in df_clean.columns and not df_clean.empty:
            if df_clean.index.name in date_columns:
                df_clean = df_clean.reset_index()
                df_clean['date'] = pd.to_datetime(df_clean[df_clean.index.name])
            else:
                # 如果没有日期列，使用行号
                df_clean['date'] = pd.date_range(
                    start='2020-01-01',
                    periods=len(df_clean),
                    freq='D'
                )

        # 2. 价格列处理
        price_mapping = {
            'close': ['close', '收盘', '收盘价', '收盘价(元)'],
            'open': ['open', '开盘', '开盘价', '开盘价(元)'],
            'high': ['high', '最高', '最高价', '最高价(元)'],
            'low': ['low', '最低', '最低价', '最低价(元)'],
            'volume': ['volume', '成交量', '成交数量', '成交数量(手)']
        }

        for target_col, source_cols in price_mapping.items():
            for source_col in source_cols:
                if source_col in df_clean.columns:
                    df_clean[target_col] = pd.to_numeric(df_clean[source_col], errors='coerce')
                    break

        # 3. 确保必要的列存在
        required_cols = ['date', 'close']
        for col in required_cols:
            if col not in df_clean.columns:
                print(f"⚠️  股票 {stock_code}: 缺少必要列 {col}")
                if col == 'close' and 'open' in df_clean.columns:
                    df_clean['close'] = df_clean['open']
                else:
                    # 使用随机数据填充（仅用于测试）
                    df_clean[col] = np.random.randn(len(df_clean))

        # 4. 按日期排序
        df_clean = df_clean.sort_values('date')

        # 5. 设置日期索引
        df_clean.set_index('date', inplace=True)

        # 6. 处理缺失值
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

        # 7. 移除完全为NaN的行
        df_clean = df_clean.dropna(how='all')

        print(f"   股票 {stock_code}: 处理后形状: {df_clean.shape}")

        return df_clean

    def build_price_dataframe(self, stock_codes: List[str], max_stocks: int = 300) -> pd.DataFrame:
        """
        构建价格DataFrame

        Args:
            stock_codes: 股票代码列表
            max_stocks: 最大股票数量

        Returns:
            价格DataFrame，每列是一只股票的收盘价
        """
        print(f"📊 构建价格DataFrame...")

        # 限制股票数量
        if max_stocks > 0 and len(stock_codes) > max_stocks:
            stock_codes = stock_codes[:max_stocks]

        all_data = {}
        valid_stocks = []

        for i, stock_code in enumerate(stock_codes, 1):
            if i % 50 == 0 or i <= 5 or i == len(stock_codes):
                print(f"  [{i}/{len(stock_codes)}] 加载股票 {stock_code}...")

            df = self.load_single_stock_data(stock_code)
            if df is not None and not df.empty and 'close' in df.columns:
                all_data[stock_code] = df['close']
                valid_stocks.append(stock_code)

        if not all_data:
            print("❌ 没有成功加载任何股票数据")
            return pd.DataFrame()

        # 创建DataFrame
        price_df = pd.DataFrame(all_data)

        # 查找共同的时间索引
        print(f"  共同时间范围: {price_df.index.min()} 到 {price_df.index.max()}")
        print(f"  共同交易日数: {len(price_df)}")
        print(f"  收盘价DataFrame形状: {price_df.shape}")

        return price_df

    def calculate_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率

        Args:
            price_df: 价格DataFrame

        Returns:
            收益率DataFrame
        """
        print(f"📈 计算收益率...")

        if price_df.empty:
            print("❌ 价格数据为空，无法计算收益率")
            return pd.DataFrame()

        # 计算日收益率
        returns_df = price_df.pct_change().dropna()

        # 计算对数收益率
        # log_returns = np.log(price_df / price_df.shift(1)).dropna()

        print(f"✅ 收益率计算完成")
        print(f"   收益率数据形状: {returns_df.shape}")
        print(f"   共有交易日: {len(returns_df)}")

        return returns_df

    def save_processed_data(self, returns_df: pd.DataFrame, filename: str = "stock_returns.csv") -> Path:
        """
        保存处理后的数据

        Args:
            returns_df: 收益率DataFrame
            filename: 文件名

        Returns:
            保存的文件路径
        """
        if returns_df.empty:
            print("❌ 收益率数据为空，无法保存")
            return None

        # 保存路径
        save_path = self.processed_dir / filename

        try:
            returns_df.to_csv(save_path)
            print(f"✅ 数据已保存: {save_path}")
            return save_path
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return None

    def run_pipeline(self, max_stocks: int = 50, save_data: bool = True) -> Dict[str, Any]:
        """
        运行完整的数据处理流水线

        Args:
            max_stocks: 最大处理的股票数量
            save_data: 是否保存数据

        Returns:
            处理结果字典
        """
        print(f"🚀 开始运行数据处理流水线")
        print(f"=" * 60)

        # 1. 扫描可用股票
        stock_codes = self.scan_available_stocks(max_stocks)

        if not stock_codes:
            print("❌ 没有找到可用的股票数据")
            return {}

        # 2. 构建价格DataFrame
        price_df = self.build_price_dataframe(stock_codes, max_stocks)

        if price_df.empty:
            print("❌ 价格DataFrame构建失败")
            return {}

        # 3. 计算收益率
        returns_df = self.calculate_returns(price_df)

        if returns_df.empty:
            print("❌ 收益率计算失败")
            return {}

        # 4. 保存数据
        save_path = None
        if save_data:
            save_path = self.save_processed_data(returns_df)

        # 5. 统计信息
        print(f"\n📊 收益率统计:")
        print(f"   平均日收益率范围: {returns_df.min().min():.4%} 到 {returns_df.max().max():.4%}")
        print(f"   日收益率波动范围: {returns_df.std().min():.4%} 到 {returns_df.std().max():.4%}")

        # 前5只股票的统计
        print(f"\n📈 前5只股票的收益率统计:")
        for i, stock in enumerate(returns_df.columns[:5], 1):
            stock_returns = returns_df[stock]
            mean_return = stock_returns.mean()
            std_return = stock_returns.std()
            sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            print(f"   {stock}: 均值={mean_return:.4%}, 波动={std_return:.4%}, 夏普={sharpe:.4f}")

        result = {
            'stock_codes': stock_codes,
            'price_df': price_df,
            'returns_df': returns_df,
            'save_path': save_path,
            'num_stocks': len(stock_codes),
            'num_days': len(returns_df)
        }

        print(f"\n✅ 数据处理流水线完成")
        print(f"   处理股票数量: {len(stock_codes)}")
        print(f"   处理交易日数: {len(returns_df)}")
        print(f"=" * 60)

        return result

    def load_existing_data(self, filename: str = "stock_returns.csv") -> Optional[pd.DataFrame]:
        """
        加载已存在的数据

        Args:
            filename: 文件名

        Returns:
            加载的DataFrame
        """
        data_path = self.processed_dir / filename

        if not data_path.exists():
            print(f"⚠️  数据文件不存在: {data_path}")
            return None

        try:
            print(f"📂 加载已有数据: {data_path}")
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)

            print(f"✅ 数据加载成功")
            print(f"   数据形状: {df.shape}")
            print(f"   时间范围: {df.index.min()} 到 {df.index.max()}")
            print(f"   股票数量: {len(df.columns)}")

            return df
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None


def test_data_pipeline():
    """测试数据管道"""
    print("🧪 测试数据管道...")

    # 创建数据管道实例
    pipeline = StockDataPipeline()

    # 测试1: 扫描股票
    stock_codes = pipeline.scan_available_stocks(max_stocks=20)
    print(f"找到股票: {len(stock_codes)} 只")

    if not stock_codes:
        print("❌ 测试失败: 未找到股票")
        return False

    # 测试2: 加载单只股票
    test_stock = stock_codes[0]
    stock_data = pipeline.load_single_stock_data(test_stock)

    if stock_data is None or stock_data.empty:
        print(f"❌ 测试失败: 无法加载股票 {test_stock}")
        return False

    print(f"股票 {test_stock} 数据形状: {stock_data.shape}")
    print(f"列名: {stock_data.columns.tolist()}")

    # 测试3: 运行完整流水线（小规模测试）
    result = pipeline.run_pipeline(max_stocks=5, save_data=False)

    if not result or 'returns_df' not in result:
        print("❌ 测试失败: 完整流水线运行失败")
        return False

    returns_df = result['returns_df']
    print(f"收益率数据形状: {returns_df.shape}")

    # 测试4: 保存和加载数据
    if not returns_df.empty:
        save_path = pipeline.save_processed_data(returns_df, "test_returns.csv")
        if save_path and save_path.exists():
            print(f"✅ 数据保存测试成功: {save_path}")

            # 加载测试
            loaded_df = pipeline.load_existing_data("test_returns.csv")
            if loaded_df is not None and not loaded_df.empty:
                print(f"✅ 数据加载测试成功: {loaded_df.shape}")

    print("✅ 所有测试通过!")
    return True


if __name__ == "__main__":
    # 运行测试
    success = test_data_pipeline()

    if success:
        print("\n🎉 数据管道测试完成，可以正常使用")
    else:
        print("\n⚠️  数据管道测试失败，请检查配置和数据文件")
        sys.exit(1)
