# utils/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class StockDataLoader:
    """沪深300多股票数据加载器"""

    def __init__(self, data_dir: str = "./data/raw/沪深300数据_baostock/历史数据"):
        self.data_dir = Path(data_dir)
        self.available_stocks = self._scan_stocks()

    def _scan_stocks(self) -> List[str]:
        """扫描数据目录，返回可用股票代码列表"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        csv_files = list(self.data_dir.glob("*.csv"))
        stock_codes = [f.stem for f in csv_files if f.stem.isdigit()]
        print(f"📁 在目录中发现 {len(stock_codes)} 只股票")
        return sorted(stock_codes)[:50]  # 先取前50只测试

    def load_single_stock(self, stock_code: str) -> Optional[pd.DataFrame]:
        """加载单只股票数据，返回包含收盘价的DataFrame"""
        file_path = self.data_dir / f"{stock_code}.csv"
        if not file_path.exists():
            print(f"⚠️  文件不存在: {file_path}")
            return None

        try:
            # 尝试多种编码读取
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"  {stock_code}: 使用编码 {encoding} 成功读取")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"❌ 无法读取文件: {file_path}")
                return None

            # 统一列名（处理中英文列名）- 修复列名映射
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'date' in col_lower or '日期' in col or '时间' in col:
                    col_mapping[col] = 'date'
                elif ('close' in col_lower or '收盘' in col) and '昨' not in col:  # 排除"昨收盘"
                    col_mapping[col] = 'close'
                elif 'open' in col_lower or '开盘' in col:
                    col_mapping[col] = 'open'
                elif 'high' in col_lower or '最高' in col:
                    col_mapping[col] = 'high'
                elif 'low' in col_lower or '最低' in col:
                    col_mapping[col] = 'low'
                elif 'volume' in col_lower or '成交' in col:
                    # 区分成交量和成交额
                    if '额' in col or 'amount' in col_lower:
                        col_mapping[col] = 'amount'
                    else:
                        col_mapping[col] = 'volume'

            df.rename(columns=col_mapping, inplace=True)

            # 确保有日期和收盘价
            if 'date' not in df.columns or 'close' not in df.columns:
                print(f"⚠️  {stock_code}: 缺少必要列 date 或 close")
                print(f"  {stock_code}: 实际列名: {df.columns.tolist()}")
                return None

            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # 只返回收盘价列
            if 'close' in df.columns:
                result_df = df[['close']].copy()
            else:
                print(f"❌  {stock_code}: 没有找到close列")
                return None

            # 确保是DataFrame
            if not isinstance(result_df, pd.DataFrame):
                result_df = pd.DataFrame({'close': result_df})

            print(f"  {stock_code}: 处理后形状: {result_df.shape}")
            print(f"  {stock_code}: 时间范围: {result_df.index.min()} 到 {result_df.index.max()}")

            return result_df

        except Exception as e:
            print(f"❌ 加载股票 {stock_code} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_multiple_stocks(self, stock_codes: List[str] = None) -> pd.DataFrame:
        """
        加载多只股票，返回收益率DataFrame
        列：股票代码，行：日期，值：日收益率
        """
        if stock_codes is None:
            stock_codes = self.available_stocks[:20]  # 默认加载前20只

        print(f"📈 开始加载 {len(stock_codes)} 只股票数据...")

        close_prices_dict = {}
        valid_stocks = []

        for i, code in enumerate(stock_codes):
            print(f"\n[{i + 1}/{len(stock_codes)}] 加载股票 {code}...")
            df = self.load_single_stock(code)

            if df is not None and not df.empty:
                # 确保我们有一个DataFrame
                if isinstance(df, pd.DataFrame):
                    if 'close' in df.columns:
                        # 提取close列作为Series
                        close_series = df['close']
                        if isinstance(close_series, pd.Series):
                            close_prices_dict[code] = close_series
                            valid_stocks.append(code)
                            print(f"  ✓ {code}: 成功加载 {len(close_series)} 个交易日数据")
                        else:
                            print(f"  ✗ {code}: close列不是Series类型")
                    else:
                        print(f"  ✗ {code}: DataFrame中没有close列")
                else:
                    print(f"  ✗ {code}: 返回的不是DataFrame")

        if not close_prices_dict:
            raise ValueError("没有成功加载任何股票数据")

        print(f"\n📊 创建收盘价DataFrame...")

        # 创建收盘价DataFrame
        try:
            # 先找到所有股票共同的时间索引
            all_indices = [series.index for series in close_prices_dict.values()]
            if not all_indices:
                raise ValueError("没有可用的时间索引")

            # 找出共同的时间范围
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            if len(common_index) == 0:
                raise ValueError("没有共同的时间索引")

            print(f"  共同时间范围: {common_index[0]} 到 {common_index[-1]}")
            print(f"  共同交易日数: {len(common_index)}")

            # 对齐所有股票的数据
            aligned_data = {}
            for code, series in close_prices_dict.items():
                aligned_series = series.reindex(common_index)
                aligned_data[code] = aligned_series

            close_prices_df = pd.DataFrame(aligned_data)
            print(f"  收盘价DataFrame形状: {close_prices_df.shape}")

        except Exception as e:
            print(f"❌ 创建收盘价DataFrame失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 计算收益率
        print(f"\n📈 计算收益率...")
        returns_df = close_prices_df.pct_change().dropna()

        # 删除任何含有NaN的行
        returns_df = returns_df.dropna()

        print(f"\n✅ 成功加载 {len(valid_stocks)} 只股票")
        print(f"   时间范围: {returns_df.index.min()} 到 {returns_df.index.max()}")
        print(f"   收益率数据形状: {returns_df.shape}")
        print(f"   共有交易日: {len(returns_df)}")

        # 基本统计信息
        if not returns_df.empty:
            mean_returns = returns_df.mean() * 100
            std_returns = returns_df.std() * 100
            print(f"\n📊 收益率统计:")
            print(f"   平均日收益率范围: {mean_returns.min():.4f}% 到 {mean_returns.max():.4f}%")
            print(f"   日收益率波动范围: {std_returns.min():.4f}% 到 {std_returns.max():.4f}%")

            # 显示前几只股票的详细信息
            print(f"\n📈 前5只股票的收益率统计:")
            for i, code in enumerate(returns_df.columns[:5]):
                returns = returns_df[code]
                if len(returns) > 0:
                    mean_return = returns.mean() * 100
                    std_return = returns.std() * 100
                    sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
                    print(f"   {code}: 均值={mean_return:.4f}%, 波动={std_return:.4f}%, 夏普={sharpe_ratio:.4f}")

        # 保存处理后的数据
        processed_dir = Path("./data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        returns_path = processed_dir / "stock_returns.csv"
        returns_df.to_csv(returns_path)
        print(f"   数据已保存: {returns_path}")

        return returns_df