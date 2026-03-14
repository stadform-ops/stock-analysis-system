# utils/technical_indicators.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """技术指标计算器"""

    def __init__(self):
        self.indicators = {}

    def add_all_indicators(self, df: pd.DataFrame, close_col: str = 'close',
                           volume_col: str = 'volume') -> pd.DataFrame:
        """添加所有技术指标"""
        df_copy = df.copy()

        # 确保有必要的列
        if close_col not in df_copy.columns:
            raise ValueError(f"数据中缺少{close_col}列")

        # 基础价格特征
        df_copy = self._add_basic_features(df_copy, close_col)

        # 移动平均
        df_copy = self._add_moving_averages(df_copy, close_col)

        # 动量指标
        df_copy = self._add_momentum_indicators(df_copy, close_col)

        # 波动率指标
        df_copy = self._add_volatility_indicators(df_copy, close_col)

        # 成交量指标
        if volume_col in df_copy.columns:
            df_copy = self._add_volume_indicators(df_copy, close_col, volume_col)

        # MACD指标
        df_copy = self._add_macd(df_copy, close_col)

        # 布林带
        df_copy = self._add_bollinger_bands(df_copy, close_col)

        # RSI
        df_copy = self._add_rsi(df_copy, close_col)

        # 删除NaN值
        df_copy = df_copy.dropna()

        print(f"📊 技术指标添加完成")
        print(f"   原始特征数: {len(df.columns)}")
        print(f"   新增特征数: {len(df_copy.columns) - len(df.columns)}")
        print(f"   总特征数: {len(df_copy.columns)}")

        return df_copy

    def _add_basic_features(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """添加基础价格特征"""
        df['returns'] = df[close_col].pct_change()
        df['log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
        df['price_change'] = df[close_col].diff()
        df['high_low_spread'] = df['high'] - df['low'] if 'high' in df.columns and 'low' in df.columns else 0

        return df

    def _add_moving_averages(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """添加移动平均线"""
        periods = [5, 10, 20, 30, 60]

        for period in periods:
            df[f'MA{period}'] = df[close_col].rolling(window=period).mean()
            df[f'MA{period}_ratio'] = df[close_col] / df[f'MA{period}']
            df[f'MA{period}_signal'] = (df[close_col] > df[f'MA{period}']).astype(int)

        # 移动平均交叉信号
        df['MA_cross_5_20'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA_cross_10_30'] = (df['MA10'] > df['MA30']).astype(int)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """添加动量指标"""
        # 收益率动量
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df[close_col].pct_change(periods=period)

        # 价格变化率
        df['roc_10'] = (df[close_col] - df[close_col].shift(10)) / df[close_col].shift(10)
        df['roc_20'] = (df[close_col] - df[close_col].shift(20)) / df[close_col].shift(20)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """添加波动率指标"""
        # 历史波动率
        for period in [5, 10, 20, 30]:
            df[f'volatility_{period}d'] = df['returns'].rolling(window=period).std() * np.sqrt(252)

        # ATR (平均真实波幅)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR_14'] = true_range.rolling(window=14).mean()

        return df

    def _add_volume_indicators(self, df: pd.DataFrame, close_col: str, volume_col: str) -> pd.DataFrame:
        """添加成交量指标"""
        # 成交量移动平均
        for period in [5, 10, 20]:
            df[f'volume_MA{period}'] = df[volume_col].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df[volume_col] / df[f'volume_MA{period}']

        # 量价关系
        df['price_volume_corr'] = df[close_col].pct_change().rolling(20).corr(df[volume_col].pct_change())

        # OBV (能量潮)
        df['obv'] = 0
        mask = df[close_col] > df[close_col].shift(1)
        df.loc[mask, 'obv'] = df[volume_col]
        df.loc[~mask, 'obv'] = -df[volume_col]
        df['obv'] = df['obv'].cumsum()

        return df

    def _add_macd(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """添加MACD指标"""
        # 计算EMA
        ema12 = df[close_col].ewm(span=12, adjust=False).mean()
        ema26 = df[close_col].ewm(span=26, adjust=False).mean()

        # MACD线
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # MACD信号
        df['MACD_cross'] = (df['MACD'] > df['MACD_signal']).astype(int)
        df['MACD_signal_change'] = df['MACD_cross'].diff().fillna(0)

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """添加布林带"""
        period = 20
        df['BB_middle'] = df[close_col].rolling(window=period).mean()
        bb_std = df[close_col].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std

        # 布林带位置
        df['BB_position'] = (df[close_col] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

        # 突破信号
        df['BB_break_upper'] = (df[close_col] > df['BB_upper']).astype(int)
        df['BB_break_lower'] = (df[close_col] < df['BB_lower']).astype(int)

        return df

    def _add_rsi(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """添加RSI指标"""
        period = 14
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # RSI信号
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)

        return df