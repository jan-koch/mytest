"""
Hyperliquid Price Data Analyzer

This module provides technical analysis tools and indicators for algorithmic
trading using Hyperliquid price data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List


class PriceAnalyzer:
    """
    Technical analysis and indicator calculator for price data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a price DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()

    def add_sma(self, period: int, column: str = 'close', name: Optional[str] = None) -> pd.DataFrame:
        """
        Add Simple Moving Average (SMA).

        Args:
            period: Number of periods for the moving average
            column: Column to calculate SMA on (default: 'close')
            name: Custom name for the SMA column

        Returns:
            DataFrame with added SMA column
        """
        col_name = name or f'sma_{period}'
        self.df[col_name] = self.df[column].rolling(window=period).mean()
        return self.df

    def add_ema(self, period: int, column: str = 'close', name: Optional[str] = None) -> pd.DataFrame:
        """
        Add Exponential Moving Average (EMA).

        Args:
            period: Number of periods for the moving average
            column: Column to calculate EMA on (default: 'close')
            name: Custom name for the EMA column

        Returns:
            DataFrame with added EMA column
        """
        col_name = name or f'ema_{period}'
        self.df[col_name] = self.df[column].ewm(span=period, adjust=False).mean()
        return self.df

    def add_rsi(self, period: int = 14, column: str = 'close') -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).

        Args:
            period: RSI period (default: 14)
            column: Column to calculate RSI on (default: 'close')

        Returns:
            DataFrame with added RSI column
        """
        delta = self.df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return self.df

    def add_macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            column: Column to calculate MACD on (default: 'close')

        Returns:
            DataFrame with MACD, signal, and histogram columns
        """
        ema_fast = self.df[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.df[column].ewm(span=slow_period, adjust=False).mean()

        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal_period, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        return self.df

    def add_bollinger_bands(
        self,
        period: int = 20,
        num_std: float = 2.0,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands.

        Args:
            period: Period for moving average (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            column: Column to calculate bands on (default: 'close')

        Returns:
            DataFrame with upper, middle, and lower band columns
        """
        sma = self.df[column].rolling(window=period).mean()
        std = self.df[column].rolling(window=period).std()

        self.df['bb_upper'] = sma + (std * num_std)
        self.df['bb_middle'] = sma
        self.df['bb_lower'] = sma - (std * num_std)
        self.df['bb_width'] = self.df['bb_upper'] - self.df['bb_lower']
        return self.df

    def add_atr(self, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR).

        Args:
            period: ATR period (default: 14)

        Returns:
            DataFrame with ATR column
        """
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        return self.df

    def add_stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.

        Args:
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            smooth_k: Smoothing period for %K (default: 3)

        Returns:
            DataFrame with %K and %D columns
        """
        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()

        stoch_k = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        self.df['stoch_k'] = stoch_k.rolling(window=smooth_k).mean()
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=d_period).mean()
        return self.df

    def add_volume_indicators(self) -> pd.DataFrame:
        """
        Add volume-based indicators.

        Returns:
            DataFrame with volume indicators
        """
        # Volume SMA
        self.df['volume_sma_20'] = self.df['volume'].rolling(window=20).mean()

        # Volume ratio
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_20']

        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i - 1]:
                obv.append(obv[-1] + self.df['volume'].iloc[i])
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i - 1]:
                obv.append(obv[-1] - self.df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        self.df['obv'] = obv

        return self.df

    def add_returns(self) -> pd.DataFrame:
        """
        Add return calculations.

        Returns:
            DataFrame with return columns
        """
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
        self.df['cumulative_returns'] = (1 + self.df['returns']).cumprod() - 1
        return self.df

    def detect_support_resistance(
        self,
        window: int = 20,
        threshold: float = 0.02
    ) -> Tuple[List[float], List[float]]:
        """
        Detect support and resistance levels.

        Args:
            window: Window size for local minima/maxima detection
            threshold: Threshold for grouping nearby levels

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        # Find local minima (support)
        local_min = self.df['low'].rolling(window=window, center=True).min()
        supports = self.df['low'][self.df['low'] == local_min].dropna().tolist()

        # Find local maxima (resistance)
        local_max = self.df['high'].rolling(window=window, center=True).max()
        resistances = self.df['high'][self.df['high'] == local_max].dropna().tolist()

        # Group nearby levels
        support_levels = self._group_levels(supports, threshold)
        resistance_levels = self._group_levels(resistances, threshold)

        return support_levels, resistance_levels

    def _group_levels(self, levels: List[float], threshold: float) -> List[float]:
        """Group nearby price levels."""
        if not levels:
            return []

        levels = sorted(levels)
        grouped = [levels[0]]

        for level in levels[1:]:
            if abs(level - grouped[-1]) / grouped[-1] > threshold:
                grouped.append(level)

        return grouped

    def calculate_volatility(self, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            window: Window size for volatility calculation

        Returns:
            Series with volatility values
        """
        returns = self.df['close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(window)
        return volatility

    def get_statistics(self) -> Dict:
        """
        Get statistical summary of the price data.

        Returns:
            Dictionary with statistical metrics
        """
        if 'returns' not in self.df.columns:
            self.add_returns()

        stats = {
            'period_start': self.df.index[0],
            'period_end': self.df.index[-1],
            'total_periods': len(self.df),
            'start_price': self.df['close'].iloc[0],
            'end_price': self.df['close'].iloc[-1],
            'min_price': self.df['low'].min(),
            'max_price': self.df['high'].max(),
            'mean_price': self.df['close'].mean(),
            'median_price': self.df['close'].median(),
            'total_return': (self.df['close'].iloc[-1] / self.df['close'].iloc[0] - 1) * 100,
            'volatility': self.df['returns'].std() * np.sqrt(len(self.df)),
            'avg_volume': self.df['volume'].mean(),
            'total_volume': self.df['volume'].sum(),
        }

        return stats

    def backtest_simple_strategy(
        self,
        buy_condition: pd.Series,
        sell_condition: pd.Series,
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Simple backtesting framework.

        Args:
            buy_condition: Boolean series indicating buy signals
            sell_condition: Boolean series indicating sell signals
            initial_capital: Starting capital for backtest

        Returns:
            Dictionary with backtest results
        """
        position = 0
        capital = initial_capital
        trades = []

        for i in range(len(self.df)):
            if buy_condition.iloc[i] and position == 0:
                # Buy
                position = capital / self.df['close'].iloc[i]
                entry_price = self.df['close'].iloc[i]
                trades.append({
                    'type': 'buy',
                    'timestamp': self.df.index[i],
                    'price': entry_price,
                    'position': position
                })
                capital = 0

            elif sell_condition.iloc[i] and position > 0:
                # Sell
                exit_price = self.df['close'].iloc[i]
                capital = position * exit_price
                pnl = capital - initial_capital
                trades.append({
                    'type': 'sell',
                    'timestamp': self.df.index[i],
                    'price': exit_price,
                    'pnl': pnl,
                    'return': (pnl / initial_capital) * 100
                })
                position = 0

        # Close any open position
        if position > 0:
            capital = position * self.df['close'].iloc[-1]

        total_return = ((capital - initial_capital) / initial_capital) * 100
        num_trades = len([t for t in trades if t['type'] == 'sell'])

        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'num_trades': num_trades,
            'trades': trades
        }

    def get_dataframe(self) -> pd.DataFrame:
        """Get the current DataFrame with all added indicators."""
        return self.df
