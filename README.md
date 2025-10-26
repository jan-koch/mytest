# Hyperliquid Price Analysis Tool

A comprehensive Python toolkit for analyzing historical price data from Hyperliquid for algorithmic trading strategies.

## Features

### API Client (`hyperliquid_client.py`)
- Fetch historical candlestick (OHLCV) data
- Support for multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d, etc.)
- Get current mid prices for all trading pairs
- Retrieve market metadata
- Pandas DataFrame integration for easy data manipulation
- Both mainnet and testnet support
- Batch fetching for multiple coins

### Technical Analysis (`hyperliquid_analyzer.py`)
- **Moving Averages**: SMA, EMA
- **Momentum Indicators**: RSI, Stochastic Oscillator
- **Trend Indicators**: MACD
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, Volume Ratio
- **Price Analysis**: Returns, Log Returns, Cumulative Returns
- **Support/Resistance Detection**: Automatic level detection
- **Backtesting Framework**: Simple strategy backtesting
- **Statistical Analysis**: Comprehensive price statistics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from hyperliquid_client import HyperliquidClient

# Initialize client (mainnet)
client = HyperliquidClient()

# Get available coins
coins = client.get_available_coins()
print(f"Available coins: {coins[:5]}")

# Get current prices
prices = client.get_all_mids()
print(f"BTC Price: ${prices['BTC']}")

# Fetch historical data (last 100 hours)
df = client.get_candles_df("BTC", "1h", lookback_periods=100)
print(df.head())
```

### Technical Analysis

```python
from hyperliquid_analyzer import PriceAnalyzer

# Get data
df = client.get_candles_df("ETH", "4h", lookback_periods=200)

# Initialize analyzer
analyzer = PriceAnalyzer(df)

# Add indicators
analyzer.add_sma(20)
analyzer.add_sma(50)
analyzer.add_rsi(14)
analyzer.add_macd()
analyzer.add_bollinger_bands()
analyzer.add_returns()

# Get enhanced dataframe
df_with_indicators = analyzer.get_dataframe()
print(df_with_indicators.tail())

# Get statistics
stats = analyzer.get_statistics()
print(f"Total Return: {stats['total_return']:.2f}%")
```

### Multi-Coin Analysis

```python
# Compare multiple coins
coins = ["BTC", "ETH", "SOL"]
data = client.get_multiple_candles_df(
    coins=coins,
    interval="1d",
    lookback_periods=30
)

for coin, df in data.items():
    if not df.empty:
        returns = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        print(f"{coin}: {returns:+.2f}%")
```

### Strategy Backtesting

```python
# Simple moving average crossover strategy
analyzer.add_sma(20)
analyzer.add_sma(50)

df_strategy = analyzer.get_dataframe()

# Generate signals
buy_signal = (df_strategy['sma_20'] > df_strategy['sma_50']) & \
             (df_strategy['sma_20'].shift(1) <= df_strategy['sma_50'].shift(1))

sell_signal = (df_strategy['sma_20'] < df_strategy['sma_50']) & \
              (df_strategy['sma_20'].shift(1) >= df_strategy['sma_50'].shift(1))

# Run backtest
results = analyzer.backtest_simple_strategy(
    buy_condition=buy_signal,
    sell_condition=sell_signal,
    initial_capital=10000
)

print(f"Total Return: {results['total_return']:.2f}%")
print(f"Number of Trades: {results['num_trades']}")
```

## API Reference

### HyperliquidClient

#### Initialization
```python
client = HyperliquidClient(testnet=False)
```
- `testnet`: Use testnet API if True (default: False)

#### Methods

**get_candles_df(coin, interval, start_time=None, end_time=None, lookback_periods=None)**
- Fetch historical candlestick data as pandas DataFrame
- `coin`: Coin symbol (e.g., "BTC", "ETH")
- `interval`: Timeframe - "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"
- `start_time`: Start time (datetime or epoch ms)
- `end_time`: End time (datetime or epoch ms)
- `lookback_periods`: Number of periods to look back
- Returns: DataFrame with columns [open, high, low, close, volume]

**get_all_mids()**
- Get current mid prices for all trading pairs
- Returns: Dictionary {coin: price}

**get_meta()**
- Get market metadata
- Returns: Market information dictionary

**get_available_coins()**
- Get list of available perpetual trading pairs
- Returns: List of coin symbols

**get_multiple_candles_df(coins, interval, ...)**
- Fetch data for multiple coins
- Returns: Dictionary {coin: DataFrame}

### PriceAnalyzer

#### Initialization
```python
analyzer = PriceAnalyzer(df)
```
- `df`: DataFrame with OHLCV data

#### Technical Indicators

**add_sma(period, column='close', name=None)**
- Add Simple Moving Average

**add_ema(period, column='close', name=None)**
- Add Exponential Moving Average

**add_rsi(period=14, column='close')**
- Add Relative Strength Index

**add_macd(fast_period=12, slow_period=26, signal_period=9)**
- Add MACD indicator

**add_bollinger_bands(period=20, num_std=2.0)**
- Add Bollinger Bands

**add_atr(period=14)**
- Add Average True Range

**add_stochastic(k_period=14, d_period=3, smooth_k=3)**
- Add Stochastic Oscillator

**add_volume_indicators()**
- Add volume-based indicators (OBV, volume ratio)

**add_returns()**
- Add return calculations

#### Analysis Methods

**detect_support_resistance(window=20, threshold=0.02)**
- Detect support and resistance levels
- Returns: (support_levels, resistance_levels)

**calculate_volatility(window=20)**
- Calculate rolling volatility
- Returns: Volatility series

**get_statistics()**
- Get statistical summary
- Returns: Dictionary with metrics

**backtest_simple_strategy(buy_condition, sell_condition, initial_capital=10000)**
- Simple strategy backtesting
- Returns: Dictionary with backtest results

## Supported Timeframes

- **Minutes**: 1m, 3m, 5m, 15m, 30m
- **Hours**: 1h, 2h, 4h, 8h, 12h
- **Days**: 1d, 3d
- **Weekly**: 1w
- **Monthly**: 1M

## Important Limitations

1. **Historical Data**: Only the most recent 5000 candles are available via the API
2. **Rate Limiting**: Use delay parameter when fetching multiple coins
3. **Data Availability**: Some coins may have limited historical data

## Examples

See `example.py` for comprehensive examples including:
1. Basic data fetching
2. Technical indicator usage
3. Multi-timeframe analysis
4. Multi-coin comparison
5. Support/resistance detection
6. Strategy backtesting
7. Volatility analysis

Run examples:
```bash
python example.py
```

### Offline Demo

If you want to test the tool without API access or explore its features with simulated data:

```bash
python demo_offline.py
```

This demonstrates all features using generated sample data, perfect for:
- Learning how the tool works
- Testing your strategies with simulated data
- Developing offline when API is not accessible
- Understanding the analysis capabilities

## Use Cases for Algorithmic Trading

### Trend Following
```python
analyzer.add_ema(12)
analyzer.add_ema(26)
# Use EMA crossovers for trend signals
```

### Mean Reversion
```python
analyzer.add_bollinger_bands(20, 2.0)
# Trade when price touches bands
```

### Momentum Trading
```python
analyzer.add_rsi(14)
# Use RSI overbought/oversold signals
```

### Volatility Trading
```python
analyzer.add_atr(14)
analyzer.calculate_volatility(window=20)
# Adjust position sizing based on volatility
```

## Advanced Usage

### Custom Time Ranges
```python
from datetime import datetime, timedelta

end_time = datetime.now()
start_time = end_time - timedelta(days=7)

df = client.get_candles_df("BTC", "15m", start_time=start_time, end_time=end_time)
```

### Combining Multiple Indicators
```python
analyzer.add_rsi(14)
analyzer.add_macd()
analyzer.add_bollinger_bands()

df = analyzer.get_dataframe()

# Complex strategy logic
buy_signal = (df['rsi_14'] < 30) & \
             (df['macd'] > df['macd_signal']) & \
             (df['close'] < df['bb_lower'])
```

### Portfolio Analysis
```python
coins = ["BTC", "ETH", "SOL", "ARB", "OP"]
data = client.get_multiple_candles_df(coins, "1d", lookback_periods=90)

for coin, df in data.items():
    analyzer = PriceAnalyzer(df)
    analyzer.add_returns()
    stats = analyzer.get_statistics()
    print(f"{coin}: {stats['total_return']:.2f}% return, {stats['volatility']*100:.2f}% vol")
```

## Troubleshooting

### API Access Issues

If you encounter 403 errors or connection issues:
- Ensure you have internet connectivity
- The Hyperliquid API may have rate limits or regional restrictions
- Try using the `demo_offline.py` script to test functionality with simulated data
- Check if you need to use a VPN or different network

### Common Issues

**Import errors**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

**Empty DataFrames**: Some coins may not have data for all timeframes. Always check if the DataFrame is empty before analysis:
```python
if not df.empty:
    # Proceed with analysis
```

**Rate Limiting**: When fetching multiple coins, use the `delay` parameter:
```python
data = client.get_multiple_candles_df(coins, interval, delay=0.2)
```

## Contributing

Feel free to extend this toolkit with additional indicators, strategies, or features!

## License

This tool is provided as-is for educational and research purposes.

## Disclaimer

This tool is for educational purposes only. Always do your own research and risk management when trading. Past performance does not guarantee future results.
