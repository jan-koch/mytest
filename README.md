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

### Price Forecasting (`hyperliquid_forecaster.py`) ⭐ NEW!
- **Foundation Model**: Amazon Chronos transformer-based forecasting
- **Multiple Model Sizes**: Tiny to Large (8M - 710M parameters)
- **Probabilistic Forecasts**: Median, mean, and confidence intervals
- **Backtesting**: Evaluate forecast accuracy on historical data
- **Rolling Forecasts**: Time-series cross-validation
- **Multi-horizon**: Forecast from hours to weeks ahead
- **Zero-shot Capability**: Works across different crypto assets

## Installation

### Basic Installation (Analysis Only)

Install core dependencies for data fetching and technical analysis:

```bash
pip install -r requirements.txt
```

### Full Installation (Including Forecasting)

To enable price forecasting with Chronos:

```bash
pip install -r requirements-forecasting.txt
```

Or install manually:

```bash
pip install requests pandas numpy
pip install torch
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

**Note**: Chronos models require PyTorch. First-time use will download models from Hugging Face (~50MB to 3GB depending on model size).

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

### Price Forecasting

```python
from hyperliquid_forecaster import PriceForecaster

# Initialize forecaster with Chronos model
forecaster = PriceForecaster(model_size='base')  # Options: tiny, mini, small, base, large

# Get historical data
df = client.get_candles_df("BTC", "1h", lookback_periods=500)

# Generate 24-hour forecast
result = forecaster.forecast_prices(
    df=df,
    prediction_length=24,
    num_samples=20
)

# View results
print(f"Current Price: ${df['close'].iloc[-1]:,.2f}")
print(f"24h Forecast: ${result['median'][-1]:,.2f}")
print(f"90% Confidence Interval: ${result['quantiles']['q10'][-1]:,.2f} - ${result['quantiles']['q90'][-1]:,.2f}")

# Get forecast as DataFrame
forecast_df = forecaster.forecast_df(df, prediction_length=24)
print(forecast_df[['mean', 'median', 'q10', 'q90']])

# Backtest the forecaster
backtest_results = forecaster.backtest(
    df=df,
    prediction_length=24,
    test_size=48
)
print(f"MAPE: {backtest_results['mape']:.2f}%")
print(f"Direction Accuracy: {backtest_results['direction_accuracy']:.2f}%")
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

### PriceForecaster

#### Initialization
```python
forecaster = PriceForecaster(model_size='base', device='auto', torch_dtype=None)
```
- `model_size`: Model size - 'tiny' (8M), 'mini' (20M), 'small' (46M), 'base' (200M), 'large' (710M)
- `device`: Device to use - 'auto', 'cpu', 'cuda', 'mps'
- `torch_dtype`: Optional dtype - 'float32', 'float16', 'bfloat16'

#### Forecasting Methods

**forecast_prices(df, prediction_length, num_samples=20, column='close', return_quantiles=True)**
- Generate price forecasts
- `df`: DataFrame with historical data
- `prediction_length`: Number of periods to forecast
- `num_samples`: Number of sample trajectories
- `column`: Column to forecast
- `return_quantiles`: Whether to return quantile predictions
- Returns: Dictionary with forecast results (median, mean, quantiles, samples)

**forecast_df(df, prediction_length, num_samples=20, column='close', include_history=False)**
- Get forecast as pandas DataFrame
- Returns: DataFrame with forecast including mean, median, and quantiles

**backtest(df, prediction_length, test_size, column='close', num_samples=20)**
- Backtest forecasting model
- `test_size`: Size of test set (last N periods)
- Returns: Dictionary with MAE, RMSE, MAPE, direction accuracy

**rolling_forecast(df, prediction_length, window_size, step_size=1, column='close')**
- Perform rolling window forecasting
- Returns: List of forecast results for each window

#### Model Sizes

| Size  | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| tiny  | ~8M       | Fastest | Good | Quick testing, prototyping |
| mini  | ~20M      | Very Fast | Better | Development, fast iteration |
| small | ~46M      | Fast | Good | Production (fast) |
| base  | ~200M     | Medium | Excellent | **Recommended for production** |
| large | ~710M     | Slow | Best | Maximum accuracy needed |

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

### Technical Analysis Examples

See `example.py` for comprehensive technical analysis examples:
1. Basic data fetching
2. Technical indicator usage
3. Multi-timeframe analysis
4. Multi-coin comparison
5. Support/resistance detection
6. Strategy backtesting
7. Volatility analysis

```bash
python example.py
```

### Price Forecasting Examples

See `example_forecasting.py` for forecasting examples:
1. Basic price forecasting
2. Forecast DataFrames
3. Backtesting forecasts
4. Multi-coin forecasting
5. Model size comparison
6. Combined technical analysis + forecasting
7. Rolling window forecasts

```bash
python example_forecasting.py
```

**Note**: Requires Chronos installation (`pip install -r requirements-forecasting.txt`)

### Offline Demos

If you want to test the tool without API access or explore features with simulated data:

**Technical Analysis Demo:**
```bash
python demo_offline.py
```

**Forecasting Demo:**
```bash
python demo_forecasting_offline.py
```

These demos use generated sample data, perfect for:
- Learning how the tools work
- Testing strategies with simulated data
- Developing offline when API is not accessible
- Understanding capabilities without installing heavy dependencies

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

### Price Forecasting
```python
# Predict future prices for position planning
forecaster = PriceForecaster(model_size='base')
forecast = forecaster.forecast_prices(df, prediction_length=24)

# Use forecast confidence intervals for risk management
lower_bound = forecast['quantiles']['q10'][-1]
upper_bound = forecast['quantiles']['q90'][-1]
expected_price = forecast['median'][-1]

# Make trading decisions based on forecast
if expected_price > current_price * 1.02:  # Expecting 2%+ increase
    # Consider long position with stop-loss at lower_bound
    pass
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
