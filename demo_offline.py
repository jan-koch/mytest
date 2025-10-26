"""
Offline demo showing how the Hyperliquid Price Analysis Tool works
using simulated data (when API is not accessible)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hyperliquid_analyzer import PriceAnalyzer

def generate_sample_data(periods=500, start_price=45000, volatility=0.02):
    """Generate sample OHLCV data for demonstration."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1h')

    # Generate random walk price data
    returns = np.random.normal(0, volatility, periods)
    prices = start_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        open_price = close * (1 + np.random.uniform(-0.005, 0.005))
        high = max(open_price, close) * (1 + abs(np.random.uniform(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.uniform(0, 0.01)))
        volume = np.random.uniform(1000, 10000)

        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def demo_technical_indicators():
    """Demonstrate technical indicators."""
    print("=" * 70)
    print("DEMO 1: Technical Indicators")
    print("=" * 70)

    # Generate sample data
    print("\nGenerating sample BTC price data (500 hourly candles)...")
    df = generate_sample_data(periods=500, start_price=45000)

    print(f"✓ Generated {len(df)} candles")
    print(f"\nFirst few rows:")
    print(df.head())

    # Initialize analyzer
    print("\n\nInitializing analyzer and adding indicators...")
    analyzer = PriceAnalyzer(df)

    # Add various indicators
    analyzer.add_sma(20)
    analyzer.add_sma(50)
    analyzer.add_ema(12)
    analyzer.add_ema(26)
    analyzer.add_rsi(14)
    analyzer.add_macd()
    analyzer.add_bollinger_bands(20, 2.0)
    analyzer.add_atr(14)
    analyzer.add_stochastic()
    analyzer.add_volume_indicators()
    analyzer.add_returns()

    df_analyzed = analyzer.get_dataframe()

    print(f"\n✓ Added technical indicators")
    print(f"\nAll columns: {df_analyzed.columns.tolist()}")

    print(f"\n\nLatest values:")
    latest = df_analyzed.iloc[-1]
    print(f"  Close Price: ${latest['close']:.2f}")
    print(f"  SMA(20): ${latest['sma_20']:.2f}")
    print(f"  SMA(50): ${latest['sma_50']:.2f}")
    print(f"  RSI(14): {latest['rsi_14']:.2f}")
    print(f"  MACD: {latest['macd']:.4f}")
    print(f"  MACD Signal: {latest['macd_signal']:.4f}")
    print(f"  BB Upper: ${latest['bb_upper']:.2f}")
    print(f"  BB Lower: ${latest['bb_lower']:.2f}")
    print(f"  ATR(14): ${latest['atr_14']:.2f}")
    print(f"  Stochastic %K: {latest['stoch_k']:.2f}")

def demo_statistics():
    """Demonstrate statistical analysis."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Statistical Analysis")
    print("=" * 70)

    df = generate_sample_data(periods=720, start_price=2500)  # 30 days of hourly data
    analyzer = PriceAnalyzer(df)
    analyzer.add_returns()

    stats = analyzer.get_statistics()

    print("\nStatistical Summary (30 days of hourly data):")
    print(f"  Period Start: {stats['period_start']}")
    print(f"  Period End: {stats['period_end']}")
    print(f"  Total Periods: {stats['total_periods']}")
    print(f"  Start Price: ${stats['start_price']:.2f}")
    print(f"  End Price: ${stats['end_price']:.2f}")
    print(f"  Min Price: ${stats['min_price']:.2f}")
    print(f"  Max Price: ${stats['max_price']:.2f}")
    print(f"  Mean Price: ${stats['mean_price']:.2f}")
    print(f"  Median Price: ${stats['median_price']:.2f}")
    print(f"  Total Return: {stats['total_return']:.2f}%")
    print(f"  Volatility: {stats['volatility']*100:.2f}%")
    print(f"  Average Volume: {stats['avg_volume']:.2f}")
    print(f"  Total Volume: {stats['total_volume']:.2f}")

def demo_support_resistance():
    """Demonstrate support/resistance detection."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Support and Resistance Detection")
    print("=" * 70)

    df = generate_sample_data(periods=500, start_price=45000)
    analyzer = PriceAnalyzer(df)

    supports, resistances = analyzer.detect_support_resistance(window=20, threshold=0.02)

    current_price = df['close'].iloc[-1]

    print(f"\nCurrent Price: ${current_price:.2f}")
    print(f"\nSupport Levels (nearest first):")
    relevant_supports = [s for s in supports if s < current_price][-5:]
    for i, level in enumerate(reversed(relevant_supports), 1):
        distance = ((current_price - level) / level) * 100
        print(f"  {i}. ${level:.2f} ({distance:.2f}% below)")

    print(f"\nResistance Levels (nearest first):")
    relevant_resistances = [r for r in resistances if r > current_price][:5]
    for i, level in enumerate(relevant_resistances, 1):
        distance = ((level - current_price) / current_price) * 100
        print(f"  {i}. ${level:.2f} ({distance:.2f}% above)")

def demo_strategy_backtest():
    """Demonstrate strategy backtesting."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Simple Moving Average Crossover Strategy Backtest")
    print("=" * 70)

    # Generate data with a trend
    print("\nGenerating sample price data...")
    df = generate_sample_data(periods=1000, start_price=2500, volatility=0.015)

    analyzer = PriceAnalyzer(df)
    analyzer.add_sma(20)
    analyzer.add_sma(50)

    df_strategy = analyzer.get_dataframe()

    print("\nStrategy: Buy when SMA(20) crosses above SMA(50), sell when it crosses below")

    # Generate signals
    buy_signal = (df_strategy['sma_20'] > df_strategy['sma_50']) & \
                 (df_strategy['sma_20'].shift(1) <= df_strategy['sma_50'].shift(1))

    sell_signal = (df_strategy['sma_20'] < df_strategy['sma_50']) & \
                  (df_strategy['sma_20'].shift(1) >= df_strategy['sma_50'].shift(1))

    # Run backtest
    initial_capital = 10000
    results = analyzer.backtest_simple_strategy(
        buy_condition=buy_signal,
        sell_condition=sell_signal,
        initial_capital=initial_capital
    )

    print(f"\nBacktest Results:")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")
    print(f"  Total Return: {results['total_return']:.2f}%")
    print(f"  Number of Trades: {results['num_trades']}")

    if results['trades']:
        print(f"\n  Sample trades:")
        for i, trade in enumerate(results['trades'][:6], 1):
            if trade['type'] == 'buy':
                print(f"    {i}. BUY  @ ${trade['price']:.2f} on {trade['timestamp']}")
            else:
                print(f"    {i}. SELL @ ${trade['price']:.2f} on {trade['timestamp']} "
                      f"(Return: {trade['return']:.2f}%)")

def demo_volatility_analysis():
    """Demonstrate volatility analysis."""
    print("\n\n" + "=" * 70)
    print("DEMO 5: Volatility Analysis")
    print("=" * 70)

    df = generate_sample_data(periods=720, start_price=45000, volatility=0.025)

    analyzer = PriceAnalyzer(df)
    analyzer.add_atr(14)
    analyzer.add_bollinger_bands(20, 2.0)

    volatility = analyzer.calculate_volatility(window=20)
    df_vol = analyzer.get_dataframe()

    print("\nLatest Volatility Metrics:")
    latest = df_vol.iloc[-1]
    print(f"  ATR(14): ${latest['atr_14']:.2f}")
    print(f"  Bollinger Band Width: ${latest['bb_width']:.2f}")
    print(f"  20-period Volatility: {volatility.iloc[-1]*100:.2f}%")

    # High volatility periods
    high_vol_threshold = volatility.quantile(0.9)
    high_vol_periods = volatility[volatility > high_vol_threshold]

    print(f"\nHigh Volatility Analysis:")
    print(f"  Threshold (90th percentile): {high_vol_threshold*100:.2f}%")
    print(f"  High volatility periods: {len(high_vol_periods)}")
    print(f"  Percentage of time: {(len(high_vol_periods)/len(volatility))*100:.1f}%")

    # Volatility distribution
    print(f"\nVolatility Distribution:")
    print(f"  Min: {volatility.min()*100:.2f}%")
    print(f"  25th percentile: {volatility.quantile(0.25)*100:.2f}%")
    print(f"  Median: {volatility.median()*100:.2f}%")
    print(f"  75th percentile: {volatility.quantile(0.75)*100:.2f}%")
    print(f"  Max: {volatility.max()*100:.2f}%")

def demo_multi_timeframe():
    """Demonstrate multi-timeframe analysis concept."""
    print("\n\n" + "=" * 70)
    print("DEMO 6: Multi-Timeframe Analysis Concept")
    print("=" * 70)

    print("\nThis demonstrates how you would analyze the same asset across timeframes:")

    timeframes = {
        '15m': (672, 15),  # 1 week of 15-min candles
        '1h': (168, 60),   # 1 week of hourly candles
        '4h': (42, 240),   # 1 week of 4-hour candles
        '1d': (30, 1440)   # 30 days
    }

    for tf_name, (periods, freq_min) in timeframes.items():
        # Simulate different timeframe data
        df = generate_sample_data(periods=periods, start_price=45000)

        analyzer = PriceAnalyzer(df)
        analyzer.add_rsi(14)
        analyzer.add_sma(20)
        analyzer.add_returns()

        df_analyzed = analyzer.get_dataframe()
        latest = df_analyzed.iloc[-1]

        print(f"\n  {tf_name} Timeframe ({periods} periods):")
        print(f"    Close: ${latest['close']:.2f}")
        print(f"    SMA(20): ${latest['sma_20']:.2f}")
        print(f"    RSI(14): {latest['rsi_14']:.2f}")
        print(f"    Latest Return: {latest['returns']*100:.2f}%")

        # Determine trend
        trend = "Bullish" if latest['close'] > latest['sma_20'] else "Bearish"
        rsi_state = "Overbought" if latest['rsi_14'] > 70 else ("Oversold" if latest['rsi_14'] < 30 else "Neutral")
        print(f"    Trend: {trend}, RSI: {rsi_state}")

def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("HYPERLIQUID PRICE ANALYSIS TOOL - OFFLINE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo uses simulated price data to showcase all features.")
    print("When connected to the Hyperliquid API, the tool fetches real market data.\n")

    try:
        demo_technical_indicators()
        demo_statistics()
        demo_support_resistance()
        demo_strategy_backtest()
        demo_volatility_analysis()
        demo_multi_timeframe()

        print("\n\n" + "=" * 70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nTo use with real Hyperliquid data, run:")
        print("  python example.py")
        print("\nOr import in your own scripts:")
        print("  from hyperliquid_client import HyperliquidClient")
        print("  from hyperliquid_analyzer import PriceAnalyzer")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
