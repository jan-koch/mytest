"""
Example usage of the Hyperliquid Price Analysis Tool

This script demonstrates various features of the Hyperliquid client and analyzer.
"""

from hyperliquid_client import HyperliquidClient
from hyperliquid_analyzer import PriceAnalyzer
from datetime import datetime, timedelta


def example_basic_usage():
    """Example: Basic data fetching and analysis."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)

    # Initialize client (mainnet)
    client = HyperliquidClient(testnet=False)

    # Get list of available coins
    print("\nFetching available coins...")
    coins = client.get_available_coins()
    print(f"Found {len(coins)} available coins")
    print(f"First 10 coins: {coins[:10]}")

    # Get current mid prices
    print("\nFetching current mid prices...")
    prices = client.get_all_mids()
    print(f"BTC: ${prices.get('BTC', 'N/A')}")
    print(f"ETH: ${prices.get('ETH', 'N/A')}")

    # Get historical data for BTC
    print("\nFetching BTC 1-hour candles (last 100 periods)...")
    df = client.get_candles_df("BTC", "1h", lookback_periods=100)
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())


def example_technical_indicators():
    """Example: Adding technical indicators."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Technical Indicators")
    print("=" * 60)

    client = HyperliquidClient()

    # Fetch data
    print("\nFetching ETH 4-hour candles...")
    df = client.get_candles_df("ETH", "4h", lookback_periods=200)

    # Initialize analyzer
    analyzer = PriceAnalyzer(df)

    # Add various indicators
    print("\nAdding technical indicators...")
    analyzer.add_sma(20)
    analyzer.add_sma(50)
    analyzer.add_ema(12)
    analyzer.add_ema(26)
    analyzer.add_rsi(14)
    analyzer.add_macd()
    analyzer.add_bollinger_bands()
    analyzer.add_atr()
    analyzer.add_returns()

    # Get the updated dataframe
    df_with_indicators = analyzer.get_dataframe()

    print("\nDataFrame columns:")
    print(df_with_indicators.columns.tolist())
    print("\nLatest data with indicators:")
    print(df_with_indicators.tail())

    # Get statistics
    print("\nStatistical Summary:")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_multiple_timeframes():
    """Example: Multi-timeframe analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multi-Timeframe Analysis")
    print("=" * 60)

    client = HyperliquidClient()

    coin = "BTC"
    timeframes = ["15m", "1h", "4h", "1d"]

    print(f"\nAnalyzing {coin} across multiple timeframes...")

    for tf in timeframes:
        print(f"\n{tf} timeframe:")
        df = client.get_candles_df(coin, tf, lookback_periods=50)

        if not df.empty:
            analyzer = PriceAnalyzer(df)
            analyzer.add_rsi(14)
            analyzer.add_returns()

            latest = analyzer.get_dataframe().iloc[-1]
            print(f"  Latest Close: ${latest['close']:.2f}")
            print(f"  RSI(14): {latest['rsi_14']:.2f}")
            print(f"  Latest Return: {latest['returns']*100:.2f}%")


def example_multiple_coins():
    """Example: Comparing multiple coins."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multiple Coin Comparison")
    print("=" * 60)

    client = HyperliquidClient()

    coins = ["BTC", "ETH", "SOL"]
    interval = "1d"
    lookback = 30

    print(f"\nFetching {interval} data for {len(coins)} coins...")
    data = client.get_multiple_candles_df(
        coins=coins,
        interval=interval,
        lookback_periods=lookback
    )

    print("\nPerformance comparison (last 30 days):")
    for coin, df in data.items():
        if not df.empty:
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            return_pct = ((end_price - start_price) / start_price) * 100
            print(f"  {coin}: {return_pct:+.2f}%")


def example_support_resistance():
    """Example: Support and resistance detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Support and Resistance Levels")
    print("=" * 60)

    client = HyperliquidClient()

    print("\nFetching BTC 1-hour data...")
    df = client.get_candles_df("BTC", "1h", lookback_periods=500)

    analyzer = PriceAnalyzer(df)
    supports, resistances = analyzer.detect_support_resistance(window=20)

    print(f"\nCurrent price: ${df['close'].iloc[-1]:.2f}")
    print(f"\nSupport levels: {[f'${s:.2f}' for s in supports[-5:]]}")
    print(f"Resistance levels: {[f'${r:.2f}' for r in resistances[-5:]]}")


def example_simple_strategy():
    """Example: Simple trading strategy backtest."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Simple Moving Average Crossover Strategy")
    print("=" * 60)

    client = HyperliquidClient()

    print("\nFetching data and setting up strategy...")
    df = client.get_candles_df("ETH", "1h", lookback_periods=1000)

    analyzer = PriceAnalyzer(df)
    analyzer.add_sma(20)
    analyzer.add_sma(50)

    df_strategy = analyzer.get_dataframe()

    # Generate signals: Buy when SMA20 crosses above SMA50, sell when it crosses below
    buy_signal = (df_strategy['sma_20'] > df_strategy['sma_50']) & \
                 (df_strategy['sma_20'].shift(1) <= df_strategy['sma_50'].shift(1))

    sell_signal = (df_strategy['sma_20'] < df_strategy['sma_50']) & \
                  (df_strategy['sma_20'].shift(1) >= df_strategy['sma_50'].shift(1))

    # Run backtest
    print("\nRunning backtest...")
    results = analyzer.backtest_simple_strategy(
        buy_condition=buy_signal,
        sell_condition=sell_signal,
        initial_capital=10000
    )

    print("\nBacktest Results:")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")
    print(f"  Total Return: {results['total_return']:.2f}%")
    print(f"  Number of Trades: {results['num_trades']}")

    if results['trades']:
        print(f"\nFirst 3 trades:")
        for trade in results['trades'][:3]:
            print(f"  {trade}")


def example_volatility_analysis():
    """Example: Volatility analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Volatility Analysis")
    print("=" * 60)

    client = HyperliquidClient()

    print("\nFetching BTC daily data...")
    df = client.get_candles_df("BTC", "1d", lookback_periods=90)

    analyzer = PriceAnalyzer(df)
    analyzer.add_atr(14)
    analyzer.add_bollinger_bands(20, 2.0)

    volatility = analyzer.calculate_volatility(window=20)

    df_vol = analyzer.get_dataframe()

    print("\nLatest volatility metrics:")
    latest = df_vol.iloc[-1]
    print(f"  ATR(14): ${latest['atr_14']:.2f}")
    print(f"  BB Width: ${latest['bb_width']:.2f}")
    print(f"  Volatility (20-day): {volatility.iloc[-1]*100:.2f}%")

    # Find high volatility periods
    high_vol_threshold = volatility.quantile(0.9)
    high_vol_periods = volatility[volatility > high_vol_threshold]

    print(f"\nHigh volatility periods (top 10%):")
    print(f"  Threshold: {high_vol_threshold*100:.2f}%")
    print(f"  Number of periods: {len(high_vol_periods)}")


def main():
    """Run all examples."""
    try:
        example_basic_usage()
        example_technical_indicators()
        example_multiple_timeframes()
        example_multiple_coins()
        example_support_resistance()
        example_simple_strategy()
        example_volatility_analysis()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have an internet connection and the API is accessible.")


if __name__ == "__main__":
    main()
