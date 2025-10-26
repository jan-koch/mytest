"""
Examples: Price Forecasting with Hyperliquid and Chronos

This script demonstrates how to use the Chronos forecasting model
with Hyperliquid price data for cryptocurrency price predictions.
"""

import pandas as pd
import numpy as np
from hyperliquid_client import HyperliquidClient
from hyperliquid_forecaster import PriceForecaster, check_chronos_installation, get_installation_instructions


def check_setup():
    """Check if Chronos is installed."""
    print("Checking setup...")
    if not check_chronos_installation():
        print("\n" + "=" * 70)
        print("⚠️  Chronos is not installed!")
        print("=" * 70)
        print(get_installation_instructions())
        return False
    print("✓ Chronos is installed and ready\n")
    return True


def example_basic_forecasting():
    """Example 1: Basic price forecasting."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Price Forecasting")
    print("=" * 70)

    # Fetch historical data
    print("\n1. Fetching BTC historical data...")
    client = HyperliquidClient()
    df = client.get_candles_df("BTC", "1h", lookback_periods=500)
    print(f"   ✓ Fetched {len(df)} hourly candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")

    # Initialize forecaster
    print("\n2. Initializing Chronos forecaster (base model)...")
    forecaster = PriceForecaster(model_size='base', device='cpu')

    # Make forecast
    print("\n3. Generating 24-hour forecast...")
    result = forecaster.forecast_prices(
        df=df,
        prediction_length=24,
        num_samples=20
    )

    print(f"\n4. Forecast Results:")
    print(f"   Forecast horizon: 24 hours")
    print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
    print(f"   Predicted price (24h): ${result['median'][-1]:,.2f}")
    print(f"   Change: {((result['median'][-1] / df['close'].iloc[-1] - 1) * 100):+.2f}%")

    print(f"\n   Price range predictions:")
    if 'quantiles' in result:
        print(f"   - 10th percentile (24h): ${result['quantiles']['q10'][-1]:,.2f}")
        print(f"   - 50th percentile (24h): ${result['quantiles']['q50'][-1]:,.2f}")
        print(f"   - 90th percentile (24h): ${result['quantiles']['q90'][-1]:,.2f}")

    # Show some intermediate predictions
    print(f"\n   Intermediate predictions:")
    checkpoints = [6, 12, 18, 24]
    for hours in checkpoints:
        idx = hours - 1
        print(f"   {hours}h: ${result['median'][idx]:,.2f} "
              f"({((result['median'][idx] / df['close'].iloc[-1] - 1) * 100):+.2f}%)")


def example_forecast_dataframe():
    """Example 2: Get forecast as DataFrame."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Forecast DataFrame")
    print("=" * 70)

    client = HyperliquidClient()
    print("\n1. Fetching ETH 4-hour data...")
    df = client.get_candles_df("ETH", "4h", lookback_periods=200)
    print(f"   ✓ Fetched {len(df)} candles")

    print("\n2. Generating forecast...")
    forecaster = PriceForecaster(model_size='small', device='cpu')

    forecast_df = forecaster.forecast_df(
        df=df,
        prediction_length=12,  # 12 * 4h = 48 hours
        num_samples=20,
        column='close',
        include_history=False
    )

    print("\n3. Forecast DataFrame:")
    print(forecast_df.head(12))

    print(f"\n4. Summary:")
    print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
    print(f"   48h forecast: ${forecast_df['median'].iloc[-1]:,.2f}")
    print(f"   Prediction interval (90%):")
    print(f"     Lower: ${forecast_df['q10'].iloc[-1]:,.2f}")
    print(f"     Upper: ${forecast_df['q90'].iloc[-1]:,.2f}")


def example_backtesting():
    """Example 3: Backtest forecasting model."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Backtesting")
    print("=" * 70)

    client = HyperliquidClient()
    print("\n1. Fetching historical data for backtesting...")
    df = client.get_candles_df("BTC", "1h", lookback_periods=1000)
    print(f"   ✓ Fetched {len(df)} hourly candles")

    print("\n2. Running backtest...")
    print("   - Training on first 952 hours")
    print("   - Testing on last 48 hours")
    print("   - Forecasting 24 hours ahead")

    forecaster = PriceForecaster(model_size='base', device='cpu')

    results = forecaster.backtest(
        df=df,
        prediction_length=24,
        test_size=48,
        column='close',
        num_samples=20
    )

    print("\n3. Backtest Results:")
    print(f"   MAE (Mean Absolute Error): ${results['mae']:,.2f}")
    print(f"   RMSE (Root Mean Squared Error): ${results['rmse']:,.2f}")
    print(f"   MAPE (Mean Absolute Percentage Error): {results['mape']:.2f}%")
    print(f"   Direction Accuracy: {results['direction_accuracy']:.2f}%")

    print(f"\n4. Sample predictions vs actuals:")
    for i in range(0, len(results['actual']), 6):
        if i < len(results['actual']):
            actual = results['actual'][i]
            predicted = results['predicted'][i]
            error = ((predicted - actual) / actual) * 100
            print(f"   Hour {i+1:2d}: Actual=${actual:,.2f}, "
                  f"Predicted=${predicted:,.2f}, Error={error:+.2f}%")


def example_multi_coin_forecast():
    """Example 4: Forecast multiple coins."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Multi-Coin Forecasting")
    print("=" * 70)

    coins = ["BTC", "ETH", "SOL"]
    client = HyperliquidClient()

    print("\n1. Fetching data for multiple coins...")
    data = client.get_multiple_candles_df(
        coins=coins,
        interval="1d",
        lookback_periods=90
    )

    print("\n2. Generating forecasts...")
    forecaster = PriceForecaster(model_size='small', device='cpu')

    results = {}
    for coin, df in data.items():
        if not df.empty:
            print(f"\n   Forecasting {coin}...")
            result = forecaster.forecast_prices(
                df=df,
                prediction_length=7,  # 7 days
                num_samples=20
            )
            results[coin] = result

    print("\n3. 7-Day Price Predictions:")
    for coin in coins:
        if coin in results:
            result = results[coin]
            current_price = data[coin]['close'].iloc[-1]
            forecast_price = result['median'][-1]
            change_pct = ((forecast_price / current_price - 1) * 100)

            print(f"\n   {coin}:")
            print(f"     Current: ${current_price:,.2f}")
            print(f"     7-day forecast: ${forecast_price:,.2f} ({change_pct:+.2f}%)")
            if 'quantiles' in result:
                print(f"     90% confidence interval: "
                      f"${result['quantiles']['q10'][-1]:,.2f} - "
                      f"${result['quantiles']['q90'][-1]:,.2f}")


def example_different_models():
    """Example 5: Compare different model sizes."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 5: Comparing Model Sizes")
    print("=" * 70)

    client = HyperliquidClient()
    print("\n1. Fetching BTC hourly data...")
    df = client.get_candles_df("BTC", "1h", lookback_periods=300)

    model_sizes = ['tiny', 'small', 'base']
    prediction_length = 12

    print(f"\n2. Comparing model sizes for {prediction_length}-hour forecast...")
    print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")

    for size in model_sizes:
        print(f"\n   {size.upper()} model:")
        forecaster = PriceForecaster(model_size=size, device='cpu')

        result = forecaster.forecast_prices(
            df=df,
            prediction_length=prediction_length,
            num_samples=10  # Fewer samples for speed
        )

        forecast_price = result['median'][-1]
        change_pct = ((forecast_price / df['close'].iloc[-1] - 1) * 100)

        print(f"     12h forecast: ${forecast_price:,.2f} ({change_pct:+.2f}%)")
        print(f"     Std deviation: ${result['std'][-1]:,.2f}")


def example_with_technical_analysis():
    """Example 6: Combine forecasting with technical analysis."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 6: Forecasting + Technical Analysis")
    print("=" * 70)

    from hyperliquid_analyzer import PriceAnalyzer

    client = HyperliquidClient()
    print("\n1. Fetching data and adding indicators...")
    df = client.get_candles_df("BTC", "1h", lookback_periods=500)

    # Add technical indicators
    analyzer = PriceAnalyzer(df)
    analyzer.add_sma(20)
    analyzer.add_rsi(14)
    analyzer.add_macd()

    df_with_indicators = analyzer.get_dataframe()

    print(f"   ✓ Data prepared with technical indicators")
    print(f"\n   Current technical state:")
    latest = df_with_indicators.iloc[-1]
    print(f"     Price: ${latest['close']:,.2f}")
    print(f"     SMA(20): ${latest['sma_20']:,.2f}")
    print(f"     RSI(14): {latest['rsi_14']:.2f}")
    print(f"     MACD: {latest['macd']:.4f}")

    # Determine trend
    trend = "Bullish" if latest['close'] > latest['sma_20'] else "Bearish"
    rsi_state = "Overbought" if latest['rsi_14'] > 70 else (
        "Oversold" if latest['rsi_14'] < 30 else "Neutral"
    )

    print(f"     Trend: {trend}")
    print(f"     RSI State: {rsi_state}")

    # Make forecast
    print("\n2. Generating price forecast...")
    forecaster = PriceForecaster(model_size='base', device='cpu')

    result = forecaster.forecast_prices(
        df=df,  # Use original df without indicators for forecasting
        prediction_length=24,
        num_samples=20
    )

    print(f"\n3. Combined Analysis:")
    forecast_price = result['median'][-1]
    price_change_pct = ((forecast_price / latest['close'] - 1) * 100)

    print(f"   24h price forecast: ${forecast_price:,.2f} ({price_change_pct:+.2f}%)")

    # Investment recommendation based on combined signals
    signals = []
    if trend == "Bullish":
        signals.append("Bullish trend")
    if rsi_state == "Oversold":
        signals.append("RSI oversold")
    elif rsi_state == "Overbought":
        signals.append("RSI overbought")
    if price_change_pct > 2:
        signals.append("Forecast indicates price increase")
    elif price_change_pct < -2:
        signals.append("Forecast indicates price decrease")

    print(f"\n   Signals: {', '.join(signals) if signals else 'Neutral'}")


def example_rolling_forecast():
    """Example 7: Rolling window forecasting."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 7: Rolling Window Forecasting")
    print("=" * 70)

    client = HyperliquidClient()
    print("\n1. Fetching extended historical data...")
    df = client.get_candles_df("BTC", "1d", lookback_periods=200)
    print(f"   ✓ Fetched {len(df)} daily candles")

    print("\n2. Performing rolling forecast...")
    print("   - Window size: 100 days")
    print("   - Prediction length: 7 days")
    print("   - Step size: 7 days")

    forecaster = PriceForecaster(model_size='small', device='cpu')

    forecasts = forecaster.rolling_forecast(
        df=df,
        prediction_length=7,
        window_size=100,
        step_size=7,
        num_samples=10
    )

    print(f"\n3. Results from {len(forecasts)} rolling forecasts:")

    # Calculate aggregate metrics
    all_errors = []
    for i, fc in enumerate(forecasts):
        if len(fc['actual']) == 7:
            metrics = PriceForecaster.calculate_metrics(fc['actual'], fc['median'])
            all_errors.append(metrics['mape'])

            if i < 3:  # Show first 3 in detail
                print(f"\n   Forecast {i+1}:")
                print(f"     Train period: {fc['train_start'].date()} to {fc['train_end'].date()}")
                print(f"     MAPE: {metrics['mape']:.2f}%")
                print(f"     Direction accuracy: {metrics['direction_accuracy']:.2f}%")

    if all_errors:
        print(f"\n4. Aggregate Performance:")
        print(f"   Average MAPE: {np.mean(all_errors):.2f}%")
        print(f"   Median MAPE: {np.median(all_errors):.2f}%")
        print(f"   Best MAPE: {np.min(all_errors):.2f}%")
        print(f"   Worst MAPE: {np.max(all_errors):.2f}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("HYPERLIQUID PRICE FORECASTING WITH CHRONOS")
    print("=" * 70)
    print("\nThese examples demonstrate price forecasting using Amazon's Chronos")
    print("foundation model integrated with Hyperliquid historical data.\n")

    # Check if Chronos is installed
    if not check_setup():
        return

    try:
        # Run examples
        example_basic_forecasting()
        example_forecast_dataframe()
        example_backtesting()
        example_multi_coin_forecast()
        example_different_models()
        example_with_technical_analysis()
        example_rolling_forecast()

        print("\n\n" + "=" * 70)
        print("ALL FORECASTING EXAMPLES COMPLETED!")
        print("=" * 70)
        print("\nNote: Forecasts are for educational purposes only.")
        print("Always conduct thorough research before making investment decisions.")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf you're getting API errors, try using the offline demo:")
        print("  python demo_forecasting_offline.py")


if __name__ == "__main__":
    main()
