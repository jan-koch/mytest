"""
Offline Demo: Price Forecasting Simulation

This demo shows how the forecasting functionality works using simulated data
and a simple forecasting algorithm (since Chronos may not be installed).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict


def generate_sample_data(periods=500, start_price=45000, volatility=0.02, trend=0.0001):
    """Generate sample OHLCV data with optional trend."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1h')

    # Generate random walk with trend
    returns = np.random.normal(trend, volatility, periods)
    prices = start_price * np.exp(np.cumsum(returns))

    # Add some autocorrelation to make it more realistic
    for i in range(1, len(prices)):
        prices[i] = 0.7 * prices[i] + 0.3 * prices[i-1]

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


class SimpleForecaster:
    """Simple forecasting model for demonstration (not using Chronos)."""

    def __init__(self, method='exponential_smoothing'):
        self.method = method

    def forecast(self, data: np.ndarray, prediction_length: int, num_samples: int = 20) -> Dict:
        """
        Simple forecast using exponential smoothing with noise.

        This is a simplified version for demo purposes only.
        The real Chronos model is much more sophisticated.
        """
        # Calculate trend and level using exponential smoothing
        alpha = 0.3  # Smoothing factor
        beta = 0.1   # Trend factor

        level = data[0]
        trend = 0

        # Simple exponential smoothing
        for value in data:
            prev_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        # Generate samples with trend and noise
        samples = []
        for _ in range(num_samples):
            forecast_values = []
            current_level = level
            current_trend = trend

            for h in range(prediction_length):
                # Add some randomness
                noise = np.random.normal(0, np.std(data[-50:]) * 0.3)
                forecast_value = current_level + current_trend * (h + 1) + noise
                forecast_values.append(forecast_value)

            samples.append(forecast_values)

        samples = np.array(samples)

        # Generate future dates
        last_date = pd.Timestamp.now()
        forecast_dates = pd.date_range(start=last_date, periods=prediction_length + 1, freq='1h')[1:]

        # Calculate quantiles
        quantiles = {
            'q10': np.quantile(samples, 0.1, axis=0),
            'q25': np.quantile(samples, 0.25, axis=0),
            'q50': np.quantile(samples, 0.5, axis=0),
            'q75': np.quantile(samples, 0.75, axis=0),
            'q90': np.quantile(samples, 0.9, axis=0),
        }

        return {
            'forecast_dates': forecast_dates,
            'mean': samples.mean(axis=0),
            'median': np.median(samples, axis=0),
            'std': samples.std(axis=0),
            'samples': samples,
            'quantiles': quantiles,
        }


def demo_basic_forecasting():
    """Demo 1: Basic forecasting."""
    print("=" * 70)
    print("DEMO 1: Basic Price Forecasting")
    print("=" * 70)

    print("\n1. Generating sample BTC price data (500 hours)...")
    df = generate_sample_data(periods=500, start_price=45000)
    print(f"   ✓ Generated {len(df)} hourly candles")
    print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")

    print("\n2. Creating simple forecaster (demo model)...")
    forecaster = SimpleForecaster()
    print("   ✓ Forecaster initialized")

    print("\n3. Generating 24-hour forecast...")
    result = forecaster.forecast(
        data=df['close'].values,
        prediction_length=24,
        num_samples=20
    )

    print(f"\n4. Forecast Results:")
    print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
    print(f"   24h forecast (median): ${result['median'][-1]:,.2f}")
    change_pct = ((result['median'][-1] / df['close'].iloc[-1] - 1) * 100)
    print(f"   Expected change: {change_pct:+.2f}%")

    print(f"\n   Confidence intervals (24h):")
    print(f"   - 10th percentile: ${result['quantiles']['q10'][-1]:,.2f}")
    print(f"   - 50th percentile: ${result['quantiles']['q50'][-1]:,.2f}")
    print(f"   - 90th percentile: ${result['quantiles']['q90'][-1]:,.2f}")

    print(f"\n   Hourly predictions:")
    for hours in [6, 12, 18, 24]:
        idx = hours - 1
        price = result['median'][idx]
        change = ((price / df['close'].iloc[-1] - 1) * 100)
        print(f"   {hours:2d}h: ${price:,.2f} ({change:+.2f}%)")


def demo_backtesting():
    """Demo 2: Backtesting."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Backtesting")
    print("=" * 70)

    print("\n1. Generating historical data...")
    df = generate_sample_data(periods=1000, start_price=45000)
    print(f"   ✓ Generated {len(df)} hourly candles")

    print("\n2. Splitting data for backtesting...")
    train_size = 952
    test_size = 48
    prediction_length = 24

    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"   Training: {len(train_df)} periods")
    print(f"   Testing: {len(test_df)} periods")

    print("\n3. Generating forecast on training data...")
    forecaster = SimpleForecaster()
    result = forecaster.forecast(
        data=train_df['close'].values,
        prediction_length=prediction_length,
        num_samples=20
    )

    print("\n4. Comparing with actual test data...")
    actual = test_df['close'].iloc[:prediction_length].values
    predicted = result['median']

    # Calculate metrics
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Direction accuracy
    actual_dir = np.diff(actual) > 0
    pred_dir = np.diff(predicted) > 0
    dir_accuracy = np.mean(actual_dir == pred_dir) * 100

    print(f"\n5. Backtest Metrics:")
    print(f"   MAE: ${mae:,.2f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Direction Accuracy: {dir_accuracy:.2f}%")

    print(f"\n6. Sample predictions vs actuals:")
    for i in range(0, prediction_length, 6):
        print(f"   Hour {i+1:2d}: Actual=${actual[i]:,.2f}, "
              f"Predicted=${predicted[i]:,.2f}, "
              f"Error={(((predicted[i] - actual[i]) / actual[i]) * 100):+.2f}%")


def demo_multi_coin_forecast():
    """Demo 3: Multi-coin forecasting."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Multi-Coin Forecasting")
    print("=" * 70)

    coins_data = {
        'BTC': generate_sample_data(periods=90, start_price=45000, volatility=0.025),
        'ETH': generate_sample_data(periods=90, start_price=2500, volatility=0.03),
        'SOL': generate_sample_data(periods=90, start_price=100, volatility=0.04),
    }

    print("\n1. Generated data for multiple coins:")
    for coin, df in coins_data.items():
        print(f"   {coin}: ${df['close'].iloc[-1]:,.2f}")

    print("\n2. Generating 7-day forecasts...")
    forecaster = SimpleForecaster()

    results = {}
    for coin, df in coins_data.items():
        result = forecaster.forecast(
            data=df['close'].values,
            prediction_length=7,
            num_samples=20
        )
        results[coin] = result

    print("\n3. 7-Day Forecasts:")
    for coin in coins_data.keys():
        current = coins_data[coin]['close'].iloc[-1]
        forecast = results[coin]['median'][-1]
        change = ((forecast / current - 1) * 100)

        print(f"\n   {coin}:")
        print(f"     Current: ${current:,.2f}")
        print(f"     7-day forecast: ${forecast:,.2f} ({change:+.2f}%)")
        print(f"     90% CI: ${results[coin]['quantiles']['q10'][-1]:,.2f} - "
              f"${results[coin]['quantiles']['q90'][-1]:,.2f}")


def demo_with_technical_analysis():
    """Demo 4: Forecasting with technical analysis."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Forecasting + Technical Analysis")
    print("=" * 70)

    print("\n1. Generating price data with indicators...")
    df = generate_sample_data(periods=500, start_price=45000)

    # Calculate simple indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]

    print(f"\n2. Current Technical State:")
    print(f"   Price: ${latest['close']:,.2f}")
    print(f"   SMA(20): ${latest['sma_20']:,.2f}")
    print(f"   SMA(50): ${latest['sma_50']:,.2f}")
    print(f"   RSI(14): {latest['rsi']:.2f}")

    trend = "Bullish" if latest['close'] > latest['sma_20'] else "Bearish"
    rsi_state = "Overbought" if latest['rsi'] > 70 else (
        "Oversold" if latest['rsi'] < 30 else "Neutral"
    )

    print(f"   Trend: {trend}")
    print(f"   RSI State: {rsi_state}")

    print("\n3. Generating 24-hour forecast...")
    forecaster = SimpleForecaster()
    result = forecaster.forecast(
        data=df['close'].values,
        prediction_length=24,
        num_samples=20
    )

    forecast_price = result['median'][-1]
    price_change = ((forecast_price / latest['close'] - 1) * 100)

    print(f"\n4. Combined Analysis:")
    print(f"   24h forecast: ${forecast_price:,.2f} ({price_change:+.2f}%)")

    # Generate signals
    signals = []
    if trend == "Bullish":
        signals.append("✓ Bullish trend")
    else:
        signals.append("✗ Bearish trend")

    if rsi_state == "Oversold":
        signals.append("✓ RSI oversold (potential buy)")
    elif rsi_state == "Overbought":
        signals.append("⚠ RSI overbought (caution)")

    if price_change > 2:
        signals.append("✓ Forecast suggests price increase")
    elif price_change < -2:
        signals.append("✗ Forecast suggests price decrease")

    print(f"\n   Technical Signals:")
    for signal in signals:
        print(f"     {signal}")


def demo_forecast_visualization_data():
    """Demo 5: Generate data for visualization."""
    print("\n\n" + "=" * 70)
    print("DEMO 5: Forecast Visualization Data")
    print("=" * 70)

    print("\n1. Generating historical and forecast data...")
    df = generate_sample_data(periods=168, start_price=45000)  # 1 week

    forecaster = SimpleForecaster()
    result = forecaster.forecast(
        data=df['close'].values,
        prediction_length=24,
        num_samples=20
    )

    print("\n2. Historical Data (last 24 hours):")
    print(df[['close']].tail(24))

    print("\n3. Forecast Data (next 24 hours):")
    forecast_df = pd.DataFrame({
        'forecast_median': result['median'],
        'forecast_q10': result['quantiles']['q10'],
        'forecast_q90': result['quantiles']['q90'],
    }, index=result['forecast_dates'])

    print(forecast_df)

    print("\n4. This data can be used with plotting libraries like:")
    print("   - matplotlib: For line charts with confidence intervals")
    print("   - plotly: For interactive candlestick + forecast charts")
    print("   - seaborn: For statistical visualizations")


def demo_model_performance():
    """Demo 6: Evaluate model performance."""
    print("\n\n" + "=" * 70)
    print("DEMO 6: Model Performance Evaluation")
    print("=" * 70)

    print("\n1. Generating test scenarios...")

    scenarios = {
        'Uptrend': generate_sample_data(200, 40000, 0.02, trend=0.0005),
        'Downtrend': generate_sample_data(200, 50000, 0.02, trend=-0.0005),
        'Volatile': generate_sample_data(200, 45000, 0.04, trend=0),
        'Stable': generate_sample_data(200, 45000, 0.01, trend=0),
    }

    forecaster = SimpleForecaster()

    print("\n2. Testing forecast performance across scenarios:\n")

    results_summary = []

    for scenario_name, df in scenarios.items():
        # Split data
        train = df.iloc[:150]
        test = df.iloc[150:174]  # 24 periods

        # Forecast
        forecast = forecaster.forecast(
            data=train['close'].values,
            prediction_length=24,
            num_samples=20
        )

        # Metrics
        actual = test['close'].values
        predicted = forecast['median']

        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        direction_correct = np.mean((np.diff(actual) > 0) == (np.diff(predicted) > 0)) * 100

        results_summary.append({
            'Scenario': scenario_name,
            'MAPE': mape,
            'Direction Acc': direction_correct
        })

        print(f"   {scenario_name}:")
        print(f"     MAPE: {mape:.2f}%")
        print(f"     Direction Accuracy: {direction_correct:.2f}%")

    print("\n3. Summary:")
    avg_mape = np.mean([r['MAPE'] for r in results_summary])
    avg_dir = np.mean([r['Direction Acc'] for r in results_summary])
    print(f"   Average MAPE: {avg_mape:.2f}%")
    print(f"   Average Direction Accuracy: {avg_dir:.2f}%")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("PRICE FORECASTING - OFFLINE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo simulates forecasting functionality using synthetic data")
    print("and a simple exponential smoothing model.")
    print("\nThe real Chronos model provides much more sophisticated forecasts!")
    print("Install Chronos to use the actual foundation model:")
    print("  pip install git+https://github.com/amazon-science/chronos-forecasting.git\n")

    try:
        demo_basic_forecasting()
        demo_backtesting()
        demo_multi_coin_forecast()
        demo_with_technical_analysis()
        demo_forecast_visualization_data()
        demo_model_performance()

        print("\n\n" + "=" * 70)
        print("ALL FORECASTING DEMOS COMPLETED!")
        print("=" * 70)
        print("\nTo use real Hyperliquid data with Chronos forecasting:")
        print("  1. Install Chronos: pip install chronos-forecasting")
        print("  2. Run: python example_forecasting.py")
        print("\nFor offline analysis demo:")
        print("  python demo_offline.py")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
