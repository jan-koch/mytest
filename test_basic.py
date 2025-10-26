"""
Basic test to verify Hyperliquid API client works
"""

from hyperliquid_client import HyperliquidClient
from hyperliquid_analyzer import PriceAnalyzer

def test_basic_functionality():
    """Test basic client functionality."""
    print("Testing Hyperliquid Price Analysis Tool")
    print("=" * 60)

    # Initialize client
    print("\n1. Initializing client...")
    client = HyperliquidClient(testnet=False)
    print("   ✓ Client initialized")

    # Get available coins
    print("\n2. Fetching available coins...")
    coins = client.get_available_coins()
    print(f"   ✓ Found {len(coins)} coins")
    print(f"   First 10: {coins[:10]}")

    # Get current prices
    print("\n3. Fetching current mid prices...")
    prices = client.get_all_mids()
    print(f"   ✓ Got prices for {len(prices)} coins")
    if 'BTC' in prices:
        print(f"   BTC: ${prices['BTC']}")
    if 'ETH' in prices:
        print(f"   ETH: ${prices['ETH']}")

    # Get historical data
    print("\n4. Fetching BTC 1-hour candles (last 50 periods)...")
    df = client.get_candles_df("BTC", "1h", lookback_periods=50)
    print(f"   ✓ Fetched {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")

    # Test analyzer
    print("\n5. Testing technical analysis features...")
    analyzer = PriceAnalyzer(df)
    analyzer.add_sma(20)
    analyzer.add_rsi(14)
    analyzer.add_returns()

    df_analyzed = analyzer.get_dataframe()
    print(f"   ✓ Added {len(df_analyzed.columns)} columns")
    print(f"   Columns: {df_analyzed.columns.tolist()}")

    # Get statistics
    print("\n6. Getting statistics...")
    stats = analyzer.get_statistics()
    print(f"   ✓ Statistics calculated")
    print(f"   Total periods: {stats['total_periods']}")
    print(f"   Price range: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}")
    print(f"   Mean price: ${stats['mean_price']:.2f}")

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
