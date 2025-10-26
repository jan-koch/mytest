"""
Hyperliquid API Client for Historical Price Data Analysis

This module provides a Python client for interacting with the Hyperliquid API,
specifically focused on fetching and analyzing historical price data for
algorithmic trading.
"""

import requests
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import time


class HyperliquidClient:
    """
    Client for interacting with the Hyperliquid API.

    Supports both mainnet and testnet environments.
    """

    MAINNET_URL = "https://api.hyperliquid.xyz"
    TESTNET_URL = "https://api.hyperliquid-testnet.xyz"

    VALID_INTERVALS = [
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "8h", "12h",
        "1d", "3d", "1w", "1M"
    ]

    def __init__(self, testnet: bool = False):
        """
        Initialize the Hyperliquid client.

        Args:
            testnet: If True, use testnet API. Default is False (mainnet).
        """
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self.info_url = f"{self.base_url}/info"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _post_request(self, data: Dict) -> Dict:
        """
        Make a POST request to the info endpoint.

        Args:
            data: Request payload

        Returns:
            API response as dictionary

        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.session.post(self.info_url, json=data)
        response.raise_for_status()
        return response.json()

    def get_all_mids(self) -> Dict[str, str]:
        """
        Get mid-prices for all trading pairs.

        Returns mid-prices with fallback to last trade price if order book is empty.

        Returns:
            Dictionary mapping coin symbols to mid-prices

        Example:
            {"BTC": "45000.5", "ETH": "2500.25"}
        """
        data = {"type": "allMids"}
        return self._post_request(data)

    def get_meta(self) -> Dict:
        """
        Get metadata about available trading pairs and market information.

        Returns:
            Market metadata including available coins and their properties
        """
        data = {"type": "meta"}
        return self._post_request(data)

    def get_spot_meta(self) -> Dict:
        """
        Get metadata for spot trading pairs.

        Returns:
            Spot market metadata
        """
        data = {"type": "spotMeta"}
        return self._post_request(data)

    def get_candles(
        self,
        coin: str,
        interval: str,
        start_time: Optional[Union[datetime, int]] = None,
        end_time: Optional[Union[datetime, int]] = None,
        lookback_periods: Optional[int] = None
    ) -> List[Dict]:
        """
        Get historical candlestick (OHLCV) data.

        Note: Only the most recent 5000 candles are available via the API.

        Args:
            coin: Coin symbol (e.g., "BTC", "ETH")
            interval: Candle interval (e.g., "1m", "1h", "1d")
            start_time: Start time as datetime or epoch milliseconds
            end_time: End time as datetime or epoch milliseconds
            lookback_periods: Number of periods to look back (alternative to start_time)

        Returns:
            List of candle dictionaries with OHLCV data

        Raises:
            ValueError: If interval is invalid or time parameters are incorrect

        Example:
            >>> client.get_candles("BTC", "1h", lookback_periods=100)
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Must be one of: {', '.join(self.VALID_INTERVALS)}"
            )

        # Handle time parameters
        if end_time is None:
            end_time = datetime.now()

        if isinstance(end_time, datetime):
            end_time_ms = int(end_time.timestamp() * 1000)
        else:
            end_time_ms = end_time

        if start_time is None and lookback_periods is not None:
            # Calculate start_time based on lookback_periods
            interval_seconds = self._interval_to_seconds(interval)
            start_time = datetime.fromtimestamp(end_time_ms / 1000) - \
                         timedelta(seconds=interval_seconds * lookback_periods)
            start_time_ms = int(start_time.timestamp() * 1000)
        elif isinstance(start_time, datetime):
            start_time_ms = int(start_time.timestamp() * 1000)
        elif start_time is not None:
            start_time_ms = start_time
        else:
            # Default to last 5000 candles
            interval_seconds = self._interval_to_seconds(interval)
            start_time = datetime.fromtimestamp(end_time_ms / 1000) - \
                         timedelta(seconds=interval_seconds * 5000)
            start_time_ms = int(start_time.timestamp() * 1000)

        data = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms
            }
        }

        return self._post_request(data)

    def get_candles_df(
        self,
        coin: str,
        interval: str,
        start_time: Optional[Union[datetime, int]] = None,
        end_time: Optional[Union[datetime, int]] = None,
        lookback_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical candlestick data as a pandas DataFrame.

        Args:
            coin: Coin symbol (e.g., "BTC", "ETH")
            interval: Candle interval (e.g., "1m", "1h", "1d")
            start_time: Start time as datetime or epoch milliseconds
            end_time: End time as datetime or epoch milliseconds
            lookback_periods: Number of periods to look back

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Example:
            >>> df = client.get_candles_df("BTC", "1h", lookback_periods=100)
        """
        candles = self.get_candles(coin, interval, start_time, end_time, lookback_periods)

        if not candles:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(candle['t'], unit='ms'),
                'open': float(candle['o']),
                'high': float(candle['h']),
                'low': float(candle['l']),
                'close': float(candle['c']),
                'volume': float(candle['v']),
            }
            for candle in candles
        ])

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        return df

    def get_multiple_candles_df(
        self,
        coins: List[str],
        interval: str,
        start_time: Optional[Union[datetime, int]] = None,
        end_time: Optional[Union[datetime, int]] = None,
        lookback_periods: Optional[int] = None,
        delay: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical candlestick data for multiple coins.

        Args:
            coins: List of coin symbols
            interval: Candle interval
            start_time: Start time
            end_time: End time
            lookback_periods: Number of periods to look back
            delay: Delay between requests in seconds (to avoid rate limiting)

        Returns:
            Dictionary mapping coin symbols to their DataFrames
        """
        result = {}
        for coin in coins:
            try:
                df = self.get_candles_df(coin, interval, start_time, end_time, lookback_periods)
                result[coin] = df
                if delay > 0 and coin != coins[-1]:  # Don't delay after last request
                    time.sleep(delay)
            except Exception as e:
                print(f"Error fetching data for {coin}: {e}")
                result[coin] = pd.DataFrame()

        return result

    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        """Convert interval string to seconds."""
        unit = interval[-1]
        value = int(interval[:-1])

        multipliers = {
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'M': 2592000  # Approximate (30 days)
        }

        return value * multipliers.get(unit, 60)

    def get_available_coins(self) -> List[str]:
        """
        Get list of available perpetual trading pairs.

        Returns:
            List of coin symbols
        """
        meta = self.get_meta()
        return [asset['name'] for asset in meta.get('universe', [])]

    def get_coin_info(self, coin: str) -> Optional[Dict]:
        """
        Get detailed information about a specific coin.

        Args:
            coin: Coin symbol

        Returns:
            Coin metadata or None if not found
        """
        meta = self.get_meta()
        for asset in meta.get('universe', []):
            if asset['name'] == coin:
                return asset
        return None
