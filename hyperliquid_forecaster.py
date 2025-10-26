"""
Hyperliquid Price Forecasting Module using Chronos

This module provides price forecasting capabilities using Amazon's Chronos
foundation model for time series forecasting, integrated with the Hyperliquid
price analysis tool.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import warnings

try:
    import torch
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    warnings.warn(
        "Chronos not installed. Install with: "
        "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
    )


class PriceForecaster:
    """
    Price forecasting using Chronos foundation models.

    This class provides an interface to use Amazon's Chronos models for
    forecasting cryptocurrency prices fetched from Hyperliquid.
    """

    AVAILABLE_MODELS = {
        'tiny': 'amazon/chronos-t5-tiny',        # ~8M parameters, fastest
        'mini': 'amazon/chronos-t5-mini',        # ~20M parameters
        'small': 'amazon/chronos-t5-small',      # ~46M parameters
        'base': 'amazon/chronos-t5-base',        # ~200M parameters, recommended
        'large': 'amazon/chronos-t5-large',      # ~710M parameters, most accurate
    }

    def __init__(
        self,
        model_size: str = 'base',
        device: str = 'auto',
        torch_dtype: Optional[str] = None
    ):
        """
        Initialize the forecaster with a Chronos model.

        Args:
            model_size: Size of the model ('tiny', 'mini', 'small', 'base', 'large')
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            torch_dtype: Torch dtype to use (None, 'float32', 'float16', 'bfloat16')

        Raises:
            ImportError: If chronos is not installed
            ValueError: If model_size is invalid
        """
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "Chronos is not installed. Install with: "
                "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )

        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_size = model_size
        self.model_name = self.AVAILABLE_MODELS[model_size]
        self.device = device
        self.pipeline = None

        # Convert torch_dtype string to torch dtype
        self.torch_dtype = None
        if torch_dtype:
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            self.torch_dtype = dtype_map.get(torch_dtype)

        print(f"Initializing Chronos model: {self.model_name}")
        self._load_model()

    def _load_model(self):
        """Load the Chronos model."""
        load_kwargs = {}

        if self.device != 'auto':
            load_kwargs['device_map'] = self.device

        if self.torch_dtype:
            load_kwargs['torch_dtype'] = self.torch_dtype

        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        print(f"✓ Model loaded successfully")

    def forecast_prices(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        num_samples: int = 20,
        column: str = 'close',
        return_quantiles: bool = True,
        quantile_levels: Optional[List[float]] = None
    ) -> Dict:
        """
        Forecast future prices based on historical data.

        Args:
            df: DataFrame with historical price data (must have datetime index)
            prediction_length: Number of periods to forecast
            num_samples: Number of sample trajectories to generate
            column: Column to forecast (default: 'close')
            return_quantiles: If True, return quantile predictions
            quantile_levels: List of quantile levels (default: [0.1, 0.25, 0.5, 0.75, 0.9])

        Returns:
            Dictionary containing:
                - 'forecast_dates': Future timestamps
                - 'median': Median forecast
                - 'mean': Mean forecast
                - 'quantiles': Quantile forecasts (if return_quantiles=True)
                - 'samples': All sample trajectories
                - 'context': Historical data used for forecasting

        Example:
            >>> forecaster = PriceForecaster('base')
            >>> result = forecaster.forecast_prices(df, prediction_length=24)
            >>> print(result['median'])
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Prepare context data
        context = torch.tensor(df[column].values, dtype=torch.float32)

        # Generate forecast
        forecast = self.pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

        # forecast shape: (num_samples, prediction_length)
        forecast_np = forecast.numpy()

        # Generate future dates
        last_date = df.index[-1]
        freq = pd.infer_freq(df.index)
        if freq is None:
            # Fallback: calculate median time difference
            time_diffs = df.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            forecast_dates = pd.date_range(
                start=last_date + median_diff,
                periods=prediction_length,
                freq=median_diff
            )
        else:
            forecast_dates = pd.date_range(
                start=last_date,
                periods=prediction_length + 1,
                freq=freq
            )[1:]  # Skip the last historical point

        # Calculate statistics
        result = {
            'forecast_dates': forecast_dates,
            'mean': forecast_np.mean(axis=0),
            'median': np.median(forecast_np, axis=0),
            'std': forecast_np.std(axis=0),
            'samples': forecast_np,
            'context': df[column].values,
            'context_dates': df.index,
        }

        # Calculate quantiles
        if return_quantiles:
            if quantile_levels is None:
                quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

            quantiles = {}
            for q in quantile_levels:
                quantiles[f'q{int(q*100)}'] = np.quantile(forecast_np, q, axis=0)

            result['quantiles'] = quantiles

        return result

    def forecast_df(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        num_samples: int = 20,
        column: str = 'close',
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Forecast future prices and return as a DataFrame.

        Args:
            df: DataFrame with historical price data
            prediction_length: Number of periods to forecast
            num_samples: Number of sample trajectories
            column: Column to forecast
            include_history: If True, include historical data in result

        Returns:
            DataFrame with forecast results including mean, median, and quantiles

        Example:
            >>> forecast_df = forecaster.forecast_df(df, prediction_length=24)
            >>> print(forecast_df[['mean', 'q10', 'q50', 'q90']])
        """
        result = self.forecast_prices(
            df=df,
            prediction_length=prediction_length,
            num_samples=num_samples,
            column=column,
            return_quantiles=True
        )

        # Create forecast DataFrame
        forecast_data = {
            'mean': result['mean'],
            'median': result['median'],
            'std': result['std'],
        }

        # Add quantiles
        if 'quantiles' in result:
            forecast_data.update(result['quantiles'])

        forecast_df = pd.DataFrame(forecast_data, index=result['forecast_dates'])

        if include_history:
            # Create historical DataFrame
            hist_df = pd.DataFrame({
                'actual': result['context'],
                'mean': result['context'],
                'median': result['context'],
            }, index=result['context_dates'])

            # Combine with forecast
            return pd.concat([hist_df, forecast_df])

        return forecast_df

    def backtest(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        test_size: int,
        column: str = 'close',
        num_samples: int = 20
    ) -> Dict:
        """
        Backtest the forecasting model on historical data.

        Args:
            df: DataFrame with historical price data
            prediction_length: Number of periods to forecast
            test_size: Size of test set (last N periods)
            column: Column to forecast
            num_samples: Number of samples for forecast

        Returns:
            Dictionary with backtest results including metrics and predictions

        Example:
            >>> results = forecaster.backtest(df, prediction_length=24, test_size=48)
            >>> print(f"MAPE: {results['mape']:.2f}%")
        """
        if test_size < prediction_length:
            raise ValueError("test_size must be >= prediction_length")

        # Split data
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        # Make prediction
        forecast_result = self.forecast_prices(
            df=train_df,
            prediction_length=prediction_length,
            num_samples=num_samples,
            column=column
        )

        # Get actual values for the forecast period
        actual_values = test_df[column].iloc[:prediction_length].values
        predicted_values = forecast_result['median']

        # Calculate metrics
        mae = np.mean(np.abs(actual_values - predicted_values))
        rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

        # Direction accuracy (did we predict up/down correctly?)
        actual_direction = np.diff(actual_values) > 0
        pred_direction = np.diff(predicted_values) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'actual': actual_values,
            'predicted': predicted_values,
            'forecast_dates': test_df.index[:prediction_length],
            'train_size': len(train_df),
            'test_size': test_size,
            'prediction_length': prediction_length,
        }

    def rolling_forecast(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        window_size: int,
        step_size: int = 1,
        column: str = 'close',
        num_samples: int = 20
    ) -> List[Dict]:
        """
        Perform rolling window forecasting.

        Args:
            df: DataFrame with historical price data
            prediction_length: Number of periods to forecast
            window_size: Size of the training window
            step_size: Number of periods to move forward each iteration
            column: Column to forecast
            num_samples: Number of samples per forecast

        Returns:
            List of forecast results for each window

        Example:
            >>> results = forecaster.rolling_forecast(
            ...     df, prediction_length=12, window_size=100, step_size=12
            ... )
        """
        forecasts = []
        total_periods = len(df)

        for start_idx in range(0, total_periods - window_size - prediction_length + 1, step_size):
            end_idx = start_idx + window_size
            train_df = df.iloc[start_idx:end_idx]

            try:
                forecast_result = self.forecast_prices(
                    df=train_df,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    column=column
                )

                # Get actual values
                actual_end_idx = min(end_idx + prediction_length, total_periods)
                actual_values = df[column].iloc[end_idx:actual_end_idx].values

                forecast_result['actual'] = actual_values
                forecast_result['train_start'] = df.index[start_idx]
                forecast_result['train_end'] = df.index[end_idx - 1]

                forecasts.append(forecast_result)

            except Exception as e:
                print(f"Error in window {start_idx}-{end_idx}: {e}")
                continue

        return forecasts

    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """
        Calculate forecasting metrics.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary with MAE, RMSE, MAPE, and direction accuracy
        """
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        if len(actual) > 1:
            actual_direction = np.diff(actual) > 0
            pred_direction = np.diff(predicted) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = None

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }


def check_chronos_installation() -> bool:
    """
    Check if Chronos is properly installed.

    Returns:
        True if Chronos is available, False otherwise
    """
    return CHRONOS_AVAILABLE


def get_installation_instructions() -> str:
    """
    Get installation instructions for Chronos.

    Returns:
        Installation instructions as a string
    """
    return """
To use the price forecasting features, install Chronos:

Method 1 (from GitHub):
    pip install git+https://github.com/amazon-science/chronos-forecasting.git

Method 2 (from PyPI):
    pip install chronos-forecasting

Additional requirements:
    - PyTorch: pip install torch
    - For GPU support, install appropriate CUDA version of PyTorch

Note: Models will be downloaded from Hugging Face on first use.
Model sizes range from ~50MB (tiny) to ~3GB (large).
"""
