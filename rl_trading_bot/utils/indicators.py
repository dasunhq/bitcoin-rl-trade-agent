"""
Technical Indicators for Trading
Implements RSI, EMA, MACD, Bollinger Bands, and other indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions (0-100 scale).
    
    Args:
        prices: Series of closing prices
        period: RSI period (default: 14)
        
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    EMA gives more weight to recent prices compared to simple moving average.
    
    Args:
        prices: Series of closing prices
        period: EMA period (default: 20)
        
    Returns:
        Series of EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Series of closing prices
        period: SMA period (default: 20)
        
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period).mean()


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    MACD is a trend-following momentum indicator showing the relationship
    between two moving averages of prices.
    
    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        
    Returns:
        Tuple of (MACD line, Signal line, MACD histogram)
    """
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (SMA) and two outer bands
    that are standard deviations away from the middle band.
    
    Args:
        prices: Series of closing prices
        period: SMA period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        
    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    middle_band = calculate_sma(prices, period)
    std_dev = prices.rolling(window=period).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR is a measure of volatility, showing how much an asset moves
    on average during a given time frame.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period (default: 14)
        
    Returns:
        Series of ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV is a momentum indicator that uses volume flow to predict
    changes in stock price.
    
    Args:
        close: Series of closing prices
        volume: Series of trading volumes
        
    Returns:
        Series of OBV values
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    The stochastic oscillator is a momentum indicator comparing a particular
    closing price to a range of prices over time.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: Lookback period (default: 14)
        
    Returns:
        Tuple of (%K line, %D line)
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()
    
    return k_percent, d_percent


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # EMAs
    df['ema_12'] = calculate_ema(df['close'], period=12)
    df['ema_26'] = calculate_ema(df['close'], period=26)
    df['ema_50'] = calculate_ema(df['close'], period=50)
    
    # SMAs
    df['sma_20'] = calculate_sma(df['close'], period=20)
    df['sma_50'] = calculate_sma(df['close'], period=50)
    
    # MACD
    macd, signal, histogram = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = histogram
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle  # Normalized width
    
    # ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # OBV
    df['obv'] = calculate_obv(df['close'], df['volume'])
    
    # Stochastic Oscillator
    k, d = calculate_stochastic_oscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = k
    df['stoch_d'] = d
    
    # Price momentum
    df['momentum'] = df['close'].pct_change(periods=10) * 100
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="h")
    
    price = 40000
    prices = [price]
    for _ in range(199):
        price = price * (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": np.random.uniform(100, 1000, 200)
    })
    
    print("Testing Technical Indicators...")
    df_with_indicators = add_all_indicators(df)
    
    print(f"\nDataFrame shape: {df_with_indicators.shape}")
    print(f"Columns: {list(df_with_indicators.columns)}")
    print(f"\nSample data (last 5 rows):")
    print(df_with_indicators.tail())
    
    print(f"\nRSI range: {df_with_indicators['rsi'].min():.2f} - {df_with_indicators['rsi'].max():.2f}")
    print(f"MACD range: {df_with_indicators['macd'].min():.2f} - {df_with_indicators['macd'].max():.2f}")
