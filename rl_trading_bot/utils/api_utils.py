"""
API Utilities for Cryptocurrency Trading
Supports Binance Testnet and Coinbase Sandbox for paper trading.
"""

import os
import time
import requests
import hmac
import hashlib
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta


class BinanceTestnetAPI:
    """
    Binance Testnet API client for paper trading.
    Connects to Binance Spot Testnet for risk-free testing.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Binance Testnet API client.
        
        Args:
            api_key: Binance testnet API key
            api_secret: Binance testnet API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binance.vision/api"
        self.headers = {"X-MBX-APIKEY": api_key}
    
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for authenticated requests."""
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def get_account_balance(self) -> Dict:
        """
        Get account balance for all assets.
        
        Returns:
            Dictionary with asset balances
        """
        endpoint = f"{self.base_url}/v3/account"
        params = {"timestamp": int(time.time() * 1000)}
        params["signature"] = self._generate_signature(params)
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract non-zero balances
            balances = {}
            for balance in data.get("balances", []):
                free = float(balance["free"])
                locked = float(balance["locked"])
                if free > 0 or locked > 0:
                    balances[balance["asset"]] = {
                        "free": free,
                        "locked": locked,
                        "total": free + locked
                    }
            return balances
        except Exception as e:
            print(f"Error fetching account balance: {e}")
            return {}
    
    def get_ticker_price(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """
        Get current ticker price for a symbol.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            
        Returns:
            Current price or None if error
        """
        endpoint = f"{self.base_url}/v3/ticker/price"
        params = {"symbol": symbol}
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            return float(data["price"])
        except Exception as e:
            print(f"Error fetching ticker price: {e}")
            return None
    
    def get_historical_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical candlestick (kline) data.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of data points (max 1000)
            start_time: Start timestamp in milliseconds
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.base_url}/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["startTime"] = start_time
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            print(f"Error fetching historical klines: {e}")
            return pd.DataFrame()
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Place a trading order.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            side: BUY or SELL
            order_type: LIMIT, MARKET, STOP_LOSS_LIMIT
            quantity: Amount to trade
            price: Limit price (required for LIMIT orders)
            
        Returns:
            Order response or None if error
        """
        endpoint = f"{self.base_url}/v3/order"
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "timestamp": int(time.time() * 1000)
        }
        
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price required for LIMIT orders")
            params["price"] = price
            params["timeInForce"] = "GTC"  # Good Till Cancel
        
        params["signature"] = self._generate_signature(params)
        
        try:
            response = requests.post(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error placing order: {e}")
            return None


class CoinbaseSandboxAPI:
    """
    Coinbase Sandbox API client for paper trading.
    Uses Coinbase Pro sandbox environment.
    """
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        """
        Initialize Coinbase Sandbox API client.
        
        Args:
            api_key: Coinbase sandbox API key
            api_secret: Coinbase sandbox API secret
            passphrase: Coinbase sandbox passphrase
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://api-public.sandbox.pro.coinbase.com"
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC SHA256 signature for authenticated requests."""
        message = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).digest()
        return signature.hex()
    
    def get_accounts(self) -> List[Dict]:
        """
        Get all trading accounts.
        
        Returns:
            List of account dictionaries
        """
        endpoint = f"{self.base_url}/accounts"
        timestamp = str(time.time())
        signature = self._generate_signature(timestamp, "GET", "/accounts")
        
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase
        }
        
        try:
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching accounts: {e}")
            return []
    
    def get_product_ticker(self, product_id: str = "BTC-USD") -> Optional[Dict]:
        """
        Get current ticker for a product.
        
        Args:
            product_id: Product identifier (e.g., BTC-USD)
            
        Returns:
            Ticker data or None if error
        """
        endpoint = f"{self.base_url}/products/{product_id}/ticker"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching ticker: {e}")
            return None
    
    def get_historical_data(
        self,
        product_id: str = "BTC-USD",
        granularity: int = 3600,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical candlestick data.
        
        Args:
            product_id: Product identifier (e.g., BTC-USD)
            granularity: Time slice in seconds (60, 300, 900, 3600, 21600, 86400)
            start: Start time in ISO 8601
            end: End time in ISO 8601
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.base_url}/products/{product_id}/candles"
        params = {"granularity": granularity}
        
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=["timestamp", "low", "high", "open", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()


def get_api_client(provider: str = "binance", **credentials) -> object:
    """
    Factory function to get API client based on provider.
    
    Args:
        provider: 'binance' or 'coinbase'
        **credentials: API credentials (api_key, api_secret, passphrase)
        
    Returns:
        API client instance
    """
    if provider.lower() == "binance":
        return BinanceTestnetAPI(
            api_key=credentials.get("api_key", ""),
            api_secret=credentials.get("api_secret", "")
        )
    elif provider.lower() == "coinbase":
        return CoinbaseSandboxAPI(
            api_key=credentials.get("api_key", ""),
            api_secret=credentials.get("api_secret", ""),
            passphrase=credentials.get("passphrase", "")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Example usage
if __name__ == "__main__":
    # Example: Fetch Binance testnet data
    print("Testing Binance Testnet API (public endpoints)...")
    client = BinanceTestnetAPI(api_key="", api_secret="")
    
    # Get current BTC price
    price = client.get_ticker_price("BTCUSDT")
    print(f"Current BTC/USDT Price: ${price:,.2f}")
    
    # Get historical data
    df = client.get_historical_klines("BTCUSDT", interval="1h", limit=100)
    print(f"\nHistorical Data Shape: {df.shape}")
    print(df.head())
