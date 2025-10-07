"""
Custom Trading Environment for Reinforcement Learning
OpenAI Gym compatible environment for cryptocurrency trading simulation.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Custom OpenAI Gym environment for cryptocurrency trading.
    
    State Space:
        - Current portfolio value
        - Cash balance
        - BTC holdings
        - Current BTC price
        - Technical indicators (RSI, EMA, MACD, etc.)
        - Recent price history (lookback window)
    
    Action Space:
        - 0: HOLD (do nothing)
        - 1: BUY (buy BTC with available cash)
        - 2: SELL (sell all BTC holdings)
    
    Reward:
        - Portfolio value change + transaction cost penalties
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,  # 0.1% fee
        lookback_window: int = 50,
        normalize_state: bool = True
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            initial_balance: Starting cash balance (USD)
            transaction_fee: Trading fee as decimal (0.001 = 0.1%)
            lookback_window: Number of past time steps to include in state
            normalize_state: Whether to normalize state values
        """
        super(TradingEnvironment, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.lookback_window = lookback_window
        self.normalize_state = normalize_state
        
        # Validate DataFrame
        required_cols = ["close", "open", "high", "low", "volume"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        
        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # State space dimension calculation
        # [cash, btc_holdings, current_price] + technical indicators + price history
        self.state_dim = 3 + 5 + lookback_window  # 3 portfolio + 5 indicators + history
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = len(self.df) - lookback_window - 1
        
        # Portfolio state
        self.cash = initial_balance
        self.btc_holdings = 0.0
        self.portfolio_values = []
        self.trades = []
        
        logger.info(f"Trading Environment initialized with {len(self.df)} data points")
        logger.info(f"State dimension: {self.state_dim}, Max steps: {self.max_steps}")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.cash = self.initial_balance
        self.btc_holdings = 0.0
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Trading action (0=HOLD, 1=BUY, 2=SELL)
            
        Returns:
            observation: Current state
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        if self.current_step >= self.max_steps:
            raise ValueError("Episode is done. Please reset the environment.")
        
        # Get current price
        current_price = self._get_current_price()
        portfolio_value_before = self._get_portfolio_value()
        
        # Execute action
        action_name = ["HOLD", "BUY", "SELL"][action]
        transaction_cost = 0.0
        
        if action == 1:  # BUY
            # Buy BTC with all available cash
            if self.cash > 0:
                btc_to_buy = self.cash / current_price
                transaction_cost = btc_to_buy * current_price * self.transaction_fee
                btc_after_fee = btc_to_buy - (transaction_cost / current_price)
                
                self.btc_holdings += btc_after_fee
                self.cash = 0.0
                
                self.trades.append({
                    "step": self.current_step,
                    "action": "BUY",
                    "price": current_price,
                    "amount": btc_after_fee,
                    "cost": transaction_cost
                })
        
        elif action == 2:  # SELL
            # Sell all BTC holdings
            if self.btc_holdings > 0:
                cash_from_sale = self.btc_holdings * current_price
                transaction_cost = cash_from_sale * self.transaction_fee
                
                self.cash += cash_from_sale - transaction_cost
                self.btc_holdings = 0.0
                
                self.trades.append({
                    "step": self.current_step,
                    "action": "SELL",
                    "price": current_price,
                    "amount": self.btc_holdings,
                    "cost": transaction_cost
                })
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        portfolio_value_after = self._get_portfolio_value()
        reward = self._calculate_reward(
            portfolio_value_before,
            portfolio_value_after,
            transaction_cost
        )
        
        # Track portfolio value
        self.portfolio_values.append(portfolio_value_after)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get next observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            "step": self.current_step,
            "action": action_name,
            "cash": self.cash,
            "btc_holdings": self.btc_holdings,
            "current_price": current_price,
            "portfolio_value": portfolio_value_after,
            "reward": reward,
            "total_trades": len(self.trades)
        }
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the current state observation.
        
        Returns:
            State vector as numpy array
        """
        idx = self.lookback_window + self.current_step
        
        # Portfolio state
        current_price = self.df.loc[idx, "close"]
        portfolio_value = self._get_portfolio_value()
        
        state = [
            self.cash / self.initial_balance,  # Normalized cash
            self.btc_holdings * current_price / self.initial_balance,  # Normalized BTC value
            current_price / 10000.0  # Normalized price (assuming BTC ~$10k-$100k)
        ]
        
        # Technical indicators (from the current position)
        indicators = self._calculate_indicators(idx)
        state.extend(indicators)
        
        # Price history (lookback window)
        price_history = self.df.loc[idx - self.lookback_window:idx, "close"].values
        if self.normalize_state:
            price_history = price_history / price_history[0]  # Normalize to first price
        state.extend(price_history.tolist())
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_indicators(self, idx: int) -> List[float]:
        """
        Calculate technical indicators for the current state.
        
        Args:
            idx: Current index in DataFrame
            
        Returns:
            List of indicator values
        """
        # Simple moving averages
        window_short = 7
        window_long = 25
        
        if idx >= window_long:
            sma_short = self.df.loc[idx - window_short:idx, "close"].mean()
            sma_long = self.df.loc[idx - window_long:idx, "close"].mean()
            current_price = self.df.loc[idx, "close"]
            
            indicators = [
                (current_price - sma_short) / current_price,  # Price vs short SMA
                (current_price - sma_long) / current_price,   # Price vs long SMA
                (sma_short - sma_long) / sma_long,            # SMA crossover signal
                self.df.loc[idx, "volume"] / self.df["volume"].mean(),  # Volume ratio
                (self.df.loc[idx, "close"] - self.df.loc[idx, "open"]) / self.df.loc[idx, "open"]  # Candle change
            ]
        else:
            indicators = [0.0] * 5
        
        return indicators
    
    def _get_current_price(self) -> float:
        """Get the current BTC price."""
        idx = self.lookback_window + self.current_step
        return self.df.loc[idx, "close"]
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + BTC holdings in USD)."""
        current_price = self._get_current_price()
        return self.cash + (self.btc_holdings * current_price)
    
    def _calculate_reward(
        self,
        portfolio_before: float,
        portfolio_after: float,
        transaction_cost: float
    ) -> float:
        """
        Calculate reward for the action taken.
        
        Args:
            portfolio_before: Portfolio value before action
            portfolio_after: Portfolio value after action
            transaction_cost: Cost of transaction
            
        Returns:
            Reward value
        """
        # Profit/loss percentage
        profit_pct = (portfolio_after - portfolio_before) / portfolio_before
        
        # Penalize transaction costs
        cost_penalty = transaction_cost / self.initial_balance
        
        # Combined reward
        reward = profit_pct - cost_penalty
        
        return reward
    
    def render(self, mode: str = "human"):
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode
        """
        if mode == "human":
            portfolio_value = self._get_portfolio_value()
            profit = portfolio_value - self.initial_balance
            profit_pct = (profit / self.initial_balance) * 100
            
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"BTC Holdings: {self.btc_holdings:.6f} BTC")
            print(f"Current Price: ${self._get_current_price():,.2f}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Profit: ${profit:,.2f} ({profit_pct:+.2f}%)")
            print(f"Total Trades: {len(self.trades)}")
            print(f"{'='*60}\n")
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the episode.
        
        Returns:
            Dictionary with performance metrics
        """
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # Hourly to yearly
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = portfolio_values / portfolio_values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "initial_balance": self.initial_balance,
            "final_portfolio_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": len(self.trades),
            "num_steps": len(portfolio_values)
        }


# Test environment
if __name__ == "__main__":
    # Create dummy data for testing
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="h")
    
    # Simulate BTC price with random walk
    price = 40000
    prices = [price]
    for _ in range(999):
        price = price * (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": np.random.uniform(100, 1000, 1000)
    })
    
    # Test environment
    env = TradingEnvironment(df, initial_balance=10000)
    
    print("Testing Trading Environment...")
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Random actions
    for i in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
        
        if done:
            break
    
    metrics = env.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}