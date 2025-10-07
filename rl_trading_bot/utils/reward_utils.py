"""
Reward Functions for Reinforcement Learning Trading Agent
Implements various reward shaping strategies for the trading environment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def simple_profit_reward(
    portfolio_value_before: float,
    portfolio_value_after: float,
    initial_balance: float = 10000.0
) -> float:
    """
    Simple reward based on portfolio value change.
    
    Args:
        portfolio_value_before: Portfolio value before action
        portfolio_value_after: Portfolio value after action
        initial_balance: Initial portfolio balance
        
    Returns:
        Reward value (percentage change)
    """
    return (portfolio_value_after - portfolio_value_before) / initial_balance


def profit_with_penalty_reward(
    portfolio_value_before: float,
    portfolio_value_after: float,
    transaction_cost: float,
    initial_balance: float = 10000.0,
    cost_penalty_multiplier: float = 2.0
) -> float:
    """
    Reward that penalizes transaction costs.
    
    Args:
        portfolio_value_before: Portfolio value before action
        portfolio_value_after: Portfolio value after action
        transaction_cost: Cost of the transaction
        initial_balance: Initial portfolio balance
        cost_penalty_multiplier: Multiplier for cost penalty
        
    Returns:
        Reward value
    """
    profit = (portfolio_value_after - portfolio_value_before) / initial_balance
    cost_penalty = (transaction_cost / initial_balance) * cost_penalty_multiplier
    
    return profit - cost_penalty


def sharpe_ratio_reward(
    returns: List[float],
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio as reward.
    
    Sharpe ratio measures risk-adjusted return.
    
    Args:
        returns: List of historical returns
        risk_free_rate: Risk-free rate of return
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    mean_return = returns_array.mean()
    std_return = returns_array.std()
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe


def sortino_ratio_reward(
    returns: List[float],
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sortino ratio as reward.
    
    Similar to Sharpe ratio but only considers downside volatility.
    
    Args:
        returns: List of historical returns
        risk_free_rate: Risk-free rate of return
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    mean_return = returns_array.mean()
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns_array[returns_array < risk_free_rate]
    if len(downside_returns) == 0:
        return mean_return  # No downside, return mean
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    
    sortino = (mean_return - risk_free_rate) / downside_std
    return sortino


def risk_adjusted_reward(
    portfolio_value_before: float,
    portfolio_value_after: float,
    returns_history: List[float],
    transaction_cost: float = 0.0,
    initial_balance: float = 10000.0,
    risk_penalty: float = 0.5
) -> float:
    """
    Risk-adjusted reward that considers both profit and volatility.
    
    Args:
        portfolio_value_before: Portfolio value before action
        portfolio_value_after: Portfolio value after action
        returns_history: Historical returns for volatility calculation
        transaction_cost: Cost of the transaction
        initial_balance: Initial portfolio balance
        risk_penalty: Weight for risk penalty
        
    Returns:
        Risk-adjusted reward
    """
    # Calculate profit
    profit = (portfolio_value_after - portfolio_value_before) / initial_balance
    
    # Calculate volatility penalty
    if len(returns_history) > 1:
        volatility = np.std(returns_history)
        risk_cost = volatility * risk_penalty
    else:
        risk_cost = 0.0
    
    # Transaction cost penalty
    cost_penalty = transaction_cost / initial_balance
    
    return profit - risk_cost - cost_penalty


def drawdown_penalty_reward(
    portfolio_value: float,
    peak_value: float,
    drawdown_penalty: float = 1.0
) -> float:
    """
    Penalize drawdowns from peak portfolio value.
    
    Args:
        portfolio_value: Current portfolio value
        peak_value: Peak portfolio value in episode
        drawdown_penalty: Multiplier for drawdown penalty
        
    Returns:
        Drawdown penalty (negative)
    """
    if peak_value == 0:
        return 0.0
    
    drawdown = (peak_value - portfolio_value) / peak_value
    return -drawdown * drawdown_penalty


def holding_time_reward(
    action: int,
    holding_time: int,
    max_holding_time: int = 100,
    penalty: float = 0.001
) -> float:
    """
    Encourage/discourage holding positions for too long.
    
    Args:
        action: Current action (0=HOLD, 1=BUY, 2=SELL)
        holding_time: Number of steps in current position
        max_holding_time: Maximum encouraged holding time
        penalty: Penalty per step over max
        
    Returns:
        Holding time adjustment
    """
    if action == 0 and holding_time > max_holding_time:  # HOLD
        excess_time = holding_time - max_holding_time
        return -excess_time * penalty
    return 0.0


def momentum_aligned_reward(
    action: int,
    price_momentum: float,
    alignment_bonus: float = 0.01
) -> float:
    """
    Reward actions aligned with price momentum.
    
    Args:
        action: Current action (0=HOLD, 1=BUY, 2=SELL)
        price_momentum: Price momentum indicator (positive/negative)
        alignment_bonus: Bonus for aligned actions
        
    Returns:
        Alignment bonus/penalty
    """
    if action == 1 and price_momentum > 0:  # BUY in uptrend
        return alignment_bonus
    elif action == 2 and price_momentum < 0:  # SELL in downtrend
        return alignment_bonus
    elif action == 1 and price_momentum < 0:  # BUY in downtrend
        return -alignment_bonus
    elif action == 2 and price_momentum > 0:  # SELL in uptrend
        return -alignment_bonus
    return 0.0


def combined_reward(
    portfolio_value_before: float,
    portfolio_value_after: float,
    transaction_cost: float,
    action: int,
    returns_history: List[float],
    peak_value: float,
    price_momentum: float = 0.0,
    initial_balance: float = 10000.0,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Combined reward function with multiple components.
    
    Args:
        portfolio_value_before: Portfolio value before action
        portfolio_value_after: Portfolio value after action
        transaction_cost: Cost of transaction
        action: Action taken
        returns_history: Historical returns
        peak_value: Peak portfolio value
        price_momentum: Price momentum indicator
        initial_balance: Initial balance
        weights: Dictionary of component weights
        
    Returns:
        Combined weighted reward
    """
    if weights is None:
        weights = {
            "profit": 1.0,
            "cost": 2.0,
            "risk": 0.3,
            "drawdown": 0.5,
            "momentum": 0.2
        }
    
    # Component rewards
    profit = (portfolio_value_after - portfolio_value_before) / initial_balance
    cost = -(transaction_cost / initial_balance) * weights["cost"]
    
    # Risk adjustment
    if len(returns_history) > 1:
        risk = -np.std(returns_history) * weights["risk"]
    else:
        risk = 0.0
    
    # Drawdown penalty
    drawdown = drawdown_penalty_reward(
        portfolio_value_after,
        peak_value,
        weights["drawdown"]
    )
    
    # Momentum alignment
    momentum = momentum_aligned_reward(
        action,
        price_momentum,
        weights["momentum"]
    )
    
    # Combine all components
    total_reward = (
        profit * weights["profit"] +
        cost +
        risk +
        drawdown +
        momentum
    )
    
    return total_reward


# Example usage
if __name__ == "__main__":
    print("Testing Reward Functions...")
    
    # Test simple profit reward
    portfolio_before = 10000
    portfolio_after = 10500
    transaction_cost = 10
    
    reward1 = simple_profit_reward(portfolio_before, portfolio_after, 10000)
    print(f"Simple profit reward: {reward1:.4f}")
    
    reward2 = profit_with_penalty_reward(
        portfolio_before,
        portfolio_after,
        transaction_cost,
        10000
    )
    print(f"Profit with penalty reward: {reward2:.4f}")
    
    # Test Sharpe ratio
    returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.03]
    sharpe = sharpe_ratio_reward(returns)
    print(f"Sharpe ratio: {sharpe:.4f}")
    
    # Test combined reward
    combined = combined_reward(
        portfolio_value_before=10000,
        portfolio_value_after=10500,
        transaction_cost=10,
        action=1,
        returns_history=[0.01, 0.02, -0.005],
        peak_value=10500,
        price_momentum=0.05,
        initial_balance=10000
    )
    print(f"Combined reward: {combined:.4f}")
