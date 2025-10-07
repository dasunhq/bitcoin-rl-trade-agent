"""
Proximal Policy Optimization (PPO) Agent for Trading
Implements PPO using Stable Baselines3 for policy-based RL.
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Optional, Dict, List, Callable
import os
import gym


class TradingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.
    Logs episode rewards, portfolio values, and other metrics.
    """
    
    def __init__(self, check_freq: int = 1000, save_path: str = "./models/", verbose: int = 1):
        """
        Initialize callback.
        
        Args:
            check_freq: Frequency to check and log metrics
            save_path: Directory to save best model
            verbose: Verbosity level
        """
        super(TradingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            True to continue training, False to stop
        """
        # Check if episode is done
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            # Get episode info
            info = self.locals.get('infos', [{}])[0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        
        # Periodic logging
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                
                if self.verbose > 0:
                    print(f"Step: {self.n_calls}")
                    print(f"  Mean Episode Reward (last 100): {mean_reward:.2f}")
                    print(f"  Mean Episode Length (last 100): {mean_length:.2f}")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    save_file = os.path.join(self.save_path, "ppo_best_model.zip")
                    self.model.save(save_file)
                    if self.verbose > 0:
                        print(f"  New best model saved! Reward: {mean_reward:.2f}")
        
        return True


class PPOAgent:
    """
    PPO Agent wrapper for trading using Stable Baselines3.
    
    PPO is a policy gradient method that uses a clipped surrogate objective
    to prevent destructively large policy updates.
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict] = None,
        device: str = "auto",
        verbose: int = 1
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: Trading environment (gym.Env)
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Maximum norm for gradient clipping
            policy_kwargs: Additional policy network arguments
            device: Device to run on ('cpu', 'cuda', or 'auto')
            verbose: Verbosity level
        """
        self.env = env
        self.verbose = verbose
        
        # Default policy network architecture
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [dict(pi=[256, 128], vf=[256, 128])],
                "activation_fn": nn.ReLU,
            }
        
        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose
        )
        
        # Training tracking
        self.episode_rewards = []
        self.training_timesteps = 0
        
        if verbose > 0:
            print(f"PPO Agent initialized")
            print(f"  Device: {self.model.device}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Policy network: {policy_kwargs}")
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total number of samples to train on
            callback: Callback function for logging/checkpointing
            log_interval: Number of episodes between logs
            save_path: Path to save final model
        """
        if self.verbose > 0:
            print(f"\nStarting PPO training for {total_timesteps} timesteps...")
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=True
        )
        
        self.training_timesteps += total_timesteps
        
        # Save final model
        if save_path:
            self.save(save_path)
        
        if self.verbose > 0:
            print(f"Training completed! Total timesteps: {self.training_timesteps}")
    
    def predict(
        self,
        state: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predict action given state.
        
        Args:
            state: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, state) where state is hidden state of policy
        """
        action, _states = self.model.predict(state, deterministic=deterministic)
        return action, _states
    
    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate the agent on an environment.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(state, deterministic=deterministic)
                state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Get final portfolio value if available
            if hasattr(env, 'get_performance_metrics'):
                metrics = env.get_performance_metrics()
                portfolio_values.append(metrics['final_portfolio_value'])
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
        }
        
        if portfolio_values:
            results['mean_portfolio_value'] = np.mean(portfolio_values)
            results['std_portfolio_value'] = np.std(portfolio_values)
        
        return results
    
    def save(self, filepath: str):
        """
        Save the PPO model.
        
        Args:
            filepath: Path to save model (without extension)
        """
        self.model.save(filepath)
        if self.verbose > 0:
            print(f"PPO model saved to {filepath}.zip")
    
    def load(self, filepath: str):
        """
        Load a saved PPO model.
        
        Args:
            filepath: Path to load model from
        """
        if not os.path.exists(f"{filepath}.zip"):
            print(f"Model file not found: {filepath}.zip")
            return
        
        self.model = PPO.load(filepath, env=self.env)
        if self.verbose > 0:
            print(f"PPO model loaded from {filepath}.zip")
    
    def get_policy_parameters(self) -> Dict:
        """
        Get current policy network parameters.
        
        Returns:
            Dictionary with policy statistics
        """
        policy = self.model.policy
        
        stats = {
            'total_parameters': sum(p.numel() for p in policy.parameters()),
            'trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
        }
        
        return stats


def create_ppo_agent(
    env: gym.Env,
    learning_rate: float = 3e-4,
    device: str = "auto"
) -> PPOAgent:
    """
    Factory function to create a PPO agent with default settings.
    
    Args:
        env: Trading environment
        learning_rate: Learning rate
        device: Device to use
        
    Returns:
        Configured PPO agent
    """
    # Optimized settings for trading
    agent = PPOAgent(
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "net_arch": [dict(pi=[256, 128], vf=[256, 128])],
            "activation_fn": nn.ReLU,
        },
        device=device,
        verbose=1
    )
    
    return agent


# Test PPO agent
if __name__ == "__main__":
    print("Testing PPO Agent...")
    
    # Create dummy environment
    from gym import spaces
    
    class DummyTradingEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(58,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)
        
        def reset(self):
            return np.random.randn(58).astype(np.float32)
        
        def step(self, action):
            state = np.random.randn(58).astype(np.float32)
            reward = np.random.randn()
            done = np.random.rand() > 0.95
            return state, reward, done, {}
    
    env = DummyTradingEnv()
    
    # Create agent
    agent = create_ppo_agent(env, learning_rate=3e-4)
    
    # Test prediction
    state = env.reset()
    action, _ = agent.predict(state)
    print(f"Test prediction - Action: {action}")
    
    # Test training (short)
    print("\nRunning short training test...")
    callback = TradingCallback(check_freq=500, verbose=1)
    agent.train(total_timesteps=2000, callback=callback, log_interval=1)
    
    print("\nPPO Agent test complete!")
    policy_params = agent.get_policy_parameters()
    print(f"Policy parameters: {policy_params}")
