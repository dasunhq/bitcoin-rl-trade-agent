"""
Double Deep Q-Network (DDQN) Agent for Trading
Implements DDQN with experience replay and target network for stable learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple, List, Optional
import os


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer for DDQN.
    Stores transitions and samples random minibatches for training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture.
    Multi-layer perceptron for Q-value approximation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers with ReLU activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout for regularization
            input_dim = hidden_dim
        
        # Output layer (Q-values for each action)
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(state)


class DDQNAgent:
    """
    Double Deep Q-Network Agent for trading.
    
    Uses two networks (online and target) to reduce overestimation bias.
    Implements experience replay for stable learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,  # HOLD, BUY, SELL
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_dims: List[int] = [256, 128, 64],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize DDQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Size of replay buffer
            batch_size: Minibatch size for training
            target_update_freq: Steps between target network updates
            hidden_dims: Hidden layer dimensions
            device: Device to run computations on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Online network (main network for action selection)
        self.online_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Target network (stable network for Q-value targets)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # Set to evaluation mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training tracking
        self.steps = 0
        self.episode_rewards = []
        self.losses = []
        
        print(f"DDQN Agent initialized on device: {self.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Network architecture: {hidden_dims}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (enables exploration)
            
        Returns:
            Selected action index
        """
        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation: best action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using a minibatch from replay buffer.
        
        Returns:
            Loss value or None if buffer is too small
        """
        # Wait until buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample minibatch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Unpack experiences
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.online_network(states).gather(1, actions)
        
        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.online_network(next_states).argmax(1, keepdim=True)
            
            # Evaluate actions using target network (Double DQN)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            
            # Compute target
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Track loss
        self.losses.append(loss.item())
        
        return loss.item()
    
    def save(self, filepath: str):
        """
        Save agent's networks and training state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent's networks and training state.
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps = checkpoint.get('steps', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        
        print(f"Model loaded from {filepath}")
        print(f"Epsilon: {self.epsilon:.4f}, Steps: {self.steps}")
    
    def get_training_stats(self) -> dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'total_experiences': len(self.replay_buffer),
            'episode_rewards': self.episode_rewards
        }


# Test DDQN agent
if __name__ == "__main__":
    print("Testing DDQN Agent...")
    
    # Create agent
    state_dim = 58  # Example: 3 portfolio + 5 indicators + 50 history
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=3,
        learning_rate=0.001,
        hidden_dims=[256, 128, 64]
    )
    
    # Simulate training
    for episode in range(5):
        state = np.random.randn(state_dim)
        episode_reward = 0
        
        for step in range(100):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Simulate environment step
            next_state = np.random.randn(state_dim)
            reward = np.random.randn()
            done = (step == 99)
            
            # Store and train
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
        
        agent.episode_rewards.append(episode_reward)
        
        if (episode + 1) % 1 == 0:
            stats = agent.get_training_stats()
            print(f"Episode {episode + 1}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Epsilon={stats['epsilon']:.4f}, "
                  f"Avg Loss={stats['avg_loss']:.4f}")
    
    print("\nTraining complete!")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total steps: {agent.steps}")
