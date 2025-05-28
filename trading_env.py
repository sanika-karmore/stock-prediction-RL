import gymnasium as gym
import numpy as np
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data + technical indicators + portfolio info
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(df.columns) + 3,),  # +3 for balance, shares, portfolio value
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.balance
        self.current_position = 0  # 0 = no position, 1 = long
        
        return self._get_observation()
    
    def _get_observation(self):
        # Get the current market data
        obs = self.df.iloc[self.current_step].values
        
        # Add portfolio information
        portfolio_obs = np.array([
            self.balance,
            self.shares_held,
            self.portfolio_value
        ])
        
        return np.concatenate([obs, portfolio_obs])
    
    def step(self, action):
        # Get current price
        current_price = float(self.df.iloc[self.current_step]['Close'])
        
        # Initialize reward
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.balance >= current_price:
                # Calculate maximum shares that can be bought
                max_shares = self.balance // current_price
                shares_to_buy = max_shares
                
                # Update positions
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.current_position = 1
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Sell all shares
                self.balance += self.shares_held * current_price
                self.shares_held = 0
                self.current_position = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        self.portfolio_value = self.balance + self.shares_held * current_price
        
        # Calculate reward
        reward = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        # Add penalties
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0
        if cash_ratio > 0.5:  # Penalize holding too much cash
            reward -= 0.1
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, {}