import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import DQN, ActorCritic, ReplayBuffer
from trading_env import StockTradingEnv
from preprocess import prepare_data
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def train_dqn(env, model, target_model, replay_buffer, optimizer, n_episodes=1000,
              batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
              epsilon_decay=0.995, target_update=10):
    
    epsilon = epsilon_start
    rewards_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            # Train if enough samples in replay buffer
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                # Compute current Q values
                current_q_values = model(states).gather(1, actions.unsqueeze(1))
                
                # Compute target Q values
                with torch.no_grad():
                    max_next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
                
                # Compute loss and update
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Update target network
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            logging.info(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return rewards_history

def train_a2c(env, model, optimizer, n_episodes=1000, gamma=0.99):
    rewards_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = model(state_tensor)
            
            # Sample action from policy
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            # Take action
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            
            # Calculate advantage
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_state_value = model(next_state_tensor)
            advantage = reward + gamma * next_state_value * (1 - done) - state_value
            
            # Calculate losses
            actor_loss = -action_dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)
            
            # Combined loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            logging.info(f"Episode {episode}, Total Reward: {total_reward:.2f}")
    
    return rewards_history

def plot_results(dqn_rewards, a2c_rewards, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_rewards, label='DQN')
    plt.plot(a2c_rewards, label='A2C')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'DQN vs A2C Performance - {symbol}')
    plt.legend()
    plt.savefig(f'training_results_{symbol}.png')
    plt.close()

def main():
    # Parameters
    symbols = ['AAPL', 'GOOGL', 'MSFT']  # Multiple stocks for better generalization
    train_start = '2010-01-01'
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    test_end = '2023-12-31'
    
    for symbol in symbols:
        logging.info(f"Training on {symbol}")
        
        # Load and preprocess data
        df_train, scaler, _ = prepare_data(symbol, train_start, train_end)
        df_test, _, _ = prepare_data(symbol, test_start, test_end)
        
        # Create environments
        train_env = StockTradingEnv(df_train)
        test_env = StockTradingEnv(df_test)
        
        # Initialize models
        state_dim = train_env.observation_space.shape[0]
        n_actions = train_env.action_space.n
        
        # DQN setup
        dqn_model = DQN(state_dim, n_actions)
        dqn_target = DQN(state_dim, n_actions)
        dqn_target.load_state_dict(dqn_model.state_dict())
        dqn_optimizer = optim.Adam(dqn_model.parameters(), lr=0.001)
        replay_buffer = ReplayBuffer(10000)
        
        # A2C setup
        a2c_model = ActorCritic(state_dim, n_actions)
        a2c_optimizer = optim.Adam(a2c_model.parameters(), lr=0.001)
        
        # Train models
        logging.info("Training DQN model...")
        dqn_rewards = train_dqn(train_env, dqn_model, dqn_target, replay_buffer, dqn_optimizer)
        
        logging.info("Training A2C model...")
        a2c_rewards = train_a2c(train_env, a2c_model, a2c_optimizer)
        
        # Plot results
        plot_results(dqn_rewards, a2c_rewards, symbol)
        
        # Save models
        torch.save({
            'model_state_dict': dqn_model.state_dict(),
            'optimizer_state_dict': dqn_optimizer.state_dict(),
            'rewards_history': dqn_rewards,
            'scaler': scaler
        }, f'dqn_model_{symbol}.pth')
        
        torch.save({
            'model_state_dict': a2c_model.state_dict(),
            'optimizer_state_dict': a2c_optimizer.state_dict(),
            'rewards_history': a2c_rewards,
            'scaler': scaler
        }, f'a2c_model_{symbol}.pth')
        
        logging.info(f"Training completed for {symbol}")

if __name__ == "__main__":
    main() 