import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, dropout_rate=0.2):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(input_dim, 128)
        self.shared_dropout1 = nn.Dropout(dropout_rate)
        self.shared_fc2 = nn.Linear(128, 64)
        self.shared_dropout2 = nn.Dropout(dropout_rate)
        
        # Actor head (policy)
        self.actor = nn.Linear(64, n_actions)
        
        # Critic head (value)
        self.critic = nn.Linear(64, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.shared_fc1(x))
        x = self.shared_dropout1(x)
        x = F.relu(self.shared_fc2(x))
        x = self.shared_dropout2(x)
        
        # Actor: action probabilities
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic: state value
        state_value = self.critic(x)
        
        return action_probs, state_value

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer) 