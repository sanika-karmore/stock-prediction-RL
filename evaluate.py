import pandas as pd
import numpy as np
import torch
from models import DQN, ActorCritic
from trading_env import StockTradingEnv
from preprocess import prepare_data
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)

def calculate_metrics(portfolio_values):
    """Calculate various trading metrics."""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Annualized return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    n_years = len(portfolio_values) / 252  # Assuming 252 trading days per year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Sharpe ratio (assuming risk-free rate of 0.02)
    excess_returns = returns - 0.02/252
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Additional metrics
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    win_rate = len([r for r in returns if r > 0]) / len(returns)
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Volatility': volatility,
        'Win Rate': win_rate
    }

def evaluate_model(model, env, model_type='DQN'):
    """Evaluate a trained model on test data."""
    state = env.reset()
    done = False
    portfolio_values = [env.portfolio_value]
    actions_taken = []
    positions = []  # Track positions (0: Hold, 1: Long)
    
    while not done:
        with torch.no_grad():
            if model_type == 'DQN':
                q_values = model(torch.FloatTensor(state))
                action = q_values.argmax().item()
            else:  # A2C
                action_probs, _ = model(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(action_probs).item()
        
        state, reward, done, info = env.step(action)
        portfolio_values.append(env.portfolio_value)
        actions_taken.append(action)
        positions.append(env.current_position)
    
    return portfolio_values, actions_taken, positions

def plot_performance_comparison(portfolio_values_dqn, portfolio_values_a2c, buy_hold_values, symbol):
    """Plot detailed performance comparison."""
    plt.figure(figsize=(15, 10))
    
    # Portfolio Values
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values_dqn, label='DQN', linewidth=2)
    plt.plot(portfolio_values_a2c, label='A2C', linewidth=2)
    plt.plot(buy_hold_values, label='Buy & Hold', linewidth=2)
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'Portfolio Value Comparison - {symbol}')
    plt.legend()
    plt.grid(True)
    
    # Returns Distribution
    plt.subplot(2, 1, 2)
    returns_dqn = np.diff(portfolio_values_dqn) / portfolio_values_dqn[:-1]
    returns_a2c = np.diff(portfolio_values_a2c) / portfolio_values_a2c[:-1]
    returns_bh = np.diff(buy_hold_values) / buy_hold_values[:-1]
    
    sns.kdeplot(returns_dqn, label='DQN Returns', linewidth=2)
    sns.kdeplot(returns_a2c, label='A2C Returns', linewidth=2)
    sns.kdeplot(returns_bh, label='Buy & Hold Returns', linewidth=2)
    plt.xlabel('Daily Returns')
    plt.ylabel('Density')
    plt.title('Returns Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'performance_analysis_{symbol}.png')
    plt.close()

def calculate_buy_hold_return(df):
    """Calculate buy and hold strategy returns."""
    initial_price = df['Close'].iloc[0]
    prices = df['Close']
    initial_balance = 10000  # Same as environment
    n_shares = initial_balance // initial_price
    portfolio_values = prices * n_shares
    return portfolio_values.tolist()

def analyze_trading_patterns(actions, positions, symbol):
    """Analyze trading patterns and generate insights."""
    total_trades = sum([1 for i in range(1, len(positions)) if positions[i] != positions[i-1]])
    hold_periods = []
    current_period = 1
    
    for i in range(1, len(positions)):
        if positions[i] == positions[i-1]:
            current_period += 1
        else:
            hold_periods.append(current_period)
            current_period = 1
    
    if hold_periods:
        avg_hold_period = np.mean(hold_periods)
        max_hold_period = np.max(hold_periods)
    else:
        avg_hold_period = current_period
        max_hold_period = current_period
    
    action_distribution = {
        'Hold': actions.count(0),
        'Buy': actions.count(1),
        'Sell': actions.count(2)
    }
    
    return {
        'Total Trades': total_trades,
        'Average Hold Period': avg_hold_period,
        'Max Hold Period': max_hold_period,
        'Action Distribution': action_distribution
    }

def main():
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    test_start = '2023-01-01'
    test_end = '2023-12-31'
    
    results = {}
    
    for symbol in symbols:
        logging.info(f"\nEvaluating models on {symbol}")
        
        # Load test data
        df_test, _, df_original = prepare_data(symbol, test_start, test_end)
        test_env = StockTradingEnv(df_test)
        
        # Load trained models
        state_dim = test_env.observation_space.shape[0]
        n_actions = test_env.action_space.n
        
        try:
            # Load DQN model
            dqn_checkpoint = torch.load(f'dqn_model_{symbol}.pth')
            dqn_model = DQN(state_dim, n_actions)
            dqn_model.load_state_dict(dqn_checkpoint['model_state_dict'])
            dqn_model.eval()
            
            # Load A2C model
            a2c_checkpoint = torch.load(f'a2c_model_{symbol}.pth')
            a2c_model = ActorCritic(state_dim, n_actions)
            a2c_model.load_state_dict(a2c_checkpoint['model_state_dict'])
            a2c_model.eval()
            
            # Evaluate models
            portfolio_values_dqn, actions_dqn, positions_dqn = evaluate_model(dqn_model, test_env, 'DQN')
            test_env.reset()
            portfolio_values_a2c, actions_a2c, positions_a2c = evaluate_model(a2c_model, test_env, 'A2C')
            
            # Calculate buy & hold performance
            buy_hold_values = calculate_buy_hold_return(df_original)
            
            # Plot results
            plot_performance_comparison(portfolio_values_dqn, portfolio_values_a2c, buy_hold_values, symbol)
            
            # Calculate metrics
            results[symbol] = {
                'DQN': {
                    'Performance': calculate_metrics(portfolio_values_dqn),
                    'Trading Patterns': analyze_trading_patterns(actions_dqn, positions_dqn, symbol)
                },
                'A2C': {
                    'Performance': calculate_metrics(portfolio_values_a2c),
                    'Trading Patterns': analyze_trading_patterns(actions_a2c, positions_a2c, symbol)
                },
                'Buy & Hold': calculate_metrics(buy_hold_values)
            }
            
            # Print results
            logging.info(f"\nResults for {symbol}:")
            
            for strategy in results[symbol]:
                logging.info(f"\n{strategy}:")
                if strategy in ['DQN', 'A2C']:
                    # Print performance metrics
                    logging.info("\nPerformance Metrics:")
                    for metric, value in results[symbol][strategy]['Performance'].items():
                        logging.info(f"{metric}: {value:.4f}")
                    
                    # Print trading patterns
                    logging.info("\nTrading Patterns:")
                    patterns = results[symbol][strategy]['Trading Patterns']
                    logging.info(f"Total Trades: {patterns['Total Trades']}")
                    logging.info(f"Average Hold Period: {patterns['Average Hold Period']:.2f} days")
                    logging.info(f"Max Hold Period: {patterns['Max Hold Period']} days")
                    logging.info("Action Distribution:")
                    for action, count in patterns['Action Distribution'].items():
                        logging.info(f"  {action}: {count}")
                else:
                    # Print buy & hold metrics
                    for metric, value in results[symbol][strategy].items():
                        logging.info(f"{metric}: {value:.4f}")
            
        except Exception as e:
            logging.error(f"Error evaluating {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 