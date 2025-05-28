import pandas as pd
import numpy as np
from ta import add_all_ta_features  # Technical analysis library

# Load historical data (e.g., AAPL)
df = pd.read_csv('AAPL.csv', parse_dates=['Date'])
df = df.sort_values('Date')

# Add technical indicators
df['SMA_20'] = df['Close'].rolling(20).mean()
df['RSI_14'] = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")['momentum_rsi']

# Split data chronologically
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]