import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)

def download_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    logging.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators manually."""
    logging.info("Adding technical indicators...")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert Volume to float to ensure proper calculations
    df['Volume'] = df['Volume'].astype(float)
    
    # Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    
    # Price changes
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Volume indicators
    volume_sma = df['Volume'].rolling(window=20).mean()
    df['Volume_SMA'] = volume_sma
    df['Volume_Ratio'] = df['Volume'].div(volume_sma)
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataframe."""
    logging.info("Handling missing values...")
    
    # Forward fill price data first
    price_cols = ['Open', 'High', 'Low', 'Close']
    if 'Adj Close' in df.columns:
        price_cols.append('Adj Close')
    
    df[price_cols] = df[price_cols].ffill()
    
    # Forward fill other columns
    df = df.ffill()
    
    # Backward fill any remaining NaNs
    df = df.bfill()
    
    return df

def normalize_features(df):
    """Normalize features using StandardScaler."""
    logging.info("Normalizing features...")
    
    scaler = StandardScaler()
    
    # Select numerical columns for normalization
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create a copy to avoid modifying the original
    df_normalized = df.copy()
    
    # Normalize the features
    df_normalized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df_normalized, scaler

def prepare_data(symbol, start_date, end_date):
    """Prepare data for training/testing."""
    try:
        # Download data
        df = download_data(symbol, start_date, end_date)
        df_original = df.copy()  # Keep original data for comparison
        
        # Add technical indicators
        df = calculate_technical_indicators(df)
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Drop rows with any remaining NaN values
        df = df.dropna()
        
        # Normalize features
        df_normalized, scaler = normalize_features(df)
        
        logging.info("Data preparation completed successfully!")
        return df_normalized, scaler, df_original
        
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise

def main():
    # Example usage
    symbol = 'AAPL'
    train_start = '2010-01-01'
    train_end = '2022-12-31'
    
    try:
        df_scaled, scaler, df_original = prepare_data(symbol, train_start, train_end)
        
        # Save preprocessed data
        df_scaled.to_csv(f'{symbol}_preprocessed.csv')
        df_original.to_csv(f'{symbol}_original.csv')
        
        # Print some statistics
        print("\nDataset Statistics:")
        print(f"Total samples: {len(df_scaled)}")
        print(f"Features: {df_scaled.columns.tolist()}")
        print("\nFeature statistics:")
        print(df_scaled.describe())
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 