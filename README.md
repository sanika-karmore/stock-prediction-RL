# Stock Market Data Preprocessing Pipeline

## Overview
This project provides a robust data preprocessing pipeline for stock market analysis. It downloads historical stock data from Yahoo Finance and calculates various technical indicators commonly used in financial analysis and algorithmic trading.

## Features
- Downloads historical stock data using Yahoo Finance API
- Calculates multiple technical indicators including:
  - Simple Moving Averages (SMA20, SMA50)
  - Exponential Moving Averages (EMA12, EMA26)
  - Moving Average Convergence Divergence (MACD)
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - Volume indicators
  - Daily and logarithmic returns
  - Volatility measures

## Technical Indicators Included
- **Moving Averages**: SMA20, SMA50, EMA12, EMA26
- **MACD**: Including MACD line, signal line, and histogram
- **RSI**: 14-period Relative Strength Index
- **Bollinger Bands**: Including upper, middle, and lower bands
- **Volume Analysis**: Volume SMA and Volume Ratio
- **Price Changes**: Daily returns and logarithmic returns
- **Volatility**: 20-day rolling standard deviation of returns

## Data Processing Features
- Automated missing value handling
- Feature normalization using StandardScaler
- Comprehensive error handling and logging
- Data validation and cleaning

## Dependencies
- pandas
- numpy
- scikit-learn
- yfinance
- logging

## Usage
```python
from preprocess import prepare_data

# Example usage
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-12-31'

# Get processed data
df_scaled, scaler, df_original = prepare_data(symbol, start_date, end_date)
```

## Output
The pipeline generates two CSV files:
- `{symbol}_preprocessed.csv`: Normalized data with all technical indicators
- `{symbol}_original.csv`: Original downloaded data

## Error Handling
The pipeline includes comprehensive error handling and logging to ensure robust operation and easy debugging.
 
