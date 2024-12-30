import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch historical data for a single ticker
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
    stock_data.columns = [col.replace(f'_{ticker}', '') for col in stock_data.columns]
    stock_data.reset_index(inplace=True)

    if 'Close' not in stock_data.columns:
        if 'Adj Close' in stock_data.columns:
            stock_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        else:
            raise ValueError("'Close' or 'Adj Close' column is missing in the fetched data.")
    
    return stock_data


# Function to add technical indicators
def add_indicators(data, macd_params=(12, 26, 9), ema_windows=[7, 10, 20]):
    short_window, long_window, signal_window = macd_params
    data['EMA_short'] = data['Close_Lag1'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close_Lag1'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    for window in ema_windows:
        data[f'EMA_{window}'] = data['Close_Lag1'].ewm(span=window, adjust=False).mean()
    return data

def add_lagged_values(df, columns, lags):
    for col in columns:
        for lag in lags:
            df[f'{col}_Lag{lag}'] = df[col].shift(lag)
    return df

def clean_data(df, drop_threshold=0.05):
    null_props = df.isnull().mean()
    df = df.drop(columns=null_props[null_props >= drop_threshold].index.tolist())
    df = df.ffill().bfill()
    return df

def preprocess_data(data, columns_to_lag, lags):
    data = add_lagged_values(data, columns_to_lag, lags)
    if 'Close' not in data.columns:
        raise ValueError("'Close' column is missing after adding lagged values.")
    data = clean_data(data)
    return data
