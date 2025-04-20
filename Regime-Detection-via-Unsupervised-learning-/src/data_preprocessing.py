import pandas as pd
import numpy as np

def load_data(depth_file_path, trade_file_path):
    depth_df = pd.read_csv(depth_file_path)
    trade_df = pd.read_csv(trade_file_path)
    return depth_df, trade_df

def preprocess_depth_data(depth_df):
    depth_df['Time'] = depth_df['Time'].str.replace(r'\s+\d+\s\w+', '', regex=True)
    depth_df['Time'] = pd.to_datetime(depth_df['Time'], errors='coerce')
    depth_df.dropna(subset=['Time'], inplace=True)
    return depth_df

def preprocess_trade_data(trade_df):
    trade_df['Time'] = trade_df['Time'].str.replace(r'\s+\d+\s\w+', '', regex=True)
    trade_df['Time'] = pd.to_datetime(trade_df['Time'], errors='coerce')
    trade_df.dropna(subset=['Time'], inplace=True)
    trade_df.sort_values('Time', inplace=True)
    return trade_df

def normalize_features(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def handle_missing_values(df):
    return df.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill

def preprocess_data(depth_file_path, trade_file_path):
    depth_df, trade_df = load_data(depth_file_path, trade_file_path)
    depth_df = preprocess_depth_data(depth_df)
    trade_df = preprocess_trade_data(trade_df)
    
    # Handle missing values
    depth_df = handle_missing_values(depth_df)
    trade_df = handle_missing_values(trade_df)
    
    return depth_df, trade_df