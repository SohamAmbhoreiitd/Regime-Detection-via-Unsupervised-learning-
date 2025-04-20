import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_features(df, feature_columns):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

def add_price_and_volume_features(depth_df, volume_features_df):
    depth_df = add_price_features(depth_df)
    volume_features_df = add_volume_features(volume_features_df)
    
    return depth_df, volume_features_df

def add_price_features(df):
    df = df.copy()
    df = df.sort_values('Time')
    df['mid_price'] = (df['BidPriceL1'] + df['AskPriceL1']) / 2
    df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
    df.set_index('Time', inplace=True)
    df['volatility_10s'] = df['log_return'].rolling('10s').std()
    df['volatility_30s'] = df['log_return'].rolling('30s').std()
    df.reset_index(inplace=True)
    return df

def add_volume_features(trades):
    trades = trades.copy()
    trades.set_index('Time', inplace=True)
    trades['buy_volume'] = trades['Quantity'].where(trades['IsMarketMaker'] == False, 0)
    trades['sell_volume'] = trades['Quantity'].where(trades['IsMarketMaker'] == True, 0)
    resampled = trades.resample('1s').agg({
        'buy_volume': 'sum',
        'sell_volume': 'sum',
        'Price': ['mean', 'sum'],
        'Quantity': 'sum'
    })
    resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
    resampled['volume_imbalance'] = resampled['buy_volume_sum'] - resampled['sell_volume_sum']
    resampled['vwap'] = resampled['Price_sum'] / resampled['Quantity_sum']
    resampled['vwap_change_10s'] = resampled['vwap'].diff(10)
    resampled['vwap_change_30s'] = resampled['vwap'].diff(30)
    resampled.reset_index(inplace=True)
    return resampled