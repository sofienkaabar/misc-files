import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Vision_Library import get_timeseries
# Generate sample data (replace with your own data)
ticker = 'dis'   # Market

# import
df = get_timeseries(ticker=ticker)

# Parameters
rsi_period = 14
bb_period = 20
bb_std = 2
hold_periods = 14  # n periods to hold

# Calculate RSI

def calculate_rsi(df, source='close', output_name='rsi', rsi_period=14, smoothed=False):
    delta = df[source].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rsi_real_lookback = (rsi_period*2)-1
    avg_gain = gain.ewm(span=rsi_real_lookback, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_real_lookback, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss 
    rsi = 100-(100/(1+rs))
    df[output_name] = rsi
    if smoothed==True:
        df[output_name] = df[output_name].ewm(span=20, adjust=False).mean()
    return df.dropna()

df = calculate_rsi(df, source='close', output_name='rsi', rsi_period=14, smoothed=False)

# Calculate Bollinger Bands on RSI
df['rsi_sma'] = df['rsi'].rolling(window=bb_period).mean()
df['rsi_std'] = df['rsi'].rolling(window=bb_period).std()
df['rsi_upper_bb'] = df['rsi_sma'] + bb_std * df['rsi_std']
df['rsi_lower_bb'] = df['rsi_sma'] - bb_std * df['rsi_std']

# Strategy 1: RSI crosses 30/70 levels
df['rsi_prev'] = df['rsi'].shift(1)
df['signal_s1_long'] = (df['rsi_prev'] < 30) & (df['rsi'] >= 30)
df['signal_s1_short'] = (df['rsi_prev'] > 70) & (df['rsi'] <= 70)

# Strategy 2: RSI crosses Bollinger Bands
df['signal_s2_long'] = (df['rsi_prev'] < df['rsi_lower_bb'].shift(1)) & (df['rsi'] >= df['rsi_lower_bb'])
df['signal_s2_short'] = (df['rsi_prev'] > df['rsi_upper_bb'].shift(1)) & (df['rsi'] <= df['rsi_upper_bb'])

# Calculate forward returns for holding n periods
df['forward_return'] = df['close'].shift(-hold_periods) / df['close'] - 1

# Extract trade returns
def get_trade_returns(df, long_col, short_col):
    long_trades = df[df[long_col]]['forward_return'].dropna()
    short_trades = -df[df[short_col]]['forward_return'].dropna()
    all_trades = pd.concat([long_trades, short_trades])
    return all_trades

s1_returns = get_trade_returns(df, 'signal_s1_long', 'signal_s1_short')
s2_returns = get_trade_returns(df, 'signal_s2_long', 'signal_s2_short')

# Performance metrics
def calculate_metrics(returns, hold_periods):
    if len(returns) == 0:
        return {k: np.nan for k in ['profit_factor', 'hit_ratio', 'sharpe', 't_stat', 'bar_efficiency']}
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
    hit_ratio = len(wins) / len(returns) * 100
    
    periods_per_year = 252 / hold_periods
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret * periods_per_year) / (std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else np.nan
    t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else np.nan
    bar_efficiency = mean_ret / hold_periods * 10000
    
    return {
        'profit_factor': round(profit_factor, 3),
        'hit_ratio': round(hit_ratio, 2),
        'sharpe': round(sharpe, 3),
        't_stat': round(t_stat, 3),
    }

metrics_s1 = calculate_metrics(s1_returns, hold_periods)
metrics_s2 = calculate_metrics(s2_returns, hold_periods)

comparison = pd.DataFrame({
    'Strategy_1_RSI_30_70': metrics_s1,
    'Strategy_2_RSI_BB': metrics_s2
}).T

print("=" * 70)
print("TRADING STRATEGY COMPARISON")
print(f"RSI Period: {rsi_period} | BB Period: {bb_period} | Hold: {hold_periods} bars")
print("=" * 70)
print(comparison.to_string())
print("=" * 70)
