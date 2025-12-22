import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def rsi(my_time_series, source='close', output_name='indicator', rsi_lookback=14, smoothed=False):
    delta = my_time_series[source].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rsi_real_lookback = (rsi_lookback*2)-1
    avg_gain = gain.ewm(span=rsi_real_lookback, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_real_lookback, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss 
    rsi = 100-(100/(1+rs))
    my_time_series[output_name] = rsi
    if smoothed==True:
        my_time_series[output_name] = my_time_series[output_name].ewm(span=20, adjust=False).mean()
    return my_time_series.dropna()

def global_median_rsi(df, source='close', output_name='indicator', min_lb=2, max_lb=31, smoothed=False):
    close = df[source]
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    all_rsis = []
    for lb in range(min_lb, max_lb + 1):
        real_lb = (lb * 2) - 1
        avg_gain = gain.ewm(span=real_lb, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(span=real_lb, min_periods=1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        all_rsis.append(rsi.values)
    # stack to shape (n_points, n_lookbacks)
    rsi_matrix = np.vstack(all_rsis).T
    # take row-wise median
    median_rsi = np.median(rsi_matrix, axis=1)
    # assign to df
    df[output_name] = median_rsi
    if smoothed:
        df[output_name] = df[output_name].ewm(span=3, adjust=False).mean()
    return df.dropna()

def get_timeseries(ticker=None, period="5y"): 
    df = yf.download(ticker, start="2024-01-01", interval="1h")
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.columns = df.columns.get_level_values(0)
    return df

def create_signals_aggressive(df, lower_barrier=30, upper_barrier=70):
    rsi = df["indicator"]
    bull = (rsi.shift(1) >= lower_barrier) & (rsi < lower_barrier)
    bear = (rsi.shift(1) <= upper_barrier) & (rsi > upper_barrier)
    df["bullish_signal"] = bull.astype(int)
    df["bearish_signal"] = bear.astype(int)
    return df

def create_signals_random_contrarian(
    df,
    lower_barrier=40,
    upper_barrier=60,
    prob=0.2,
    seed=None):
    if seed is not None:
        np.random.seed(seed)
    rsi = df["indicator"]
    bullish_signal = np.zeros(len(df), dtype=int)
    bearish_signal = np.zeros(len(df), dtype=int)
    oversold = rsi < lower_barrier
    overbought = rsi > upper_barrier
    for i in range(len(df)):
        if oversold.iloc[i] and np.random.rand() < prob:
            bullish_signal[i] = 1

        if overbought.iloc[i] and np.random.rand() < prob:
            bearish_signal[i] = 1
    df = df.copy()
    df["bullish_signal"] = bullish_signal
    df["bearish_signal"] = bearish_signal
    return df

def create_signals_conservative(df, lower_barrier=30, upper_barrier=70):
    rsi = df["indicator"]
    bull = (rsi.shift(1) <= lower_barrier) & (rsi > lower_barrier)
    bear = (rsi.shift(1) >= upper_barrier) & (rsi < upper_barrier)
    df["bullish_signal"] = bull.astype(int)
    df["bearish_signal"] = bear.astype(int)
    return df

def create_performance_columns(df, bull_hold=10, bear_hold=10):
    df = df.copy()
    df["bullish_perf"] = np.nan
    df["bearish_perf"] = np.nan

    closes = df["close"].values
    n = len(df)

    for i in range(n):
        entry_price = closes[i]

        # ---------- bullish ----------
        if df.iloc[i]["bullish_signal"] == 1:
            exit_i = i + bull_hold
            if exit_i < n:
                exit_price = closes[exit_i]
                df.iloc[exit_i, df.columns.get_loc("bullish_perf")] = (
                    exit_price - entry_price
                )

        # ---------- bearish ----------
        if df.iloc[i]["bearish_signal"] == 1:
            exit_i = i + bear_hold
            if exit_i < n:
                exit_price = closes[exit_i]
                df.iloc[exit_i, df.columns.get_loc("bearish_perf")] = (
                    entry_price - exit_price
                )

    return df

def swing_detect(my_time_series, swing_lookback=20):
    my_time_series['swing_low']  = my_time_series['close'].rolling(window=swing_lookback, min_periods=1, center=True).min()
    my_time_series['swing_low']  = my_time_series.apply(lambda row: row['close'] if row['close'] == row['swing_low'] else 0, axis=1)
    my_time_series['swing_low']  = my_time_series['swing_low'].replace(0, np.nan)
    my_time_series['swing_high'] = my_time_series['close'].rolling(window=swing_lookback, min_periods=1, center=True).max()
    my_time_series['swing_high'] = my_time_series.apply(lambda row: row['close'] if row['close'] == row['swing_high'] else 0, axis=1)
    my_time_series['swing_high'] = my_time_series['swing_high'].replace(0, np.nan)
    return my_time_series

def extract_trades(df):
    rows = df.dropna(subset=["bullish_perf", "bearish_perf"], how="all")
    trades = []
    for idx, row in rows.iterrows():
        if not np.isnan(row["bullish_perf"]):
            trades.append({"idx": idx, "type": "bull", "pnl": row["bullish_perf"]})
        elif not np.isnan(row["bearish_perf"]):
            trades.append({"idx": idx, "type": "bear", "pnl": row["bearish_perf"]})
    return pd.DataFrame(trades)

def hit_ratio(trades):
    return (trades["pnl"] > 0).mean()

def signal_chart_indicator(my_time_series, indicator, window=500, lower_barrier=30, upper_barrier=70, plot_type='bars',
                             barriers=True, indicator_label='Indicator'): 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ]     
    if plot_type == 'line':
        ax1.plot(sample['close'], color='black') 
    else:
        print('Choose between bars or candlesticks')           
    for i in my_time_series.index:
        if my_time_series.loc[i, 'bullish_signal'] == 1:
            ax1.annotate('', xy=(i, my_time_series.loc[i, 'close']), xytext=(i, my_time_series.loc[i, 'close']-1),
                        arrowprops=dict(facecolor='green', shrink=0.05))
        elif my_time_series.loc[i, 'bearish_signal'] == 1:
            ax1.annotate('', xy=(i, my_time_series.loc[i, 'close']), xytext=(i, my_time_series.loc[i, 'close']+1),
                        arrowprops=dict(facecolor='red', shrink=0.05))    
    ax2.plot(sample.index, sample[indicator], label=indicator_label, color='blue')
    from matplotlib.lines import Line2D
    bullish_signal = Line2D([0], [0], marker='^', color='w', label='Buy signal', markerfacecolor='green', markersize=10)
    bearish_signal = Line2D([0], [0], marker='v', color='w', label='Sell signal', markerfacecolor='red', markersize=10)
    ax1.legend(handles=[bullish_signal, bearish_signal])
    if barriers == True:
        ax2.axhline(y=lower_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=upper_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=50, color='black', linestyle='dashed')        
    ax1.grid()
    ax2.legend()
    ax2.grid()
    
def trade_statistics(df, pnl_col="pnl"):
    pnl = df[pnl_col]

    # Wins and losses
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    # Profit Factor
    total_win = wins.sum()
    total_loss = abs(losses.sum())
    profit_factor = total_win / total_loss if total_loss != 0 else np.inf

    # Expectancy
    win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Sharpe ratio (per trade)
    sharpe = pnl.mean() / pnl.std() if pnl.std() != 0 else np.nan

    return {
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }

def optimal_horizon_after_swings(df, max_horizon=50, log_returns=True):
    """
    Finds the forward horizon that maximizes directionally-correct
    median return after swing lows and swing highs.
    """

    closes = df["close"].values
    n = len(closes)

    # swing exists where value is NOT NaN
    swing_low_pos  = np.where(df["swing_low"].notna())[0]
    swing_high_pos = np.where(df["swing_high"].notna())[0]

    def forward_returns(positions, h):
        rets = []
        for p in positions:
            if p + h < n:
                if log_returns:
                    r = np.log(closes[p + h] / closes[p])
                else:
                    r = closes[p + h] / closes[p] - 1
                rets.append(r)
        return np.array(rets, dtype=float)

    low_medians  = {}
    high_medians = {}

    for h in range(1, max_horizon + 1):
        low_rets  = forward_returns(swing_low_pos, h)
        high_rets = forward_returns(swing_high_pos, h)

        # swing lows: positive returns desirable
        if len(low_rets) > 0:
            low_medians[h] = np.nanmedian(low_rets)

        # swing highs: negative returns desirable → flip sign
        if len(high_rets) > 0:
            high_medians[h] = np.nanmedian(-high_rets)

    best_low_h  = max(low_medians,  key=low_medians.get)  if low_medians  else None
    best_high_h = max(high_medians, key=high_medians.get) if high_medians else None

    return best_low_h, best_high_h

def global_performance_metrics(df):
    """
    Computes global hit ratio, cumulative return, and profit factor
    from bullish_perf and bearish_perf columns.
    """

    # collect all trades
    bull = df["bullish_perf"].dropna().values
    bear = df["bearish_perf"].dropna().values

    trades = np.concatenate([bull, bear])

    if len(trades) == 0:
        return {
            "hit_ratio": np.nan,
            "cumulative_return": 0.0,
            "profit_factor": np.nan,
            "n_trades": 0
        }

    wins   = trades[trades > 0]
    losses = trades[trades < 0]

    hit_ratio = len(wins) / len(trades)
    cumulative_return = trades.sum()

    if losses.sum() != 0:
        profit_factor = wins.sum() / abs(losses.sum())
    else:
        profit_factor = np.inf

    return {
        "hit_ratio": hit_ratio,
        "cumulative_return": cumulative_return,
        "profit_factor": profit_factor,
        "n_trades": len(trades)
    }