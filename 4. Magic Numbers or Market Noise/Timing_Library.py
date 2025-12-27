import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

def get_timeseries(ticker=None): 
    df = yf.download(ticker, start="2024-03-01", interval="60m")
    df = df.rename(columns={"Close": "close", 'High': 'high', 'Low': 'low', 'Open': 'open'})
    df.drop(['Volume'], axis=1, inplace=True)
    df.columns = df.columns.get_level_values(0)
    return df

def ohlc_plot(my_time_series, window=250, plot_type='bars', chart_type='ohlc'):
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ] 
    # create a plot
    fig, ax = plt.subplots(figsize = (10, 5))  
    # thin black bars for better long-term visualization
    if plot_type == 'bars':
        for i in sample.index:  
            plt.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                plt.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='black', linewidth=1)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                plt.vlines(x=i, ymin = sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='black', linewidth=1)  
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                plt.vlines(x=i, ymin = sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=1)  
    # regular candlesticks for better interpretation
    elif plot_type == 'candlesticks':
        for i in sample.index:  
            plt.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                plt.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='green', linewidth=3)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                plt.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='red', linewidth=3)   
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                plt.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+0.5, color='black', linewidth=3)  
    # simple line chart using the open prices (to choose close, switch the below argument)
    elif plot_type == 'line':
        if chart_type == 'ohlc':
            plt.plot(sample['open'], color='black')
        elif chart_type == 'simple_economic_indicator':
            plt.plot(sample['value'], color='black')
        elif chart_type == 'simple_financial':
            plt.plot(sample['close'], color='black')
    else:
        print('Choose between bars or candlesticks')           
    plt.grid()
    plt.show()
    plt.tight_layout()
    
def td_setup(df, source='close', perfected_source_low='low', perfected_source_high='high', final_step=9, 
              difference=4, buy_column='buy_setup', sell_column='sell_setup'):
    df[buy_column] = 0
    df[sell_column] = 0    
    # show perfected and imperfected setups
    for i in range(4, len(df)):
        # bullish setup
        if df[source].iloc[i] < df[source].iloc[i-difference]:
            df.at[df.index[i], buy_column] = df[buy_column].iloc[i-1] + 1 if df[buy_column].iloc[i-1] < final_step else 0
        else:
            df.at[df.index[i], buy_column] = 0
        # bearish setup
        if df[source].iloc[i] > df[source].iloc[i-difference]:
            df.at[df.index[i], sell_column] = df[sell_column].iloc[i-1] + 1 if df[sell_column].iloc[i-1] < final_step else 0
        else:
            df.at[df.index[i], sell_column] = 0  
    return df

def signal_chart(my_time_series, window, td=9, choice='bars', chart_type='ohlc'): 
    
    sample = my_time_series.iloc[-window:, ]

    if chart_type == 'ohlc':
        ohlc_plot(sample, window, plot_type=choice)
        ax = plt.gca()   # ← FIX

        for i in sample.index:

            # bullish countdown → green 13 below low
            if sample.loc[i, 'buy_setup'] == td:
                ax.text(
                    i,
                    sample.loc[i, 'low'] * 0.999,
                    f'{td}',
                    color='green',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='top'
                )

            # bearish countdown → red 13 above high
            elif sample.loc[i, 'sell_setup'] == td:
                ax.text(
                    i,
                    sample.loc[i, 'high'] * 1.001,
                    f'{td}',
                    color='red',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='bottom'
                )

    plt.tight_layout()
 
def performance(df, holding_period=4, signal_value=9):
    
    returns = []
    
    # We iterate through the dataframe
    # Optimization Tip: Loop is slow for large DF, but fine for backtesting logic
    for i in range(len(df) - holding_period):
        
        # LONG SIGNAL
        if df.iloc[i]['buy_setup'] == signal_value:
            entry = df.iloc[i]['close']
            exit_ = df.iloc[i + holding_period]['close']
            ret = ((exit_ - entry) / entry)  * 100000
            returns.append(ret)

        # SHORT SIGNAL
        elif df.iloc[i]['sell_setup'] == signal_value:
            entry = df.iloc[i]['close']
            exit_ = df.iloc[i + holding_period]['close']
            ret = ((entry - exit_) / entry)  * 100000
            returns.append(ret)

    returns = np.array(returns)

    if len(returns) == 0:
        return {
            "trades": 0,
            "expectancy": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "bar_efficiency": 0,
            "raw_returns": np.array([])
        }

    # --- Metric 1: Expectancy ---
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    win_rate = len(wins) / len(returns)
    loss_rate = 1 - win_rate

    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    # --- Metric 2: Profit Factor ---
    # Gross Profit / Gross Loss
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0  # Use abs() to make loss positive
    
    if gross_loss == 0:
        profit_factor = float('inf') # Infinite profit factor if no losses
    else:
        profit_factor = gross_profit / gross_loss

    # --- Metric 3: Bar Efficiency ---
    # Net Profit / Total Bars Held
    # Total bars held = number of trades * holding_period
    total_bars = len(returns) * holding_period
    net_profit = returns.sum()
    bar_efficiency = net_profit / total_bars if total_bars > 0 else 0

    return {
        "trades": len(returns),
        "expectancy": expectancy, # Assuming standardizing to pips/points
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "bar_efficiency": bar_efficiency,
        "raw_returns": returns # Needed for the statistical tests later
    }

def compare_strategies(stats_9_4, stats_8_3):
    """
    Compares two strategy dictionaries (outputs from the performance function).
    
    Null Hypothesis (H0): There is no difference between the 9-4 and 8-3 setups.
    """
    
    r9 = stats_9_4['raw_returns']
    r8 = stats_8_3['raw_returns']

    if len(r9) < 2 or len(r8) < 2:
        return "Insufficient data for statistical testing (Need N > 2)"

    # --- Metric 4: Two-Sample t-test (Independent) ---
    # Tests if the *average* return is significantly different.
    # equal_var=False performs Welch's t-test (safer as variances likely differ)
    t_stat, p_value_t = stats.ttest_ind(r9, r8, equal_var=False)

    # --- Metric 5: Kolmogorov-Smirnov Test (K-S Test) ---
    # Tests if the *shape* of the distribution is significantly different.
    # (e.g., does one have fatter tails or different skew?)
    ks_stat, p_value_ks = stats.ks_2samp(r9, r8)

    # Interpretation
    alpha = 0.05
    t_sig = "Significant" if p_value_t < alpha else "Not Significant"
    ks_sig = "Significant" if p_value_ks < alpha else "Not Significant"

    return {
        "t_test_stat": t_stat,
        "t_test_p_value": p_value_t,
        "t_test_result": f"{t_sig} Difference in Means",
        
        "ks_stat": ks_stat,
        "ks_p_value": p_value_ks,
        "ks_test_result": f"{ks_sig} Difference in Distributions"
    }