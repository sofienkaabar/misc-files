import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def get_timeseries(ticker=None): 
    df = yf.download(ticker, start="2024-03-01", interval="60m")
    df = df.rename(columns={"Close": "close", 'High': 'high', 'Low': 'low', 'Open': 'open'})
    df.drop(['Volume'], axis=1, inplace=True)
    df.columns = df.columns.get_level_values(0)
    return df

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

def signal_chart(my_time_series, window, choice='bars', chart_type='ohlc'): 
    
    sample = my_time_series.iloc[-window:, ]

    if chart_type == 'ohlc':
        ohlc_plot(sample, window, plot_type=choice)
        ax = plt.gca()   # ← FIX

        for i in sample.index:

            # bullish countdown → green 13 below low
            if sample.loc[i, 'buy_countdown'] == 13:
                ax.text(
                    i,
                    sample.loc[i, 'low'] * 0.999,
                    '13',
                    color='green',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='top'
                )

            # bearish countdown → red 13 above high
            elif sample.loc[i, 'sell_countdown'] == 13:
                ax.text(
                    i,
                    sample.loc[i, 'high'] * 1.001,
                    '13',
                    color='red',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='bottom'
                )

    plt.tight_layout()
 
def signal_chart_indicator(df, window):

    sample = df.iloc[-window:]

    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # =========================
    # PRICE PANEL
    # =========================
    ax_price.plot(
        sample.index,
        sample['close'],
        color='black',
        linewidth=1.2
    )

    for i in sample.index:

        price = sample.loc[i, 'close']

        # bullish 13 → below close
        if sample.loc[i, 'buy_countdown'] == 13:
            ax_price.text(
                i,
                price * 0.999,
                '13',
                color='green',
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='top'
            )

        # bearish 13 → above close
        elif sample.loc[i, 'sell_countdown'] == 13:
            ax_price.text(
                i,
                price * 1.001,
                '13',
                color='red',
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='bottom'
            )

    ax_price.set_title("Close Price")
    ax_price.grid(alpha=0.2)

    # =========================
    # RSI PANEL
    # =========================
    ax_rsi.plot(
        sample.index,
        sample['indicator'],
        color='blue',
        linewidth=1.2
    )

    ax_rsi.axhline(40, color='red', alpha=0.3)
    ax_rsi.axhline(60, color='red', alpha=0.3)
    ax_rsi.set_ylim(0, 100)

    ax_rsi.set_title("RSI")
    ax_rsi.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()
        
def performance_from_countdown(
    df,
    holding_period=5,
    signal_value=13
):
    """
    df must contain:
    ['Close', 'buy_countdown', 'sell_countdown']
    """

    returns = []
    directions = []

    for i in range(len(df) - holding_period):

        # LONG
        if df.iloc[i]['buy_countdown'] == signal_value:
            entry = df.iloc[i]['close']
            exit_ = df.iloc[i + holding_period]['close']
            ret = (exit_ - entry) / entry
            returns.append(ret)
            directions.append('long')

        # SHORT
        elif df.iloc[i]['sell_countdown'] == signal_value:
            entry = df.iloc[i]['close']
            exit_ = df.iloc[i + holding_period]['close']
            ret = (entry - exit_) / entry
            returns.append(ret)
            directions.append('short')

    returns = np.array(returns)

    if len(returns) == 0:
        raise ValueError("No trades generated.")

    # Expectancy
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    expectancy = (
        (len(wins) / len(returns)) * wins.mean() -
        (len(losses) / len(returns)) * abs(losses.mean())
        if len(losses) > 0 else wins.mean()
    )

    # Sharpe (non-annualized)
    sharpe = returns.mean() / returns.std(ddof=1)

    return {
        "trades": len(returns),
        "expectancy": expectancy * 100000,
        "win_rate": len(wins) / len(returns),
    }

def performance_from_countdown_rsi(
    df,
    holding_period=5,
    signal_value=13,
    lower_barrier=30,
    upper_barrier=70
    
):
    """
    df must contain:
    ['Close', 'buy_countdown', 'sell_countdown']
    """

    returns = []
    directions = []

    for i in range(len(df) - holding_period):

        # LONG
        if df.iloc[i]['buy_countdown'] == signal_value and df.iloc[i]['indicator'] < lower_barrier:
            entry = df.iloc[i]['close']
            exit_ = df.iloc[i + holding_period]['close']
            ret = (exit_ - entry) / entry
            returns.append(ret)
            directions.append('long')

        # SHORT
        elif df.iloc[i]['sell_countdown'] == signal_value and df.iloc[i]['indicator'] > upper_barrier:
            entry = df.iloc[i]['close']
            exit_ = df.iloc[i + holding_period]['close']
            ret = (entry - exit_) / entry
            returns.append(ret)
            directions.append('short')

    returns = np.array(returns)

    if len(returns) == 0:
        raise ValueError("No trades generated.")

    # Expectancy
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    expectancy = (
        (len(wins) / len(returns)) * wins.mean() -
        (len(losses) / len(returns)) * abs(losses.mean())
        if len(losses) > 0 else wins.mean()
    )

    return {
        "trades": len(returns),
        "expectancy": expectancy * 100000,
        "win_rate": len(wins) / len(returns),
    }




def calculate_td_countdown(df):
    # Initialize output columns with 0
    df['buy_countdown'] = 0
    df['sell_countdown'] = 0

    # --- Buy Countdown State ---
    buy_count = 0
    buy_active = False
    buy_node_8_close = None

    # --- Sell Countdown State ---
    sell_count = 0
    sell_active = False
    sell_node_8_close = None

    # Loop through the dataframe starting at index 2 (need 2 bars lookback)
    for i in range(2, len(df)):
        
        # Current and historical values
        close_price = df['close'].iloc[i]
        
        # Ensure we have access to high/low. If not, use close as fallback (though less accurate)
        low_price_2_ago = df['low'].iloc[i-2] if 'low' in df.columns else df['close'].iloc[i-2]
        high_price_2_ago = df['high'].iloc[i-2] if 'high' in df.columns else df['close'].iloc[i-2]

        prev_close_1 = df['close'].iloc[i-1]
        prev_close_2 = df['close'].iloc[i-2]
        
        # ==========================================
        # TD BUY COUNTDOWN
        # ==========================================
        
        # 1. CANCELLATION: Contrary Sell Setup 9
        if df['sell_setup'].iloc[i] == 9:
            buy_active = False
            buy_count = 0
            buy_node_8_close = None

        # 2. RECYCLE / START: Buy Setup 9
        # "The close of the ninth bar must be lower than the previous two bars"
        if df['buy_setup'].iloc[i] == 9:
            if close_price < prev_close_1 and close_price < prev_close_2:
                # Reset/Start Countdown at 1
                buy_active = True
                buy_count = 1
                buy_node_8_close = None # Reset node 8
                df.loc[df.index[i], 'buy_countdown'] = 1
                # Important: If we start a new count here, we skip the standard increment logic below
            else:
                # If setup 9 appears but fails price check, what happens?
                # Usually standard TD cancels the old count but doesn't start a new one, 
                # or ignores it. Based on "Recycle occurs", we assume it resets.
                buy_active = False
                buy_count = 0
        
        # 3. COUNTING (Only if active and didn't just recycle this same bar)
        elif buy_active:
            # Rule: Close must be below the LOW of 2 bars earlier
            if close_price < low_price_2_ago:
                
                # If we are at 12, check for the Qualifier
                if buy_count == 12:
                    # Rule: Bar 13 must be <= Bar 8 Close
                    if close_price <= buy_node_8_close:
                        # SUCCESS: Mark 13, then STOP.
                        df.loc[df.index[i], 'buy_countdown'] = 13
                        
                        # Stop counting immediately
                        buy_active = False 
                        buy_count = 0 
                    else:
                        # Condition met (price < low[i-2]) BUT Qualifier failed.
                        # We stay at 12 (waiting).
                        # We write 12 to show we are still in the countdown waiting for the dip.
                        df.loc[df.index[i], 'buy_countdown'] = 12
                
                else:
                    # Normal increment
                    buy_count += 1
                    df.loc[df.index[i], 'buy_countdown'] = buy_count
                    
                    # Capture Bar 8 Close
                    if buy_count == 8:
                        buy_node_8_close = close_price
            
            else:
                # Bar did not qualify. 
                # Since "does NOT have to be consecutive", we simply do nothing.
                # The count stays the same (e.g., at 5).
                # We write 0 to the column to indicate this specific bar isn't part of the count.
                df.loc[df.index[i], 'buy_countdown'] = 0

        # ==========================================
        # TD SELL COUNTDOWN (Symmetrical)
        # ==========================================
        
        # 1. CANCELLATION
        if df['buy_setup'].iloc[i] == 9:
            sell_active = False
            sell_count = 0
            sell_node_8_close = None

        # 2. RECYCLE
        if df['sell_setup'].iloc[i] == 9:
            if close_price > prev_close_1 and close_price > prev_close_2:
                sell_active = True
                sell_count = 1
                sell_node_8_close = None
                df.loc[df.index[i], 'sell_countdown'] = 1
            else:
                sell_active = False
                sell_count = 0
        
        # 3. COUNTING
        elif sell_active:
            # Rule: Close > High of 2 bars earlier
            if close_price > high_price_2_ago:
                
                if sell_count == 12:
                    # Qualifier: Bar 13 >= Bar 8 Close
                    if close_price >= sell_node_8_close:
                        df.loc[df.index[i], 'sell_countdown'] = 13
                        # STOP
                        sell_active = False
                        sell_count = 0
                    else:
                        df.loc[df.index[i], 'sell_countdown'] = 12
                else:
                    sell_count += 1
                    df.loc[df.index[i], 'sell_countdown'] = sell_count
                    
                    if sell_count == 8:
                        sell_node_8_close = close_price
            else:
                df.loc[df.index[i], 'sell_countdown'] = 0

    return df