from Quantitative_Research_2_Master_Library import get_timeseries, rsi, calculate_td_countdown, td_setup, signal_chart_indicator, performance_from_countdown_rsi

ticker = 'EURUSD=X'
    
df = get_timeseries(ticker=ticker)

df = td_setup(df, source='close', final_step=9, difference=4, buy_column='buy_setup', sell_column='sell_setup')

df = calculate_td_countdown(df)

df = rsi(df, source='close', output_name='indicator', rsi_lookback=14, smoothed=False)

df = df.reset_index()

signal_chart_indicator(df, 2500)

stats = performance_from_countdown_rsi(df, holding_period=50, lower_barrier=40, upper_barrier=60)

for k, v in stats.items():
    if k != 'returns':
        print(f"{k}: {v}")