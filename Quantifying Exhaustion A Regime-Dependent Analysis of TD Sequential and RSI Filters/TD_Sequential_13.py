from Quantitative_Research_2_Master_Library import get_timeseries, calculate_td_countdown, td_setup, signal_chart, performance_from_countdown

ticker = 'USDMXN=X'
    
df = get_timeseries(ticker=ticker)

df = td_setup(df, source='close', final_step=9, difference=4, buy_column='buy_setup', sell_column='sell_setup')

df = calculate_td_countdown(df)

df = df.reset_index()

signal_chart(df, 2500, choice='line', chart_type='ohlc')

stats = performance_from_countdown(df, holding_period=50)

for k, v in stats.items():
    if k != 'returns':
        print(f"{k}: {v}")