from Timing_Library import get_timeseries, td_setup, signal_chart, performance, compare_strategies

hp = 20

ticker = 'USDSGD=X'
    
df_9 = get_timeseries(ticker=ticker)
df_9 = td_setup(df_9, source='close', final_step=9, difference=4, buy_column='buy_setup', sell_column='sell_setup')
df_9 = df_9.reset_index()
#signal_chart(df_9, 500, td=9, choice='line', chart_type='ohlc')
results_9 = performance(df_9, holding_period=hp, signal_value=9)
     
df_8 = get_timeseries(ticker=ticker)
df_8 = td_setup(df_8, source='close', final_step=8, difference=3, buy_column='buy_setup', sell_column='sell_setup')
df_8 = df_8.reset_index()
#signal_chart(df_8, 500, td=8, choice='line', chart_type='ohlc')
results_8 = performance(df_8, holding_period=hp, signal_value=8)
  
print("--- TD 9-4 Performance ---")
print(f"Profit Factor: {results_9['profit_factor']:.2f}")
print(f"Bar Efficiency: {results_9['bar_efficiency']:.2f}")
print(f"Hit Ratio: {results_9['win_rate']:.4f}")
print(f"Expectancy: {results_9['expectancy']:.2f}")

print("\n--- TD 8-3 Performance ---")
print(f"Profit Factor: {results_8['profit_factor']:.2f}")
print(f"Bar Efficiency: {results_8['bar_efficiency']:.2f}")
print(f"Hit Ratio: {results_8['win_rate']:.4f}")
print(f"Expectancy: {results_8['expectancy']:.2f}")

# 4. Run Statistical Comparison
comparison = compare_strategies(results_9, results_8)
print("\n--- Statistical Comparison ---")
print(f"T-Test (Means): {comparison['t_test_result']} (p={comparison['t_test_p_value']:.4f})")
print(f"KS-Test (Dist): {comparison['ks_test_result']} (p={comparison['ks_p_value']:.4f})")