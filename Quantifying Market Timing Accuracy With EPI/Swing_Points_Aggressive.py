import numpy as np
from research_1_libraries import get_timeseries, rsi, create_signals_aggressive, create_performance_columns
from research_1_libraries import optimal_horizon_after_swings, swing_detect, global_performance_metrics
import matplotlib.pyplot as plt

df = get_timeseries(ticker='USDCAD=X', period="5y")

df = rsi(df, source='close', output_name='indicator', rsi_lookback=5, smoothed=False)

df = create_signals_aggressive(df, lower_barrier=20, upper_barrier=80)

df = swing_detect(df, swing_lookback=10)

def signal_swing_precision(df, window=3):
    total_signals = 0
    valid_signals = 0

    # swing exists where value is NOT NaN
    swing_low_pos  = set(np.where(df["swing_low"].notna())[0])
    swing_high_pos = set(np.where(df["swing_high"].notna())[0])

    n = len(df)

    for pos in range(n):
        row = df.iloc[pos]

        # ---------- BULLISH SIGNAL ----------
        if row["bullish_signal"] == 1:
            total_signals += 1

            for k in range(-window, window + 1):
                check_pos = pos + k
                if 0 <= check_pos < n and check_pos in swing_low_pos:
                    valid_signals += 1
                    break

        # ---------- BEARISH SIGNAL ----------
        if row["bearish_signal"] == 1:
            total_signals += 1

            for k in range(-window, window + 1):
                check_pos = pos + k
                if 0 <= check_pos < n and check_pos in swing_high_pos:
                    valid_signals += 1
                    break

    precision = valid_signals / total_signals if total_signals > 0 else np.nan
    return precision, valid_signals, total_signals


precision, valid, total = signal_swing_precision(df, window=0)

print("SIGNAL–SWING PRECISION")
print("Valid signals:", valid)
print("Total signals:", total)
print("Precision:", round(precision, 2))

df_to_plot = df.tail(500)

def plot_signals_and_swings(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- price ----
    ax.plot(df.index, df["close"], label="Price", color="black")

    # ---- dynamic offset (VERY small) ----
    price_range = df["close"].max() - df["close"].min()
    offset = price_range * 0.002

    # ---- swing points (NON-NaN values) ----
    swing_highs = df["swing_high"].notna()
    swing_lows  = df["swing_low"].notna()

    ax.scatter(
        df.index[swing_highs],
        df.loc[swing_highs, "swing_high"] + offset,
        s=50,
        marker="o",
        label="Swing High"
    )

    ax.scatter(
        df.index[swing_lows],
        df.loc[swing_lows, "swing_low"] - offset,
        s=50,
        marker="o",
        label="Swing Low"
    )

    # ---- bullish signals ----
    bulls = df["bullish_signal"] == 1
    ax.scatter(
        df.index[bulls],
        df.loc[bulls, "close"] - 2 * offset,
        marker="^",
        s=70,
        label="Bullish Signal"
    )

    # ---- bearish signals ----
    bears = df["bearish_signal"] == 1
    ax.scatter(
        df.index[bears],
        df.loc[bears, "close"] + 2 * offset,
        marker="v",
        s=70,
        label="Bearish Signal"
    )

    ax.legend()
    ax.set_title("Price, Swing Points, and Trading Signals")
    ax.grid(True)

    plt.show()

df_to_plot  = df_to_plot.reset_index()

#plot_signals_and_swings(df_to_plot)

best_low_h, best_high_h = optimal_horizon_after_swings(df, max_horizon=50, log_returns=True)

df = create_performance_columns(df, bull_hold=10, bear_hold=10)

stats = global_performance_metrics(df)

for k, v in stats.items():
    print(f"{k}: {v}")
    
'''    
def compute_equity_curve(df):
    """
    Builds a simple equity curve from bullish_perf and bearish_perf.
    """

    # combine trade returns into a single series
    trades = (
        df["bullish_perf"]
        .fillna(0)
        .add(df["bearish_perf"].fillna(0))
    )

    equity_curve = trades.cumsum()

    return equity_curve

df["equity"] = compute_equity_curve(df)

plt.figure(figsize=(10, 4))
plt.plot(df.index, df["equity"])
plt.title("Equity Curve")
plt.grid(True)
plt.show()
'''
