"""
Queue Imbalance / Microprice Prediction — Educational Simulation
================================================================
Pure numpy/pandas/matplotlib. No real data. No HFT pretence.
The goal: understand the mechanics honestly.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SEED            = 42
N_STEPS         = 2000       # number of order-book snapshots (event time, not clock time)
TICK_SIZE       = 0.01       # price grid spacing
INITIAL_PRICE   = 100.0      # starting mid price
SPREAD_TICKS    = 2          # bid-ask spread in ticks (best ask = best bid + SPREAD_TICKS * TICK_SIZE)
BASE_DEPTH      = 200        # baseline queue depth (shares) at best bid/ask
DEPTH_NOISE     = 80         # standard deviation of depth fluctuations
IMBALANCE_EDGE  = 0.12       # how much true imbalance predicts next move (signal strength)
PRICE_NOISE_STD = 0.5        # noise on next-tick price move (in ticks)

# Strategy parameters
IMBALANCE_THRESHOLD = 0.25   # |I_t| above this generate a signal
HALF_SPREAD_COST    = SPREAD_TICKS / 2 * TICK_SIZE  # cost of crossing the spread (one side)
POSITION_LIMIT      = 1      # max position (shares); keeps it clean for educational purposes

rng = np.random.default_rng(SEED)

def simulate_order_book(n_steps, initial_price, tick_size, spread_ticks,
                        base_depth, depth_noise, imbalance_edge,
                        price_noise_std, rng):
    """
    Simulate a stylised top-of-book time series.

    WHAT IS REALISTIC:
    - Imbalance has some predictive power over next-tick mid-price direction.
    - Queue sizes fluctuate around a mean.
    - The spread is fixed (simplification of a real spread process).

    WHAT IS SIMPLIFIED / MISSING:
    - No queue position (we don't know where our limit order sits in the queue).
    - No order cancellations modelled explicitly.
    - Spread is constant real spreads widen in uncertainty and narrow in calm.
    - Price evolves in clock time here, not true event time.
    - No market impact from our own trades.
    - No tick-size regime effects (e.g. inverted markets, locked markets).
    - No adverse selection from informed order flow.
    """
    half_spread = spread_ticks * tick_size / 2

    mid_prices   = np.zeros(n_steps)
    bid_sizes    = np.zeros(n_steps)
    ask_sizes    = np.zeros(n_steps)
    imbalances   = np.zeros(n_steps)
    microprices  = np.zeros(n_steps)
    realised_move = np.zeros(n_steps)  # next-step mid price change

    mid = initial_price

    for t in range(n_steps):
        # Queue sizes: positive, mean-reverting around base_depth
        vb = max(10.0, base_depth + depth_noise * rng.standard_normal())
        va = max(10.0, base_depth + depth_noise * rng.standard_normal())

        # Imbalance: I_t in [-1, +1]
        I = (vb - va) / (vb + va)

        # Mid price
        mid_prices[t]  = mid
        bid_sizes[t]   = vb
        ask_sizes[t]   = va
        imbalances[t]  = I

        # Microprice: weighted mid using queue sizes
        # mp = ask * vb/(vb+va) + bid * va/(vb+va)
        bid_price = mid - half_spread
        ask_price = mid + half_spread
        mp = ask_price * (vb / (vb + va)) + bid_price * (va / (vb + va))
        microprices[t] = mp

        # Next-tick mid price move (in ticks):
        # true signal is imbalance_edge * I, plus noise
        if t < n_steps - 1:
            next_move_ticks = imbalance_edge * I + price_noise_std * rng.standard_normal()
            next_move = next_move_ticks * tick_size
            realised_move[t] = next_move
            mid = mid + next_move

    return pd.DataFrame({
        'mid':          mid_prices,
        'bid_size':     bid_sizes,
        'ask_size':     ask_sizes,
        'imbalance':    imbalances,
        'microprice':   microprices,
        'next_move':    realised_move,
    })


def run_strategy(df, threshold, half_spread_cost, position_limit):
    """
    Simple imbalance-based prediction strategy.

    Signal logic:
        I_t > +threshold   predict up   buy
        I_t < -threshold  predict down sell
        |I_t| <= threshold no trade

    Execution assumption (SIMPLIFIED / UNREALISTIC):
        We assume immediate fill at mid price  half spread.
        In reality:
        - We don't know our queue position.
        - We may not get filled at all if the queue moves against us.
        - Latency means the signal may be stale by the time we act.
        - Market impact (even tiny) is ignored here.

    Cost model:
        Each trade crosses half the spread once.
        This is a floor estimate — real costs include exchange fees,
        market impact, and adverse selection.
    """
    n = len(df)
    position    = 0
    cash        = 0.0
    trade_log   = []
    pnl_series  = np.zeros(n)

    for t in range(n - 1):   # can't trade on last bar (no next_move known)
        I   = df['imbalance'].iloc[t]
        mid = df['mid'].iloc[t]

        signal = 0
        if I >  threshold and position <  position_limit:
            signal = +1
        elif I < -threshold and position > -position_limit:
            signal = -1

        if signal != 0:
            # Simplified execution: fill at mid, deduct half-spread cost
            fill_price = mid
            cost = half_spread_cost  # always pay half spread on entry
            cash -= signal * fill_price + cost * abs(signal)
            position += signal
            trade_log.append({'t': t, 'signal': signal, 'fill': fill_price})

        # Mark-to-market pnl
        pnl_series[t] = cash + position * mid

    # Flatten position at end
    if position != 0:
        final_mid = df['mid'].iloc[-1]
        cash += position * final_mid - half_spread_cost
        position = 0
    final_pnl = cash

    # Compute trade-by-trade stats
    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    # Hit rate: did the next move agree with the signal?
    hits = 0
    total_trades = len(trade_log)
    for tr in trade_log:
        t = tr['t']
        if t < n - 1:
            move = df['next_move'].iloc[t]
            if tr['signal'] * move > 0:
                hits += 1
    hit_rate = hits / total_trades if total_trades > 0 else np.nan

    # Cumulative pnl series (mark-to-market)
    pnl_series[-1] = final_pnl
    cum_pnl = pd.Series(pnl_series).cumsum()

    return {
        'cum_pnl':       cum_pnl,
        'final_pnl':     final_pnl,
        'total_trades':  total_trades,
        'hit_rate':      hit_rate,
        'trade_df':      trade_df,
    }


def make_chart(df, results):
    fig = plt.figure(figsize=(14, 10), facecolor='#0f1117')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # mid price + microprice
    ax2 = fig.add_subplot(gs[1, 0])   # imbalance
    ax3 = fig.add_subplot(gs[1, 1])   # queue sizes
    ax4 = fig.add_subplot(gs[2, :])   # cumulative PnL

    text_color  = '#e0e0e0'
    grid_color  = '#2a2a3a'
    accent1     = '#4fc3f7'
    accent2     = '#ff8a65'
    accent3     = '#a5d6a7'

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors=text_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.grid(color=grid_color, linewidth=0.5, linestyle='--')

    x = np.arange(len(df))

    # Panel 1: mid price vs microprice
    ax1.plot(x, df['mid'],        color=accent1, lw=1.0, label='Mid Price',   alpha=0.9)
    ax1.plot(x, df['microprice'], color=accent2, lw=0.8, label='Microprice', alpha=0.7, linestyle='--')
    ax1.set_title('Mid Price vs Microprice', color=text_color, fontsize=10, pad=6)
    ax1.set_ylabel('Price', color=text_color, fontsize=8)
    leg = ax1.legend(fontsize=8, facecolor='#1e2530', labelcolor=text_color)

    # Panel 2: imbalance
    ax2.fill_between(x, df['imbalance'], 0,
                     where=df['imbalance'] > 0, color=accent3, alpha=0.6, label='Bid-heavy')
    ax2.fill_between(x, df['imbalance'], 0,
                     where=df['imbalance'] < 0, color=accent2, alpha=0.6, label='Ask-heavy')
    ax2.axhline( IMBALANCE_THRESHOLD, color='white', lw=0.7, linestyle=':')
    ax2.axhline(-IMBALANCE_THRESHOLD, color='white', lw=0.7, linestyle=':')
    ax2.set_title('Order Book Imbalance I_t', color=text_color, fontsize=10, pad=6)
    ax2.set_ylabel('I_t', color=text_color, fontsize=8)
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(fontsize=7, facecolor='#1e2530', labelcolor=text_color)

    # Panel 3: queue sizes
    ax3.plot(x, df['bid_size'], color=accent3, lw=0.7, label='Bid Size', alpha=0.8)
    ax3.plot(x, df['ask_size'], color=accent2, lw=0.7, label='Ask Size', alpha=0.8)
    ax3.set_title('Best Bid/Ask Queue Sizes', color=text_color, fontsize=10, pad=6)
    ax3.set_ylabel('Shares', color=text_color, fontsize=8)
    ax3.legend(fontsize=7, facecolor='#1e2530', labelcolor=text_color)

    # Panel 4: cumulative PnL
    cum = results['cum_pnl']
    ax4.plot(x, cum, color=accent1, lw=1.2)
    ax4.axhline(0, color='white', lw=0.6, linestyle='--')
    ax4.fill_between(x, cum, 0, where=cum >= 0, color=accent3, alpha=0.3)
    ax4.fill_between(x, cum, 0, where=cum  < 0, color=accent2, alpha=0.3)
    ax4.set_title('Cumulative Mark-to-Market PnL', color=text_color, fontsize=10, pad=6)
    ax4.set_ylabel('PnL ($)', color=text_color, fontsize=8)
    ax4.set_xlabel('Event Step', color=text_color, fontsize=8)

    stats = (f"Trades: {results['total_trades']}    "
             f"Hit Rate: {results['hit_rate']:.1%}    "
             f"Final PnL: ${results['final_pnl']:.2f}")
    fig.text(0.5, 0.01, stats, ha='center', va='bottom',
             color=text_color, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e2530', edgecolor=grid_color))

    fig.suptitle('Queue Imbalance / Microprice Strategy Educational Simulation',
                 color='white', fontsize=13, fontweight='bold', y=0.98)

    plt.savefig('/home/claude/simulation_chart.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("Chart saved.")


if __name__ == '__main__':
    df = simulate_order_book(
        n_steps        = N_STEPS,
        initial_price  = INITIAL_PRICE,
        tick_size      = TICK_SIZE,
        spread_ticks   = SPREAD_TICKS,
        base_depth     = BASE_DEPTH,
        depth_noise    = DEPTH_NOISE,
        imbalance_edge = IMBALANCE_EDGE,
        price_noise_std= PRICE_NOISE_STD,
        rng            = rng,
    )

    results = run_strategy(
        df              = df,
        threshold       = IMBALANCE_THRESHOLD,
        half_spread_cost= HALF_SPREAD_COST,
        position_limit  = POSITION_LIMIT,
    )

    print(f"Total trades  : {results['total_trades']}")
    print(f"Hit rate      : {results['hit_rate']:.1%}")
    print(f"Final PnL     : ${results['final_pnl']:.2f}")

    make_chart(df, results)
