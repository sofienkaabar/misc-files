"""
Fourier-based time series forecasting with proper methodology.

Key principles:
- Use log price for better residual behavior
- Rolling linear regression for trend (extrapolatable)
- Hann-windowed Fourier on residuals only
- Fixed small k components (no re-optimization)
- Always benchmark against dumb baselines
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ForecastResult:
    forecast: np.ndarray          # Final forecast (log price)
    trend_forecast: np.ndarray    # Trend component
    residual_forecast: np.ndarray # Fourier residual component
    persistence: np.ndarray       # Baseline: last value repeated
    slope_continuation: np.ndarray # Baseline: linear extrapolation


def rolling_linear_regression(y: np.ndarray, window: int) -> tuple[float, float]:
    """
    Fit linear regression on the last `window` points.
    Returns (intercept, slope) where intercept is at t=window-1.
    """
    n = min(len(y), window)
    y_win = y[-n:]
    t = np.arange(n)
    
    t_mean = t.mean()
    y_mean = y_win.mean()
    
    slope = np.sum((t - t_mean) * (y_win - y_mean)) / np.sum((t - t_mean) ** 2)
    intercept = y_mean - slope * t_mean
    
    # Return level at end of window and slope
    level = intercept + slope * (n - 1)
    return level, slope


def extract_top_k_fourier(residual: np.ndarray, k: int) -> list[tuple[float, float, float]]:
    """
    Apply Hann window and extract top k Fourier components.
    Returns list of (frequency_idx, amplitude, phase) tuples.
    """
    n = len(residual)
    
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed = residual * window
    
    # FFT
    fft = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n)
    
    # Get magnitudes (skip DC component at index 0)
    magnitudes = np.abs(fft[1:])
    
    # Find top k components by magnitude
    top_indices = np.argsort(magnitudes)[-k:][::-1]
    
    components = []
    for idx in top_indices:
        actual_idx = idx + 1  # Offset for skipped DC
        amp = 2 * np.abs(fft[actual_idx]) / (n * np.mean(window))  # Correct for window
        phase = np.angle(fft[actual_idx])
        freq = freqs[actual_idx]
        components.append((freq, amp, phase))
    
    return components


def project_fourier_components(
    components: list[tuple[float, float, float]],
    n_history: int,
    n_forecast: int
) -> np.ndarray:
    """
    Extend sinusoids forward from end of history window.
    """
    t_forecast = np.arange(n_history, n_history + n_forecast)
    projection = np.zeros(n_forecast)
    
    for freq, amp, phase in components:
        projection += amp * np.cos(2 * np.pi * freq * t_forecast + phase)
    
    return projection


def forecast_fourier(
    prices: np.ndarray,
    n_steps: int,
    trend_window: int = 60,
    fourier_window: int = 120,
    k_components: int = 5
) -> ForecastResult:
    """
    Forecast n_steps ahead using trend + Fourier residual decomposition.
    
    Parameters:
    -----------
    prices : array
        Raw price series
    n_steps : int
        Number of steps to forecast
    trend_window : int
        Window for rolling linear regression (default 60)
    fourier_window : int
        Window for Fourier analysis (default 120)
    k_components : int
        Number of Fourier components to keep (default 5, typical range 3-10)
    
    Returns:
    --------
    ForecastResult with forecast and baselines
    """
    # Work in log space
    log_prices = np.log(prices)
    
    # Step 1: Fit trend via rolling linear regression
    level, slope = rolling_linear_regression(log_prices, trend_window)
    
    # Step 2: Compute residual = log_price - trend
    n = len(log_prices)
    trend_window_actual = min(n, trend_window)
    t_trend = np.arange(trend_window_actual)
    trend_in_window = (level - slope * (trend_window_actual - 1)) + slope * t_trend
    
    # Use fourier_window for residual analysis
    fourier_n = min(n, fourier_window)
    log_recent = log_prices[-fourier_n:]
    
    # Reconstruct trend over fourier window
    trend_full = level + slope * (np.arange(fourier_n) - (fourier_n - 1))
    residual = log_recent - trend_full
    
    # Step 3: Fourier decomposition on residual (with Hann window)
    components = extract_top_k_fourier(residual, k_components)
    
    # Step 4: Project components forward
    residual_forecast = project_fourier_components(components, fourier_n, n_steps)
    
    # Step 5: Extrapolate trend
    t_forecast = np.arange(1, n_steps + 1)
    trend_forecast = level + slope * t_forecast
    
    # Step 6: Combine
    forecast = trend_forecast + residual_forecast
    
    # Baselines
    persistence = np.full(n_steps, log_prices[-1])
    slope_continuation = log_prices[-1] + slope * t_forecast
    
    return ForecastResult(
        forecast=forecast,
        trend_forecast=trend_forecast,
        residual_forecast=residual_forecast,
        persistence=persistence,
        slope_continuation=slope_continuation
    )


def evaluate_forecast(actual: np.ndarray, result: ForecastResult) -> dict:
    """
    Compare forecast against baselines using RMSE.
    """
    log_actual = np.log(actual)
    
    def rmse(pred, true):
        return np.sqrt(np.mean((pred - true) ** 2))
    
    return {
        'fourier_rmse': rmse(result.forecast, log_actual),
        'persistence_rmse': rmse(result.persistence, log_actual),
        'slope_rmse': rmse(result.slope_continuation, log_actual),
        'fourier_vs_persistence': rmse(result.forecast, log_actual) / rmse(result.persistence, log_actual),
        'fourier_vs_slope': rmse(result.forecast, log_actual) / rmse(result.slope_continuation, log_actual),
    }

import yfinance as yf
import pandas as pd

def get_timeseries(ticker: str = None, ema_span= None) -> pd.DataFrame: 
    df = yf.download(ticker, start="2000-01-01", interval="1d")
    df = df[["Close", "Low", "High", "Open"]].rename(columns={"Close": "close", "Low": "low", "High": "high", "Open": "open"})
    df.columns = df.columns.get_level_values(0)
    if ema_span is not None:
        df['close_raw'] = df['close']        
        df['close'] = df['close'].ewm(span=ema_span, adjust=False).mean()
    return df

symbol = "TSLA"
df = get_timeseries(ticker=symbol)    
df = df['close']
prices = np.array(df)
# Demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Split: use first 280 for fitting, last 20 for validation
    n_forecast = 50
    len_df = len(prices)
    train_prices = prices[:len_df-n_forecast]
    test_prices = prices[len_df-n_forecast:]
    
    
    # Forecast
    result = forecast_fourier(
        train_prices,
        n_steps=n_forecast,
        trend_window=20,
        fourier_window=50,
        k_components=50
    )
    
    # Evaluate
    metrics = evaluate_forecast(test_prices, result)
    
    print("Forecast Evaluation (RMSE in log space):")
    print(f"  Fourier model:     {metrics['fourier_rmse']:.6f}")
    print(f"  Persistence:       {metrics['persistence_rmse']:.6f}")
    print(f"  Slope continuation:{metrics['slope_rmse']:.6f}")
    print()
    print("Relative performance (< 1.0 means Fourier wins):")
    print(f"  vs Persistence:    {metrics['fourier_vs_persistence']:.3f}")
    print(f"  vs Slope:          {metrics['fourier_vs_slope']:.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    
    # Top plot: Full context + forecast
    ax1 = axes[0]
    n_context = 60  # Show last 60 points of training data for context
    
    t_context = np.arange(-n_context, 0)
    t_forecast = np.arange(0, n_forecast)
    
    # Training context
    ax1.plot(t_context, train_prices[-n_context:], 'k-', lw=1.5, label='History')
    
    # Actual test data
    ax1.plot(t_forecast, test_prices, 'ko-', lw=2, markersize=5, label='Actual')
    
    # Forecasts (convert from log space)
    ax1.plot(t_forecast, np.exp(result.forecast), 'b-', lw=2, label='Fourier')
    ax1.plot(t_forecast, np.exp(result.persistence), 'r--', lw=1.5, alpha=0.7, label='Persistence')
    ax1.plot(t_forecast, np.exp(result.slope_continuation), 'g--', lw=1.5, alpha=0.7, label='Slope')
    
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Steps from forecast origin')
    ax1.set_ylabel('Price')
    ax1.set_title('Forecast Comparison')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Decomposition of Fourier forecast
    ax2 = axes[1]
    
    ax2.plot(t_forecast, np.exp(result.trend_forecast), 'c-', lw=2, label='Trend extrapolation')
    ax2.plot(t_forecast, result.residual_forecast, 'm-', lw=2, label='Residual projection')
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Steps from forecast origin')
    ax2.set_ylabel('Value')
    ax2.set_title('Forecast Decomposition (Trend in price space, Residual in log space)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add metrics annotation
    metrics_text = (
        f"RMSE (log): Fourier={metrics['fourier_rmse']:.4f}, "
        f"Persist={metrics['persistence_rmse']:.4f}, Slope={metrics['slope_rmse']:.4f}\n"
        f"Fourier vs Persist: {metrics['fourier_vs_persistence']:.3f}x, "
        f"Fourier vs Slope: {metrics['fourier_vs_slope']:.3f}x"
    )
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    
    print("\nPlot saved to forecast_comparison.png")