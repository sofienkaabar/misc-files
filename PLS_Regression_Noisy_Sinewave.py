import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# 1. Generate Synthetic Data
t = np.linspace(0, 50, 1000)
data = np.sin(t) + np.random.normal(0, 0.05, size=1000)  # Sine wave + noise

# 2. Create Lagged Features
def create_lags(series, n_lags):
    X, y = [], []
    for i in range(len(series) - n_lags):
        X.append(series[i:i + n_lags])
        y.append(series[i + n_lags])
    return np.array(X), np.array(y)

n_lags = 200  # Use 20 previous points to predict the next one
X, y = create_lags(data, n_lags)

# 3. Split into Train and Test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Apply PLS Regression
# We'll use 3 components to show how PLS reduces 20 features to 3 latent variables
pls = PLSRegression(n_components=200)
pls.fit(X_train, y_train)

# 5. Predict
y_pred = pls.predict(X_test)

# 6. Visualize
plt.figure(figsize=(12, 5))
plt.plot(y_test[:100], label="Actual (Test)", color='dodgerblue', alpha=0.7)
plt.plot(y_pred[:100], label="PLS Prediction", color='crimson', linestyle='--')
plt.title(f"PLS Regression: Lagged Sine Wave Prediction ({n_lags} Lags -> 3 Components)")
plt.legend()
plt.show()

print(f"Test MSE: {mean_squared_error(y_test, y_pred):.5f}")