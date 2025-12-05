import numpy as np
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt

# True relationship
np.random.seed(0)
X = np.linspace(0, 10, 200)
y = 3 * X + np.sin(X) + np.random.normal(0, 0.2, 200)

X = X.reshape(-1, 1)

# Symbolic Regression Model
est = SymbolicRegressor(
    population_size=1000,
    generations=20,
    function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log'],
    metric='mse',
    verbose=1
)

est.fit(X, y)

print("\nDiscovered Formula:")
print(est._program)

# Plot comparison
y_pred = est.predict(X)
plt.plot(X, y, label='True data')
plt.plot(X, y_pred, label='Symbolic prediction')
plt.legend()
plt.show()