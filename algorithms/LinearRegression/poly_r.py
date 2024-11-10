from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

from mlr import MultipleLinearRegression

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.rand(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)


mlr = MultipleLinearRegression()
mlr.fit(X_poly, y)
y_pred = mlr.predict(X_poly)


plt.scatter(X, y, c='blue', label='Data')
plt.scatter(X, y_pred, c='orange', label='Polynomial fit (degree=2)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

print("Intercept:", mlr.intercept_)
print("Coefficients:", mlr.coef_)