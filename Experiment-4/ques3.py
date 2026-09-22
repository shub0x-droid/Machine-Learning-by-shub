import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([1, 4, 9, 15, 27, 38, 50, 68]) 

lin_reg = LinearRegression()
lin_reg.fit(X, y)
r2_linear = r2_score(y, lin_reg.predict(X))

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
r2_poly = r2_score(y, poly_reg.predict(X_poly))

print(f"Standard Linear Regression R2 Score: {r2_linear:.4f}")
print(f"Polynomial Regression (Degree 2) R2 Score: {r2_poly:.4f}")
