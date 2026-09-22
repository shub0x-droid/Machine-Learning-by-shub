import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_multi = np.array([
    [500, 1],
    [800, 2],
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4]
])
y_price = np.array([150, 200, 240, 280, 350, 400])

model = LinearRegression()
model.fit(X_multi, y_price)

y_pred = model.predict(X_multi)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"R2 Score: {r2_score(y_price, y_pred):.4f}")
