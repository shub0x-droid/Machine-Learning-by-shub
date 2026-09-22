import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_area = np.array([500, 800, 1000, 1200, 1500, 1800, 2000, 2500]).reshape(-1, 1)
y_price = np.array([150, 200, 240, 280, 350, 400, 450, 560])

model = LinearRegression()
model.fit(X_area, y_price)
y_pred = model.predict(X_area)

print(f"R2 Score: {r2_score(y_price, y_pred):.4f}")

plt.scatter(X_area, y_price, color='blue', label='Actual')
plt.plot(X_area, y_pred, color='red', label='Regression Line')
plt.xlabel('House Area (sq ft)')
plt.ylabel('Price ($1000)')
plt.title('House Price Prediction')
plt.legend()
plt.show()
