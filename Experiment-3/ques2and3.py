import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Mock numerical data
X = np.array([[25, 50000], [30, np.nan], [np.nan, 75000], [40, 90000]])

# Impute missing values first
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# StandardScaler
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X_imputed)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_imputed)

print("--- Original (Imputed) Array ---")
print(X_imputed)

print("\n--- StandardScaler Output (Mean ≈ 0, Std ≈ 1) ---")
print(X_std)

print("\n--- MinMaxScaler Output (Range strictly between [0, 1]) ---")
print(X_minmax)
