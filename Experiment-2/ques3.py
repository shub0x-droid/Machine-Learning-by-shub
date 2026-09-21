import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

corr_unstack = corr_matrix.unstack()
corr_unstack = corr_unstack[corr_unstack < 1.0]
max_corr_pair = corr_unstack.idxmax()
max_corr_val = corr_unstack.max()

print(f"Strongest Positive Correlation: {max_corr_pair} with a value of {max_corr_val:.4f}")
