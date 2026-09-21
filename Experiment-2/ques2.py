import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

plt.figure(figsize=(15, 10))
df.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplots of Wine Dataset Features")
plt.tight_layout()
plt.show()
