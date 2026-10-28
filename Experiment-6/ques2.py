
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=2,
    random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Visualization (Max Depth = 2, Entropy)")
plt.show()
