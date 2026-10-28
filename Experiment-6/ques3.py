from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

depths = [2, 3, 5, None]

train_accuracies = []
test_accuracies = []

for depth in depths:
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    model.fit(x_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(x_train))
    test_acc = accuracy_score(y_test, model.predict(x_test))

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

depth_labels = ["2", "3", "5", "None"]

plt.figure(figsize=(9, 5))
plt.plot(depth_labels, train_accuracies, marker="o",
         label="Training Accuracy", linewidth=2)
plt.plot(depth_labels, test_accuracies, marker="s",
         label="Testing Accuracy", linewidth=2)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy for Different Tree Depths")
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
