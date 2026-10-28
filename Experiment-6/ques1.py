depths = [2, 3, 5, None]

for depth in depths:
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Max Depth = {depth}, Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

    plt.figure(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Decision Tree Visualization (Max Depth = {depth})")
    plt.show()
