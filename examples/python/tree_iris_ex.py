from pydatamunge import datamunge as dm

iris = dm.DataFrame.iris()
print(f"iris: {iris.nrows()} rows x {iris.ncols()} cols\n")

model = dm.DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width")
model.print_summary()

print("\nConfusion matrix (rows = actual, cols = predicted):")
print(model.confusion_matrix().to_string())

print("\nMisclassified rows:")
predictions = model.predict(iris)
misclassified = 0
for i in range(iris.nrows()):
    actual = iris.string_at("Species", i)
    if predictions[i] == actual:
        continue
    misclassified += 1
    print(f"  row {i}: Petal.Length={iris.numeric_at('Petal.Length', i)} "
          f"Petal.Width={iris.numeric_at('Petal.Width', i)}  actual={actual}  predicted={predictions[i]}")
print(f"{misclassified} of {iris.nrows()} misclassified ({100.0 * misclassified / iris.nrows()}%)")

model.plot_classification(iris, "Petal.Length", "Petal.Width").save("tree_iris_classification.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg")
print("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg")

# A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
shallow = dm.DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2)
print(f"\nDepth-2 tree training accuracy: {shallow.training_accuracy() * 100.0}% ({shallow.leaf_count()} leaves)")
shallow.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions_depth2.svg")
print("Saved tree_iris_decision_regions_depth2.svg")
