from pydatamunge import datamunge as dm

iris = dm.DataFrame.iris()
print(f"iris: {iris.nrows()} rows x {iris.ncols()} cols\n")

model = dm.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width")
model.print_summary()

print("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):")
print(model.confusion_matrix().to_string())

print("\nLeave-one-out misclassified rows:")
fitted = model.predict(iris)
misclassified = 0
for i in range(iris.nrows()):
    actual = iris.string_at("Species", i)
    if fitted[i] == actual:
        continue
    misclassified += 1
    print(f"  row {i}: Petal.Length={iris.numeric_at('Petal.Length', i)} "
          f"Petal.Width={iris.numeric_at('Petal.Width', i)}  actual={actual}  predicted={fitted[i]}")
print(f"{misclassified} of {iris.nrows()} misclassified ({100.0 * misclassified / iris.nrows()}%)")

model.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg")
print("\nSaved knn_iris_decision_regions_k5.svg")

# k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
# every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
# underfit boundary). KNNClassifier(data, formula, k, metric, weighted, standardize)
k1 = dm.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 1)
print(f"\nk=1  leave-one-out accuracy: {k1.training_accuracy() * 100.0}%")
k1.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k1.svg")
print("Saved knn_iris_decision_regions_k1.svg")

k25 = dm.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 25)
print(f"\nk=25 leave-one-out accuracy: {k25.training_accuracy() * 100.0}%")
k25.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k25.svg")
print("Saved knn_iris_decision_regions_k25.svg")
