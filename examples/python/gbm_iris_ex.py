from pydatamunge import datamunge as dm

iris = dm.DataFrame.iris()
print(f"iris: {iris.nrows()} rows x {iris.ncols()} cols\n")

model = dm.GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

model.plot_training_deviance().save("gbm_iris_training_deviance.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg")
print("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg")

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# GBMClassifier(data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
few = dm.GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5)
print(f"\n5-round ensemble:   training accuracy={few.training_accuracy() * 100.0}%"
      f"  deviance={few.training_deviance()[-1]}")
print(f"100-round ensemble: training accuracy={model.training_accuracy() * 100.0}%"
      f"  deviance={model.training_deviance()[-1]}")
few.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg")
print("Saved gbm_iris_decision_regions_5rounds.svg")
