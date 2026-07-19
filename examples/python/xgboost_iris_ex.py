from pydatamunge import datamunge as dm

iris = dm.DataFrame.iris()
print(f"iris: {iris.nrows()} rows x {iris.ncols()} cols\n")

model = dm.XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

model.plot_training_deviance().save("xgboost_iris_training_deviance.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg")
print("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg")

# XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
# grows the same deep trees but keeps every leaf weight small, producing a much smoother
# decision boundary than the lightly-regularized default.
# XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
#                    min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
heavy = dm.XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)
print(f"\nlambda=1 (default):   training accuracy={model.training_accuracy() * 100.0}%"
      f"  deviance={model.training_deviance()[-1]}")
print(f"lambda=50 (heavy L2): training accuracy={heavy.training_accuracy() * 100.0}%"
      f"  deviance={heavy.training_deviance()[-1]}")
heavy.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg")
print("Saved xgboost_iris_decision_regions_heavy_lambda.svg")
