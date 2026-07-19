1;

datamunge;

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n\n", DataFrame_nrows(iris), DataFrame_ncols(iris));

model = DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width");
DecisionTreeClassifier_print_summary(model);

printf("\nConfusion matrix (rows = actual, cols = predicted):\n");
printf("%s\n", DataFrame_to_string(DecisionTreeClassifier_confusion_matrix(model)));

printf("\nMisclassified rows:\n");
predictions = DecisionTreeClassifier_predict(model, iris);
misclassified = 0;
n = DataFrame_nrows(iris);
for i = 0:(n - 1)
  actual = DataFrame_string_at(iris, "Species", i);
  pred = predictions{i + 1};
  if !strcmp(pred, actual)
    misclassified = misclassified + 1;
    printf("  row %d: Petal.Length=%g Petal.Width=%g  actual=%s  predicted=%s\n", i, ...
           DataFrame_numeric_at(iris, "Petal.Length", i), DataFrame_numeric_at(iris, "Petal.Width", i), actual, pred);
  end
end
printf("%d of %d misclassified (%g%%)\n", misclassified, n, 100.0 * misclassified / n);

Plot_save(DecisionTreeClassifier_plot_classification(model, iris, "Petal.Length", "Petal.Width"), "tree_iris_classification.svg");
Plot_save(DecisionTreeClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "tree_iris_decision_regions.svg");
printf("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n");

% A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
shallow = DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2);
printf("\nDepth-2 tree training accuracy: %g%% (%d leaves)\n", DecisionTreeClassifier_training_accuracy(shallow) * 100.0, DecisionTreeClassifier_leaf_count(shallow));
Plot_save(DecisionTreeClassifier_plot_decision_regions(shallow, "Petal.Length", "Petal.Width"), "tree_iris_decision_regions_depth2.svg");
printf("Saved tree_iris_decision_regions_depth2.svg\n");
