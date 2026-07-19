1;

datamunge;

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n\n", DataFrame_nrows(iris), DataFrame_ncols(iris));

model = RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width");
RandomForestClassifier_print_summary(model);

printf("\nConfusion matrix (rows = actual, cols = predicted):\n");
printf("%s\n", DataFrame_to_string(RandomForestClassifier_confusion_matrix(model)));

printf("\nMisclassified rows:\n");
predictions = RandomForestClassifier_predict(model, iris);
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

Plot_save(RandomForestClassifier_plot_classification(model, iris, "Petal.Length", "Petal.Width"), "forest_iris_classification.svg");
Plot_save(RandomForestClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "forest_iris_decision_regions.svg");
printf("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg\n");

% A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
% are added -- the defining random forest effect a single decision tree cannot demonstrate.
% RandomForestClassifier(data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf,
%                         max_features, criterion, bootstrap, sample_fraction, seed)
small_forest = RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
printf("\n5-tree forest:   training accuracy=%g%%  OOB accuracy=%g%%\n", ...
       RandomForestClassifier_training_accuracy(small_forest) * 100.0, RandomForestClassifier_oob_accuracy(small_forest) * 100.0);
printf("100-tree forest: training accuracy=%g%%  OOB accuracy=%g%%\n", ...
       RandomForestClassifier_training_accuracy(model) * 100.0, RandomForestClassifier_oob_accuracy(model) * 100.0);
Plot_save(RandomForestClassifier_plot_decision_regions(small_forest, "Petal.Length", "Petal.Width"), "forest_iris_decision_regions_5trees.svg");
printf("Saved forest_iris_decision_regions_5trees.svg\n");
