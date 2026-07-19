1;

datamunge;

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n\n", DataFrame_nrows(iris), DataFrame_ncols(iris));

model = GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width");
GBMClassifier_print_summary(model);

printf("\nConfusion matrix (rows = actual, cols = predicted):\n");
printf("%s\n", DataFrame_to_string(GBMClassifier_confusion_matrix(model)));

printf("\nMisclassified rows:\n");
predictions = GBMClassifier_predict(model, iris);
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

Plot_save(GBMClassifier_plot_training_deviance(model), "gbm_iris_training_deviance.svg");
Plot_save(GBMClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "gbm_iris_decision_regions.svg");
printf("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n");

% A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
% at the training loss, gradually sharpening the decision boundary.
% GBMClassifier(data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
few = GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
few_dev = GBMClassifier_training_deviance(few);
model_dev = GBMClassifier_training_deviance(model);
printf("\n5-round ensemble:   training accuracy=%g%%  deviance=%g\n", GBMClassifier_training_accuracy(few) * 100.0, few_dev{end});
printf("100-round ensemble: training accuracy=%g%%  deviance=%g\n", GBMClassifier_training_accuracy(model) * 100.0, model_dev{end});
Plot_save(GBMClassifier_plot_decision_regions(few, "Petal.Length", "Petal.Width"), "gbm_iris_decision_regions_5rounds.svg");
printf("Saved gbm_iris_decision_regions_5rounds.svg\n");
