1;

datamunge;

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n\n", DataFrame_nrows(iris), DataFrame_ncols(iris));

model = XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width");
XGBoostClassifier_print_summary(model);

printf("\nConfusion matrix (rows = actual, cols = predicted):\n");
printf("%s\n", DataFrame_to_string(XGBoostClassifier_confusion_matrix(model)));

printf("\nMisclassified rows:\n");
predictions = XGBoostClassifier_predict(model, iris);
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

Plot_save(XGBoostClassifier_plot_training_deviance(model), "xgboost_iris_training_deviance.svg");
Plot_save(XGBoostClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "xgboost_iris_decision_regions.svg");
printf("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg\n");

% XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
% grows the same deep trees but keeps every leaf weight small, producing a much smoother
% decision boundary than the lightly-regularized default.
% XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
%                    min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
heavy = XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0);
model_dev = XGBoostClassifier_training_deviance(model);
heavy_dev = XGBoostClassifier_training_deviance(heavy);
printf("\nlambda=1 (default):   training accuracy=%g%%  deviance=%g\n", XGBoostClassifier_training_accuracy(model) * 100.0, model_dev{end});
printf("lambda=50 (heavy L2): training accuracy=%g%%  deviance=%g\n", XGBoostClassifier_training_accuracy(heavy) * 100.0, heavy_dev{end});
Plot_save(XGBoostClassifier_plot_decision_regions(heavy, "Petal.Length", "Petal.Width"), "xgboost_iris_decision_regions_heavy_lambda.svg");
printf("Saved xgboost_iris_decision_regions_heavy_lambda.svg\n");
