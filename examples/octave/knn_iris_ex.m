1;

datamunge;

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n\n", DataFrame_nrows(iris), DataFrame_ncols(iris));

model = KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width");
KNNClassifier_print_summary(model);

printf("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n");
printf("%s\n", DataFrame_to_string(KNNClassifier_confusion_matrix(model)));

printf("\nLeave-one-out misclassified rows:\n");
fitted = KNNClassifier_predict(model, iris);
misclassified = 0;
n = DataFrame_nrows(iris);
for i = 0:(n - 1)
  actual = DataFrame_string_at(iris, "Species", i);
  pred = fitted{i + 1};
  if !strcmp(pred, actual)
    misclassified = misclassified + 1;
    printf("  row %d: Petal.Length=%g Petal.Width=%g  actual=%s  predicted=%s\n", i, ...
           DataFrame_numeric_at(iris, "Petal.Length", i), DataFrame_numeric_at(iris, "Petal.Width", i), actual, pred);
  end
end
printf("%d of %d misclassified (%g%%)\n", misclassified, n, 100.0 * misclassified / n);

Plot_save(KNNClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "knn_iris_decision_regions_k5.svg");
printf("\nSaved knn_iris_decision_regions_k5.svg\n");

% k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
% every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
% underfit boundary). KNNClassifier(data, formula, k, metric, weighted, standardize)
k1 = KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 1);
printf("\nk=1  leave-one-out accuracy: %g%%\n", KNNClassifier_training_accuracy(k1) * 100.0);
Plot_save(KNNClassifier_plot_decision_regions(k1, "Petal.Length", "Petal.Width"), "knn_iris_decision_regions_k1.svg");
printf("Saved knn_iris_decision_regions_k1.svg\n");

k25 = KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 25);
printf("\nk=25 leave-one-out accuracy: %g%%\n", KNNClassifier_training_accuracy(k25) * 100.0);
Plot_save(KNNClassifier_plot_decision_regions(k25, "Petal.Length", "Petal.Width"), "knn_iris_decision_regions_k25.svg");
printf("Saved knn_iris_decision_regions_k25.svg\n");
