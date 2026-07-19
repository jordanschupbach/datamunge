1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n\n", DataFrame_nrows(iris), DataFrame_ncols(iris));

printf("=== RBF kernel (default) ===\n");
rbf_model = SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
SVM_print_summary(rbf_model);
printf("\nConfusion matrix (rows = actual, cols = predicted):\n");
printf("%s\n", DataFrame_to_string(SVM_confusion_matrix(rbf_model)));

printf("\n=== Linear kernel, for comparison ===\n");
linear_model = SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear");
printf("Training accuracy: %g%%\n", SVM_training_accuracy(linear_model) * 100.0);
printf("Support vectors: %d\n", SVM_num_support_vectors(linear_model));

newdata = DataFrame_empty();
DataFrame_add_numeric_column(newdata, "Sepal.Length", dv([5.1, 6.0, 6.5, 6.2]));
DataFrame_add_numeric_column(newdata, "Sepal.Width", dv([3.5, 2.7, 3.0, 2.8]));
DataFrame_add_numeric_column(newdata, "Petal.Length", dv([1.4, 4.5, 5.5, 4.8]));
DataFrame_add_numeric_column(newdata, "Petal.Width", dv([0.2, 1.5, 2.0, 1.8]));

printf("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n");
printf("%s\n", DataFrame_to_string(SVM_predict_frame(rbf_model, newdata)));
