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

model = LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
LDA_print_summary(model);

printf("\nConfusion matrix (rows = actual, cols = predicted):\n");
printf("%s\n", DataFrame_to_string(LDA_confusion_matrix(model)));

newdata = DataFrame_empty();
DataFrame_add_numeric_column(newdata, "Sepal.Length", dv([5.1, 6.0, 6.5, 6.2]));
DataFrame_add_numeric_column(newdata, "Sepal.Width", dv([3.5, 2.7, 3.0, 2.8]));
DataFrame_add_numeric_column(newdata, "Petal.Length", dv([1.4, 4.5, 5.5, 4.8]));
DataFrame_add_numeric_column(newdata, "Petal.Width", dv([0.2, 1.5, 2.0, 1.8]));

printf("\nPredictions for new flowers:\n");
printf("%s\n", DataFrame_to_string(LDA_predict_frame(model, newdata)));

LDA_save_discriminant_plot(model, "lda_iris_discriminants.svg");
printf("\nSaved discriminant plot as lda_iris_discriminants.svg\n");
