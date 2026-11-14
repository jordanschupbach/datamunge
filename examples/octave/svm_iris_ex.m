% CMD: just TARGET=svm_iris_ex run-octave
% Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.

formula = 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width';
iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n\n', iris.nrows (), iris.ncols ());

printf ('=== RBF kernel (default) ===\n');
rbf_model = datamunge.SVM (iris, formula);
rbf_model.print_summary ();
printf ('\nConfusion matrix (rows = actual, cols = predicted):\n%s\n', rbf_model.confusion_matrix ().to_string ());

printf ('\n=== Linear kernel, for comparison ===\n');
linear_model = datamunge.SVM (iris, formula, 'linear');
printf ('Training accuracy: %.16g%%\n', linear_model.training_accuracy () * 100);
printf ('Support vectors: %d\n', linear_model.num_support_vectors ());

newdata = datamunge.DataFrame ();
newdata.add_numeric_column ('Sepal.Length', [5.1 6.0 6.5 6.2]);
newdata.add_numeric_column ('Sepal.Width', [3.5 2.7 3.0 2.8]);
newdata.add_numeric_column ('Petal.Length', [1.4 4.5 5.5 4.8]);
newdata.add_numeric_column ('Petal.Width', [0.2 1.5 2.0 1.8]);
printf ('\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n%s\n', ...
        rbf_model.predict_frame (newdata).to_string ());
