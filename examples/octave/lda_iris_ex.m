% CMD: just TARGET=lda_iris_ex run-octave
% Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.

iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n\n', iris.nrows (), iris.ncols ());

model = datamunge.LDA (iris, 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width');
model.print_summary ();

printf ('\nConfusion matrix (rows = actual, cols = predicted):\n%s\n', model.confusion_matrix ().to_string ());

newdata = datamunge.DataFrame ();
newdata.add_numeric_column ('Sepal.Length', [5.1 6.0 6.5 6.2]);
newdata.add_numeric_column ('Sepal.Width', [3.5 2.7 3.0 2.8]);
newdata.add_numeric_column ('Petal.Length', [1.4 4.5 5.5 4.8]);
newdata.add_numeric_column ('Petal.Width', [0.2 1.5 2.0 1.8]);
printf ('\nPredictions for new flowers:\n%s\n', model.predict_frame (newdata).to_string ());

model.save_discriminant_plot ('lda_iris_discriminants.svg');
printf ('\nSaved discriminant plot as lda_iris_discriminants.svg\n');
