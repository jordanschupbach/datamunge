// CMD: just TARGET=lda_iris_ex run-dart
// Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
import 'package:datamunge/datamunge.dart';

void main() {
  final iris = Data.iris()!;
  print('iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n');

  final model = Lda(iris, 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width');
  model.printSummary();

  print('\nConfusion matrix (rows = actual, cols = predicted):\n${model.confusionMatrix()!.toString()}');

  final newdata = DataFrame();
  newdata.addNumericColumn('Sepal.Length', [5.1, 6.0, 6.5, 6.2]);
  newdata.addNumericColumn('Sepal.Width', [3.5, 2.7, 3.0, 2.8]);
  newdata.addNumericColumn('Petal.Length', [1.4, 4.5, 5.5, 4.8]);
  newdata.addNumericColumn('Petal.Width', [0.2, 1.5, 2.0, 1.8]);
  print('\nPredictions for new flowers:\n${model.predictFrame(newdata)!.toString()}');

  model.saveDiscriminantPlot('lda_iris_discriminants.svg');
  print('\nSaved discriminant plot as lda_iris_discriminants.svg');
}
