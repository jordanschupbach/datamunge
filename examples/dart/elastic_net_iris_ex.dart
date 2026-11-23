// CMD: just TARGET=elastic_net_iris_ex run-dart
// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
import 'package:datamunge/datamunge.dart';

void main() {
  const formula = 'Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width';

  final iris = Data.iris()!;
  print('iris: ${iris.nrows()} rows x ${iris.ncols()} cols');
  print('formula: $formula\n');

  print('=================== Ridge ===================');
  final ridge = Ridge(iris, formula);
  ridge.printSummary();

  print('\n=================== Lasso ===================');
  final lasso = Lasso(iris, formula);
  lasso.printSummary();

  print('\n================= Elastic Net =================');
  // ElasticNet(data, formula, alpha=0.5, lambda, nLambda, cvFolds, standardize, seed)
  final elastic = ElasticNet(iris, formula, 0.5);
  elastic.printSummary();

  print("\nSaved figures showing how each model's coefficients respond to the regularization strength, "
      'and the cross-validation curve used to pick it:');

  final figures = <(RPlot path, RPlot cv, String name, String label)>[
    (ridge.plotCoefficientPath(), ridge.plotCvCurve(), 'ridge', 'ridge:      '),
    (lasso.plotCoefficientPath(), lasso.plotCvCurve(), 'lasso', 'lasso:      '),
    (elastic.plotCoefficientPath(), elastic.plotCvCurve(), 'elasticnet', 'elastic net:'),
  ];
  for (final (path, cv, name, label) in figures) {
    path.save('elastic_net_${name}_path.svg');
    cv.save('elastic_net_${name}_cv.svg');
    print('  $label elastic_net_${name}_path.svg, elastic_net_${name}_cv.svg');
  }
}
