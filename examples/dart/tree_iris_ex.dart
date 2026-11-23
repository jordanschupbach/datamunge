// CMD: just TARGET=tree_iris_ex run-dart
// Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
import 'package:datamunge/datamunge.dart';

/// A double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
/// when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
String fmt(double x) {
  if (x.isNaN) return 'NaN';
  if (x.isInfinite) return x < 0 ? '-∞' : '∞';
  var s = x.toString();
  final negative = s.startsWith('-');
  if (negative) s = s.substring(1);
  var mantissa = s;
  var exponent = 0;
  final e = s.indexOf('e');
  if (e >= 0) {
    mantissa = s.substring(0, e);
    exponent = int.parse(s.substring(e + 1));
  }
  final parts = mantissa.split('.');
  var digits = parts[0] + (parts.length > 1 ? parts[1] : '');
  var point = parts[0].length + exponent;
  while (digits.startsWith('0')) {
    digits = digits.substring(1);
    point--;
  }
  while (digits.endsWith('0')) {
    digits = digits.substring(0, digits.length - 1);
  }
  final sign = negative ? '-' : '';
  if (digits.isEmpty) return '${sign}0';
  final exp10 = point - 1;
  if (exp10 < -4 || exp10 >= 15) {
    final rest = digits.substring(1);
    final magnitude = exp10.abs().toString().padLeft(2, '0');
    return '$sign${digits[0]}${rest.isEmpty ? '' : '.$rest'}E${exp10 < 0 ? '-' : '+'}$magnitude';
  }
  if (point <= 0) return '${sign}0.${'0' * -point}$digits';
  if (point >= digits.length) return '$sign$digits${'0' * (point - digits.length)}';
  return '$sign${digits.substring(0, point)}.${digits.substring(point)}';
}

void main() {
  final iris = Data.iris()!;
  print('iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n');

  final model = DecisionTreeClassifier(iris, 'Species ~ Petal.Length + Petal.Width');
  model.printSummary();

  print('\nConfusion matrix (rows = actual, cols = predicted):');
  print(model.confusionMatrix()!.toString());

  print('\nMisclassified rows:');
  final predictions = model.predict(iris);
  final n = iris.nrows();
  var misclassified = 0;
  for (var i = 0; i < n; i++) {
    final (actual, predicted) = (iris.stringAt('Species', i), predictions[i]);
    if (predicted == actual) continue;
    misclassified++;
    print('  row $i: Petal.Length=${fmt(iris.numericAt('Petal.Length', i))} '
        'Petal.Width=${fmt(iris.numericAt('Petal.Width', i))}  actual=$actual  predicted=$predicted');
  }
  print('$misclassified of $n misclassified (${fmt(100.0 * misclassified / n)}%)');

  model.plotClassification(iris, 'Petal.Length', 'Petal.Width').save('tree_iris_classification.svg');
  model.plotDecisionRegions('Petal.Length', 'Petal.Width').save('tree_iris_decision_regions.svg');
  print('\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg');

  // A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
  final shallow = DecisionTreeClassifier(iris, 'Species ~ Petal.Length + Petal.Width', 2);
  print('\nDepth-2 tree training accuracy: ${fmt(shallow.trainingAccuracy() * 100)}% (${shallow.leafCount()} leaves)');
  shallow.plotDecisionRegions('Petal.Length', 'Petal.Width').save('tree_iris_decision_regions_depth2.svg');
  print('Saved tree_iris_decision_regions_depth2.svg');
}
