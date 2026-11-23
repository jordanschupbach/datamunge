// CMD: just TARGET=knn_iris_ex run-dart
// k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
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

  final model = KnnClassifier(iris, 'Species ~ Petal.Length + Petal.Width');
  model.printSummary();

  print('\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):');
  print(model.confusionMatrix()!.toString());

  print('\nLeave-one-out misclassified rows:');
  final fitted = model.predict(iris);
  final n = iris.nrows();
  var misclassified = 0;
  for (var i = 0; i < n; i++) {
    final (actual, predicted) = (iris.stringAt('Species', i), fitted[i]);
    if (predicted == actual) continue;
    misclassified++;
    print('  row $i: Petal.Length=${fmt(iris.numericAt('Petal.Length', i))} '
        'Petal.Width=${fmt(iris.numericAt('Petal.Width', i))}  actual=$actual  predicted=$predicted');
  }
  print('$misclassified of $n misclassified (${fmt(100.0 * misclassified / n)}%)');

  model.plotDecisionRegions('Petal.Length', 'Petal.Width').save('knn_iris_decision_regions_k5.svg');
  print('\nSaved knn_iris_decision_regions_k5.svg');

  // k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
  // every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
  // underfit boundary). KnnClassifier(data, formula, k, metric, weighted, standardize)
  for (final k in [1, 25]) {
    final knn = KnnClassifier(iris, 'Species ~ Petal.Length + Petal.Width', k);
    final file = 'knn_iris_decision_regions_k$k.svg';
    print('\nk=${'$k'.padRight(2)} leave-one-out accuracy: ${fmt(knn.trainingAccuracy() * 100)}%');
    knn.plotDecisionRegions('Petal.Length', 'Petal.Width').save(file);
    print('Saved $file');
  }
}
