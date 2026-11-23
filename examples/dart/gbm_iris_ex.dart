// CMD: just TARGET=gbm_iris_ex run-dart
// Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
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

  final model = GbmClassifier(iris, 'Species ~ Petal.Length + Petal.Width');
  model.printSummary();

  print('\nConfusion matrix (rows = actual, cols = predicted):\n${model.confusionMatrix()!.toString()}');

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

  model.plotTrainingDeviance().save('gbm_iris_training_deviance.svg');
  model.plotDecisionRegions('Petal.Length', 'Petal.Width').save('gbm_iris_decision_regions.svg');
  print('\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg');

  // A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
  // at the training loss, gradually sharpening the decision boundary.
  // GbmClassifier(data, formula, nTrees, learningRate, maxDepth, minSamplesSplit,
  //               minSamplesLeaf, subsample, seed)
  final few = GbmClassifier(iris, 'Species ~ Petal.Length + Petal.Width', 5);
  print('\n5-round ensemble:   training accuracy=${fmt(100 * few.trainingAccuracy())}%  '
      'deviance=${fmt(few.trainingDeviance().last)}');
  print('100-round ensemble: training accuracy=${fmt(100 * model.trainingAccuracy())}%  '
      'deviance=${fmt(model.trainingDeviance().last)}');
  few.plotDecisionRegions('Petal.Length', 'Petal.Width').save('gbm_iris_decision_regions_5rounds.svg');
  print('Saved gbm_iris_decision_regions_5rounds.svg');
}
