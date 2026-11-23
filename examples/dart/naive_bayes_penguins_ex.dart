// CMD: just TARGET=naive_bayes_penguins_ex run-dart
// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
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
  // Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
  // when fitting, matching every other formula-based model in this library.
  final penguins = Data.penguins()!;
  print('penguins: ${penguins.nrows()} rows x ${penguins.ncols()} cols\n');

  // A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
  // predictors in one formula -- each modeled independently given the class, per the naive
  // Bayes assumption.
  final model = NaiveBayesClassifier(penguins, 'species ~ bill_length_mm + bill_depth_mm + island + sex');
  model.printSummary();

  print('\nConfusion matrix (rows = actual, cols = predicted):\n${model.confusionMatrix()!.toString()}');

  print('\nMisclassified rows:');
  final predictions = model.predict(penguins);
  var misclassified = 0;
  const columns = ['species', 'bill_length_mm', 'bill_depth_mm', 'island', 'sex'];
  for (var i = 0; i < penguins.nrows(); i++) {
    if (columns.any((c) => penguins.isNull(c, i))) continue;
    final (actual, predicted) = (penguins.stringAt('species', i), predictions[i]);
    if (predicted == actual) continue;
    misclassified++;
    print('  row $i: bill_length=${fmt(penguins.numericAt('bill_length_mm', i))} '
        'bill_depth=${fmt(penguins.numericAt('bill_depth_mm', i))} '
        'island=${penguins.stringAt('island', i)} sex=${penguins.stringAt('sex', i)}  '
        'actual=$actual  predicted=$predicted');
  }
  print('$misclassified misclassified (of ${penguins.nrows()} rows, some incomplete)');

  // PlotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
  // two-predictor model (bill measurements alone) just for visualization.
  final billOnly = NaiveBayesClassifier(penguins, 'species ~ bill_length_mm + bill_depth_mm');
  print('\nbill-measurements-only model training accuracy: ${fmt(100 * billOnly.trainingAccuracy())}%');
  billOnly.plotClassification(penguins, 'bill_length_mm', 'bill_depth_mm').save('naive_bayes_penguins_classification.svg');
  billOnly.plotDecisionRegions('bill_length_mm', 'bill_depth_mm').save('naive_bayes_penguins_decision_regions.svg');
  print('Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg');
}
