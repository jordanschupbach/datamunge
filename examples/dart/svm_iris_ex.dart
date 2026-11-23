// CMD: just TARGET=svm_iris_ex run-dart
// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
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
  const formula = 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width';
  final iris = Data.iris()!;
  print('iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n');

  print('=== RBF kernel (default) ===');
  final rbfModel = Svm(iris, formula);
  rbfModel.printSummary();
  print('\nConfusion matrix (rows = actual, cols = predicted):\n${rbfModel.confusionMatrix()!.toString()}');

  print('\n=== Linear kernel, for comparison ===');
  final linearModel = Svm(iris, formula, 'linear');
  print('Training accuracy: ${fmt(linearModel.trainingAccuracy() * 100)}%');
  print('Support vectors: ${linearModel.numSupportVectors()}');

  final newdata = DataFrame();
  newdata.addNumericColumn('Sepal.Length', [5.1, 6.0, 6.5, 6.2]);
  newdata.addNumericColumn('Sepal.Width', [3.5, 2.7, 3.0, 2.8]);
  newdata.addNumericColumn('Petal.Length', [1.4, 4.5, 5.5, 4.8]);
  newdata.addNumericColumn('Petal.Width', [0.2, 1.5, 2.0, 1.8]);
  print('\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n'
      '${rbfModel.predictFrame(newdata)!.toString()}');
}
