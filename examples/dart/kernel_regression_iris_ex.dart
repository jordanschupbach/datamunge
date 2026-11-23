// CMD: just TARGET=kernel_regression_iris_ex run-dart
// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
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
  const formula = 'Petal.Length ~ Petal.Width';

  final iris = Data.iris()!;
  print('iris: ${iris.nrows()} rows x ${iris.ncols()} cols');
  print('formula: $formula\n');

  final model = KernelRegression(iris, formula); // bandwidth auto-selected via leave-one-out CV
  model.printSummary();

  model.plotFit(iris).save('kernel_regression_iris_fit.svg');
  model.plotCvCurve().save('kernel_regression_iris_cv.svg');
  print('\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg');

  // A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
  // bias-variance story. KernelRegression(data, formula, kernel, bandwidth, nBandwidth, standardize)
  final small = KernelRegression(iris, formula, 'gaussian', 0.05);
  print('\nbandwidth=0.05 (too small): LOO R-squared=${fmt(small.rSquared())}  LOO RMSE=${fmt(small.rmse())}');
  small.plotFit(iris).save('kernel_regression_iris_fit_small_bandwidth.svg');

  final large = KernelRegression(iris, formula, 'gaussian', 5.0);
  print('bandwidth=5.0 (too large):  LOO R-squared=${fmt(large.rSquared())}  LOO RMSE=${fmt(large.rmse())}');
  large.plotFit(iris).save('kernel_regression_iris_fit_large_bandwidth.svg');

  print('bandwidth=${fmt(model.bandwidth())} (CV-selected): LOO R-squared=${fmt(model.rSquared())}  '
      'LOO RMSE=${fmt(model.rmse())}');
  print('\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg');
}
