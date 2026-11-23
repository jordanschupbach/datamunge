// CMD: just TARGET=glm_iris_ex run-dart
// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
import 'dart:math' as math;

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
  final rows = List.generate(iris.nrows(), (i) => i);

  // Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
  // the other two on these predictors, which sends logistic regression's coefficients toward
  // +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
  // binary split for a demo.
  final kept = rows.where((i) => const ['versicolor', 'virginica'].contains(iris.stringAt('Species', i))).toList();
  final isVirginica = [for (final i in kept) iris.stringAt('Species', i) == 'virginica' ? 1.0 : 0.0];
  final petalLength = [for (final i in kept) iris.numericAt('Petal.Length', i)];
  final petalWidth = [for (final i in kept) iris.numericAt('Petal.Width', i)];

  final sub = DataFrame();
  sub.addNumericColumn('Petal.Length', petalLength);
  sub.addNumericColumn('Petal.Width', petalWidth);
  sub.addNumericColumn('is_virginica', isVirginica);

  print('=================== Logistic regression (binomial, logit link) ===================');
  final logit = Glm(sub, 'is_virginica ~ Petal.Length + Petal.Width', 'binomial');
  logit.printSummary();

  final fitted = logit.fittedValues();
  final n = isVirginica.length;
  final correct = Iterable.generate(n).where((i) => (fitted[i] >= 0.5) == (isVirginica[i] >= 0.5)).length;
  print('\nResubstitution accuracy at 0.5 threshold: ${fmt(100.0 * correct / n)}%');

  logit.saveDiagnosticPlots('glm_logistic_iris');
  print('\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg');

  // Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
  // classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
  {
    const gridN = 100;
    final widthMean = petalWidth.fold(0.0, (a, b) => a + b) / petalWidth.length;
    final plMin = petalLength.reduce(math.min) - 0.3, plMax = petalLength.reduce(math.max) + 0.3;
    final grid = DataFrame();
    grid.addNumericColumn('Petal.Length', List.generate(gridN, (i) => plMin + (plMax - plMin) * i / (gridN - 1)));
    grid.addNumericColumn('Petal.Width', List.filled(gridN, widthMean));
    final curveFrame = logit.predictFrame(grid, 'confidence')!;
    print('\nPredicted-probability curve (first 5 rows):');
    print(curveFrame.toString(5));
  }

  // Poisson regression, for contrast: same IRLS engine, different family/link.
  print('\n=================== Poisson regression (log link) ===================');
  final countData = DataFrame();
  countData.addNumericColumn('Sepal.Width', [for (final i in rows) iris.numericAt('Sepal.Width', i)]);
  countData.addNumericColumn('Petal.Length', [for (final i in rows) iris.numericAt('Petal.Length', i)]);
  countData.addNumericColumn('count', [for (final i in rows) (iris.numericAt('Sepal.Length', i) + 0.5).floorToDouble()]);
  Glm(countData, 'count ~ Sepal.Width + Petal.Length', 'poisson').printSummary();
}
