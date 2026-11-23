// CMD: just TARGET=hypothesis_tests_ex run-dart
// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
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

void printResult(String label, HypothesisTestResult r) {
  final alternative = switch (r.alternative) {
    Alternative.less => 'less',
    Alternative.greater => 'greater',
    _ => 'two.sided',
  };
  final line = StringBuffer('$label: statistic=${fmt(r.statistic)}');
  if (r.parameter1 != 0) line.write(', df1=${fmt(r.parameter1)}');
  if (r.parameter2 != 0) line.write(', df2=${fmt(r.parameter2)}');
  line.write(', p=${fmt(r.pValue)} ($alternative)');
  if (r.hasConfInt) line.write(', CI=[${fmt(r.confIntLower)}, ${fmt(r.confIntUpper)}]');
  print('$line -- ${r.method}');
}

List<double> columnForSpecies(DataFrame df, String column, String species) => [
      for (var i = 0; i < df.nrows(); i++)
        if (df.stringAt('Species', i) == species) df.numericAt(column, i),
    ];

List<double> column(DataFrame df, String column) => [for (var i = 0; i < df.nrows(); i++) df.numericAt(column, i)];

void main() {
  final iris = Data.iris()!;

  final setosaPetal = columnForSpecies(iris, 'Petal.Length', 'setosa');
  final versicolorPetal = columnForSpecies(iris, 'Petal.Length', 'versicolor');
  final virginicaPetal = columnForSpecies(iris, 'Petal.Length', 'virginica');

  print('=================== t-tests: petal length, setosa vs. versicolor ===================');
  printResult('Welch two-sample t-test', tTestTwoSample(setosaPetal, versicolorPetal));
  printResult('Wilcoxon rank-sum test', wilcoxonRankSumTest(setosaPetal, versicolorPetal));

  print('\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================');
  final allPetal = [...setosaPetal, ...versicolorPetal, ...virginicaPetal];
  final sizes = [setosaPetal.length, versicolorPetal.length, virginicaPetal.length];
  printResult('One-way ANOVA', oneWayAnova(allPetal, sizes));
  printResult('Kruskal-Wallis', kruskalWallisTest(allPetal, sizes));

  print('\n=================== Correlation: sepal length vs. petal length ===================');
  final sepalLength = column(iris, 'Sepal.Length');
  final petalLength = column(iris, 'Petal.Length');
  printResult('Pearson correlation', pearsonCorrelationTest(sepalLength, petalLength));
  printResult('Spearman correlation', spearmanCorrelationTest(sepalLength, petalLength));

  print('\n=================== F-test: petal length variance, setosa vs. virginica ===================');
  printResult('F test', fTestVariance(setosaPetal, virginicaPetal));

  print('\n=================== Normality: is sepal length normally distributed within setosa? ===================');
  final setosaSepal = columnForSpecies(iris, 'Sepal.Length', 'setosa');
  printResult('Shapiro-Francia', shapiroFranciaTest(setosaSepal));
  printResult('KS vs. fitted normal', ksTestOneSampleNormal(setosaSepal, 5.006, 0.3525));

  print('\n=================== Chi-squared / Fisher: is petal length "long" independent of species? ===================');
  final medianAll = ([...allPetal]..sort())[allPetal.length ~/ 2];
  int countLong(List<double> v) => v.where((x) => x > medianAll).length;
  {
    final setosaLong = countLong(setosaPetal);
    final setosaShort = setosaPetal.length - setosaLong;
    final versicolorLong = countLong(versicolorPetal);
    final versicolorShort = versicolorPetal.length - versicolorLong;
    print('table: setosa=[$setosaLong,$setosaShort] versicolor=[$versicolorLong,$versicolorShort]');
    final table = [setosaLong, setosaShort, versicolorLong, versicolorShort].map((c) => c.toDouble()).toList();
    printResult('Chi-squared independence', chiSquaredTestIndependence(table, 2, 2));
    printResult("Fisher's exact test", fisherExactTest2x2(setosaLong, setosaShort, versicolorLong, versicolorShort));
  }

  print('\n=================== Proportion / binomial: fraction of "long" petals overall ===================');
  {
    final longCount = countLong(allPetal);
    final n = allPetal.length;
    printResult('One-sample proportion test (vs 0.5)', proportionTestOneSample(longCount, n, 0.5));
    printResult('Exact binomial test (vs 0.5)', binomialTest(longCount, n, 0.5));
  }
}
