// CMD: just TARGET=timeseries_ex run-dart
// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
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

/// System.Random(seed) of .NET (the seeded, Knuth subtractive generator), so the simulated data
/// match the C# example's exactly.
class DotNetRandom {
  DotNetRandom(int seed) {
    var mj = 161803398 - (seed == -2147483648 ? _mbig : seed.abs());
    _seeds[55] = mj;
    var mk = 1;
    for (var i = 1; i < 55; i++) {
      final ii = (21 * i) % 55;
      _seeds[ii] = mk;
      mk = mj - mk;
      if (mk < 0) mk += _mbig;
      mj = _seeds[ii];
    }
    for (var k = 1; k < 5; k++) {
      for (var i = 1; i < 56; i++) {
        _seeds[i] -= _seeds[1 + (i + 30) % 55];
        if (_seeds[i] < 0) _seeds[i] += _mbig;
      }
    }
  }

  static const _mbig = 2147483647;
  final _seeds = List<int>.filled(56, 0);
  var _inext = 0, _inextp = 21;

  double _sample() {
    _inext = _inext + 1 >= 56 ? 1 : _inext + 1;
    _inextp = _inextp + 1 >= 56 ? 1 : _inextp + 1;
    var value = _seeds[_inext] - _seeds[_inextp];
    if (value == _mbig) value--;
    if (value < 0) value += _mbig;
    _seeds[_inext] = value;
    return value * (1.0 / _mbig);
  }

  /// Random.NextDouble(): [0, 1).
  double nextDouble() => _sample();

  /// Random.Next(minValue, maxValue): [minValue, maxValue).
  int next(int minValue, int maxValue) => (_sample() * (maxValue - minValue)).toInt() + minValue;
}

var rng = DotNetRandom(42);
void seed(int n) => rng = DotNetRandom(n);
double uniform() => 1 - rng.nextDouble(); // (0, 1]: avoids log(0) in the Box-Muller transform.
double gauss() => math.sqrt(-2 * math.log(uniform())) * math.cos(2 * math.pi * uniform());
String joined(List<double> v) => v.map(fmt).join(', ');

void main() {
  print('=================== ARIMA(1,1,1) on a simulated random walk with drift ===================');
  seed(42);
  final y = <double>[];
  {
    var level = 100.0, prevShock = 0.0;
    for (var i = 0; i < 150; i++) {
      final shock = gauss();
      level += 0.3 + 0.4 * prevShock + shock; // integrated ARMA(0,1) increments with drift
      y.add(level);
      prevShock = shock;
    }
  }

  final model = Arima(
      y,
      ArimaOptions()
        ..p = 1
        ..d = 1
        ..q = 1
        ..dePopulationSize = 80
        ..deMaxGenerations = 400);

  print('AR coefficient: ${fmt(model.arCoefficients()[0])}');
  print('MA coefficient: ${fmt(model.maCoefficients()[0])}');
  print('sigma^2: ${fmt(model.sigma2())}, AIC: ${fmt(model.aic())}, BIC: ${fmt(model.bic())}');

  final (forecastValues, stdErrors) = model.forecastWithIntervals(6);
  print('6-step forecast: ${joined(forecastValues)}');
  print('forecast std. errors: ${joined(stdErrors)}');

  print('\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================');
  seed(7);
  final s = <double>[];
  {
    var prev = 0.0;
    for (var i = 0; i < 120; i++) {
      prev = 0.5 * prev + gauss();
      s.add(20 + 0.2 * i + 5 * math.sin(2 * math.pi * i / 12) + prev);
    }
  }

  final model2 = Arima(
      s,
      ArimaOptions()
        ..p = 1
        ..seasonalP = 1
        ..seasonalD = 1
        ..seasonalPeriod = 12
        ..dePopulationSize = 100
        ..deMaxGenerations = 500);
  print('AR coefficient: ${fmt(model2.arCoefficients()[0])}, '
      'seasonal AR coefficient: ${fmt(model2.seasonalArCoefficients()[0])}');
  print('12-step forecast: ${joined(model2.forecast(12))}');

  print('\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================');
  const seasonalShape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6];
  seed(11);
  final y2 = List.generate(48, (i) => (100 + 2 * i) * seasonalShape[i % 12] + gauss() * 3);

  final esModel = ExponentialSmoothing(
      y2,
      ExponentialSmoothingOptions()
        ..trend = TrendType.additive
        ..seasonal = SeasonalType.multiplicative
        ..seasonalPeriod = 12
        ..dePopulationSize = 60
        ..deMaxGenerations = 300);
  print('alpha=${fmt(esModel.alpha())} beta=${fmt(esModel.beta())} gamma=${fmt(esModel.gamma())}');
  print('sigma^2: ${fmt(esModel.sigma2())}, AIC: ${fmt(esModel.aic())}');
  print('12-month forecast: ${joined(esModel.forecast(12))}');

  print('\n=================== Simple exponential smoothing vs. Holt\'s linear trend ===================');
  const flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];

  final ses = ExponentialSmoothing(flat, ExponentialSmoothingOptions());
  print('SES alpha= ${fmt(ses.alpha())}');
  print('SES 5-step forecast: ${joined(ses.forecast(5))}');

  final holt = ExponentialSmoothing(flat, ExponentialSmoothingOptions()..trend = TrendType.additive);
  print('Holt alpha=${fmt(holt.alpha())} beta=${fmt(holt.beta())}');
  print('Holt 5-step forecast: ${joined(holt.forecast(5))}');
}
