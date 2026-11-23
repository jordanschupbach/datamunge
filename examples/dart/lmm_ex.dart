// CMD: just TARGET=lmm_ex run-dart
// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
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

final rng = DotNetRandom(2024);
double uniform() => 1 - rng.nextDouble(); // (0, 1]: avoids log(0) in the Box-Muller transform.
double gauss() => math.sqrt(-2 * math.log(uniform())) * math.cos(2 * math.pi * uniform());

void main() {
  print('=================== Random intercept on a real dataset (penguins) ===================');
  Lmm(Data.penguins()!, 'body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)').printSummary();

  print('\n=================== Random intercept + slope on a simulated multi-school dataset ===================');
  // 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
  // and the return on an extra hour of study vary by school -- the textbook case for a
  // random-intercept-and-slope model.
  const nSchools = 30;
  final schoolIntercept = List.generate(nSchools, (_) => gauss() * 6);
  final schoolSlope = List.generate(nSchools, (_) => gauss() * 1.2);
  const trueIntercept = 60.0;
  const trueSlope = 3.0;

  final school = <double>[], studyHours = <double>[], score = <double>[];
  for (var s = 0; s < nSchools; s++) {
    final nStudents = 15 + rng.next(0, 21);
    for (var student = 0; student < nStudents; student++) {
      final hours = rng.nextDouble() * 10;
      school.add(s.toDouble());
      studyHours.add(hours);
      score.add(trueIntercept + schoolIntercept[s] + (trueSlope + schoolSlope[s]) * hours + gauss() * 4);
    }
  }
  final df = DataFrame();
  df.addNumericColumn('school', school);
  df.addNumericColumn('study_hours', studyHours);
  df.addNumericColumn('score', score);

  final model = Lmm(df, 'score ~ study_hours + (1 + study_hours | school)');
  model.printSummary();

  print('\nTrue generating values: intercept=${fmt(trueIntercept)}, slope=${fmt(trueSlope)}, '
      'random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0');

  print('\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---');
  final labels = model.groupLabels();
  for (var idx = 0; idx < 3; idx++) {
    final re = model.randomEffectsForGroup(idx);
    print('school ${labels[idx]}: intercept shift=${fmt(re[0])}, slope shift=${fmt(re[1])}');
  }

  print('\n--- Prediction: population-level vs. school-adjusted ---');
  final population = DataFrame();
  final school0 = DataFrame();
  population.addNumericColumn('study_hours', [5.0]);
  school0.addNumericColumn('study_hours', [5.0]);
  school0.addNumericColumn('school', [0.0]);
  print('5 study hours, unseen school:      ${fmt(model.predict(population)[0])} (fixed effects only)');
  print("5 study hours, school 0 (known):    ${fmt(model.predict(school0)[0])} (fixed effects + school 0's BLUP)");
}
