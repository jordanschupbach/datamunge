// CMD: just TARGET=glmm_ex run-dart
// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
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

final rng = DotNetRandom(4242);
double uniform() => rng.nextDouble();
double gauss() {
  final u1 = 1 - uniform(), u2 = uniform(); // 1 - u keeps log() away from 0
  return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
}

void main() {
  print('=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================');
  // Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
  // confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
  // is not tied to location, so this is a well-behaved fit.
  {
    final penguins = Data.penguins()!;
    final isMale = <double>[], bodyMass = <double>[];
    final island = <String>[];
    for (var i = 0; i < penguins.nrows(); i++) {
      if (const ['sex', 'body_mass_g', 'island'].any((c) => penguins.isNull(c, i))) continue;
      isMale.add(penguins.stringAt('sex', i) == 'male' ? 1.0 : 0.0);
      bodyMass.add(penguins.numericAt('body_mass_g', i));
      island.add(penguins.stringAt('island', i));
    }
    final sexDf = DataFrame();
    sexDf.addNumericColumn('is_male', isMale);
    sexDf.addNumericColumn('body_mass_g', bodyMass);
    sexDf.addStringColumn('island', island);
    Glmm(sexDf, 'is_male ~ body_mass_g + (1 | island)', 'binomial').printSummary();
  }

  print('\n=================== Poisson mixed model on simulated multi-site count data ===================');
  // 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
  // the baseline traffic and the promo's effectiveness vary by store.
  const nStores = 25;
  final storeEffect = List.generate(nStores, (_) => gauss() * 0.4);
  const trueIntercept = 2.0;
  const trueSlope = 0.3;

  final store = <double>[], promo = <double>[], visits = <double>[];
  for (var s = 0; s < nStores; s++) {
    final nDays = 15 + rng.next(0, 11);
    for (var day = 0; day < nDays; day++) {
      final promoIntensity = uniform() * 3;
      final lam = math.exp(trueIntercept + storeEffect[s] + trueSlope * promoIntensity);
      // Knuth's Poisson sampler.
      var (threshold, k, p) = (math.exp(-lam), 0, 1.0);
      do {
        k++;
        p *= uniform();
      } while (p > threshold);
      store.add(s.toDouble());
      promo.add(promoIntensity);
      visits.add(k - 1.0);
    }
  }
  final df = DataFrame();
  df.addNumericColumn('store', store);
  df.addNumericColumn('promo', promo);
  df.addNumericColumn('visits', visits);
  final storeModel = Glmm(df, 'visits ~ promo + (1 | store)', 'poisson');

  storeModel.printSummary();

  print('\nTrue generating values: intercept=${fmt(trueIntercept)}, slope=${fmt(trueSlope)}, '
      'random-intercept SD (log scale)=0.4');

  print('\n--- BLUPs for a few stores ---');
  final labels = storeModel.groupLabels();
  for (var idx = 0; idx < 3; idx++) {
    print('store ${labels[idx]}: intercept shift=${fmt(storeModel.randomEffectsForGroup(idx)[0])}');
  }

  print('\n--- Prediction: population-level vs. store-adjusted ---');
  final population = DataFrame();
  final store0 = DataFrame();
  population.addNumericColumn('promo', [1.5]);
  store0.addNumericColumn('promo', [1.5]);
  store0.addNumericColumn('store', [0.0]);
  print('promo=1.5, unseen store:   ${fmt(storeModel.predict(population)[0])} expected visits (fixed effects only)');
  print("promo=1.5, store 0 (known): ${fmt(storeModel.predict(store0)[0])} expected visits "
      "(fixed effects + store 0's BLUP)");
}
