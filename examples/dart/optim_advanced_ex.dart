// CMD: just TARGET=optim_advanced_ex run-dart
// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
// AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Dart.
//
// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From Dart, extend the interface class and override its virtual
// methods; the optimizer then calls the Dart overrides. Coordinates are updated in place through
// a Ref (the C++ parameter is a non-const std::vector<double>&).
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

/// A double through C#'s "G<precision>" format: [precision] significant digits, trailing zeros
/// dropped, in E notation (E-05, E+06) when the exponent is below -5 or at least [precision].
String fmtG(double x, int precision) {
  if (x.isNaN) return 'NaN';
  if (x == 0) return '0';
  final scientific = x.toStringAsExponential(precision - 1); // d.ddddde[+-]x, correctly rounded
  final e = scientific.indexOf('e');
  var mantissa = scientific.substring(0, e);
  final exponent = int.parse(scientific.substring(e + 1));
  String trim(String s) => s.contains('.') ? s.replaceFirst(RegExp(r'\.?0+$'), '') : s;
  if (exponent < -5 || exponent >= precision) {
    mantissa = trim(mantissa);
    final magnitude = exponent.abs().toString().padLeft(2, '0');
    return '${mantissa}E${exponent < 0 ? '-' : '+'}$magnitude';
  }
  return trim(x.toStringAsFixed(precision - 1 - exponent));
}

void main() {
  print('=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================');
  // Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
  // then check the acquisition function favors an unexplored region.
  {
    const xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    final ys = [for (final x in xs) (x - 1) * (x - 1)];
    final surrogate = RbfGaussianProcessSurrogate(1.0, 1e-6);
    surrogate.fit([for (final x in xs) [x]], ys);
    final incumbent = math.min(0.0, ys.reduce(math.min)); // best observed value so far
    print('incumbent (best observed y): ${fmt(incumbent)}');
    for (final x in [0.25, 1.0, 2.5]) {
      print('acquisition(x=${x.toStringAsFixed(2)}) = ${surrogate.acquisition([x], incumbent).toStringAsFixed(6)}');
    }
  }

  print('\n=================== Options structs + optimizer construction (all 10 new classes) ===================');
  // Every new Options struct and optimizer class constructs cleanly, proving
  // the binding for the whole batch is wired correctly.
  final pg = ProximalGradient(ProximalGradientOptions()..stepSize = 0.05);
  print('ProximalGradient: $pg');
  final fista = Fista(FistaOptions());
  print('FISTA: $fista');
  final lm = LevenbergMarquardt(LevenbergMarquardtOptions()..maxIterations = 200);
  print('LevenbergMarquardt: $lm');
  final newton = Newton(NewtonOptions());
  print('Newton: $newton');
  final trn = TrustRegionNewton(TrustRegionNewtonOptions());
  print('TrustRegionNewton: $trn');
  final al = AugmentedLagrangian(AugmentedLagrangianOptions());
  print('AugmentedLagrangian: $al');
  final sqp = Sqp(SqpOptions());
  print('SQP: $sqp');
  final ip = InteriorPoint(InteriorPointOptions());
  print('InteriorPoint: $ip');
  final bo = BayesianOptimization(BayesianOptimizationOptions()..initialSamples = 8);
  print('BayesianOptimization: $bo');

  print('\n=================== Dart objectives (subclassing the C++ interfaces) ===================');
  // Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
  {
    final start = Ref([5.0, 5.0]);
    final value = newton.optimize(Quadratic(), start);
    print('Newton: f = ${fmtG(value, 6)} at (${start.value[0].toStringAsFixed(6)}, '
        '${start.value[1].toStringAsFixed(6)}) (exact minimum 0 at (1, -2))');
  }

  // Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
  {
    final parameters = Ref([1.0, 0.1]);
    final sse = lm.optimize(Exponential([0.0, 0.5, 1.0, 1.5, 2.0]), parameters);
    print('LevenbergMarquardt: residual = ${fmtG(sse, 3)}, a = ${parameters.value[0].toStringAsFixed(6)}, '
        'b = ${parameters.value[1].toStringAsFixed(6)} (exact: 2, 0.5)');
  }

  // An interface method a subclass leaves out throws when the optimizer calls it.
  {
    Object? err;
    try {
      final origin = Ref([0.0, 0.0]);
      newton.optimize(Incomplete(), origin);
    } catch (e) {
      err = e;
    }
    print('missing gradient(): ok=${err == null ? 'True' : 'False'}, '
        "error mentions it: ${err != null && err.toString().contains('gradient') ? 'True' : 'False'}");
  }
}

class Quadratic extends HessianFunction {
  @override
  double evaluate(List<double> coordinates) {
    final c = coordinates;
    return (c[0] - 1) * (c[0] - 1) + 10 * (c[1] + 2) * (c[1] + 2);
  }

  @override
  List<double> gradient(List<double> coordinates) {
    final c = coordinates;
    return [2 * (c[0] - 1), 20 * (c[1] + 2)];
  }

  @override
  List<List<double>> hessian(List<double> coordinates) => [
        [2.0, 0.0],
        [0.0, 20.0],
      ];
}

class Exponential extends ResidualFunction {
  Exponential(this.ts);

  final List<double> ts;

  @override
  List<double> residuals(List<double> coordinates) {
    final p = coordinates;
    return [for (final t in ts) p[0] * math.exp(p[1] * t) - 2 * math.exp(0.5 * t)];
  }

  @override
  List<List<double>> jacobian(List<double> coordinates) {
    final p = coordinates;
    return [
      for (final t in ts) [math.exp(p[1] * t), p[0] * t * math.exp(p[1] * t)],
    ];
  }
}

class Incomplete extends HessianFunction {
  @override
  double evaluate(List<double> coordinates) => 0.0;
}
