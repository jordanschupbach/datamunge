// CMD: just TARGET=ode_ex run-dart
// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
import 'dart:math' as math;

import 'package:datamunge/datamunge.dart';

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

/// Approximate C's %g: six significant digits, no padding.
String fmtG6(double x) => fmtG(x, 6);

List<double> lastState(OdeSolution sol) => sol.stateAt(sol.size() - 1);

/// The time vector followed by the requested state components (0-based), one value per step.
(List<double> times, List<List<double>> components) solutionSeries(OdeSolution sol, List<int> components) {
  final steps = List.generate(sol.size(), (i) => i);
  final states = steps.map(sol.stateAt).toList();
  return (steps.map(sol.timeAt).toList(), [for (final c in components) states.map((s) => s[c]).toList()]);
}

OdeSolver rk4Solver(double step) => OdeSolver(OdeOptions()
  ..method = StepMethod.rk4
  ..stepSize = step);

// NOTE: like the Lua version this example uses the built-in named systems via solveBuiltin.

void main() {
  print('=================== Every named built-in system (no director needed) ===================');
  {
    final solver = OdeSolver();
    final systems = <(String name, List<double> params, List<double> y0)>[
      ('exponential_decay', [1.0], [1.0]),
      ('logistic_growth', [1.0, 1.0], [0.5]),
      ('harmonic_oscillator', [1.0], [1.0, 0.0]),
      ('van_der_pol', [1.0], [2.0, 0.0]),
      ('lorenz', [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0]),
    ];
    for (final (name, parameters, y0) in systems) {
      final sol = solver.solveBuiltin(name, parameters, y0, 0.0, 1.0);
      print('${name.padRight(22)} steps=${'${sol.stepsTaken}'.padRight(6)} '
          'final state=[${lastState(sol).map(fmtG6).join(', ')}]');
    }
  }

  print('\n=================== Harmonic oscillator (energy conservation) ===================');
  {
    final sol = rk4Solver(0.01).solveBuiltin('harmonic_oscillator', [1.0], [1.0, 0.0], 0.0, 20.0);
    final last = lastState(sol);
    final (x, v) = (last[0], last[1]);
    print('x(20) = ${x.toStringAsFixed(6)} (cos(20) = ${math.cos(20.0).toStringAsFixed(6)})');
    print('energy x^2+v^2 = ${(x * x + v * v).toStringAsFixed(6)} (should stay near 1.0)');
    final (ts, series) = solutionSeries(sol, [0, 1]);
    final p = RPlot.plot(ts, series[0], 'l', 'x(t)');
    p.lines(ts, series[1], 'v(t)');
    p.title('Harmonic Oscillator').xLabel('t').yLabel('state');
    p.saveSvg('ode_harmonic_oscillator_dart.svg');
    print('wrote ode_harmonic_oscillator_dart.svg');
  }

  print('\n=================== Lorenz attractor (phase plane) ===================');
  {
    final sol = rk4Solver(0.005).solveBuiltin('lorenz', [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0], 0.0, 25.0);
    print('steps_taken = ${sol.stepsTaken}');
    final (_, series) = solutionSeries(sol, [0, 2]);
    final p = RPlot.plot(series[0], series[1], 'l', 'trajectory');
    p.title('Lorenz Attractor (x-z phase plane)').xLabel('x').yLabel('z');
    p.saveSvg('ode_lorenz_phase_plane_dart.svg');
    print('wrote ode_lorenz_phase_plane_dart.svg');
  }
}
