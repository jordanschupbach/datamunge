// CMD: just TARGET=autodiff_ex run-dart
// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
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

// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// Var.index() is the Var's position in the adjoint vector Tape.backward returns.

Dual rosenbrockDual(Dual x0, Dual x1) {
  final a = Dual(1.0, 0.0).subtract(x0);
  final b = x1.subtract(x0.multiply(x0));
  return a.multiply(a).add(b.multiply(b).multiplyScalar(100.0));
}

double rosenbrockValue(double x0, double x1) =>
    math.pow(1.0 - x0, 2).toDouble() + 100.0 * math.pow(x1 - x0 * x0, 2).toDouble();

/// Build the Rosenbrock graph on tape; returns the output Var.
Var rosenbrockVar(Tape tape, Var x0, Var x1) {
  final a = Var(tape, 1.0).subtract(x0);
  final b = x1.subtract(x0.multiply(x0));
  return a.multiply(a).add(b.multiply(b).multiplyScalar(100.0));
}

HyperDual rosenbrockHyperDual(HyperDual x0, HyperDual x1) {
  final a = HyperDual(1.0, 0.0, 0.0, 0.0).subtract(x0);
  final b = x1.subtract(x0.multiply(x0));
  return a.multiply(a).add(b.multiply(b).multiplyScalar(100.0));
}

double adjoint(List<double> adjoints, Var v) => adjoints[v.index()];

void main() {
  print('=================== Forward mode: scalar derivative ===================');
  {
    final x0 = Dual(2.0, 1.0); // seed derivative = 1 to read df/dx directly
    final f = x0.multiply(x0).multiply(x0).subtract(x0.multiplyScalar(2.0)); // x^3 - 2x
    print("f(x) = x^3 - 2x, f'(2) = ${fmt(f.derivative())} (exact: 10)");
  }

  print('\n=================== Reverse mode: build a graph by hand ===================');
  {
    final tape = Tape();
    final a = Var(tape, 2.0);
    final b = Var(tape, 3.0);
    final y = a.multiply(b).add(a.sin());
    print('y = a*b + sin(a) at a=2, b=3 -> y = ${fmt(y.value())}');
    final adjoints = tape.backward(y);
    print('dy/da = ${fmt(adjoint(adjoints, a))} (exact: b + cos(a))');
    print('dy/db = ${fmt(adjoint(adjoints, b))} (exact: a)');
  }

  print('\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================');
  {
    const px = 0.0, py = 0.0;
    // Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
    final gx = rosenbrockDual(Dual(px, 1.0), Dual(py, 0.0)).derivative();
    final gy = rosenbrockDual(Dual(px, 0.0), Dual(py, 1.0)).derivative();
    // Reverse mode: one Tape/Var pass computes every partial at once.
    final tape = Tape();
    final vx = Var(tape, px);
    final vy = Var(tape, py);
    final grad = tape.backward(rosenbrockVar(tape, vx, vy));
    print('f(0,0) = ${fmt(rosenbrockValue(px, py))}');
    print('gradient (forward mode): [${fmt(gx)}, ${fmt(gy)}]');
    print('gradient (reverse mode): [${fmt(adjoint(grad, vx))}, ${fmt(adjoint(grad, vy))}]');
  }

  print('\n=================== Jacobian of a vector-valued function ===================');
  // f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
  {
    List<Dual> vectorFn(Dual x, Dual y) => [x.multiply(x), x.multiply(y), y.multiply(y).multiply(y)];
    final jac = List.generate(3, (_) => List.filled(2, 0.0));
    for (final (col, sx, sy) in [(0, 1.0, 0.0), (1, 0.0, 1.0)]) {
      final outputs = vectorFn(Dual(2.0, sx), Dual(3.0, sy));
      for (var row = 0; row < outputs.length; row++) {
        jac[row][col] = outputs[row].derivative();
      }
    }
    print('f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:');
    for (var row = 0; row < 3; row++) {
      print('  [${fmt(jac[row][0])}, ${fmt(jac[row][1])}]');
    }
  }

  print('\n=================== Hessian via second-order forward mode (HyperDual) ===================');
  // One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
  {
    const mx = 1.0, my = 1.0; // the Rosenbrock function's minimum
    const seeds = [(x: 1.0, y: 0.0), (x: 0.0, y: 1.0)];
    final h = List.generate(2, (_) => List.filled(2, 0.0));
    for (var i = 0; i < 2; i++) {
      for (var j = 0; j < 2; j++) {
        final (e1, e2) = (seeds[i], seeds[j]);
        h[i][j] = rosenbrockHyperDual(HyperDual(mx, e1.x, e2.x, 0.0), HyperDual(my, e1.y, e2.y, 0.0)).eps1eps2();
      }
    }
    print('Hessian of the Rosenbrock function at its minimum (1,1):');
    for (var i = 0; i < 2; i++) {
      print('  [${fmt(h[i][0])}, ${fmt(h[i][1])}]');
    }
  }

  print('\n=================== Gradient descent driven by reverse-mode gradients ===================');
  {
    final point = [-1.2, 1.0]; // the classic Rosenbrock start
    const learningRate = 0.001;
    const nSteps = 2000;
    for (var step = 0; step < nSteps; step++) {
      final tape = Tape();
      final vx = Var(tape, point[0]);
      final vy = Var(tape, point[1]);
      final vf = rosenbrockVar(tape, vx, vy);
      final grad = tape.backward(vf);
      final loss = vf.value();
      point[0] -= learningRate * adjoint(grad, vx);
      point[1] -= learningRate * adjoint(grad, vy);
      if (step == 0 || step == nSteps - 1) {
        print('step $step: loss = ${fmt(loss)}, x = [${fmt(point[0])}, ${fmt(point[1])}]');
      }
    }
    print('(true minimum is at [1, 1] with loss 0)');
  }
}
