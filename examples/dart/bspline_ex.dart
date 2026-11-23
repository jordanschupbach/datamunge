// CMD: just TARGET=bspline_ex run-dart
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
import 'dart:math' as math;

import 'package:datamunge/datamunge.dart';

void main() {
  // Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
  final x = <double>[], y = <double>[], z = <double>[];
  for (var iy = 0; iy < 8; iy++) {
    for (var ix = 0; ix < 8; ix++) {
      final xv = ix / 7.0, yv = iy / 7.0;
      x.add(xv);
      y.add(yv);
      z.add(math.sin(xv) + yv * yv);
    }
  }
  final data = DataFrame();
  data.addNumericColumn('x', x);
  data.addNumericColumn('y', y);
  data.addNumericColumn('z', z);

  final surface = Lm(data, 'z ~ bs(x, y)');
  print('Fitted z ~ bs(x, y)');
  surface.printSummary();

  final newPoints = DataFrame();
  newPoints.addNumericColumn('x', [0.25, 0.75]);
  newPoints.addNumericColumn('y', [0.50, 0.25]);
  final pred = surface.predict(newPoints);
  print('Predictions: [${pred[0].toStringAsFixed(6)}, ${pred[1].toStringAsFixed(6)}]');
}
