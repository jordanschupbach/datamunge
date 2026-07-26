module app;

import std.stdio : writeln, writefln;
import std.math : sin;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

void main() {
  // Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
  double[] x, y, z;
  for (size_t iy = 0; iy < 8; iy++) {
    for (size_t ix = 0; ix < 8; ix++) {
      double xv = ix / 7.0;
      double yv = iy / 7.0;
      x ~= xv;
      y ~= yv;
      z ~= sin(xv) + yv * yv;
    }
  }

  auto data = new DataFrame();
  data.add_numeric_column("x", dv(x));
  data.add_numeric_column("y", dv(y));
  data.add_numeric_column("z", dv(z));

  auto surface = new LM(data, "z ~ bs(x, y)");
  writeln("Fitted z ~ bs(x, y)");
  surface.print_summary();

  auto new_points = new DataFrame();
  new_points.add_numeric_column("x", dv([0.25, 0.75]));
  new_points.add_numeric_column("y", dv([0.50, 0.25]));
  auto pred = surface.predict(new_points);
  writefln("Predictions: [%.6f, %.6f]", pred[0], pred[1]);
}
