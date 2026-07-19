module app;

import std.stdio : writeln;
import datamunge;

Dual rosenbrock_dual(Dual x0, Dual x1) {
  auto a = (new Dual(1.0, 0.0)).subtract(x0);
  auto b = x1.subtract(x0.multiply(x0));
  return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0));
}

double rosenbrock_value(double x0, double x1) {
  double a = 1.0 - x0;
  double b = x1 - x0 * x0;
  return a * a + 100.0 * b * b;
}

void main() {
  writeln("=================== Forward mode: scalar derivative ===================");
  auto x0 = new Dual(2.0, 1.0);  // seed derivative = 1 to read df/dx directly
  auto f = x0.multiply(x0).multiply(x0).subtract(x0.multiply_scalar(2.0));  // x^3 - 2x
  writeln("f(x) = x^3 - 2x, f'(2) = ", f.derivative(), " (exact: 10)");

  writeln("\n=================== Reverse mode: build a graph by hand ===================");
  auto tape = new Tape();
  auto a = new Var(tape, 2.0);
  auto b = new Var(tape, 3.0);
  auto y = a.multiply(b).add(a.sin());
  writeln("y = a*b + sin(a) at a=2, b=3 -> y = ", y.value());
  auto adjoint = tape.backward(y);
  writeln("dy/da = ", adjoint[a.index()], " (exact: b + cos(a))");
  writeln("dy/db = ", adjoint[b.index()], " (exact: a)");

  writeln("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================");
  double px = 0.0;
  double py = 0.0;

  double gx = rosenbrock_dual(new Dual(px, 1.0), new Dual(py, 0.0)).derivative();
  double gy = rosenbrock_dual(new Dual(px, 0.0), new Dual(py, 1.0)).derivative();

  auto tape2 = new Tape();
  auto vx = new Var(tape2, px);
  auto vy = new Var(tape2, py);
  auto va = (new Var(tape2, 1.0)).subtract(vx);
  auto vb = vy.subtract(vx.multiply(vx));
  auto vf = va.multiply(va).add(vb.multiply(vb).multiply_scalar(100.0));
  auto grad_rev = tape2.backward(vf);

  writeln("f(0,0) = ", rosenbrock_value(px, py));
  writeln("gradient (forward mode): [", gx, ", ", gy, "]");
  writeln("gradient (reverse mode): [", grad_rev[vx.index()], ", ", grad_rev[vy.index()], "]");

  writeln("\n=================== Jacobian of a vector-valued function ===================");
  double vx_val = 2.0;
  double vy_val = 3.0;

  double[][3] jac;
  foreach (col; 0 .. 3) jac[col] = [0.0, 0.0];
  int[3] cols = [0, 1, 2];
  struct Seed { int col; double sx; double sy; }
  Seed[2] seeds = [Seed(0, 1.0, 0.0), Seed(1, 0.0, 1.0)];
  foreach (s; seeds) {
    auto dx = new Dual(vx_val, s.sx);
    auto dy = new Dual(vy_val, s.sy);
    auto ox = dx.multiply(dx);
    auto oy = dx.multiply(dy);
    auto oz = dy.multiply(dy).multiply(dy);
    jac[0][s.col] = ox.derivative();
    jac[1][s.col] = oy.derivative();
    jac[2][s.col] = oz.derivative();
  }
  writeln("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
  foreach (row; jac) writeln("  ", row);

  writeln("\n=================== Hessian via second-order forward mode (HyperDual) ===================");
  double mx = 1.0;
  double my = 1.0;

  HyperDual rosenbrock_hyperdual(HyperDual hx0, HyperDual hx1) {
    auto ha = (new HyperDual(1.0, 0.0, 0.0, 0.0)).subtract(hx0);
    auto hb = hx1.subtract(hx0.multiply(hx0));
    return ha.multiply(ha).add(hb.multiply(hb).multiply_scalar(100.0));
  }

  double[2][2] hseeds = [[1.0, 0.0], [0.0, 1.0]];
  double[2][2] hmat;
  foreach (i; 0 .. 2) {
    foreach (j; 0 .. 2) {
      auto hx = new HyperDual(mx, hseeds[i][0], hseeds[j][0], 0.0);
      auto hy = new HyperDual(my, hseeds[i][1], hseeds[j][1], 0.0);
      hmat[i][j] = rosenbrock_hyperdual(hx, hy).eps1eps2();
    }
  }
  writeln("Hessian of the Rosenbrock function at its minimum (1,1):");
  foreach (row; hmat) writeln("  ", row);

  writeln("\n=================== Gradient descent driven by reverse-mode gradients ===================");
  double[2] point = [-1.2, 1.0];
  double learning_rate = 0.001;
  int n_steps = 2000;
  foreach (step; 0 .. n_steps) {
    auto t = new Tape();
    auto vx2 = new Var(t, point[0]);
    auto vy2 = new Var(t, point[1]);
    auto va2 = (new Var(t, 1.0)).subtract(vx2);
    auto vb2 = vy2.subtract(vx2.multiply(vx2));
    auto vf2 = va2.multiply(va2).add(vb2.multiply(vb2).multiply_scalar(100.0));
    auto grad = t.backward(vf2);
    double loss = vf2.value();
    point[0] -= learning_rate * grad[vx2.index()];
    point[1] -= learning_rate * grad[vy2.index()];
    if (step == 0 || step == n_steps - 1) {
      writeln("step ", step, ": loss = ", loss, ", x = [", point[0], ", ", point[1], "]");
    }
  }
  writeln("(true minimum is at [1, 1] with loss 0)");
}
