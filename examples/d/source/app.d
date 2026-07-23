module app;

import std.stdio : writeln;
import std.math : exp, abs;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

DVectorVector dvv(double[][] rows) {
  auto m = new DVectorVector();
  foreach (row; rows) m.push_back(dv(row));
  return m;
}

// (a) HessianFunction subclass -- a 2-D quadratic bowl with a known minimum, run through Newton.
// f(x, y) = (x - 3)^2 + 2*(y + 1)^2 + 5, minimized at (3, -1) with f = 5.
class QuadraticBowl2D : HessianFunction {
  override double evaluate(DVector coords) {
    double dx = coords[0] - 3.0;
    double dy = coords[1] + 1.0;
    return dx * dx + 2.0 * dy * dy + 5.0;
  }
  override DVector gradient(DVector coords) {
    auto g = new DVector();
    g.push_back(2.0 * (coords[0] - 3.0));
    g.push_back(4.0 * (coords[1] + 1.0));
    return g;
  }
  override DVectorVector hessian(DVector coords) {
    return dvv([[2.0, 0.0], [0.0, 4.0]]);
  }
}

// (b) ResidualFunction subclass -- fit A*exp(-k*t)+c to synthetic noiseless data via
// LevenbergMarquardt, then confirm the recovered parameters match the generating ones.
class ExpDecayResiduals : ResidualFunction {
  double[] ts;
  double[] ys;
  this(double[] t, double[] y) { super(); ts = t; ys = y; }

  override DVector residuals(DVector p) {
    auto r = new DVector();
    double A = p[0], k = p[1], c = p[2];
    foreach (i, t; ts) r.push_back(A * exp(-k * t) + c - ys[i]);
    return r;
  }
  override DVectorVector jacobian(DVector p) {
    double A = p[0], k = p[1];
    double[][] rows;
    foreach (t; ts) {
      double e = exp(-k * t);
      rows ~= [e, -A * t * e, 1.0];
    }
    return dvv(rows);
  }
}

// (c) BayesianOptimization with the ready-to-use RBFGaussianProcessSurrogate -- no
// subclassing of BayesianSurrogate needed. Objective is an ArbitraryFunction subclass.
class Bowl1D : ArbitraryFunction {
  override double evaluate(DVector x) {
    double d = x[0] - 2.0;
    return d * d + 1.0;
  }
}

void main() {
  writeln("=================== HessianFunction: Newton on a 2-D quadratic bowl ===================");
  auto qb = new QuadraticBowl2D();
  auto qx = dv([0.0, 0.0]);
  double value = (new Newton()).optimize(qb, qx);
  writeln("Newton: f=", value, " x=[", qx[0], ",", qx[1], "] (true minimum: f=5 at [3, -1])");

  writeln("\n=================== ResidualFunction: LevenbergMarquardt curve fit ===================");
  double trueA = 2.5, trueK = 0.3, trueC = 1.0;
  double[] ts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
  double[] ys;
  foreach (t; ts) ys ~= trueA * exp(-trueK * t) + trueC;

  auto resid = new ExpDecayResiduals(ts, ys);
  auto params = dv([1.0, 1.0, 0.0]);
  double lmValue = (new LevenbergMarquardt()).optimize(resid, params);
  writeln("LevenbergMarquardt: residual-sq=", lmValue, " A=", params[0], " k=", params[1], " c=", params[2]);
  writeln("(generating parameters: A=", trueA, " k=", trueK, " c=", trueC, ")");

  writeln("\n=================== BayesianOptimization + RBFGaussianProcessSurrogate ===================");
  auto bowl = new Bowl1D();
  auto bx = dv([0.0]);
  auto lower = dv([-5.0]);
  auto upper = dv([5.0]);
  auto surrogate = new RBFGaussianProcessSurrogate();
  auto bo_options = new BayesianOptimizationOptions();
  bo_options.initial_samples = 8;
  bo_options.max_iterations = 50;
  bo_options.seed = 42;
  double boValue = (new BayesianOptimization(bo_options)).optimize(bowl, bx, lower, upper, surrogate);
  writeln("BayesianOptimization: f=", boValue, " x=[", bx[0], "] (true minimum: f=1 at [2])");
}
