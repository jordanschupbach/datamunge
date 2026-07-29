module app;

import std.stdio : writeln;
import std.math : sin, cos, PI, sqrt;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

class QuadraticBowl : DifferentiableFunction {
  // A plain 3-D quadratic bowl with a known minimum.
  double[] target;
  this(double[] t) { super(); target = t; }
  override double evaluate(DVector coords) {
    double s = 0.0;
    foreach (i, t; target) s += (coords[i] - t) * (coords[i] - t);
    return s;
  }
  override DVector gradient(DVector coords) {
    auto g = new DVector();
    foreach (i, t; target) g.push_back(2.0 * (coords[i] - t));
    return g;
  }
}

class RosenbrockFn : DifferentiableFunction {
  // The classic Rosenbrock "banana" function -- a much harder landscape.
  override double evaluate(DVector x) {
    double a = 1.0 - x[0];
    double b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
  }
  override DVector gradient(DVector x) {
    auto g = new DVector();
    g.push_back(-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]));
    g.push_back(200.0 * (x[1] - x[0] * x[0]));
    return g;
  }
}

class LinearRegressionLoss : DifferentiableSeparableFunction {
  // Ordinary least squares as a sum of per-example losses -- the textbook case for SGD.
  double[][] xs;
  double[] ys;
  this(double[][] xData, double[] yData) { super(); xs = xData; ys = yData; }
  override size_t num_functions() const { return ys.length; }
  double predict(DVector w, size_t i) {
    double s = 0.0;
    for (size_t j = 0; j < w.size(); j++) s += w[j] * xs[i][j];
    return s;
  }
  override double evaluate_term(DVector w, size_t i) {
    double err = predict(w, i) - ys[i];
    return err * err;
  }
  override DVector gradient_term(DVector w, size_t i) {
    double err = predict(w, i) - ys[i];
    auto g = new DVector();
    for (size_t j = 0; j < w.size(); j++) g.push_back(2.0 * err * xs[i][j]);
    return g;
  }
}

class BumpyFunction : ArbitraryFunction {
  // A bumpy, multimodal landscape -- no gradient available.
  override double evaluate(DVector x) {
    double bowl = (x[0] - 3.0) ^^ 2 + (x[1] + 1.0) ^^ 2;
    double ripples = 5.0 * sin(x[0]) * cos(x[1]);
    return bowl + ripples;
  }
}

class RastriginFunction : ArbitraryFunction {
  // The classic Rastrigin function -- highly multimodal, global minimum f=0 at the origin.
  override double evaluate(DVector x) {
    double total = 10.0 * x.size();
    for (size_t i = 0; i < x.size(); i++) {
      double xi = x[i];
      total += xi * xi - 10.0 * cos(2.0 * PI * xi);
    }
    return total;
  }
}

void main() {
  writeln("=================== DifferentiableFunction: three optimizers, one bowl ===================");
  double[] target = [4.0, -2.0, 1.0];

  auto f = new QuadraticBowl(target);
  auto x = dv([0.0, 0.0, 0.0]);
  auto gd_options = new GradientDescentOptions();
  gd_options.step_size = 0.1;
  gd_options.momentum = 0.0;
  gd_options.max_iterations = 1000;
  gd_options.tolerance = 1e-10;
  double value = (new GradientDescent(gd_options)).optimize(f, x);
  writeln("GradientDescent: f=", value, " x=[", x[0], ",", x[1], ",", x[2], "]");

  f = new QuadraticBowl(target);
  x = dv([0.0, 0.0, 0.0]);
  value = (new Adam()).optimize(f, x);
  writeln("Adam:             f=", value, " x=[", x[0], ",", x[1], ",", x[2], "]");

  f = new QuadraticBowl(target);
  x = dv([0.0, 0.0, 0.0]);
  value = (new LBFGS()).optimize(f, x);
  writeln("LBFGS:            f=", value, " x=[", x[0], ",", x[1], ",", x[2], "] (converges in far fewer iterations)");

  writeln("\n=================== LBFGS on the Rosenbrock function ===================");
  auto rf = new RosenbrockFn();
  auto rx = dv([-1.2, 1.0]);
  value = (new LBFGS()).optimize(rf, rx);
  writeln("f=", value, " x=[", rx[0], ",", rx[1], "] (true minimum: f=0 at [1, 1])");

  writeln("\n=================== DifferentiableSeparableFunction: SGD vs. closed-form LM ===================");
  auto iris = DataFrame.iris();
  double[] sepal_length;
  double[] sepal_width;
  double[] petal_length;
  auto n = iris.nrows();
  for (size_t i = 0; i < n; i++) {
    sepal_length ~= iris.numeric_at("Sepal.Length", i);
    sepal_width ~= iris.numeric_at("Sepal.Width", i);
    petal_length ~= iris.numeric_at("Petal.Length", i);
  }

  double mean_of(double[] v) {
    double s = 0.0;
    foreach (e; v) s += e;
    return s / v.length;
  }
  double stddev_of(double[] v, double mean) {
    double s = 0.0;
    foreach (e; v) s += (e - mean) * (e - mean);
    return sqrt(s / v.length);
  }

  double mean1 = mean_of(sepal_length);
  double std1 = stddev_of(sepal_length, mean1);
  double mean2 = mean_of(sepal_width);
  double std2 = stddev_of(sepal_width, mean2);

  double[][] x_std;
  for (size_t i = 0; i < n; i++) {
    x_std ~= [1.0, (sepal_length[i] - mean1) / std1, (sepal_width[i] - mean2) / std2];
  }
  auto loss = new LinearRegressionLoss(x_std, petal_length);

  auto w_std = dv([0.0, 0.0, 0.0]);
  auto sgd_options = new SGDOptions();
  sgd_options.step_size = 0.01;
  sgd_options.max_epochs = 300;
  sgd_options.batch_size = 8;
  (new SGD(sgd_options)).optimize(loss, w_std);

  double[] w = [w_std[0] - w_std[1] * mean1 / std1 - w_std[2] * mean2 / std2, w_std[1] / std1, w_std[2] / std2];
  writeln("SGD weights (intercept, Sepal.Length, Sepal.Width): ", w);

  auto lm = new LM(iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
  auto lmCoefs = lm.coefficients();
  double[] lmCoefArr;
  for (size_t i = 0; i < lmCoefs.size(); i++) lmCoefArr ~= lmCoefs[i];
  writeln("LM  weights (intercept, Sepal.Length, Sepal.Width): ", lmCoefArr, " (closed-form OLS, for comparison)");

  writeln("\n=================== ArbitraryFunction: derivative-free SimulatedAnnealing ===================");
  auto bf = new BumpyFunction();
  auto bx = dv([0.0, 0.0]);
  auto sa_options = new SimulatedAnnealingOptions();
  sa_options.initial_temperature = 10.0;
  sa_options.cooling_rate = 0.999;
  sa_options.max_iterations = 20000;
  sa_options.step_std_dev = 0.5;
  value = (new SimulatedAnnealing(sa_options)).optimize(bf, bx);
  writeln("f=", value, " x=[", bx[0], ",", bx[1], "] (found without ever computing a gradient)");

  writeln("\n=================== Population-based methods on the Rastrigin function ===================");
  auto lower = dv([-5.12, -5.12]);
  auto upper = dv([5.12, 5.12]);

  auto rg = new RastriginFunction();
  auto px1 = dv([3.0, -4.0]);
  auto pso_options = new PSOOptions();
  pso_options.topology = "global";
  pso_options.inertia_strategy = "constant";
  value = (new PSO(pso_options)).optimize(rg, px1, lower, upper);
  writeln("PSO (global topology, constant inertia):    f=", value, " x=[", px1[0], ",", px1[1], "]");

  rg = new RastriginFunction();
  auto px2 = dv([3.0, -4.0]);
  pso_options = new PSOOptions();
  pso_options.topology = "ring";
  pso_options.inertia_strategy = "linear_decay";
  value = (new PSO(pso_options)).optimize(rg, px2, lower, upper);
  writeln("PSO (ring topology, linear-decay inertia):  f=", value, " x=[", px2[0], ",", px2[1], "]");

  rg = new RastriginFunction();
  auto px3 = dv([3.0, -4.0]);
  auto de_options = new DEOptions();
  de_options.mutation_strategy = "rand1";
  de_options.crossover_strategy = "binomial";
  value = (new DifferentialEvolution(de_options)).optimize(rg, px3, lower, upper);
  writeln("DE (rand1/binomial):                        f=", value, " x=[", px3[0], ",", px3[1], "]");

  rg = new RastriginFunction();
  auto px4 = dv([3.0, -4.0]);
  de_options = new DEOptions();
  de_options.mutation_strategy = "best1";
  de_options.crossover_strategy = "exponential";
  value = (new DifferentialEvolution(de_options)).optimize(rg, px4, lower, upper);
  writeln("DE (best1/exponential):                     f=", value, " x=[", px4[0], ",", px4[1], "]");

  rg = new RastriginFunction();
  auto px5 = dv([3.0, -4.0]);
  auto ga_options = new GAOptions();
  ga_options.selection_strategy = "tournament";
  ga_options.crossover_strategy = "blend";
  value = (new GeneticAlgorithm(ga_options)).optimize(rg, px5, lower, upper);
  writeln("GA (tournament/blend, elitism on):           f=", value, " x=[", px5[0], ",", px5[1], "]");

  rg = new RastriginFunction();
  auto px6 = dv([3.0, -4.0]);
  ga_options = new GAOptions();
  ga_options.selection_strategy = "rank";
  ga_options.crossover_strategy = "uniform";
  ga_options.elitism = false;
  value = (new GeneticAlgorithm(ga_options)).optimize(rg, px6, lower, upper);
  writeln("GA (rank/uniform, elitism off):              f=", value, " x=[", px6[0], ",", px6[1], "] (true minimum: f=0 at [0, 0])");
}
