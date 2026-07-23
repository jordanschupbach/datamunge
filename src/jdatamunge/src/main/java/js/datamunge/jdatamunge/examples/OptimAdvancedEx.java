package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.ArbitraryFunction;
import js.datamunge.jdatamunge.BayesianOptimization;
import js.datamunge.jdatamunge.BayesianOptimizationOptions;
import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.DVectorVector;
import js.datamunge.jdatamunge.DifferentiableFunction;
import js.datamunge.jdatamunge.GradientDescent;
import js.datamunge.jdatamunge.GradientDescentOptions;
import js.datamunge.jdatamunge.HessianFunction;
import js.datamunge.jdatamunge.LevenbergMarquardt;
import js.datamunge.jdatamunge.LevenbergMarquardtOptions;
import js.datamunge.jdatamunge.Newton;
import js.datamunge.jdatamunge.NewtonOptions;
import js.datamunge.jdatamunge.RBFGaussianProcessSurrogate;
import js.datamunge.jdatamunge.ResidualFunction;

/**
 * Exercises the newly un-ignored/director-enabled datamunge::optim module for Java: subclassing
 * DifferentiableFunction/HessianFunction/ResidualFunction directly in Java (SWIG director
 * callbacks from C++ back into user-defined Java classes) and running them through the matching
 * optimizer, plus BayesianOptimization with the ready-made RBFGaussianProcessSurrogate.
 */
public class OptimAdvancedEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  /** Simple 2-D paraboloid f(x) = (x0-2)^2 + (x1+1)^2 with an exact gradient -- the base
   *  ArbitraryFunction/DifferentiableFunction director extension point, run through plain
   *  GradientDescent. This is the load-bearing check that director callbacks work at all for
   *  this module in Java (previously the whole optim module was %ignore'd here). */
  static class Paraboloid extends DifferentiableFunction {
    @Override
    public double evaluate(DVector x) {
      double dx = x.get(0) - 2.0;
      double dy = x.get(1) + 1.0;
      return dx * dx + dy * dy;
    }

    @Override
    public DVector gradient(DVector x) {
      DVector g = new DVector(2, 0.0);
      g.set(0, 2.0 * (x.get(0) - 2.0));
      g.set(1, 2.0 * (x.get(1) + 1.0));
      return g;
    }
  }

  /** A 2-D quadratic bowl f(x) = (x-target)^T A (x-target) with a known minimum, exposed with
   *  its exact (constant) Hessian -- the natural example for Newton's method, which needs
   *  curvature information at every step. This is one of the 10 newest optimizer/interface
   *  pairs (HessianFunction + Newton), never previously bound for Java. */
  static class QuadraticBowlHessian extends HessianFunction {
    private final double[] target;
    private final double[][] a = {{4.0, 0.5}, {0.5, 2.0}};

    QuadraticBowlHessian(double[] target) {
      this.target = target;
    }

    @Override
    public double evaluate(DVector x) {
      double d0 = x.get(0) - target[0];
      double d1 = x.get(1) - target[1];
      return d0 * (a[0][0] * d0 + a[0][1] * d1) + d1 * (a[1][0] * d0 + a[1][1] * d1);
    }

    @Override
    public DVector gradient(DVector x) {
      double d0 = x.get(0) - target[0];
      double d1 = x.get(1) - target[1];
      DVector g = new DVector(2, 0.0);
      g.set(0, 2.0 * (a[0][0] * d0 + a[0][1] * d1));
      g.set(1, 2.0 * (a[1][0] * d0 + a[1][1] * d1));
      return g;
    }

    @Override
    public DVectorVector hessian(DVector x) {
      DVector row0 = new DVector(new double[] {2.0 * a[0][0], 2.0 * a[0][1]});
      DVector row1 = new DVector(new double[] {2.0 * a[1][0], 2.0 * a[1][1]});
      return new DVectorVector(new DVector[] {row0, row1});
    }
  }

  /** Residuals for fitting A*exp(-k*t) + c to synthetic noise-free data -- the classic
   *  nonlinear-least-squares curve fit, and the textbook use case for LevenbergMarquardt
   *  (ResidualFunction is standalone, not derived from ArbitraryFunction). */
  static class ExpDecayResidual extends ResidualFunction {
    private final double[] ts;
    private final double[] ys;

    ExpDecayResidual(double[] ts, double[] ys) {
      this.ts = ts;
      this.ys = ys;
    }

    @Override
    public DVector residuals(DVector p) {
      double a = p.get(0);
      double k = p.get(1);
      double c = p.get(2);
      DVector r = new DVector(ts.length, 0.0);
      for (int i = 0; i < ts.length; i++) {
        r.set(i, a * Math.exp(-k * ts[i]) + c - ys[i]);
      }
      return r;
    }

    @Override
    public DVectorVector jacobian(DVector p) {
      double a = p.get(0);
      double k = p.get(1);
      DVector[] rows = new DVector[ts.length];
      for (int i = 0; i < ts.length; i++) {
        double e = Math.exp(-k * ts[i]);
        rows[i] = new DVector(new double[] {e, -a * ts[i] * e, 1.0});
      }
      return new DVectorVector(rows);
    }
  }

  /** A simple 1-D bowl with a known minimum at x=1.7, no gradient exposed -- exactly the
   *  black-box setting BayesianOptimization targets. */
  static class Shifted1DBowl extends ArbitraryFunction {
    @Override
    public double evaluate(DVector x) {
      double d = x.get(0) - 1.7;
      return d * d + 0.1 * Math.sin(10.0 * x.get(0));
    }
  }

  public static void run() {
    System.out.println("=================== DifferentiableFunction: GradientDescent on a paraboloid ===================");
    Paraboloid f1 = new Paraboloid();
    DVector x1 = new DVector(new double[] {0.0, 0.0});
    GradientDescentOptions gdOptions = new GradientDescentOptions();
    gdOptions.setStep_size(0.1);
    gdOptions.setMax_iterations(500);
    gdOptions.setTolerance(1e-10);
    double value1 = new GradientDescent(gdOptions).optimize(f1, x1);
    System.out.printf(
        "GradientDescent:    f=%s x=[%s, %s] (true minimum: f=0 at [2.0, -1.0])%n",
        value1, x1.get(0), x1.get(1));

    System.out.println();
    System.out.println("=================== HessianFunction: Newton's method on a quadratic bowl ===================");
    double[] target = {3.0, -1.5};
    QuadraticBowlHessian f2 = new QuadraticBowlHessian(target);
    DVector x2 = new DVector(new double[] {0.0, 0.0});
    NewtonOptions newtonOptions = new NewtonOptions();
    newtonOptions.setMax_iterations(50);
    newtonOptions.setTolerance(1e-10);
    double value2 = new Newton(newtonOptions).optimize(f2, x2);
    System.out.printf(
        "Newton:             f=%s x=[%s, %s] (true minimum: f=0 at [%s, %s])%n",
        value2, x2.get(0), x2.get(1), target[0], target[1]);

    System.out.println();
    System.out.println("=================== ResidualFunction: LevenbergMarquardt curve fit ===================");
    double trueA = 5.0;
    double trueK = 0.7;
    double trueC = 1.0;
    double[] ts = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    double[] ys = new double[ts.length];
    for (int i = 0; i < ts.length; i++) {
      ys[i] = trueA * Math.exp(-trueK * ts[i]) + trueC;
    }
    ExpDecayResidual residualFn = new ExpDecayResidual(ts, ys);
    DVector p = new DVector(new double[] {1.0, 0.1, 0.0});
    LevenbergMarquardtOptions lmOptions = new LevenbergMarquardtOptions();
    lmOptions.setMax_iterations(200);
    lmOptions.setTolerance(1e-12);
    double value3 = new LevenbergMarquardt(lmOptions).optimize(residualFn, p);
    System.out.printf("LevenbergMarquardt: sum-sq-residual=%s params(A,k,c)=[%s, %s, %s]%n",
        value3, p.get(0), p.get(1), p.get(2));
    System.out.printf("                    true params(A,k,c)=[%s, %s, %s]%n", trueA, trueK, trueC);

    System.out.println();
    System.out.println("=================== BayesianOptimization with RBFGaussianProcessSurrogate ===================");
    Shifted1DBowl f4 = new Shifted1DBowl();
    DVector x4 = new DVector(new double[] {0.0});
    DVector lower = new DVector(new double[] {-3.0});
    DVector upper = new DVector(new double[] {5.0});
    RBFGaussianProcessSurrogate surrogate = new RBFGaussianProcessSurrogate(1.0, 1e-6);
    BayesianOptimizationOptions boOptions = new BayesianOptimizationOptions();
    boOptions.setInitial_samples(10);
    boOptions.setMax_iterations(60);
    boOptions.setSeed(java.math.BigInteger.valueOf(42));
    double value4 = new BayesianOptimization(boOptions).optimize(f4, x4, lower, upper, surrogate);
    System.out.printf(
        "BayesianOptimization: f=%s x=[%s] (target minimum near x=1.7)%n", value4, x4.get(0));
  }
}
