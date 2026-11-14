// CMD: just TARGET=optim_advanced_ex run-java
// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Java.
import datamunge.AugmentedLagrangian;
import datamunge.AugmentedLagrangianOptions;
import datamunge.BayesianOptimization;
import datamunge.BayesianOptimizationOptions;
import datamunge.FISTA;
import datamunge.FISTAOptions;
import datamunge.HessianFunction;
import datamunge.InteriorPoint;
import datamunge.InteriorPointOptions;
import datamunge.LevenbergMarquardt;
import datamunge.LevenbergMarquardtOptions;
import datamunge.NativeObject;
import datamunge.Newton;
import datamunge.NewtonOptions;
import datamunge.ProximalGradient;
import datamunge.ProximalGradientOptions;
import datamunge.RBFGaussianProcessSurrogate;
import datamunge.Ref;
import datamunge.ResidualFunction;
import datamunge.SQP;
import datamunge.SQPOptions;
import datamunge.TrustRegionNewton;
import datamunge.TrustRegionNewtonOptions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From Java, extend the interface class and override its virtual
// methods; the optimizer then calls the Java overrides. The coordinates are passed in a
// datamunge.Ref and updated in place.
public class OptimAdvancedEx {
  // f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
  static class Quadratic extends HessianFunction {
    @Override
    public double evaluate(double[] c) {
      return Math.pow(c[0] - 1, 2) + 10 * Math.pow(c[1] + 2, 2);
    }

    @Override
    public double[] gradient(double[] c) {
      return new double[] {2 * (c[0] - 1), 20 * (c[1] + 2)};
    }

    @Override
    public List<double[]> hessian(double[] c) {
      return List.of(new double[] {2.0, 0.0}, new double[] {0.0, 20.0});
    }
  }

  // Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
  static class Exponential extends ResidualFunction {
    private final double[] ts;

    Exponential(double[] ts) {
      this.ts = ts;
    }

    @Override
    public double[] residuals(double[] p) {
      return Arrays.stream(ts).map(t -> p[0] * Math.exp(p[1] * t) - 2 * Math.exp(0.5 * t)).toArray();
    }

    @Override
    public List<double[]> jacobian(double[] p) {
      List<double[]> rows = new ArrayList<>();
      for (double t : ts) {
        rows.add(new double[] {Math.exp(p[1] * t), p[0] * t * Math.exp(p[1] * t)});
      }
      return rows;
    }
  }

  // Leaves gradient and hessian out.
  static class Incomplete extends HessianFunction {
    @Override
    public double evaluate(double[] c) {
      return 0.0;
    }
  }

  public static void main(String[] args) {
    System.out.println(
        "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate)"
            + " ===================");
    // Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
    // then check the acquisition function favors an unexplored region.
    double[] xs = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};
    double[] ys = Arrays.stream(xs).map(x -> (x - 1) * (x - 1)).toArray();
    var surrogate = new RBFGaussianProcessSurrogate(1.0, 1e-6);
    surrogate.fit(Arrays.stream(xs).mapToObj(x -> new double[] {x}).toList(), ys);
    double incumbent = Math.min(0.0, Arrays.stream(ys).min().orElse(0.0)); // best observed so far
    System.out.println("incumbent (best observed y): " + incumbent);
    for (double x : new double[] {0.25, 1.0, 2.5}) {
      System.out.println(
          String.format(
              Locale.ROOT,
              "acquisition(x=%.2f) = %.6f",
              x,
              surrogate.acquisition(new double[] {x}, incumbent)));
    }

    System.out.println(
        "\n=================== Options structs + optimizer construction (all 10 new classes)"
            + " ===================");
    // Every new Options struct and optimizer class constructs cleanly, proving
    // the binding for the whole batch is wired correctly.
    var pgOptions = new ProximalGradientOptions();
    pgOptions.setStepSize(0.05);
    var pg = new ProximalGradient(pgOptions);
    var fista = new FISTA(new FISTAOptions());
    var lmOptions = new LevenbergMarquardtOptions();
    lmOptions.setMaxIterations(200);
    var lm = new LevenbergMarquardt(lmOptions);
    var newton = new Newton(new NewtonOptions());
    var trn = new TrustRegionNewton(new TrustRegionNewtonOptions());
    var al = new AugmentedLagrangian(new AugmentedLagrangianOptions());
    var sqp = new SQP(new SQPOptions());
    var ip = new InteriorPoint(new InteriorPointOptions());
    var boOptions = new BayesianOptimizationOptions();
    boOptions.setInitialSamples(8);
    var bo = new BayesianOptimization(boOptions);
    for (NativeObject optimizer : List.of(pg, fista, lm, newton, trn, al, sqp, ip, bo)) {
      System.out.println(optimizer.getClass().getSimpleName() + ": " + optimizer);
    }

    System.out.println(
        "\n=================== Java objectives (subclassing the C++ interfaces) ===================");
    // Newton's method on the quadratic bowl.
    var start = new Ref<>(new double[] {5.0, 5.0});
    double value = newton.optimize(new Quadratic(), start);
    System.out.println(
        String.format(
            Locale.ROOT,
            "Newton: f = %s at (%.6f, %.6f) (exact minimum 0 at (1, -2))",
            formatG(value, 6),
            start.value[0],
            start.value[1]));

    // Levenberg-Marquardt fitting y = a * exp(b t).
    var params = new Ref<>(new double[] {1.0, 0.1});
    double sse = lm.optimize(new Exponential(new double[] {0.0, 0.5, 1.0, 1.5, 2.0}), params);
    System.out.println(
        String.format(
            Locale.ROOT,
            "LevenbergMarquardt: residual = %s, a = %.6f, b = %.6f (exact: 2, 0.5)",
            formatG(sse, 3),
            params.value[0],
            params.value[1]));

    // An interface method a subclass leaves out throws when the optimizer calls it.
    boolean ok = true;
    boolean mentions = false;
    try {
      newton.optimize(new Incomplete(), new Ref<>(new double[] {0.0, 0.0}));
    } catch (RuntimeException e) {
      ok = false;
      mentions = e.getMessage() != null && e.getMessage().contains("gradient");
    }
    System.out.println("missing gradient(): ok=" + ok + ", error mentions it: " + mentions);
  }

  // C-style %g (Java's %g never switches to the shortest form nor strips trailing zeros).
  static String formatG(double value, int precision) {
    if (value == 0.0) {
      return "0";
    }
    int exponent = (int) Math.floor(Math.log10(Math.abs(value)));
    String mantissa;
    if (exponent < -4 || exponent >= precision) {
      String s = String.format(Locale.ROOT, "%." + (precision - 1) + "e", value);
      int e = s.indexOf('e');
      mantissa = stripZeros(s.substring(0, e));
      int exp = Integer.parseInt(s.substring(e + 1));
      return mantissa + (exp < 0 ? "e-" : "e+") + String.format("%02d", Math.abs(exp));
    }
    return stripZeros(
        String.format(Locale.ROOT, "%." + Math.max(0, precision - 1 - exponent) + "f", value));
  }

  static String stripZeros(String s) {
    return s.contains(".") ? s.replaceAll("0+$", "").replaceAll("\\.$", "") : s;
  }
}
