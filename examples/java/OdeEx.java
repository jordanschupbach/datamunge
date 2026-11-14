// CMD: just TARGET=ode_ex run-java
// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
import datamunge.ODEOptions;
import datamunge.ODESolution;
import datamunge.ODESolver;
import datamunge.RPlot;
import datamunge.StepMethod;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Collectors;

public class OdeEx {
  static double[] lastState(ODESolution sol) {
    return sol.stateAt(sol.size() - 1);
  }

  // The times of every step.
  static double[] times(ODESolution sol) {
    double[] times = new double[(int) sol.size()];
    for (int i = 0; i < times.length; i++) {
      times[i] = sol.timeAt(i);
    }
    return times;
  }

  // One state component (0-based) at every step.
  static double[] component(ODESolution sol, int c) {
    double[] values = new double[(int) sol.size()];
    for (int i = 0; i < values.length; i++) {
      values[i] = sol.stateAt(i)[c];
    }
    return values;
  }

  static ODESolver rk4Solver(double step) {
    var options = new ODEOptions();
    options.setMethod(StepMethod.RK4);
    options.setStepSize(step);
    return new ODESolver(options);
  }

  // C's %g (6 significant digits, trailing zeros dropped) for moderately sized values.
  static String g(double v) {
    return new BigDecimal(v).round(new MathContext(6)).stripTrailingZeros().toPlainString();
  }

  record NamedSystem(String name, double[] params, double[] y0) {}

  public static void main(String[] args) {
    // NOTE: like the Lua version this example uses the built-in named systems via solveBuiltin.

    System.out.println(
        "=================== Every named built-in system (no director needed) ===================");
    var solver = new ODESolver();
    var systems =
        new NamedSystem[] {
          new NamedSystem("exponential_decay", new double[] {1.0}, new double[] {1.0}),
          new NamedSystem("logistic_growth", new double[] {1.0, 1.0}, new double[] {0.5}),
          new NamedSystem("harmonic_oscillator", new double[] {1.0}, new double[] {1.0, 0.0}),
          new NamedSystem("van_der_pol", new double[] {1.0}, new double[] {2.0, 0.0}),
          new NamedSystem(
              "lorenz", new double[] {10.0, 28.0, 8.0 / 3}, new double[] {1.0, 1.0, 1.0}),
        };
    for (var system : systems) {
      var sol = solver.solveBuiltin(system.name(), system.params(), system.y0(), 0.0, 1.0);
      String state =
          Arrays.stream(lastState(sol)).mapToObj(OdeEx::g).collect(Collectors.joining(", "));
      System.out.println(
          String.format(
              Locale.ROOT,
              "%-22s steps=%-6d final state=[%s]",
              system.name(),
              sol.getStepsTaken(),
              state));
    }

    System.out.println(
        "\n=================== Harmonic oscillator (energy conservation) ===================");
    var sol =
        rk4Solver(0.01)
            .solveBuiltin(
                "harmonic_oscillator", new double[] {1.0}, new double[] {1.0, 0.0}, 0.0, 20.0);
    double[] last = lastState(sol);
    double x = last[0];
    double v = last[1];
    System.out.println(
        String.format(Locale.ROOT, "x(20) = %.6f (cos(20) = %.6f)", x, Math.cos(20.0)));
    System.out.println(
        String.format(Locale.ROOT, "energy x^2+v^2 = %.6f (should stay near 1.0)", x * x + v * v));
    double[] ts = times(sol);
    var p = RPlot.plot(ts, component(sol, 0), "l", "x(t)");
    p.lines(ts, component(sol, 1), "v(t)");
    p.title("Harmonic Oscillator")
        .xLabel("t")
        .yLabel("state")
        .saveSvg("ode_harmonic_oscillator_java.svg");
    System.out.println("wrote ode_harmonic_oscillator_java.svg");

    System.out.println(
        "\n=================== Lorenz attractor (phase plane) ===================");
    sol =
        rk4Solver(0.005)
            .solveBuiltin(
                "lorenz",
                new double[] {10.0, 28.0, 8.0 / 3},
                new double[] {1.0, 1.0, 1.0},
                0.0,
                25.0);
    System.out.println("steps_taken = " + sol.getStepsTaken());
    RPlot.plot(component(sol, 0), component(sol, 2), "l", "trajectory")
        .title("Lorenz Attractor (x-z phase plane)")
        .xLabel("x")
        .yLabel("z")
        .saveSvg("ode_lorenz_phase_plane_java.svg");
    System.out.println("wrote ode_lorenz_phase_plane_java.svg");
  }
}
