module app;

import std.stdio : writeln, writefln;
import std.math : exp, cos;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

DVector last_state(ODESolution sol) {
  return sol.state_at(sol.size() - 1);
}

// NOTE: the Python ode example demonstrates a live, user-supplied RHS by subclassing
// datamunge.RHS (a SWIG director). RHS is director-enabled ONLY in the Python binding (see
// rhs.hpp's doc comment: a virtual method returning std::vector<double> cannot be wrapped as a
// working director on the other backends). So the D binding -- like Lua/Ruby -- can only use the
// built-in named systems via solve_builtin(). The method-comparison section below therefore uses
// the built-in "exponential_decay" system (dy/dt = -y) rather than a hand-written RHS, but sweeps
// the integrator across every StepMethod exactly as the Python version does.

void main() {
  writeln("=================== dy/dt = -y, y(0) = 1 (built-in exponential_decay across methods) ===================");
  double t_end = 2.0;
  double exact = exp(-t_end);
  writefln("exact y(%.1f) = %.6f", t_end, exact);

  struct Config {
    string name;
    StepMethod method;
    size_t order;
    double step;
    bool adaptive;
  }
  Config[] configs = [
    Config("Euler", StepMethod.Euler, 4, 0.01, false),
    Config("Midpoint/RK2", StepMethod.Midpoint, 4, 0.01, false),
    Config("RK4", StepMethod.RK4, 4, 0.01, false),
    Config("RK45 adaptive", StepMethod.RK45, 4, 0.1, true),
    Config("Adams-Bashforth", StepMethod.AdamsBashforth, 4, 0.01, false),
    Config("Adams-Bashforth-Moulton", StepMethod.AdamsMoulton, 4, 0.01, false),
  ];
  writefln("%-28s%-14s%-14s%-8s%s", "method", "y(2)", "abs.error", "steps", "evals");
  foreach (c; configs) {
    auto options = new ODEOptions();
    options.method = c.method;
    options.multistep_order = c.order;
    options.step_size = c.step;
    if (c.adaptive) {
      options.abs_tol = 1e-8;
      options.rel_tol = 1e-8;
    }
    auto solver = new ODESolver(options);
    auto sol = solver.solve_builtin("exponential_decay", dv([1.0]), dv([1.0]), 0.0, t_end);
    double y_final = last_state(sol)[0];
    double err = y_final - exact;
    if (err < 0) err = -err;
    writefln("%-28s%-14.6f%-14.6f%-8d%d", c.name, y_final, err, sol.steps_taken, sol.function_evaluations);
  }

  writeln("\n=================== Every named built-in system ===================");
  auto solver = new ODESolver();
  struct System {
    string name;
    double[] params;
    double[] y0;
  }
  System[] systems = [
    System("exponential_decay", [1.0], [1.0]),
    System("logistic_growth", [1.0, 1.0], [0.5]),
    System("harmonic_oscillator", [1.0], [1.0, 0.0]),
    System("van_der_pol", [1.0], [2.0, 0.0]),
    System("lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0]),
  ];
  foreach (s; systems) {
    auto sol = solver.solve_builtin(s.name, dv(s.params), dv(s.y0), 0.0, 1.0);
    auto fs = last_state(sol);
    string parts;
    for (size_t i = 0; i < fs.size(); i++) {
      import std.format : format;
      parts ~= (i == 0 ? "" : ", ") ~ format("%g", fs[i]);
    }
    writefln("%-22s steps=%-6d final state=[%s]", s.name, sol.steps_taken, parts);
  }

  writeln("\n=================== Harmonic oscillator (energy conservation) ===================");
  auto options = new ODEOptions();
  options.method = StepMethod.RK4;
  options.step_size = 0.01;
  solver = new ODESolver(options);
  auto sol = solver.solve_builtin("harmonic_oscillator", dv([1.0]), dv([1.0, 0.0]), 0.0, 20.0);
  auto fin = last_state(sol);
  double x = fin[0];
  double v = fin[1];
  writefln("x(20) = %.6f (cos(20) = %.6f)", x, cos(20.0));
  writefln("energy x^2+v^2 = %.6f (should stay near 1.0)", x * x + v * v);

  double[] t_series, x_series, v_series;
  for (size_t i = 0; i < sol.size(); i++) {
    t_series ~= sol.time_at(i);
    auto st = sol.state_at(i);
    x_series ~= st[0];
    v_series ~= st[1];
  }
  auto plot = RPlot.plot(dv(t_series), dv(x_series), "l", "x(t)");
  plot.lines(dv(t_series), dv(v_series), "v(t)");
  plot.title("Harmonic Oscillator").x_label("t").y_label("state");
  plot.save_svg("ode_harmonic_oscillator_d.svg");
  writeln("wrote ode_harmonic_oscillator_d.svg");

  writeln("\n=================== Lorenz attractor (phase plane) ===================");
  options = new ODEOptions();
  options.method = StepMethod.RK4;
  options.step_size = 0.005;
  solver = new ODESolver(options);
  sol = solver.solve_builtin("lorenz", dv([10.0, 28.0, 8.0 / 3.0]), dv([1.0, 1.0, 1.0]), 0.0, 25.0);
  writefln("steps_taken = %d", sol.steps_taken);

  double[] xs, zs;
  for (size_t i = 0; i < sol.size(); i++) {
    auto st = sol.state_at(i);
    xs ~= st[0];
    zs ~= st[2];
  }
  plot = RPlot.plot(dv(xs), dv(zs), "l", "trajectory");
  plot.title("Lorenz Attractor (x-z phase plane)").x_label("x").y_label("z");
  plot.save_svg("ode_lorenz_phase_plane_d.svg");
  writeln("wrote ode_lorenz_phase_plane_d.svg");
}
