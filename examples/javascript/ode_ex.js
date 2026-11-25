// CMD: just TARGET=ode_ex run-javascript
// ODEs: a live custom RHS across every step method, the built-in named systems, a
// harmonic-oscillator energy check and a Lorenz phase plot.

const datamunge = require("../../index.js");

// A live, user-supplied RHS subclassing datamunge.RHS. The JavaScript binding supports SWIG
// directors, so evaluate() below is called back from the C++ solver at every stage.
class ExponentialDecay extends datamunge.RHS {
  constructor(k) {
    super();
    this.k = k;
  }

  evaluate(t, y) {
    return [-this.k * y[0]];
  }
}

function lastState(sol) {
  return sol.state_at(sol.size() - 1);
}

// The time vector followed by the requested state components (0-based), one value per step.
function solutionSeries(sol, ...components) {
  const ts = [];
  const series = components.map(() => []);
  for (let i = 0; i < sol.size(); i++) {
    ts.push(sol.time_at(i));
    const state = sol.state_at(i);
    components.forEach((c, j) => series[j].push(state[c]));
  }
  return [ts, series];
}

function rk4Solver(step) {
  const options = new datamunge.ODEOptions();
  options.method = datamunge.StepMethod.RK4;
  options.step_size = step;
  return new datamunge.ODESolver(options);
}

console.log("=================== dy/dt = -y, y(0) = 1 (live custom RHS) ===================");
const decay = new ExponentialDecay(1.0);
const tEnd = 2.0;
const exact = Math.exp(-tEnd);
console.log(`exact y(${tEnd.toFixed(1)}) = ${exact.toFixed(6)}`);

const configs = [
  ["Euler", datamunge.StepMethod.Euler, 4, 0.01, null, null],
  ["Midpoint/RK2", datamunge.StepMethod.Midpoint, 4, 0.01, null, null],
  ["RK4", datamunge.StepMethod.RK4, 4, 0.01, null, null],
  ["RK45 adaptive", datamunge.StepMethod.RK45, 4, 0.1, 1e-8, 1e-8],
  ["Adams-Bashforth", datamunge.StepMethod.AdamsBashforth, 4, 0.01, null, null],
  ["Adams-Bashforth-Moulton", datamunge.StepMethod.AdamsMoulton, 4, 0.01, null, null],
];
console.log(`${"method".padEnd(28)}${"y(2)".padEnd(14)}${"abs.error".padEnd(14)}${"steps".padEnd(8)}evals`);
for (const [name, method, order, step, absTol, relTol] of configs) {
  const options = new datamunge.ODEOptions();
  options.method = method;
  options.multistep_order = order;
  options.step_size = step;
  if (absTol !== null) {
    options.abs_tol = absTol;
    options.rel_tol = relTol;
  }
  const sol = new datamunge.ODESolver(options).solve(decay, [1.0], 0.0, tEnd);
  const yFinal = lastState(sol)[0];
  console.log(
    `${name.padEnd(28)}${yFinal.toFixed(6).padEnd(14)}${Math.abs(yFinal - exact).toFixed(6).padEnd(14)}` +
      `${String(sol.steps_taken).padEnd(8)}${sol.function_evaluations}`
  );
}

console.log("\n=================== Every named built-in system (no director needed) ===================");
const solver = new datamunge.ODESolver();
for (const [system, params, y0] of [
  ["exponential_decay", [1.0], [1.0]],
  ["logistic_growth", [1.0, 1.0], [0.5]],
  ["harmonic_oscillator", [1.0], [1.0, 0.0]],
  ["van_der_pol", [1.0], [2.0, 0.0]],
  ["lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0]],
]) {
  const sol = solver.solve_builtin(system, params, y0, 0.0, 1.0);
  const state = Array.from(lastState(sol)).join(", ");
  console.log(`${system.padEnd(22)} steps=${String(sol.steps_taken).padEnd(6)} final state=[${state}]`);
}

console.log("\n=================== Harmonic oscillator (energy conservation) ===================");
let sol = rk4Solver(0.01).solve_builtin("harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0);
const [x, v] = lastState(sol);
console.log(`x(20) = ${x.toFixed(6)} (cos(20) = ${Math.cos(20).toFixed(6)})`);
console.log(`energy x^2+v^2 = ${(x * x + v * v).toFixed(6)} (should stay near 1.0)`);

const [ts, [xs, vs]] = solutionSeries(sol, 0, 1);
let plot = datamunge.RPlot.plot(ts, xs, "l", "x(t)");
plot.lines(ts, vs, "v(t)");
plot.title("Harmonic Oscillator").x_label("t").y_label("state");
plot.save_svg("ode_harmonic_oscillator_javascript.svg");
console.log("wrote ode_harmonic_oscillator_javascript.svg");

console.log("\n=================== Lorenz attractor (phase plane) ===================");
sol = rk4Solver(0.005).solve_builtin("lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0], 0.0, 25.0);
console.log(`steps_taken = ${sol.steps_taken}`);

const [, [lx, lz]] = solutionSeries(sol, 0, 2);
plot = datamunge.RPlot.plot(lx, lz, "l", "trajectory");
plot.title("Lorenz Attractor (x-z phase plane)").x_label("x").y_label("z");
plot.save_svg("ode_lorenz_phase_plane_javascript.svg");
console.log("wrote ode_lorenz_phase_plane_javascript.svg");
