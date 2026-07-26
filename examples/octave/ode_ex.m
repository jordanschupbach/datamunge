1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function s = state_str(sol, idx)
  datamunge;
  % state_at() returns std::vector<double>, which arrives as a native Octave cell array.
  st = ODESolution_state_at(sol, idx);
  parts = {};
  for i = 1:numel(st)
    parts{end + 1} = sprintf("%g", st{i});
  end
  s = strjoin(parts, ", ");
endfunction

function idx = last_index(sol)
  datamunge;
  idx = ODESolution_size(sol) - 1;
endfunction

% NOTE: the Python/Ruby ode examples also demonstrate a live, user-supplied RHS by subclassing
% datamunge.RHS (a SWIG director). datamunge::ode::RHS::evaluate() returns std::vector<double>,
% and RHS is not director-enabled in the Octave binding (rhs.hpp documents that only Python's
% backend can generate a working director for a vector<double>-returning virtual method); this
% example therefore uses the built-in named systems via solve_builtin(), like every non-Python
% binding.

printf("=================== Every named built-in system (no director needed) ===================\n");
solver = ODESolver();
systems = { ...
  {"exponential_decay", [1.0], [1.0]}, ...
  {"logistic_growth", [1.0, 1.0], [0.5]}, ...
  {"harmonic_oscillator", [1.0], [1.0, 0.0]}, ...
  {"van_der_pol", [1.0], [2.0, 0.0]}, ...
  {"lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0]} ...
};
for i = 1:numel(systems)
  s = systems{i};
  sol = ODESolver_solve_builtin(solver, s{1}, dv(s{2}), dv(s{3}), 0.0, 1.0);
  printf("%-22s steps=%-6d final state=[%s]\n", ...
         s{1}, ODESolution_steps_taken_get(sol), state_str(sol, last_index(sol)));
end

printf("\n=================== Harmonic oscillator (energy conservation) ===================\n");
options = ODEOptions();
ODEOptions_method_set(options, StepMethod_RK4);
ODEOptions_step_size_set(options, 0.01);
solver = ODESolver(options);
sol = ODESolver_solve_builtin(solver, "harmonic_oscillator", dv([1.0]), dv([1.0, 0.0]), 0.0, 20.0);
final = ODESolution_state_at(sol, last_index(sol));
xval = final{1};
vval = final{2};
printf("x(20) = %.6f (cos(20) = %.6f)\n", xval, cos(20));
printf("energy x^2+v^2 = %.6f (should stay near 1.0)\n", xval * xval + vval * vval);

nsteps = ODESolution_size(sol);
t_series = [];
x_series = [];
v_series = [];
for i = 0:(nsteps - 1)
  t_series(end + 1) = ODESolution_time_at(sol, i);
  st = ODESolution_state_at(sol, i);
  x_series(end + 1) = st{1};
  v_series(end + 1) = st{2};
end
plot = RPlot_plot(dv(t_series), dv(x_series), "l", "x(t)");
RPlot_lines(plot, dv(t_series), dv(v_series), "v(t)");
Plot_title(plot, "Harmonic Oscillator");
Plot_x_label(plot, "t");
Plot_y_label(plot, "state");
Plot_save_svg(plot, "ode_harmonic_oscillator_octave.svg");
printf("wrote ode_harmonic_oscillator_octave.svg\n");

printf("\n=================== Lorenz attractor (phase plane) ===================\n");
options = ODEOptions();
ODEOptions_method_set(options, StepMethod_RK4);
ODEOptions_step_size_set(options, 0.005);
solver = ODESolver(options);
sol = ODESolver_solve_builtin(solver, "lorenz", dv([10.0, 28.0, 8.0 / 3.0]), dv([1.0, 1.0, 1.0]), 0.0, 25.0);
printf("steps_taken = %d\n", ODESolution_steps_taken_get(sol));

nsteps = ODESolution_size(sol);
xs = [];
zs = [];
for i = 0:(nsteps - 1)
  st = ODESolution_state_at(sol, i);
  xs(end + 1) = st{1};
  zs(end + 1) = st{3};
end
plot = RPlot_plot(dv(xs), dv(zs), "l", "trajectory");
Plot_title(plot, "Lorenz Attractor (x-z phase plane)");
Plot_x_label(plot, "x");
Plot_y_label(plot, "z");
Plot_save_svg(plot, "ode_lorenz_phase_plane_octave.svg");
printf("wrote ode_lorenz_phase_plane_octave.svg\n");
