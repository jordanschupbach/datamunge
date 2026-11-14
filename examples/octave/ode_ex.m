% CMD: just TARGET=ode_ex run-octave
% ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
1;

function state = last_state (sol)
  state = sol.state_at (sol.size_ () - 1);
end

% The time vector followed by the requested state components (1-based), one value per step.
function [ts, components] = solution_series (sol, varargin)
  n = sol.size_ ();
  ts = zeros (1, n);
  states = zeros (n, numel (sol.state_at (0)));
  for i = 0:n-1
    ts(i+1) = sol.time_at (i);
    states(i+1, :) = sol.state_at (i);
  end
  components = cellfun (@(c) states(:, c)', varargin, 'UniformOutput', false);
end

function solver = rk4_solver (step)
  options = datamunge.ODEOptions ();
  options.method = datamunge.StepMethod.RK4;
  options.step_size = step;
  solver = datamunge.ODESolver (options);
end

% NOTE: like the Lua version this example uses the built-in named systems via solve_builtin.

printf ('=================== Every named built-in system (no director needed) ===================\n');
solver = datamunge.ODESolver ();
systems = {
  'exponential_decay', 1.0, 1.0
  'logistic_growth', [1.0 1.0], 0.5
  'harmonic_oscillator', 1.0, [1.0 0.0]
  'van_der_pol', 1.0, [2.0 0.0]
  'lorenz', [10.0 28.0 8.0/3], [1.0 1.0 1.0]
};
for i = 1:rows (systems)
  [name, params, y0] = systems{i, :};
  sol = solver.solve_builtin (name, params, y0, 0.0, 1.0);
  state = strjoin (arrayfun (@(x) sprintf ('%g', x), last_state (sol), 'UniformOutput', false), ', ');
  printf ('%-22s steps=%-6d final state=[%s]\n', name, sol.steps_taken, state);
end

printf ('\n=================== Harmonic oscillator (energy conservation) ===================\n');
sol = rk4_solver (0.01).solve_builtin ('harmonic_oscillator', 1.0, [1.0 0.0], 0.0, 20.0);
last = last_state (sol);
x = last(1);
v = last(2);
printf ('x(20) = %.6f (cos(20) = %.6f)\n', x, cos (20.0));
printf ('energy x^2+v^2 = %.6f (should stay near 1.0)\n', x * x + v * v);
[ts, series] = solution_series (sol, 1, 2);
p = datamunge.RPlot.plot (ts, series{1}, 'l', 'x(t)');
p.lines (ts, series{2}, 'v(t)');
p.title ('Harmonic Oscillator').x_label ('t').y_label ('state');
p.save_svg ('ode_harmonic_oscillator_octave.svg');
printf ('wrote ode_harmonic_oscillator_octave.svg\n');

printf ('\n=================== Lorenz attractor (phase plane) ===================\n');
sol = rk4_solver (0.005).solve_builtin ('lorenz', [10.0 28.0 8.0/3], [1.0 1.0 1.0], 0.0, 25.0);
printf ('steps_taken = %d\n', sol.steps_taken);
[~, series] = solution_series (sol, 1, 3);
p = datamunge.RPlot.plot (series{1}, series{2}, 'l', 'trajectory');
p.title ('Lorenz Attractor (x-z phase plane)').x_label ('x').y_label ('z');
p.save_svg ('ode_lorenz_phase_plane_octave.svg');
printf ('wrote ode_lorenz_phase_plane_octave.svg\n');
