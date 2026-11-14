# CMD: just TARGET=ode_ex run-ruby
# ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
require "datamunge"

def last_state(sol) = sol.state_at(sol.size - 1)

# The time vector followed by the requested state components (0-based), one value per step.
def solution_series(sol, *components)
  states = Array.new(sol.size) { |i| sol.state_at(i) }
  [Array.new(sol.size) { |i| sol.time_at(i) }, components.map { |c| states.map { |s| s[c] } }]
end

def rk4_solver(step)
  options = Datamunge::ODEOptions.new
  options.method_ = :RK4
  options.step_size = step
  Datamunge::ODESolver.new(options)
end

# NOTE: like the Lua version this example uses the built-in named systems via solve_builtin.

puts "=================== Every named built-in system (no director needed) ==================="
solver = Datamunge::ODESolver.new
[
  ["exponential_decay", [1.0], [1.0]],
  ["logistic_growth", [1.0, 1.0], [0.5]],
  ["harmonic_oscillator", [1.0], [1.0, 0.0]],
  ["van_der_pol", [1.0], [2.0, 0.0]],
  ["lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0]],
].each do |name, params, y0|
  sol = solver.solve_builtin(name, params, y0, 0.0, 1.0)
  state = last_state(sol).map { |v| format("%g", v) }.join(", ")
  puts format("%-22s steps=%-6d final state=[%s]", name, sol.steps_taken, state)
end

puts "\n=================== Harmonic oscillator (energy conservation) ==================="
sol = rk4_solver(0.01).solve_builtin("harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0)
x, v = last_state(sol)
printf("x(20) = %.6f (cos(20) = %.6f)\n", x, Math.cos(20.0))
printf("energy x^2+v^2 = %.6f (should stay near 1.0)\n", x * x + v * v)
ts, (xs, vs) = solution_series(sol, 0, 1)
p = Datamunge::RPlot.plot(ts, xs, "l", "x(t)")
p.lines(ts, vs, "v(t)")
p.title("Harmonic Oscillator").x_label("t").y_label("state")
p.save_svg("ode_harmonic_oscillator_ruby.svg")
puts "wrote ode_harmonic_oscillator_ruby.svg"

puts "\n=================== Lorenz attractor (phase plane) ==================="
sol = rk4_solver(0.005).solve_builtin("lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0], 0.0, 25.0)
puts "steps_taken = #{sol.steps_taken}"
_, (xs, zs) = solution_series(sol, 0, 2)
p = Datamunge::RPlot.plot(xs, zs, "l", "trajectory")
p.title("Lorenz Attractor (x-z phase plane)").x_label("x").y_label("z")
p.save_svg("ode_lorenz_phase_plane_ruby.svg")
puts "wrote ode_lorenz_phase_plane_ruby.svg"
