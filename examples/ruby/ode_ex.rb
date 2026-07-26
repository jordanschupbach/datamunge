require "octruby"

# NOTE: the Python ode example also demonstrates a live, user-supplied RHS by subclassing
# datamunge.RHS (a SWIG director). Although Ruby's binding supports directors in general,
# datamunge::ode::RHS is deliberately NOT given %feature("director") in octruby.i (only the
# Python binding enables it -- see rhs.hpp's doc comment), so an RHS subclassed in Ruby would
# never dispatch back into Ruby. This example therefore uses the built-in named systems via
# solve_builtin(), exactly as the Lua template does.

def dv(values)
  Datamunge::DVector.new(values)
end

def last_state(sol)
  sol.state_at(sol.size - 1)
end

puts "=================== Every named built-in system (no director needed) ==================="
solver = Datamunge::ODESolver.new
systems = [
  ["exponential_decay", [1.0], [1.0]],
  ["logistic_growth", [1.0, 1.0], [0.5]],
  ["harmonic_oscillator", [1.0], [1.0, 0.0]],
  ["van_der_pol", [1.0], [2.0, 0.0]],
  ["lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0]],
]
systems.each do |name, params, y0|
  sol = solver.solve_builtin(name, dv(params), dv(y0), 0.0, 1.0)
  puts format("%-22s steps=%-6d final state=%s", name, sol.steps_taken, last_state(sol).to_a.inspect)
end

puts "\n=================== Harmonic oscillator (energy conservation) ==================="
options = Datamunge::ODEOptions.new
options.method = Datamunge::StepMethod_RK4
options.step_size = 0.01
solver = Datamunge::ODESolver.new(options)
sol = solver.solve_builtin("harmonic_oscillator", dv([1.0]), dv([1.0, 0.0]), 0.0, 20.0)
final = last_state(sol)
x = final[0]
v = final[1]
puts format("x(20) = %.6f (cos(20) = %.6f)", x, Math.cos(20))
puts format("energy x^2+v^2 = %.6f (should stay near 1.0)", x * x + v * v)

t_series = (0...sol.size).map { |i| sol.time_at(i) }
x_series = (0...sol.size).map { |i| sol.state_at(i)[0] }
v_series = (0...sol.size).map { |i| sol.state_at(i)[1] }

plot = Datamunge::RPlot.plot(dv(t_series), dv(x_series), "l", "x(t)")
plot.lines(dv(t_series), dv(v_series), "v(t)")
plot.title("Harmonic Oscillator").x_label("t").y_label("state")
plot.save_svg("ode_harmonic_oscillator_rb.svg")
puts "wrote ode_harmonic_oscillator_rb.svg"

puts "\n=================== Lorenz attractor (phase plane) ==================="
options = Datamunge::ODEOptions.new
options.method = Datamunge::StepMethod_RK4
options.step_size = 0.005
solver = Datamunge::ODESolver.new(options)
sol = solver.solve_builtin("lorenz", dv([10.0, 28.0, 8.0 / 3.0]), dv([1.0, 1.0, 1.0]), 0.0, 25.0)
puts format("steps_taken = %d", sol.steps_taken)

xs = (0...sol.size).map { |i| sol.state_at(i)[0] }
zs = (0...sol.size).map { |i| sol.state_at(i)[2] }
plot = Datamunge::RPlot.plot(dv(xs), dv(zs), "l", "trajectory")
plot.title("Lorenz Attractor (x-z phase plane)").x_label("x").y_label("z")
plot.save_svg("ode_lorenz_phase_plane_rb.svg")
puts "wrote ode_lorenz_phase_plane_rb.svg"
