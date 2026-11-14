# CMD: just TARGET=ode_ex run-julia

# ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
using Datamunge
using Printf

"Approximate C's %g: six significant digits, no padding."
fmt_g(x) = @sprintf("%.6g", x)

last_state(sol) = state_at(sol, size(sol) - 1)

# NOTE: like the Lua version this example uses the built-in named systems via solve_builtin.

println("=================== Every named built-in system (no director needed) ===================")
let solver = ODESolver()
    for (name, params, y0) in (("exponential_decay", [1.0], [1.0]),
                               ("logistic_growth", [1.0, 1.0], [0.5]),
                               ("harmonic_oscillator", [1.0], [1.0, 0.0]),
                               ("van_der_pol", [1.0], [2.0, 0.0]),
                               ("lorenz", [10.0, 28.0, 8 / 3], [1.0, 1.0, 1.0]))
        sol = solve_builtin(solver, name, params, y0, 0.0, 1.0)
        println("$(rpad(name, 22)) steps=$(rpad(sol.steps_taken, 6)) final state=[$(join(fmt_g.(last_state(sol)), ", "))]")
    end
end

"The time vector followed by one vector per requested state component (1-based)."
function solution_series(sol, components...)
    n = size(sol)
    states = [state_at(sol, i) for i in 0:n-1]
    ([time_at(sol, i) for i in 0:n-1], (getindex.(states, c) for c in components)...)
end

function rk4_solver(step)
    options = ODEOptions()
    options.method = :RK4
    options.step_size = step
    ODESolver(options)
end

println("\n=================== Harmonic oscillator (energy conservation) ===================")
let sol = solve_builtin(rk4_solver(0.01), "harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0)
    x, v = last_state(sol)
    @printf("x(20) = %.6f (cos(20) = %.6f)\n", x, cos(20.0))
    @printf("energy x^2+v^2 = %.6f (should stay near 1.0)\n", x^2 + v^2)
    ts, xs, vs = solution_series(sol, 1, 2)
    p = plot(RPlot, ts, xs, "l", "x(t)")
    lines(p, ts, vs, "v(t)")
    y_label(x_label(title(p, "Harmonic Oscillator"), "t"), "state")
    save_svg(p, "ode_harmonic_oscillator_julia.svg")
    println("wrote ode_harmonic_oscillator_julia.svg")
end

println("\n=================== Lorenz attractor (phase plane) ===================")
let sol = solve_builtin(rk4_solver(0.005), "lorenz", [10.0, 28.0, 8 / 3], [1.0, 1.0, 1.0], 0.0, 25.0)
    println("steps_taken = $(sol.steps_taken)")
    _, xs, zs = solution_series(sol, 1, 3)
    p = plot(RPlot, xs, zs, "l", "trajectory")
    y_label(x_label(title(p, "Lorenz Attractor (x-z phase plane)"), "x"), "z")
    save_svg(p, "ode_lorenz_phase_plane_julia.svg")
    println("wrote ode_lorenz_phase_plane_julia.svg")
end
