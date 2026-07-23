import math

from pydatamunge import datamunge as dm


# A live, user-supplied RHS subclassing datamunge.RHS -- this director-based path only works in
# Python (and, nominally, Ruby): every other language binding can only use solve_builtin()'s
# fixed set of named systems. See rhs.hpp's doc comment for why.
class ExponentialDecay(dm.RHS):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def evaluate(self, t, y):
        return dm.DVector([-self.k * y[0]])


def last_state(sol):
    return sol.state_at(sol.size() - 1)


print("=================== dy/dt = -y, y(0) = 1 (live custom RHS) ===================")
decay = ExponentialDecay(1.0)
t_end = 2.0
exact = math.exp(-t_end)
print(f"exact y({t_end}) = {exact:.6f}")

configs = [
    ("Euler", dm.StepMethod_Euler, 4, 0.01, None, None),
    ("Midpoint/RK2", dm.StepMethod_Midpoint, 4, 0.01, None, None),
    ("RK4", dm.StepMethod_RK4, 4, 0.01, None, None),
    ("RK45 adaptive", dm.StepMethod_RK45, 4, 0.1, 1e-8, 1e-8),
    ("Adams-Bashforth", dm.StepMethod_AdamsBashforth, 4, 0.01, None, None),
    ("Adams-Bashforth-Moulton", dm.StepMethod_AdamsMoulton, 4, 0.01, None, None),
]
print(f"{'method':<28}{'y(2)':<14}{'abs.error':<14}{'steps':<8}evals")
for name, method, order, step, abs_tol, rel_tol in configs:
    options = dm.ODEOptions()
    options.method = method
    options.multistep_order = order
    options.step_size = step
    if abs_tol is not None:
        options.abs_tol = abs_tol
        options.rel_tol = rel_tol
    solver = dm.ODESolver(options)
    sol = solver.solve(decay, dm.DVector([1.0]), 0.0, t_end)
    y_final = last_state(sol)[0]
    print(f"{name:<28}{y_final:<14.6f}{abs(y_final - exact):<14.6f}{sol.steps_taken:<8}{sol.function_evaluations}")

print("\n=================== Every named built-in system (no director needed) ===================")
solver = dm.ODESolver()
for system, params, y0 in [
    ("exponential_decay", dm.DVector([1.0]), dm.DVector([1.0])),
    ("logistic_growth", dm.DVector([1.0, 1.0]), dm.DVector([0.5])),
    ("harmonic_oscillator", dm.DVector([1.0]), dm.DVector([1.0, 0.0])),
    ("van_der_pol", dm.DVector([1.0]), dm.DVector([2.0, 0.0])),
    ("lorenz", dm.DVector([10.0, 28.0, 8.0 / 3.0]), dm.DVector([1.0, 1.0, 1.0])),
]:
    sol = solver.solve_builtin(system, params, y0, 0.0, 1.0)
    print(f"{system:<22} steps={sol.steps_taken:<6} final state={list(last_state(sol))}")

print("\n=================== Harmonic oscillator (energy conservation) ===================")
options = dm.ODEOptions()
options.method = dm.StepMethod_RK4
options.step_size = 0.01
solver = dm.ODESolver(options)
sol = solver.solve_builtin("harmonic_oscillator", dm.DVector([1.0]), dm.DVector([1.0, 0.0]), 0.0, 20.0)
x, v = last_state(sol)
print(f"x(20) = {x:.6f} (cos(20) = {math.cos(20):.6f})")
print(f"energy x^2+v^2 = {x * x + v * v:.6f} (should stay near 1.0)")

t_series = [sol.time_at(i) for i in range(sol.size())]
x_series = [sol.state_at(i)[0] for i in range(sol.size())]
v_series = [sol.state_at(i)[1] for i in range(sol.size())]

plot = dm.RPlot.plot(dm.DVector(t_series), dm.DVector(x_series), "l", "x(t)")
plot.lines(dm.DVector(t_series), dm.DVector(v_series), "v(t)")
plot.title("Harmonic Oscillator").x_label("t").y_label("state")
plot.save_svg("ode_harmonic_oscillator_py.svg")
print("wrote ode_harmonic_oscillator_py.svg")

print("\n=================== Lorenz attractor (phase plane) ===================")
options = dm.ODEOptions()
options.method = dm.StepMethod_RK4
options.step_size = 0.005
solver = dm.ODESolver(options)
sol = solver.solve_builtin("lorenz", dm.DVector([10.0, 28.0, 8.0 / 3.0]), dm.DVector([1.0, 1.0, 1.0]), 0.0, 25.0)
print(f"steps_taken = {sol.steps_taken}")

xs = [sol.state_at(i)[0] for i in range(sol.size())]
zs = [sol.state_at(i)[2] for i in range(sol.size())]
plot = dm.RPlot.plot(dm.DVector(xs), dm.DVector(zs), "l", "trajectory")
plot.title("Lorenz Attractor (x-z phase plane)").x_label("x").y_label("z")
plot.save_svg("ode_lorenz_phase_plane_py.svg")
print("wrote ode_lorenz_phase_plane_py.svg")
