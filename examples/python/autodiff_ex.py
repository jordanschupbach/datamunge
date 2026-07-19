# Note: the C++ autodiff_ex.cpp additionally demonstrates datamunge::autodiff::derivative/
# gradient_forward/gradient_reverse/jacobian_forward/hessian -- generic C++ template driver
# functions that work over any callable, but templates can't cross the SWIG boundary, so
# they aren't bound in any scripting language. This port demonstrates the same ideas
# (forward-mode gradients via repeated single-seed Dual evaluations, reverse-mode gradients
# via one Tape/Var backward pass, Hessian entries via HyperDual) using only the bound
# named-method API.
from pydatamunge import datamunge as dm


def rosenbrock_dual(x0, x1):
    a = dm.Dual(1.0, 0.0).subtract(x0)
    b = x1.subtract(x0.multiply(x0))
    return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0))


def rosenbrock_value(x0, x1):
    a = 1.0 - x0
    b = x1 - x0 * x0
    return a * a + 100.0 * b * b


print("=================== Forward mode: scalar derivative ===================")
x0 = dm.Dual(2.0, 1.0)  # seed derivative = 1 to read df/dx directly
f = x0.multiply(x0).multiply(x0).subtract(x0.multiply_scalar(2.0))  # x^3 - 2x
print(f"f(x) = x^3 - 2x, f'(2) = {f.derivative()} (exact: 10)")

print("\n=================== Reverse mode: build a graph by hand ===================")
tape = dm.Tape()
a = dm.Var(tape, 2.0)
b = dm.Var(tape, 3.0)
y = a.multiply(b).add(a.sin())
print(f"y = a*b + sin(a) at a=2, b=3 -> y = {y.value()}")
adjoint = tape.backward(y)
print(f"dy/da = {adjoint[a.index()]} (exact: b + cos(a))")
print(f"dy/db = {adjoint[b.index()]} (exact: a)")

print("\n=================== Forward vs reverse mode agree on the Rosenbrock function "
      "===================")
px, py = 0.0, 0.0

# Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
gx = rosenbrock_dual(dm.Dual(px, 1.0), dm.Dual(py, 0.0)).derivative()
gy = rosenbrock_dual(dm.Dual(px, 0.0), dm.Dual(py, 1.0)).derivative()

# Reverse mode: one Tape/Var pass computes every partial at once.
tape2 = dm.Tape()
vx = dm.Var(tape2, px)
vy = dm.Var(tape2, py)
va = dm.Var(tape2, 1.0).subtract(vx)
vb = vy.subtract(vx.multiply(vx))
vf = va.multiply(va).add(vb.multiply(vb).multiply_scalar(100.0))
grad_rev = tape2.backward(vf)

print(f"f(0,0) = {rosenbrock_value(px, py)}")
print(f"gradient (forward mode): [{gx}, {gy}]")
print(f"gradient (reverse mode): [{grad_rev[vx.index()]}, {grad_rev[vy.index()]}]")

print("\n=================== Jacobian of a vector-valued function ===================")
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
vals = {"x": 2.0, "y": 3.0}


def vector_fn(x, y):
    return [x.multiply(x), x.multiply(y), y.multiply(y).multiply(y)]


jac = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
for col, seed_x, seed_y in ((0, 1.0, 0.0), (1, 0.0, 1.0)):
    outputs = vector_fn(dm.Dual(vals["x"], seed_x), dm.Dual(vals["y"], seed_y))
    for row, out in enumerate(outputs):
        jac[row][col] = out.derivative()
print("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:")
for row in jac:
    print(" ", row)

print("\n=================== Hessian via second-order forward mode (HyperDual) ===================")
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
mx, my = 1.0, 1.0  # the Rosenbrock function's minimum


def rosenbrock_hyperdual(x0, x1):
    a = dm.HyperDual(1.0, 0.0, 0.0, 0.0).subtract(x0)
    b = x1.subtract(x0.multiply(x0))
    return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0))


seeds = [(1.0, 0.0), (0.0, 1.0)]
H = [[0.0, 0.0], [0.0, 0.0]]
for i, (e1x, e1y) in enumerate(seeds):
    for j, (e2x, e2y) in enumerate(seeds):
        hx = dm.HyperDual(mx, e1x, e2x, 0.0)
        hy = dm.HyperDual(my, e1y, e2y, 0.0)
        H[i][j] = rosenbrock_hyperdual(hx, hy).eps1eps2()
print("Hessian of the Rosenbrock function at its minimum (1,1):")
for row in H:
    print(" ", row)

print("\n=================== Gradient descent driven by reverse-mode gradients ===================")
point = [-1.2, 1.0]  # the classic Rosenbrock starting point
learning_rate = 0.001
n_steps = 2000
for step in range(n_steps):
    t = dm.Tape()
    vx = dm.Var(t, point[0])
    vy = dm.Var(t, point[1])
    va = dm.Var(t, 1.0).subtract(vx)
    vb = vy.subtract(vx.multiply(vx))
    vf = va.multiply(va).add(vb.multiply(vb).multiply_scalar(100.0))
    grad = t.backward(vf)
    loss = vf.value()
    point[0] -= learning_rate * grad[vx.index()]
    point[1] -= learning_rate * grad[vy.index()]
    if step == 0 or step == n_steps - 1:
        print(f"step {step}: loss = {loss}, x = [{point[0]}, {point[1]}]")
print("(true minimum is at [1, 1] with loss 0)")
