# CMD: just TARGET=autodiff_ex run-julia

# Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
using Datamunge

# Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
# (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
# bound. This port demonstrates the same ideas with the bound named-method API.
# Native vectors are 0-based (Var indices), so adjoints are read at index(v) + 1.

function rosenbrock_dual(x0, x1)
    a = subtract(Dual(1.0, 0.0), x0)
    b = subtract(x1, multiply(x0, x0))
    add(multiply(a, a), multiply_scalar(multiply(b, b), 100.0))
end

rosenbrock_value(x0, x1) = (1.0 - x0)^2 + 100.0 * (x1 - x0 * x0)^2

"Build the Rosenbrock graph on `tape`; returns the output Var."
function rosenbrock_var(tape, x0, x1)
    a = subtract(Var(tape, 1.0), x0)
    b = subtract(x1, multiply(x0, x0))
    add(multiply(a, a), multiply_scalar(multiply(b, b), 100.0))
end

println("=================== Forward mode: scalar derivative ===================")
let x0 = Dual(2.0, 1.0)  # seed derivative = 1 to read df/dx directly
    f = subtract(multiply(multiply(x0, x0), x0), multiply_scalar(x0, 2.0))  # x^3 - 2x
    println("f(x) = x^3 - 2x, f'(2) = $(derivative(f)) (exact: 10)")
end

println("\n=================== Reverse mode: build a graph by hand ===================")
let tape = Tape(), a = Var(tape, 2.0), b = Var(tape, 3.0)
    y = add(multiply(a, b), sin(a))
    println("y = a*b + sin(a) at a=2, b=3 -> y = $(value(y))")
    adjoint = backward(tape, y)
    println("dy/da = $(adjoint[index(a) + 1]) (exact: b + cos(a))")
    println("dy/db = $(adjoint[index(b) + 1]) (exact: a)")
end

println("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================")
let px = 0.0, py = 0.0
    # Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
    gx = derivative(rosenbrock_dual(Dual(px, 1.0), Dual(py, 0.0)))
    gy = derivative(rosenbrock_dual(Dual(px, 0.0), Dual(py, 1.0)))
    # Reverse mode: one Tape/Var pass computes every partial at once.
    tape = Tape()
    vx = Var(tape, px)
    vy = Var(tape, py)
    grad = backward(tape, rosenbrock_var(tape, vx, vy))
    println("f(0,0) = $(rosenbrock_value(px, py))")
    println("gradient (forward mode): [$gx, $gy]")
    println("gradient (reverse mode): [$(grad[index(vx) + 1]), $(grad[index(vy) + 1])]")
end

println("\n=================== Jacobian of a vector-valued function ===================")
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
vector_fn(x, y) = [multiply(x, x), multiply(x, y), multiply(multiply(y, y), y)]
let jac = zeros(3, 2)
    for (col, sx, sy) in ((1, 1.0, 0.0), (2, 0.0, 1.0))
        for (row, output) in enumerate(vector_fn(Dual(2.0, sx), Dual(3.0, sy)))
            jac[row, col] = derivative(output)
        end
    end
    println("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:")
    for row in 1:3
        println("  [$(jac[row, 1]), $(jac[row, 2])]")
    end
end

println("\n=================== Hessian via second-order forward mode (HyperDual) ===================")
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
function rosenbrock_hyperdual(x0, x1)
    a = subtract(HyperDual(1.0, 0.0, 0.0, 0.0), x0)
    b = subtract(x1, multiply(x0, x0))
    add(multiply(a, a), multiply_scalar(multiply(b, b), 100.0))
end
let mx = 1.0, my = 1.0  # the Rosenbrock function's minimum
    seeds = ((1.0, 0.0), (0.0, 1.0))
    h = zeros(2, 2)
    for i in 1:2, j in 1:2
        (e1x, e1y), (e2x, e2y) = seeds[i], seeds[j]
        h[i, j] = eps1eps2(rosenbrock_hyperdual(HyperDual(mx, e1x, e2x, 0.0), HyperDual(my, e1y, e2y, 0.0)))
    end
    println("Hessian of the Rosenbrock function at its minimum (1,1):")
    for i in 1:2
        println("  [$(h[i, 1]), $(h[i, 2])]")
    end
end

println("\n=================== Gradient descent driven by reverse-mode gradients ===================")
let point = [-1.2, 1.0], learning_rate = 0.001, n_steps = 2000  # the classic Rosenbrock start
    for step in 0:n_steps-1
        tape = Tape()
        vx = Var(tape, point[1])
        vy = Var(tape, point[2])
        vf = rosenbrock_var(tape, vx, vy)
        grad = backward(tape, vf)
        loss = value(vf)
        point[1] -= learning_rate * grad[index(vx) + 1]
        point[2] -= learning_rate * grad[index(vy) + 1]
        if step == 0 || step == n_steps - 1
            println("step $step: loss = $loss, x = [$(point[1]), $(point[2])]")
        end
    end
    println("(true minimum is at [1, 1] with loss 0)")
end
