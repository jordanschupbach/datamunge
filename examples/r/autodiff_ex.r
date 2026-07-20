# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Note: the C++ autodiff_ex.cpp additionally demonstrates datamunge::autodiff::derivative/
# gradient_forward/gradient_reverse/jacobian_forward/hessian -- generic C++ template driver
# functions that work over any callable, but templates can't cross the SWIG boundary, so
# they aren't bound in any scripting language. This port demonstrates the same ideas
# (forward-mode gradients via repeated single-seed Dual evaluations, reverse-mode gradients
# via one Tape/Var backward pass, Hessian entries via HyperDual) using only the bound
# named-method API.
library(datamunger)

rosenbrock_dual <- function(x0, x1) {
  a <- Dual_subtract(Dual(1.0, 0.0), x0)
  b <- Dual_subtract(x1, Dual_multiply(x0, x0))
  Dual_add(Dual_multiply(a, a), Dual_multiply_scalar(Dual_multiply(b, b), 100.0))
}

rosenbrock_value <- function(x0, x1) {
  a <- 1.0 - x0
  b <- x1 - x0 * x0
  a * a + 100.0 * b * b
}

cat("=================== Forward mode: scalar derivative ===================\n")
x0 <- Dual(2.0, 1.0)  # seed derivative = 1 to read df/dx directly
f <- Dual_subtract(Dual_multiply(Dual_multiply(x0, x0), x0), Dual_multiply_scalar(x0, 2.0))  # x^3 - 2x
cat("f(x) = x^3 - 2x, f'(2) =", Dual_derivative(f), "(exact: 10)\n")

cat("\n=================== Reverse mode: build a graph by hand ===================\n")
tape <- Tape()
a <- Var(tape, 2.0)
b <- Var(tape, 3.0)
y <- Var_add(Var_multiply(a, b), Var_sin(a))
cat("y = a*b + sin(a) at a=2, b=3 -> y =", Var_value(y), "\n")
adjoint <- Tape_backward(tape, y)
cat("dy/da =", adjoint[Var_index(a) + 1], "(exact: b + cos(a))\n")
cat("dy/db =", adjoint[Var_index(b) + 1], "(exact: a)\n")

cat("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n")
px <- 0.0
py <- 0.0

# Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
gx <- Dual_derivative(rosenbrock_dual(Dual(px, 1.0), Dual(py, 0.0)))
gy <- Dual_derivative(rosenbrock_dual(Dual(px, 0.0), Dual(py, 1.0)))

# Reverse mode: one Tape/Var pass computes every partial at once.
tape2 <- Tape()
vx <- Var(tape2, px)
vy <- Var(tape2, py)
va <- Var_subtract(Var(tape2, 1.0), vx)
vb <- Var_subtract(vy, Var_multiply(vx, vx))
vf <- Var_add(Var_multiply(va, va), Var_multiply_scalar(Var_multiply(vb, vb), 100.0))
grad_rev <- Tape_backward(tape2, vf)

cat("f(0,0) =", rosenbrock_value(px, py), "\n")
cat("gradient (forward mode): [", gx, ",", gy, "]\n")
cat("gradient (reverse mode): [", grad_rev[Var_index(vx) + 1], ",", grad_rev[Var_index(vy) + 1], "]\n")

cat("\n=================== Jacobian of a vector-valued function ===================\n")
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
vx_val <- 2.0
vy_val <- 3.0

vector_fn <- function(x, y) list(Dual_multiply(x, x), Dual_multiply(x, y), Dual_multiply(Dual_multiply(y, y), y))

jac <- matrix(0.0, nrow = 3, ncol = 2)
seeds <- list(c(1.0, 0.0), c(0.0, 1.0))
for (col in 1:2) {
  seed <- seeds[[col]]
  outputs <- vector_fn(Dual(vx_val, seed[1]), Dual(vy_val, seed[2]))
  for (row in 1:3) jac[row, col] <- Dual_derivative(outputs[[row]])
}
cat("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n")
for (row in 1:3) cat(" ", jac[row, ], "\n")

cat("\n=================== Hessian via second-order forward mode (HyperDual) ===================\n")
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
mx <- 1.0
my <- 1.0  # the Rosenbrock function's minimum

rosenbrock_hyperdual <- function(x0, x1) {
  a <- HyperDual_subtract(HyperDual(1.0, 0.0, 0.0, 0.0), x0)
  b <- HyperDual_subtract(x1, HyperDual_multiply(x0, x0))
  HyperDual_add(HyperDual_multiply(a, a), HyperDual_multiply_scalar(HyperDual_multiply(b, b), 100.0))
}

seeds <- list(c(1.0, 0.0), c(0.0, 1.0))
H <- matrix(0.0, nrow = 2, ncol = 2)
for (i in 1:2) {
  for (j in 1:2) {
    e1 <- seeds[[i]]
    e2 <- seeds[[j]]
    hx <- HyperDual(mx, e1[1], e2[1], 0.0)
    hy <- HyperDual(my, e1[2], e2[2], 0.0)
    H[i, j] <- HyperDual_eps1eps2(rosenbrock_hyperdual(hx, hy))
  }
}
cat("Hessian of the Rosenbrock function at its minimum (1,1):\n")
for (row in 1:2) cat(" ", H[row, ], "\n")

cat("\n=================== Gradient descent driven by reverse-mode gradients ===================\n")
point <- c(-1.2, 1.0)  # the classic Rosenbrock starting point
learning_rate <- 0.0005
n_steps <- 100000
for (step in 0:(n_steps - 1)) {
  t <- Tape()
  vx <- Var(t, point[1])
  vy <- Var(t, point[2])
  va <- Var_subtract(Var(t, 1.0), vx)
  vb <- Var_subtract(vy, Var_multiply(vx, vx))
  vf <- Var_add(Var_multiply(va, va), Var_multiply_scalar(Var_multiply(vb, vb), 100.0))
  grad <- Tape_backward(t, vf)
  loss <- Var_value(vf)
  point[1] <- point[1] - learning_rate * grad[Var_index(vx) + 1]
  point[2] <- point[2] - learning_rate * grad[Var_index(vy) + 1]
  if (step == 0 || step == n_steps - 1) {
    cat("step", step, ": loss =", loss, ", x = [", point[1], ",", point[2], "]\n")
  }
}
cat("(true minimum is at [1, 1] with loss 0)\n")

