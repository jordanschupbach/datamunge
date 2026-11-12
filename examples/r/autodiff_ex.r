# CMD: nix develop .#r --command Rscript examples/r/autodiff_ex.r
# Note: the C++ autodiff_ex.cpp additionally demonstrates datamunge::autodiff::derivative/
# gradient_forward/gradient_reverse/jacobian_forward/hessian -- generic C++ template driver
# functions that work over any callable, but templates aren't bound in any scripting
# language. This port demonstrates the same ideas
# (forward-mode gradients via repeated single-seed Dual evaluations, reverse-mode gradients
# via one Tape/Var backward pass, Hessian entries via HyperDual) using only the bound
# named-method API.
library(datamunger)

rosenbrock_dual <- function(x0, x1) {
  a <- Dual$new(1.0, 0.0)$subtract(x0)
  b <- x1$subtract(x0$multiply(x0))
  a$multiply(a)$add(b$multiply(b)$multiply_scalar(100.0))
}

rosenbrock_value <- function(x0, x1) {
  a <- 1.0 - x0
  b <- x1 - x0 * x0
  a * a + 100.0 * b * b
}

cat("=================== Forward mode: scalar derivative ===================\n")
x0 <- Dual$new(2.0, 1.0) # seed derivative = 1 to read df/dx directly
f <- x0$multiply(x0)$multiply(x0)$subtract(x0$multiply_scalar(2.0)) # x^3 - 2x
cat("f(x) = x^3 - 2x, f'(2) =", f$derivative(), "(exact: 10)\n")

cat(
  "\n=================== Reverse mode: build a graph by hand ===================\n"
)
tape <- Tape$new()
a <- Var$new(tape, 2.0)
b <- Var$new(tape, 3.0)
y <- a$multiply(b)$add(a$sin())
cat("y = a*b + sin(a) at a=2, b=3 -> y =", y$value(), "\n")
adjoint <- tape$backward(y)
cat("dy/da =", adjoint[a$index() + 1], "(exact: b + cos(a))\n")
cat("dy/db =", adjoint[b$index() + 1], "(exact: a)\n")

cat(
  "\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n"
)
px <- 0.0
py <- 0.0

# Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
gx <- rosenbrock_dual(Dual$new(px, 1.0), Dual$new(py, 0.0))$derivative()
gy <- rosenbrock_dual(Dual$new(px, 0.0), Dual$new(py, 1.0))$derivative()

# Reverse mode: one Tape/Var pass computes every partial at once.
tape2 <- Tape$new()
vx <- Var$new(tape2, px)
vy <- Var$new(tape2, py)
va <- Var$new(tape2, 1.0)$subtract(vx)
vb <- vy$subtract(vx$multiply(vx))
vf <- va$multiply(va)$add(vb$multiply(vb)$multiply_scalar(100.0))
grad_rev <- tape2$backward(vf)

cat("f(0,0) =", rosenbrock_value(px, py), "\n")
cat("gradient (forward mode): [", gx, ",", gy, "]\n")
cat(
  "gradient (reverse mode): [",
  grad_rev[vx$index() + 1],
  ",",
  grad_rev[vy$index() + 1],
  "]\n"
)

cat(
  "\n=================== Jacobian of a vector-valued function ===================\n"
)
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
vx_val <- 2.0
vy_val <- 3.0

vector_fn <- function(x, y) list(
  x$multiply(x),
  x$multiply(y),
  y$multiply(y)$multiply(y)
)

jac <- matrix(0.0, nrow = 3, ncol = 2)
seeds <- list(c(1.0, 0.0), c(0.0, 1.0))
for (col in 1:2) {
  seed <- seeds[[col]]
  outputs <- vector_fn(Dual$new(vx_val, seed[1]), Dual$new(vy_val, seed[2]))
  for (row in 1:3) jac[row, col] <- outputs[[row]]$derivative()
}
cat("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n")
for (row in 1:3) cat(" ", jac[row, ], "\n")

cat(
  "\n=================== Hessian via second-order forward mode (HyperDual) ===================\n"
)
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
mx <- 1.0
my <- 1.0 # the Rosenbrock function's minimum

rosenbrock_hyperdual <- function(x0, x1) {
  a <- HyperDual$new(1.0, 0.0, 0.0, 0.0)$subtract(x0)
  b <- x1$subtract(x0$multiply(x0))
  a$multiply(a)$add(b$multiply(b)$multiply_scalar(100.0))
}

seeds <- list(c(1.0, 0.0), c(0.0, 1.0))
H <- matrix(0.0, nrow = 2, ncol = 2)
for (i in 1:2) {
  for (j in 1:2) {
    e1 <- seeds[[i]]
    e2 <- seeds[[j]]
    hx <- HyperDual$new(mx, e1[1], e2[1], 0.0)
    hy <- HyperDual$new(my, e1[2], e2[2], 0.0)
    H[i, j] <- rosenbrock_hyperdual(hx, hy)$eps1eps2()
  }
}
cat("Hessian of the Rosenbrock function at its minimum (1,1):\n")
for (row in 1:2) cat(" ", H[row, ], "\n")

cat(
  "\n=================== Gradient descent driven by reverse-mode gradients ===================\n"
)
point <- c(-1.2, 1.0) # the classic Rosenbrock starting point
learning_rate <- 0.0005
n_steps <- 100000
for (step in 0:(n_steps - 1)) {
  t <- Tape$new()
  vx <- Var$new(t, point[1])
  vy <- Var$new(t, point[2])
  va <- Var$new(t, 1.0)$subtract(vx)
  vb <- vy$subtract(vx$multiply(vx))
  vf <- va$multiply(va)$add(vb$multiply(vb)$multiply_scalar(100.0))
  grad <- t$backward(vf)
  loss <- vf$value()
  point[1] <- point[1] - learning_rate * grad[vx$index() + 1]
  point[2] <- point[2] - learning_rate * grad[vy$index() + 1]
  if (step == 0 || step == n_steps - 1) {
    cat(
      "step",
      step,
      ": loss =",
      loss,
      ", x = [",
      point[1],
      ",",
      point[2],
      "]\n"
    )
  }
}
cat("(true minimum is at [1, 1] with loss 0)\n")
