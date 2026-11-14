# CMD: just TARGET=autodiff_ex run-ruby
# Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
require "datamunge"
include Datamunge

# Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
# (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
# bound. This port demonstrates the same ideas with the bound named-method API.
# Var#index is the Var's position in the adjoint vector Tape#backward returns.

def rosenbrock_dual(x0, x1)
  a = Dual.new(1.0, 0.0).subtract(x0)
  b = x1.subtract(x0.multiply(x0))
  a.multiply(a).add(b.multiply(b).multiply_scalar(100.0))
end

def rosenbrock_value(x0, x1) = (1.0 - x0)**2 + 100.0 * (x1 - x0 * x0)**2

# Build the Rosenbrock graph on tape; returns the output Var.
def rosenbrock_var(tape, x0, x1)
  a = Var.new(tape, 1.0).subtract(x0)
  b = x1.subtract(x0.multiply(x0))
  a.multiply(a).add(b.multiply(b).multiply_scalar(100.0))
end

def rosenbrock_hyper_dual(x0, x1)
  a = HyperDual.new(1.0, 0.0, 0.0, 0.0).subtract(x0)
  b = x1.subtract(x0.multiply(x0))
  a.multiply(a).add(b.multiply(b).multiply_scalar(100.0))
end

def adjoint(adjoints, v) = adjoints[v.index]

puts "=================== Forward mode: scalar derivative ==================="
x0 = Dual.new(2.0, 1.0) # seed derivative = 1 to read df/dx directly
f = x0.multiply(x0).multiply(x0).subtract(x0.multiply_scalar(2.0)) # x^3 - 2x
puts "f(x) = x^3 - 2x, f'(2) = #{f.derivative} (exact: 10)"

puts "\n=================== Reverse mode: build a graph by hand ==================="
tape = Tape.new
a = Var.new(tape, 2.0)
b = Var.new(tape, 3.0)
y = a.multiply(b).add(a.sin)
puts "y = a*b + sin(a) at a=2, b=3 -> y = #{y.value}"
adj = tape.backward(y)
puts "dy/da = #{adjoint(adj, a)} (exact: b + cos(a))"
puts "dy/db = #{adjoint(adj, b)} (exact: a)"

puts "\n=================== Forward vs reverse mode agree on the Rosenbrock function ==================="
px, py = 0.0, 0.0
# Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
gx = rosenbrock_dual(Dual.new(px, 1.0), Dual.new(py, 0.0)).derivative
gy = rosenbrock_dual(Dual.new(px, 0.0), Dual.new(py, 1.0)).derivative
# Reverse mode: one Tape/Var pass computes every partial at once.
tape = Tape.new
vx = Var.new(tape, px)
vy = Var.new(tape, py)
grad = tape.backward(rosenbrock_var(tape, vx, vy))
puts "f(0,0) = #{rosenbrock_value(px, py)}"
puts "gradient (forward mode): [#{gx}, #{gy}]"
puts "gradient (reverse mode): [#{adjoint(grad, vx)}, #{adjoint(grad, vy)}]"

puts "\n=================== Jacobian of a vector-valued function ==================="
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
vector_fn = ->(x, y) { [x.multiply(x), x.multiply(y), y.multiply(y).multiply(y)] }
jac = Array.new(3) { Array.new(2) }
[[0, 1.0, 0.0], [1, 0.0, 1.0]].each do |col, sx, sy|
  vector_fn.(Dual.new(2.0, sx), Dual.new(3.0, sy)).each_with_index { |out, row| jac[row][col] = out.derivative }
end
puts "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:"
jac.each { |r| puts "  [#{r[0]}, #{r[1]}]" }

puts "\n=================== Hessian via second-order forward mode (HyperDual) ==================="
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
mx, my = 1.0, 1.0 # the Rosenbrock function's minimum
seeds = [[1.0, 0.0], [0.0, 1.0]]
h = seeds.map do |e1|
  seeds.map do |e2|
    rosenbrock_hyper_dual(HyperDual.new(mx, e1[0], e2[0], 0.0), HyperDual.new(my, e1[1], e2[1], 0.0)).eps1eps2
  end
end
puts "Hessian of the Rosenbrock function at its minimum (1,1):"
h.each { |r| puts "  [#{r[0]}, #{r[1]}]" }

puts "\n=================== Gradient descent driven by reverse-mode gradients ==================="
point = [-1.2, 1.0] # the classic Rosenbrock start
learning_rate = 0.001
n_steps = 2000
n_steps.times do |step|
  tape = Tape.new
  vx = Var.new(tape, point[0])
  vy = Var.new(tape, point[1])
  vf = rosenbrock_var(tape, vx, vy)
  grad = tape.backward(vf)
  loss = vf.value
  point[0] -= learning_rate * adjoint(grad, vx)
  point[1] -= learning_rate * adjoint(grad, vy)
  puts "step #{step}: loss = #{loss}, x = [#{point[0]}, #{point[1]}]" if step.zero? || step == n_steps - 1
end
puts "(true minimum is at [1, 1] with loss 0)"
