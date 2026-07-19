require "octruby"

# Note: the C++ autodiff_ex.cpp additionally demonstrates datamunge::autodiff::derivative/
# gradient_forward/gradient_reverse/jacobian_forward/hessian -- generic C++ template driver
# functions that work over any callable, but templates can't cross the SWIG boundary, so
# they aren't bound in any scripting language. This port demonstrates the same ideas
# (forward-mode gradients via repeated single-seed Dual evaluations, reverse-mode gradients
# via one Tape/Var backward pass, Hessian entries via HyperDual) using only the bound
# named-method API.

def rosenbrock_dual(x0, x1)
  a = Datamunge::Dual.new(1.0, 0.0).subtract(x0)
  b = x1.subtract(x0.multiply(x0))
  a.multiply(a).add(b.multiply(b).multiply_scalar(100.0))
end

def rosenbrock_value(x0, x1)
  a = 1.0 - x0
  b = x1 - x0 * x0
  a * a + 100.0 * b * b
end

puts "=================== Forward mode: scalar derivative ==================="
x0 = Datamunge::Dual.new(2.0, 1.0)  # seed derivative = 1 to read df/dx directly
f = x0.multiply(x0).multiply(x0).subtract(x0.multiply_scalar(2.0))  # x^3 - 2x
puts "f(x) = x^3 - 2x, f'(2) = #{f.derivative} (exact: 10)"

puts "\n=================== Reverse mode: build a graph by hand ==================="
tape = Datamunge::Tape.new
a = Datamunge::Var.new(tape, 2.0)
b = Datamunge::Var.new(tape, 3.0)
y = a.multiply(b).add(a.sin)
puts "y = a*b + sin(a) at a=2, b=3 -> y = #{y.value}"
adjoint = tape.backward(y)
puts "dy/da = #{adjoint[a.index]} (exact: b + cos(a))"
puts "dy/db = #{adjoint[b.index]} (exact: a)"

puts "\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n"
px = 0.0
py = 0.0

# Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
gx = rosenbrock_dual(Datamunge::Dual.new(px, 1.0), Datamunge::Dual.new(py, 0.0)).derivative
gy = rosenbrock_dual(Datamunge::Dual.new(px, 0.0), Datamunge::Dual.new(py, 1.0)).derivative

# Reverse mode: one Tape/Var pass computes every partial at once.
tape2 = Datamunge::Tape.new
vx = Datamunge::Var.new(tape2, px)
vy = Datamunge::Var.new(tape2, py)
va = Datamunge::Var.new(tape2, 1.0).subtract(vx)
vb = vy.subtract(vx.multiply(vx))
vf = va.multiply(va).add(vb.multiply(vb).multiply_scalar(100.0))
grad_rev = tape2.backward(vf)

puts "f(0,0) = #{rosenbrock_value(px, py)}"
puts "gradient (forward mode): [#{gx}, #{gy}]"
puts "gradient (reverse mode): [#{grad_rev[vx.index]}, #{grad_rev[vy.index]}]"

puts "\n=================== Jacobian of a vector-valued function ==================="
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
vals = { "x" => 2.0, "y" => 3.0 }

vector_fn = lambda do |x, y|
  [x.multiply(x), x.multiply(y), y.multiply(y).multiply(y)]
end

jac = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
[[0, 1.0, 0.0], [1, 0.0, 1.0]].each do |col, seed_x, seed_y|
  outputs = vector_fn.call(Datamunge::Dual.new(vals["x"], seed_x), Datamunge::Dual.new(vals["y"], seed_y))
  outputs.each_with_index { |out, row| jac[row][col] = out.derivative }
end
puts "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:"
jac.each { |row| puts "  #{row}" }

puts "\n=================== Hessian via second-order forward mode (HyperDual) ===================\n"
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
mx = 1.0
my = 1.0  # the Rosenbrock function's minimum

rosenbrock_hyperdual = lambda do |x0, x1|
  a = Datamunge::HyperDual.new(1.0, 0.0, 0.0, 0.0).subtract(x0)
  b = x1.subtract(x0.multiply(x0))
  a.multiply(a).add(b.multiply(b).multiply_scalar(100.0))
end

seeds = [[1.0, 0.0], [0.0, 1.0]]
h = [[0.0, 0.0], [0.0, 0.0]]
seeds.each_with_index do |(e1x, e1y), i|
  seeds.each_with_index do |(e2x, e2y), j|
    hx = Datamunge::HyperDual.new(mx, e1x, e2x, 0.0)
    hy = Datamunge::HyperDual.new(my, e1y, e2y, 0.0)
    h[i][j] = rosenbrock_hyperdual.call(hx, hy).eps1eps2
  end
end
puts "Hessian of the Rosenbrock function at its minimum (1,1):"
h.each { |row| puts "  #{row}" }

puts "\n=================== Gradient descent driven by reverse-mode gradients ===================\n"
point = [-1.2, 1.0]  # the classic Rosenbrock starting point
learning_rate = 0.001
n_steps = 2000
n_steps.times do |step|
  t = Datamunge::Tape.new
  vx = Datamunge::Var.new(t, point[0])
  vy = Datamunge::Var.new(t, point[1])
  va = Datamunge::Var.new(t, 1.0).subtract(vx)
  vb = vy.subtract(vx.multiply(vx))
  vf = va.multiply(va).add(vb.multiply(vb).multiply_scalar(100.0))
  grad = t.backward(vf)
  loss = vf.value
  point[0] -= learning_rate * grad[vx.index]
  point[1] -= learning_rate * grad[vy.index]
  puts "step #{step}: loss = #{loss}, x = [#{point[0]}, #{point[1]}]" if step == 0 || step == n_steps - 1
end
puts "(true minimum is at [1, 1] with loss 0)"
