local dm = require("datamunge")

-- Note: the C++ autodiff_ex.cpp additionally demonstrates datamunge::autodiff::derivative/
-- gradient_forward/gradient_reverse/jacobian_forward/hessian -- generic C++ template driver
-- functions that work over any callable, but templates can't cross the SWIG boundary, so
-- they aren't bound in any scripting language. This port demonstrates the same ideas
-- (forward-mode gradients via repeated single-seed Dual evaluations, reverse-mode gradients
-- via one Tape/Var backward pass, Hessian entries via HyperDual) using only the bound
-- named-method API.

local function rosenbrock_dual(x0, x1)
  local a = dm.Dual(1.0, 0.0):subtract(x0)
  local b = x1:subtract(x0:multiply(x0))
  return a:multiply(a):add(b:multiply(b):multiply_scalar(100.0))
end

local function rosenbrock_value(x0, x1)
  local a = 1.0 - x0
  local b = x1 - x0 * x0
  return a * a + 100.0 * b * b
end

print("=================== Forward mode: scalar derivative ===================")
local x0 = dm.Dual(2.0, 1.0)  -- seed derivative = 1 to read df/dx directly
local f = x0:multiply(x0):multiply(x0):subtract(x0:multiply_scalar(2.0))  -- x^3 - 2x
print("f(x) = x^3 - 2x, f'(2) = " .. f:derivative() .. " (exact: 10)")

print("\n=================== Reverse mode: build a graph by hand ===================")
local tape = dm.Tape()
local a = dm.Var(tape, 2.0)
local b = dm.Var(tape, 3.0)
local y = a:multiply(b):add(a:sin())
print("y = a*b + sin(a) at a=2, b=3 -> y = " .. y:value())
local adjoint = tape:backward(y)
print("dy/da = " .. adjoint[a:index()] .. " (exact: b + cos(a))")
print("dy/db = " .. adjoint[b:index()] .. " (exact: a)")

print("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================")
local px, py = 0.0, 0.0

-- Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
local gx = rosenbrock_dual(dm.Dual(px, 1.0), dm.Dual(py, 0.0)):derivative()
local gy = rosenbrock_dual(dm.Dual(px, 0.0), dm.Dual(py, 1.0)):derivative()

-- Reverse mode: one Tape/Var pass computes every partial at once.
local tape2 = dm.Tape()
local vx = dm.Var(tape2, px)
local vy = dm.Var(tape2, py)
local va = dm.Var(tape2, 1.0):subtract(vx)
local vb = vy:subtract(vx:multiply(vx))
local vf = va:multiply(va):add(vb:multiply(vb):multiply_scalar(100.0))
local grad_rev = tape2:backward(vf)

print("f(0,0) = " .. rosenbrock_value(px, py))
print("gradient (forward mode): [" .. gx .. ", " .. gy .. "]")
print("gradient (reverse mode): [" .. grad_rev[vx:index()] .. ", " .. grad_rev[vy:index()] .. "]")

print("\n=================== Jacobian of a vector-valued function ===================")
-- f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
local vx_val, vy_val = 2.0, 3.0

local function vector_fn(x, y)
  return {x:multiply(x), x:multiply(y), y:multiply(y):multiply(y)}
end

local jac = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}
local cols = {{0, 1.0, 0.0}, {1, 0.0, 1.0}}
for _, c in ipairs(cols) do
  local col, seed_x, seed_y = c[1], c[2], c[3]
  local outputs = vector_fn(dm.Dual(vx_val, seed_x), dm.Dual(vy_val, seed_y))
  for row = 1, 3 do
    jac[row][col + 1] = outputs[row]:derivative()
  end
end
print("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:")
for _, row in ipairs(jac) do print("  [" .. row[1] .. ", " .. row[2] .. "]") end

print("\n=================== Hessian via second-order forward mode (HyperDual) ===================")
-- One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
local mx, my = 1.0, 1.0  -- the Rosenbrock function's minimum

local function rosenbrock_hyperdual(x0, x1)
  local a = dm.HyperDual(1.0, 0.0, 0.0, 0.0):subtract(x0)
  local b = x1:subtract(x0:multiply(x0))
  return a:multiply(a):add(b:multiply(b):multiply_scalar(100.0))
end

local seeds = {{1.0, 0.0}, {0.0, 1.0}}
local h = {{0.0, 0.0}, {0.0, 0.0}}
for i = 1, 2 do
  for j = 1, 2 do
    local e1x, e1y = seeds[i][1], seeds[i][2]
    local e2x, e2y = seeds[j][1], seeds[j][2]
    local hx = dm.HyperDual(mx, e1x, e2x, 0.0)
    local hy = dm.HyperDual(my, e1y, e2y, 0.0)
    h[i][j] = rosenbrock_hyperdual(hx, hy):eps1eps2()
  end
end
print("Hessian of the Rosenbrock function at its minimum (1,1):")
for _, row in ipairs(h) do print("  [" .. row[1] .. ", " .. row[2] .. "]") end

print("\n=================== Gradient descent driven by reverse-mode gradients ===================")
local point = {-1.2, 1.0}  -- the classic Rosenbrock starting point
local learning_rate = 0.001
local n_steps = 2000
for step = 0, n_steps - 1 do
  local t = dm.Tape()
  local vx2 = dm.Var(t, point[1])
  local vy2 = dm.Var(t, point[2])
  local va2 = dm.Var(t, 1.0):subtract(vx2)
  local vb2 = vy2:subtract(vx2:multiply(vx2))
  local vf2 = va2:multiply(va2):add(vb2:multiply(vb2):multiply_scalar(100.0))
  local grad = t:backward(vf2)
  local loss = vf2:value()
  point[1] = point[1] - learning_rate * grad[vx2:index()]
  point[2] = point[2] - learning_rate * grad[vy2:index()]
  if step == 0 or step == n_steps - 1 then
    print("step " .. step .. ": loss = " .. loss .. ", x = [" .. point[1] .. ", " .. point[2] .. "]")
  end
end
print("(true minimum is at [1, 1] with loss 0)")
