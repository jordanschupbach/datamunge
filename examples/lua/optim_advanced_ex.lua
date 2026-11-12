-- Exercises the 10 new datamunge::optim classes (ProximalGradient, FISTA,
-- LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP,
-- InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Lua.
--
-- Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
-- EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
-- ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
-- implement the objective. From Lua, subclass the interface with Class:extend{...}; the
-- optimizer then calls the Lua methods. Coordinates are updated in place through a dm.ref()
-- cell (the C++ parameter is a non-const std::vector<double>&).

local dm = require("datamunge")

local function dv(t)
  local v = {}
  for i, x in ipairs(t) do v[i] = x end
  return v
end

local function dvv(rows)
  local m = {}
  for i, row in ipairs(rows) do m[i] = dv(row) end
  return m
end

local function to_table(v)
  local t = {}
  for i = 1, #v do t[i] = v[i] end
  return t
end

print("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================")
-- Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
-- then check the acquisition function favors an unexplored region.
local xs = { 0.0, 0.5, 1.0, 1.5, 2.0, 3.0 }
local ys = {}
for i, x in ipairs(xs) do ys[i] = (x - 1.0) ^ 2 end

local surrogate = dm.RBFGaussianProcessSurrogate(1.0, 1e-6)
local points = dvv((function()
  local rows = {}
  for i, x in ipairs(xs) do rows[i] = { x } end
  return rows
end)())
surrogate:fit(points, dv(ys))

local incumbent = 0.0 -- best observed value so far (min of ys)
for _, y in ipairs(ys) do if y < incumbent then incumbent = y end end
print("incumbent (best observed y): " .. incumbent)
for _, x in ipairs({ 0.25, 1.0, 2.5 }) do
  local acq = surrogate:acquisition(dv({ x }), incumbent)
  print(("acquisition(x=%.2f) = %.6f"):format(x, acq))
end

print("\n=================== Options structs + optimizer construction (all 10 new classes) ===================")
-- Every new Options struct and optimizer class constructs cleanly, proving
-- the SWIG binding for the whole batch is wired correctly.
local pg_opts = dm.ProximalGradientOptions()
pg_opts.step_size = 0.05
local pg = dm.ProximalGradient(pg_opts)
print("ProximalGradient: " .. tostring(pg))

local fista_opts = dm.FISTAOptions()
local fista = dm.FISTA(fista_opts)
print("FISTA: " .. tostring(fista))

local lm_opts = dm.LevenbergMarquardtOptions()
lm_opts.max_iterations = 200
local lm = dm.LevenbergMarquardt(lm_opts)
print("LevenbergMarquardt: " .. tostring(lm))

local newton_opts = dm.NewtonOptions()
local newton = dm.Newton(newton_opts)
print("Newton: " .. tostring(newton))

local trn_opts = dm.TrustRegionNewtonOptions()
local trn = dm.TrustRegionNewton(trn_opts)
print("TrustRegionNewton: " .. tostring(trn))

local al_opts = dm.AugmentedLagrangianOptions()
local al = dm.AugmentedLagrangian(al_opts)
print("AugmentedLagrangian: " .. tostring(al))

local sqp_opts = dm.SQPOptions()
local sqp = dm.SQP(sqp_opts)
print("SQP: " .. tostring(sqp))

local ip_opts = dm.InteriorPointOptions()
local ip = dm.InteriorPoint(ip_opts)
print("InteriorPoint: " .. tostring(ip))

local bo_opts = dm.BayesianOptimizationOptions()
bo_opts.initial_samples = 8
local bo = dm.BayesianOptimization(bo_opts)
print("BayesianOptimization: " .. tostring(bo))

print("\n=================== Lua objectives (subclassing the C++ interfaces) ===================")
-- Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
local Quadratic = dm.HessianFunction:extend{}
function Quadratic:evaluate(c) return (c[1] - 1) ^ 2 + 10 * (c[2] + 2) ^ 2 end
function Quadratic:gradient(c) return { 2 * (c[1] - 1), 20 * (c[2] + 2) } end
function Quadratic:hessian() return { { 2, 0 }, { 0, 20 } } end
local start = dm.ref({ 5.0, 5.0 })
local value = newton:optimize(Quadratic(), start)
print(("Newton: f = %.6g at (%.6f, %.6f) (exact minimum 0 at (1, -2))"):format(value, start.value[1], start.value[2]))

-- Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
local ts = { 0.0, 0.5, 1.0, 1.5, 2.0 }
local Exponential = dm.ResidualFunction:extend{}
function Exponential:residuals(p)
  local r = {}
  for i, t in ipairs(ts) do r[i] = p[1] * math.exp(p[2] * t) - 2 * math.exp(0.5 * t) end
  return r
end
function Exponential:jacobian(p)
  local rows = {}
  for i, t in ipairs(ts) do rows[i] = { math.exp(p[2] * t), p[1] * t * math.exp(p[2] * t) } end
  return rows
end
local params = dm.ref({ 1.0, 0.1 })
local sse = lm:optimize(Exponential(), params)
print(("LevenbergMarquardt: residual = %.3g, a = %.6f, b = %.6f (exact: 2, 0.5)"):format(sse, params.value[1], params.value[2]))

-- An interface method a subclass leaves out raises a Lua error when the optimizer calls it.
local Incomplete = dm.HessianFunction:extend{ evaluate = function() return 0 end }
local ok, err = pcall(function() return newton:optimize(Incomplete(), dm.ref({ 0.0, 0.0 })) end)
print("missing gradient(): ok=" .. tostring(ok) .. ", error mentions it: " .. tostring(tostring(err):find("gradient", 1, true) ~= nil))
