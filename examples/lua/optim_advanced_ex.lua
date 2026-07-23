-- Exercises the 10 new datamunge::optim classes (ProximalGradient, FISTA,
-- LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP,
-- InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Lua.
--
-- IMPORTANT FINDING: every one of these 10 optimizers takes a *director*
-- interface (ProximalFunction / HessianFunction / EqualityConstrainedFunction
-- / InequalityConstrainedFunction / ResidualFunction, or plain
-- ArbitraryFunction for BayesianOptimization's objective) as its first
-- argument, and the datamunge C++ library ships NO concrete implementation
-- of any of them -- every language's example defines its own objective by
-- subclassing. Lua's SWIG backend is stock upstream SWIG (not the swig-jse
-- fork other languages use here), and stock SWIG's Lua module
-- (Source/Modules/lua.cxx) has ZERO director code generation -- confirmed by
-- grepping the generated build/datamungelua/datamunge_lua_wrap.cxx for
-- "director": 0 matches, vs. 177/156 for this project's Ruby/Perl wrappers.
-- Concretely: these abstract classes get no constructor wrapper at all
-- (`_wrap_class_HessianFunction` etc. has a literal `0` in the constructor
-- slot), so `dm.HessianFunction()` fails with
-- "attempt to call a table value (field 'HessianFunction')" -- there is no
-- way to subclass them from Lua. This is a pre-existing, whole-module gap
-- (already noted for the original ArbitraryFunction/DifferentiableFunction
-- family in this project's Lua notes) and applies identically to all 10 new
-- classes: none of them can be driven with a custom objective from Lua.
--
-- What CAN be verified from Lua: the binding wired in cleanly (every new
-- class/Options-struct constructs without error) and RBFGaussianProcessSurrogate
-- -- the one NEW class that is concrete (not abstract) -- works correctly
-- end to end.

local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function dvv(rows)
  local m = dm.DVectorVector(#rows)
  for i, row in ipairs(rows) do m[i - 1] = dv(row) end
  return m
end

local function to_table(v)
  local t = {}
  for i = 0, v:size() - 1 do t[i + 1] = v[i] end
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

print("\n=================== Director/subclass check (the load-bearing capability) ===================")
-- Every optimizer above needs a user-supplied director-interface object
-- (HessianFunction, ResidualFunction, ProximalFunction, ArbitraryFunction,
-- ...) as its objective/residual/constraint argument. Confirm empirically
-- that these interfaces cannot even be instantiated from Lua, let alone
-- subclassed and called back into.
for _, name in ipairs({ "HessianFunction", "ResidualFunction", "ProximalFunction",
                        "ArbitraryFunction", "EqualityConstrainedFunction",
                        "InequalityConstrainedFunction" }) do
  local ok, err = pcall(function() return dm[name]() end)
  print(("dm.%s(): ok=%s err=%s"):format(name, tostring(ok), tostring(err)))
end
print([[
==> CONCLUSION: Lua's stock-SWIG backend generates zero director code
    (Source/Modules/lua.cxx has no director support at all, unlike the
    swig-jse fork used by Python/Ruby/Perl/etc. in this project). None of
    ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
    AugmentedLagrangian, SQP, or InteriorPoint can be run end-to-end from
    Lua, since all of them require a user-defined subclass of a pure
    interface type that Lua cannot even instantiate (no constructor is
    generated for abstract director-eligible classes without director
    support). BayesianOptimization is in the same boat for its `ArbitraryFunction&`
    objective argument, even though its surrogate (RBFGaussianProcessSurrogate)
    is a ready-to-use concrete class that DOES work, as demonstrated above.
    This is not a bug introduced by this batch of 10 classes -- it is the
    same pre-existing, whole-optim-module limitation already recorded for
    Lua (matches R, C#, PHP, Go, Java as non-director languages here).]])
