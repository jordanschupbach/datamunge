local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function to_table(v)
  local t = {}
  for i = 0, v:size() - 1 do t[i + 1] = v[i] end
  return t
end

local function last_state(sol) return sol:state_at(sol:size() - 1) end

-- NOTE: the Python/Ruby ode examples also demonstrate a live, user-supplied RHS by subclassing
-- datamunge.RHS (a SWIG director). Stock SWIG's Lua backend generates no director code, so RHS
-- cannot be subclassed from Lua; this example uses the built-in named systems via solve_builtin().

print("=================== Every named built-in system (no director needed) ===================")
local solver = dm.ODESolver()
local systems = {
  { "exponential_decay", { 1.0 }, { 1.0 } },
  { "logistic_growth", { 1.0, 1.0 }, { 0.5 } },
  { "harmonic_oscillator", { 1.0 }, { 1.0, 0.0 } },
  { "van_der_pol", { 1.0 }, { 2.0, 0.0 } },
  { "lorenz", { 10.0, 28.0, 8.0 / 3.0 }, { 1.0, 1.0, 1.0 } },
}
for _, s in ipairs(systems) do
  local sol = solver:solve_builtin(s[1], dv(s[2]), dv(s[3]), 0.0, 1.0)
  local parts = {}
  for _, x in ipairs(to_table(last_state(sol))) do parts[#parts + 1] = string.format("%g", x) end
  print(string.format("%-22s steps=%-6d final state=[%s]", s[1], sol.steps_taken, table.concat(parts, ", ")))
end

print("\n=================== Harmonic oscillator (energy conservation) ===================")
local options = dm.ODEOptions()
options.method = dm.StepMethod_RK4
options.step_size = 0.01
solver = dm.ODESolver(options)
local sol = solver:solve_builtin("harmonic_oscillator", dv({ 1.0 }), dv({ 1.0, 0.0 }), 0.0, 20.0)
local final = last_state(sol)
local x, v = final[0], final[1]
print(string.format("x(20) = %.6f (cos(20) = %.6f)", x, math.cos(20)))
print(string.format("energy x^2+v^2 = %.6f (should stay near 1.0)", x * x + v * v))

local t_series, x_series, v_series = {}, {}, {}
for i = 0, sol:size() - 1 do
  t_series[i + 1] = sol:time_at(i)
  local st = sol:state_at(i)
  x_series[i + 1] = st[0]
  v_series[i + 1] = st[1]
end
local plot = dm.RPlot.plot(dv(t_series), dv(x_series), "l", "x(t)")
plot:lines(dv(t_series), dv(v_series), "v(t)")
plot:title("Harmonic Oscillator"):x_label("t"):y_label("state")
plot:save_svg("ode_harmonic_oscillator_lua.svg")
print("wrote ode_harmonic_oscillator_lua.svg")

print("\n=================== Lorenz attractor (phase plane) ===================")
options = dm.ODEOptions()
options.method = dm.StepMethod_RK4
options.step_size = 0.005
solver = dm.ODESolver(options)
sol = solver:solve_builtin("lorenz", dv({ 10.0, 28.0, 8.0 / 3.0 }), dv({ 1.0, 1.0, 1.0 }), 0.0, 25.0)
print(string.format("steps_taken = %d", sol.steps_taken))

local xs, zs = {}, {}
for i = 0, sol:size() - 1 do
  local st = sol:state_at(i)
  xs[i + 1] = st[0]
  zs[i + 1] = st[2]
end
plot = dm.RPlot.plot(dv(xs), dv(zs), "l", "trajectory")
plot:title("Lorenz Attractor (x-z phase plane)"):x_label("x"):y_label("z")
plot:save_svg("ode_lorenz_phase_plane_lua.svg")
print("wrote ode_lorenz_phase_plane_lua.svg")
