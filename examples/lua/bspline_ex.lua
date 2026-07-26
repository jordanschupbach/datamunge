local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

-- Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
local x, y, z = {}, {}, {}
for iy = 0, 7 do
  for ix = 0, 7 do
    local xv, yv = ix / 7.0, iy / 7.0
    x[#x + 1] = xv
    y[#y + 1] = yv
    z[#z + 1] = math.sin(xv) + yv * yv
  end
end

local data = dm.DataFrame()
data:add_numeric_column("x", dv(x))
data:add_numeric_column("y", dv(y))
data:add_numeric_column("z", dv(z))

local surface = dm.LM(data, "z ~ bs(x, y)")
print("Fitted z ~ bs(x, y)")
surface:print_summary()

local new_points = dm.DataFrame()
new_points:add_numeric_column("x", dv({ 0.25, 0.75 }))
new_points:add_numeric_column("y", dv({ 0.50, 0.25 }))
local pred = surface:predict(new_points)
print(string.format("Predictions: [%.6f, %.6f]", pred[0], pred[1]))
