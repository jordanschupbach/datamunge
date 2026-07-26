-- Demonstrates dm.RPlot / dm.RLayout -- the R-base-graphics-style plotting library.
local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function sv(t)
  local v = dm.SVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function dvv(rows)
  local m = dm.DVectorVector(#rows)
  for i, r in ipairs(rows) do m[i - 1] = dv(r) end
  return m
end

local function rgb(r, g, b)
  local c = dm.RGB()
  c.r, c.g, c.b = r, g, b
  return c
end

-- plot(x, y, type = "p") then abline() layered on afterward.
local scatter = dm.RPlot.plot(dv({ 1.0, 2.0, 3.0, 4.0, 5.0 }), dv({ 2.1, 3.9, 6.2, 7.8, 10.1 }), "p", "observed")
scatter:abline(0.0, 2.0, rgb(220, 38, 38), 1.5)
scatter:title("plot() + abline()"):x_label("x"):y_label("y")
scatter:save_svg("lua_rplot_scatter.svg")

-- hist(): equal-width binning over the data range.
local hist = dm.RPlot.hist(dv({ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7 }), 6, "counts")
hist:save_svg("lua_rplot_hist.svg")

-- barplot(): categorical positions with x tick labels.
local bars = dm.RPlot.barplot(dv({ 23.0, 41.0, 12.0 }), sv({ "A", "B", "C" }))
bars:title("barplot()")
bars:save_svg("lua_rplot_barplot.svg")

-- boxplot(): Tukey five-number summary per group.
local groups = dvv({ { 2, 4, 4, 4, 5, 5, 7, 9 }, { 1, 2, 2, 2, 3, 3, 3, 3, 4, 20 } })
local box = dm.RPlot.boxplot(groups, sv({ "low variance", "has outlier" }))
box:title("boxplot()")
box:save_svg("lua_rplot_boxplot.svg")

-- pie(): wedge areas proportional to value, axes hidden automatically.
local pie = dm.RPlot.pie(dv({ 35.0, 25.0, 20.0, 20.0 }), sv({ "Q1", "Q2", "Q3", "Q4" }))
pie:title("pie()")
pie:save_svg("lua_rplot_pie.svg")

-- curve() is omitted here: it samples a Callback (a SWIG director), which the Lua binding
-- cannot subclass. Every other RPlot method below is director-free.

-- qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
local residuals = { -2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3 }
local qq = dm.RPlot.qqnorm(dv(residuals))
qq:qqline(dv(residuals))
qq:save_svg("lua_rplot_qqnorm.svg")

-- par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
local layout = dm.RLayout.create(1, 2)
layout:add(scatter)
layout:add(hist)
layout:save_svg("lua_rplot_layout.svg")

print("Wrote 7 SVGs to lua_rplot_*.svg")
