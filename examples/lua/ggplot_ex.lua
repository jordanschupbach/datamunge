-- Demonstrates dm.GGPlot -- the ggplot2-style grammar-of-graphics library.
local dm = require("datamunge")

local function rgb(r, g, b)
  local c = dm.RGB()
  c.r, c.g, c.b = r, g, b
  return c
end

local iris = dm.DataFrame.iris()

-- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
local scatter = dm.GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
scatter:geom_point()
scatter:labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
scatter:theme_minimal()
scatter:save_svg("lua_ggplot_point.svg")

-- + geom_smooth(): an lm() fit line, reusing stats::LM internally.
local smooth = dm.GGPlot(iris, "Sepal.Length", "Petal.Length")
smooth:geom_point(rgb(156, 163, 175), 2.5)
smooth:geom_smooth()
smooth:labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
smooth:save_svg("lua_ggplot_smooth.svg")

-- geom_bar(): counts a discrete column (stat = "count").
local bar = dm.GGPlot(iris, "Species")
bar:geom_bar()
bar:labs("Observations per Species", "Species", "Count")
bar:theme_bw()
bar:save_svg("lua_ggplot_bar.svg")

-- geom_boxplot(): grouped by a discrete x column.
local box = dm.GGPlot(iris, "Species", "Petal.Width")
box:geom_boxplot()
box:labs("Petal Width by Species", "Species", "Petal Width")
box:save_svg("lua_ggplot_boxplot.svg")

-- geom_histogram() + geom_density(): distribution of a single numeric column.
local hist = dm.GGPlot(iris, "Sepal.Length")
hist:geom_histogram(20)
hist:labs("Distribution of Sepal Length", "Sepal Length", "Count")
hist:save_svg("lua_ggplot_histogram.svg")

local density = dm.GGPlot(iris, "Sepal.Length")
density:geom_density()
density:labs("Density of Sepal Length", "Sepal Length", "Density")
density:save_svg("lua_ggplot_density.svg")

-- facet_wrap(): one panel per Species, composed via RLayout under the hood.
local faceted = dm.GGPlot(iris, "Petal.Length", "Petal.Width")
faceted:geom_point()
faceted:facet_wrap("Species")
faceted:labs("Petal Dimensions", "Petal Length", "Petal Width")
faceted:save_svg("lua_ggplot_facet.svg")

-- theme_classic(): another built-in theme. (scale_color_manual is omitted here -- the
-- std::vector<RGB> overload it needs is not part of the Lua binding's exposed surface.)
local classic = dm.GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
classic:geom_point()
classic:theme_classic()
classic:save_svg("lua_ggplot_classic.svg")

print("Wrote 7 SVGs to lua_ggplot_*.svg")
