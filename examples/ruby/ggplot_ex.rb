require "octruby"

# Demonstrates Datamunge::GGPlot -- the ggplot2-style grammar-of-graphics library.

def rgb(r, g, b)
  color = Datamunge::RGB.new
  color.r = r
  color.g = g
  color.b = b
  color
end

iris = Datamunge::DataFrame.iris

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
scatter = Datamunge::GGPlot.new(iris, "Sepal.Length", "Sepal.Width", "Species")
scatter.geom_point
scatter.labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
scatter.theme_minimal
scatter.save_svg("rb_ggplot_point.svg")

# + geom_smooth(): an lm() fit line, reusing stats::LM internally.
smooth = Datamunge::GGPlot.new(iris, "Sepal.Length", "Petal.Length")
smooth.geom_point(rgb(156, 163, 175), 2.5)
smooth.geom_smooth
smooth.labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
smooth.save_svg("rb_ggplot_smooth.svg")

# geom_bar(): counts a discrete column (stat = "count").
bar = Datamunge::GGPlot.new(iris, "Species")
bar.geom_bar
bar.labs("Observations per Species", "Species", "Count")
bar.theme_bw
bar.save_svg("rb_ggplot_bar.svg")

# geom_boxplot(): grouped by a discrete x column.
box = Datamunge::GGPlot.new(iris, "Species", "Petal.Width")
box.geom_boxplot
box.labs("Petal Width by Species", "Species", "Petal Width")
box.save_svg("rb_ggplot_boxplot.svg")

# geom_histogram() + geom_density(): distribution of a single numeric column.
hist = Datamunge::GGPlot.new(iris, "Sepal.Length")
hist.geom_histogram(20)
hist.labs("Distribution of Sepal Length", "Sepal Length", "Count")
hist.save_svg("rb_ggplot_histogram.svg")

density = Datamunge::GGPlot.new(iris, "Sepal.Length")
density.geom_density
density.labs("Density of Sepal Length", "Sepal Length", "Density")
density.save_svg("rb_ggplot_density.svg")

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
faceted = Datamunge::GGPlot.new(iris, "Petal.Length", "Petal.Width")
faceted.geom_point
faceted.facet_wrap("Species")
faceted.labs("Petal Dimensions", "Petal Length", "Petal Width")
faceted.save_svg("rb_ggplot_facet.svg")

# theme_classic(): another built-in theme. (scale_color_manual is omitted here -- the
# std::vector<RGB> overload it needs has no %template(RGBVector) in the Ruby binding's
# exposed surface, so no ordered list of RGB colors can be constructed for it.)
classic = Datamunge::GGPlot.new(iris, "Sepal.Length", "Sepal.Width", "Species")
classic.geom_point
classic.theme_classic
classic.save_svg("rb_ggplot_classic.svg")

puts "Wrote 7 SVGs to rb_ggplot_*.svg"
