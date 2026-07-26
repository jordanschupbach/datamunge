require "octruby"

# Demonstrates Datamunge::RPlot / Datamunge::RLayout -- the R-base-graphics-style plotting library.

def dv(values)
  Datamunge::DVector.new(values.map(&:to_f))
end

def sv(values)
  Datamunge::SVector.new(values)
end

def dvv(rows)
  m = Datamunge::DVectorVector.new(rows.length)
  rows.each_with_index { |r, i| m[i] = dv(r) }
  m
end

def rgb(r, g, b)
  color = Datamunge::RGB.new
  color.r = r
  color.g = g
  color.b = b
  color
end

# curve() samples a Callback (a SWIG director) over a range. Unlike Lua, Ruby's binding
# supports directors, so we subclass Datamunge::Callback directly.
class SineWave < Datamunge::Callback
  def call(x)
    Math.sin(x)
  end
end

# plot(x, y, type = "p") then abline() layered on afterward.
scatter = Datamunge::RPlot.plot(dv([1.0, 2.0, 3.0, 4.0, 5.0]), dv([2.1, 3.9, 6.2, 7.8, 10.1]), "p", "observed")
scatter.abline(0.0, 2.0, rgb(220, 38, 38), 1.5)
scatter.title("plot() + abline()").x_label("x").y_label("y")
scatter.save_svg("rb_rplot_scatter.svg")

# hist(): equal-width binning over the data range.
hist = Datamunge::RPlot.hist(dv([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7]), 6, "counts")
hist.save_svg("rb_rplot_hist.svg")

# barplot(): categorical positions with x tick labels.
bars = Datamunge::RPlot.barplot(dv([23.0, 41.0, 12.0]), sv(["A", "B", "C"]))
bars.title("barplot()")
bars.save_svg("rb_rplot_barplot.svg")

# boxplot(): Tukey five-number summary per group.
groups = dvv([[2, 4, 4, 4, 5, 5, 7, 9], [1, 2, 2, 2, 3, 3, 3, 3, 4, 20]])
box = Datamunge::RPlot.boxplot(groups, sv(["low variance", "has outlier"]))
box.title("boxplot()")
box.save_svg("rb_rplot_boxplot.svg")

# pie(): wedge areas proportional to value, axes hidden automatically.
pie = Datamunge::RPlot.pie(dv([35.0, 25.0, 20.0, 20.0]), sv(["Q1", "Q2", "Q3", "Q4"]))
pie.title("pie()")
pie.save_svg("rb_rplot_pie.svg")

# curve(): samples a Ruby-subclassed Callback director over a range.
sine = SineWave.new
curve = Datamunge::RPlot.curve(sine, 0.0, 2.0 * Math::PI, 200, "sin(x)")
curve.title("curve()")
curve.save_svg("rb_rplot_curve.svg")

# qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3]
qq = Datamunge::RPlot.qqnorm(dv(residuals))
qq.qqline(dv(residuals))
qq.save_svg("rb_rplot_qqnorm.svg")

# par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
layout = Datamunge::RLayout.create(1, 2)
layout.add(scatter)
layout.add(hist)
layout.save_svg("rb_rplot_layout.svg")

puts "Wrote 8 SVGs to rb_rplot_*.svg"
