require "octruby"

# Fits a two-dimensional B-spline surface with the bs(x, y) formula term.

# Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
x, y, z = [], [], []
(0...8).each do |iy|
  (0...8).each do |ix|
    xv = ix / 7.0
    yv = iy / 7.0
    x << xv
    y << yv
    z << Math.sin(xv) + yv * yv
  end
end

data = Datamunge::DataFrame.new
data.add_numeric_column("x", x)
data.add_numeric_column("y", y)
data.add_numeric_column("z", z)

surface = Datamunge::LM.new(data, "z ~ bs(x, y)")
puts "Fitted z ~ bs(x, y)"
surface.print_summary

new_points = Datamunge::DataFrame.new
new_points.add_numeric_column("x", [0.25, 0.75])
new_points.add_numeric_column("y", [0.50, 0.25])
pred = surface.predict(new_points)
puts format("Predictions: [%.6f, %.6f]", pred[0], pred[1])
